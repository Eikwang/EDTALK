"""Render loop for realtime streaming inference.

Key design: producer-consumer decoupling via FrameBuffer/OutputStreamer.
----------------------------------------------------------------------------
- RenderLoop (producer) renders frames as fast as possible and pushes them
  into a bounded FrameBuffer. It does NOT pace the camera output.
- OutputStreamer (consumer) owns the single output pacing clock and pulls
  frames from the FrameBuffer at a steady target_fps.
- Audio-driven lip coefficients are chosen by wall-clock audio position with
  a forward-looking buffer-depth compensation, so the lip shape displayed
  ~depth/fps seconds from now matches the audio that will be heard then.
- Audio playback is triggered by OutputStreamer when it pops the frame
  marked as the start of a new audio segment.
"""
import time
import threading
from typing import Optional

import numpy as np
import torch


class RenderLoop:
    """Main rendering loop that produces frames as fast as GPU allows.

    The loop continuously renders frames and pushes them into a FrameBuffer.
    Output pacing is handled by OutputStreamer, which consumes from the same
    buffer.

    Thread model: Runs in a separate thread, independent of API server and
    the OutputStreamer consumer thread.
    """

    # Seconds to keep holding the last lip coefficient after the audio
    # finishes before returning to idle (closed mouth) mode.
    AUDIO_END_GRACE_S = 1.0

    # Lip frames per Audio2Lip chunk (125 = 5s at 25fps). Small enough
    # that one chunk (~100-200ms GPU) never starves the render loop.
    LIP_CHUNK_FRAMES = 125

    # Fixed Audio2Lip input batch size (LIP_CHUNK_FRAMES + 2 halo + 1).
    # MUST match ModelManager.LIP_INFER_BATCH.
    LIP_INFER_BATCH = 128

    # Buffer depth at or above which adaptive face repair is allowed.
    # The buffer depth is the real-time measure of production headroom:
    # repair costs ~9ms/frame of synchronous GPU wait, so it is only
    # affordable when production is comfortably ahead of consumption.
    REPAIR_MIN_BUFFER = 6

    def __init__(self, target_fps: int = 25):
        """Initialize RenderLoop.

        Args:
            target_fps: Target output frame rate (default: 25)
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._character_data = None
        self._segment_lock = threading.Lock()
        self._frame_index = 0
        self.actual_fps = 0.0

        # Injected components (set by main.py before start())
        self.model_manager = None
        self.frame_buffer = None
        self.output_streamer = None
        self.audio_buffer = None
        self.audio_processor = None
        self.compositor = None
        self.face_repair = None
        self.face_repair_mode: str = 'off'  # 'off' | 'on' | 'adaptive'
        self.gaze_corrector = None
        self.gaze_mode: str = 'off'  # 'off' | 'on' | 'adaptive'
        self.audio_player = None

        # Video clock: wall-clock anchored position in the segment.
        self._video_start_time = time.perf_counter()

        # Audio-driven lip coefficient state
        self._lip_coeffs: Optional[torch.Tensor] = None
        self._audio_start_time: float = 0.0
        self._total_lip_frames: int = 0
        self._lip_filled: int = 0
        self._audio_lock = threading.Lock()

        # Pending PCM for the current audio segment. Playback is triggered
        # by OutputStreamer at display time of the segment's first frame.
        self._pending_pcm: Optional[np.ndarray] = None
        self._pending_audio_start: bool = False
        self._pending_lock = threading.Lock()

        # Serializes full-audio pushes (each push resets the player).
        self._push_lock = threading.Lock()

        # Idle lip coefficient cache
        self._idle_lip_cache: Optional[torch.Tensor] = None
        self._idle_lip_cache_valid: bool = False

        self._debug_audio_chunks = 0

        # Frame timing histogram (last N samples)
        self._timing_window = 240
        self._timings = {
            'gen': [],
            'gaze': [],
            'comp': [],
            'repair': [],
            'put': [],
            'total': [],
        }
        self._timing_lock = threading.Lock()

    def set_character_data(self, character_data):
        """Set or replace the character data pack."""
        with self._segment_lock:
            self._character_data = character_data
            self._video_start_time = time.perf_counter()

    def get_current_pack(self):
        """Get the current character data pack (thread-safe)."""
        with self._segment_lock:
            return self._character_data

    def switch_segment(self, segment: str) -> bool:
        """Switch to a different segment within the current character."""
        with self._segment_lock:
            if self._character_data is None:
                raise ValueError("No character data loaded")
            ok = self._character_data.switch_segment(segment)
            if ok:
                self._video_start_time = time.perf_counter()
                self._flush_buffer()
                if self.gaze_corrector is not None:
                    self.gaze_corrector.reset()
            return ok

    def start(self):
        """Start the rendering loop in a separate thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the rendering loop.

        Closing the frame buffer first wakes any threads blocked on put/get,
        so the producer and consumer can shut down cleanly.
        """
        self._running = False
        if self.frame_buffer is not None:
            try:
                self.frame_buffer.close()
            except Exception as e:
                print(f"[RenderLoop] Error closing frame buffer: {e}")
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        """Producer loop: render as fast as possible."""
        frame_count = 0
        _fps_timer_start = time.perf_counter()
        _fps_frame_count = 0
        _last_status_time = time.perf_counter()

        while self._running:
            self._process_frame()
            frame_count += 1
            _fps_frame_count += 1

            # Calculate production FPS over 1-second windows
            fps_elapsed = time.perf_counter() - _fps_timer_start
            if fps_elapsed >= 1.0:
                self.actual_fps = _fps_frame_count / fps_elapsed
                _fps_frame_count = 0
                _fps_timer_start = time.perf_counter()

            # Print status every 5 seconds
            now = time.perf_counter()
            if now - _last_status_time >= 5.0:
                _last_status_time = now
                mode = "audio" if self._lip_coeffs is not None else "idle"
                depth = 0
                if self.frame_buffer is not None:
                    depth = self.frame_buffer.depth
                timing = self._timing_summary()
                print(f"[RenderLoop] FPS:{self.actual_fps:.1f}  "
                      f"frame:{frame_count}  mode:{mode}  buffer:{depth}  {timing}")

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def process_full_audio(self, pcm_float: np.ndarray):
        """Run Audio2Lip on a full audio segment.

        The audio is processed in chunks so the GPU is never monopolized.
        Playback is not started here; instead the PCM is stored and playback
        is triggered by OutputStreamer when the segment's first frame is
        popped for display.
        """
        if self.audio_processor is None or self.model_manager is None:
            return

        with self._push_lock:
            self._debug_audio_chunks += 1
            self._idle_lip_cache_valid = False

            # Flush old frames and re-arm preroll for the new segment.
            self._flush_buffer()
            if self.output_streamer is not None:
                self.output_streamer.reset_preroll()

            # Run Audio2Lip on the audio as-is (no padding)
            mel_input, bs, T = self.audio_processor.pcm_to_mel(pcm_float)
            if mel_input is None or T <= 0:
                print("[RenderLoop] process_full_audio: empty audio, ignored")
                return

            pcm_duration = len(pcm_float) / 16000.0
            lip_duration = T / self.target_fps
            print(f"[RenderLoop] Full audio: pcm={pcm_duration:.3f}s, T={T} frames, "
                  f"lip_dur={lip_duration:.3f}s")

            # Progressive lip buffer
            lip_all = torch.zeros(T, 20, dtype=torch.float32)
            with self._audio_lock:
                self._audio_start_time = 0.0
                self._lip_coeffs = None
                self._total_lip_frames = T
                self._lip_filled = 0
                self._last_rendered_lip_idx = -1

            with self._pending_lock:
                self._pending_pcm = pcm_float
                self._pending_audio_start = True

            chunk = self.LIP_CHUNK_FRAMES
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                hs, he = max(0, s - 1), min(T, e + 1)
                mel_chunk = mel_input[0, hs:he]
                mel_chunk = self._pad_mel_chunk(mel_chunk)
                lip = self.model_manager.infer_lip(
                    mel_chunk.reshape(-1, 1, 80, 16), 1, mel_chunk.shape[0])
                keep = lip[s - hs: s - hs + (e - s)]
                lip_all[s:e] = keep.detach().to(torch.float32)

                with self._audio_lock:
                    self._lip_coeffs = lip_all
                    self._lip_filled = e

                # Yield to the render thread between chunks
                if e < T:
                    time.sleep(0.004)

            duration_s = T / self.target_fps
            print(f"[RenderLoop] Ready: {T} lip coefficients, "
                  f"audio duration: {pcm_duration:.2f}s, "
                  f"lip duration: {duration_s:.2f}s")

    def _get_audio_elapsed(self) -> float:
        """Get elapsed audio playback time in seconds using wall-clock."""
        if self._audio_start_time <= 0:
            return 0.0
        return time.perf_counter() - self._audio_start_time

    def _output_latency(self) -> float:
        """Audio output latency (callback -> DAC), for lip-sync offset."""
        ap = self.audio_player
        if ap is None:
            return 0.0
        try:
            lat = getattr(ap, 'output_latency_s', 0.0)
        except Exception:
            return 0.0
        if isinstance(lat, (int, float)) and 0.0 < lat < 1.0:
            return float(lat)
        return 0.0

    def _buffer_latency_s(self) -> float:
        """Forward-looking latency introduced by the output buffer.

        Frames are displayed ~depth/fps seconds after they are produced, so
        lip coefficients are chosen for the audio position at display time.
        """
        if self.frame_buffer is None:
            return 0.0
        depth = self.frame_buffer.depth
        # Clamp to a small multiple of interval to avoid over-correction
        # during fill / flush transient states.
        max_depth = max(8, self.frame_buffer.capacity)
        depth = min(depth, max_depth)
        return depth / max(1, self.target_fps)

    def _get_lip_coeff_for_current_audio(self) -> Optional[tuple]:
        """Get the lip coefficient for the audio position at display time."""
        with self._audio_lock:
            if self._lip_coeffs is None or self._total_lip_frames == 0:
                return None

            # Compensate for DAC latency and buffer depth so the lip shape
            # matches the audio when the frame is actually displayed.
            elapsed = (self._get_audio_elapsed()
                       - self._output_latency()
                       - self._buffer_latency_s())
            if elapsed < 0:
                elapsed = 0.0

            audio_end = self._total_lip_frames / self.target_fps
            if elapsed > audio_end + self.AUDIO_END_GRACE_S:
                self._lip_coeffs = None
                self._lip_filled = 0
                return None

            frame_idx = int(elapsed * self.target_fps)
            frame_idx = max(0, frame_idx)
            if frame_idx >= self._total_lip_frames:
                frame_idx = self._total_lip_frames - 1

            filled = self._lip_filled if self._lip_filled > 0 else self._total_lip_frames
            if frame_idx >= filled:
                frame_idx = filled - 1
            if frame_idx < 0:
                return None

            return (self._lip_coeffs[frame_idx:frame_idx + 1], frame_idx)

    def _pad_mel_chunk(self, mel_rows: torch.Tensor) -> torch.Tensor:
        """Pad a mel chunk [n, 1, 80, 16] to the fixed inference batch size."""
        target = self.LIP_INFER_BATCH
        n = mel_rows.shape[0]
        if n >= target:
            return mel_rows
        pad = mel_rows[-1:].expand(target - n, *mel_rows.shape[1:])
        return torch.cat([mel_rows, pad], dim=0)

    def _get_idle_lip_coeff(self) -> Optional[torch.Tensor]:
        """Get a closed-mouth lip coefficient for idle mode."""
        if self._idle_lip_cache_valid and self._idle_lip_cache is not None:
            return self._idle_lip_cache

        if (self.audio_processor is not None
                and self.model_manager is not None):
            idle_pcm = self.audio_processor.consume_idle_pcm()
            mel_input, bs, mel_T = self.audio_processor.pcm_to_mel(idle_pcm)

            if mel_input is not None:
                mel_input_flat = self._pad_mel_chunk(
                    mel_input.view(-1, 1, 80, 16))
                lip_coeffs = self.model_manager.infer_lip(
                    mel_input_flat, 1, mel_input_flat.shape[0])
                lip_coeffs = lip_coeffs[:1]
                self._idle_lip_cache = lip_coeffs.detach().clone()
                self._idle_lip_cache_valid = True
                return self._idle_lip_cache

        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _video_idx(self, total_frames) -> int:
        """Wall-clock position in the current segment (wraps at the end)."""
        try:
            total = int(total_frames)
        except (TypeError, ValueError):
            return max(0, self._frame_index)
        if total <= 0:
            return 0
        elapsed = time.perf_counter() - self._video_start_time
        return int(elapsed * self.target_fps) % total

    def _process_frame(self) -> None:
        """Render one frame and push it into the output buffer."""
        t_total_0 = time.perf_counter()
        timings = {}

        with self._segment_lock:
            character_data = self._character_data

        if character_data is None:
            print("[RenderLoop] ERROR: No character data!")
            return

        total = character_data.total_frames
        video_idx = self._video_idx(total)

        # Pick lip coefficient: audio-driven or idle
        lip_result = self._get_lip_coeff_for_current_audio()
        if lip_result is not None:
            alpha_D_lip, lip_idx = lip_result
        else:
            alpha_D_lip = self._get_idle_lip_coeff()

        # Load frame data (never blocks on batch I/O)
        character_data.ensure_loaded(video_idx)
        orifra, cropped, meta, facemark, pre_mask = character_data.get_frame(video_idx)

        # Generator inference — output stays ON GPU (CHW BGR uint8) so face
        # repair can run GPU-inline (IOBinding, zero copy). Single GPU->CPU
        # download happens once, after repair, below.
        rendered_gpu_chw = None
        t_gen_0 = time.perf_counter()
        rendered_c = None
        if self.model_manager is not None and cropped is not None and alpha_D_lip is not None:
            rendered = self.model_manager.infer_frame(cropped, alpha_D_lip, cropped)
            rendered_c = rendered.clamp(-1, 1)

            # Gaze correction on the rendered output (Route C', GPU-inline).
            # Must run AFTER the generator (it erases input-side corrections)
            # and BEFORE the compositor (mask includes eye regions).
            if self.gaze_corrector is not None and self._gaze_allowed():
                t_gaze_0 = time.perf_counter()
                rendered_c = self.gaze_corrector.correct(
                    rendered_c[0], facemark).unsqueeze(0)
                timings['gaze'] = (time.perf_counter() - t_gaze_0) * 1000.0

            rendered_post = (rendered_c[0] * 127.5 + 127.5).to(torch.uint8)
            rendered_gpu_chw = rendered_post[[2, 1, 0], :, :].contiguous()
        if 'gaze' not in timings:
            timings['gaze'] = 0.0
        timings['gen'] = (time.perf_counter() - t_gen_0) * 1000.0

        # CPU prep: resolve mask and resize ori crop for blending.
        t_comp_0 = time.perf_counter()
        prepared_crop = None
        mask_256 = None
        if self.compositor is not None and facemark is not None:
            mask_256 = self._resolve_mask(facemark, pre_mask)
            prepared_crop = self.compositor.prepare_crop(orifra, meta)
        timings['comp'] = (time.perf_counter() - t_comp_0) * 1000.0

        # Face repair
        t_repair_0 = time.perf_counter()
        if (self.face_repair is not None and rendered_gpu_chw is not None
                and self._repair_allowed()):
            # GPU-inline: tensor in, tensor out; no numpy round trip.
            rendered_gpu_chw = self.face_repair.repair_gpu(rendered_gpu_chw)
        timings['repair'] = (time.perf_counter() - t_repair_0) * 1000.0

        # Single GPU->CPU download of the (optionally repaired) render.
        rendered_np = None
        if rendered_gpu_chw is not None:
            rendered_np = rendered_gpu_chw.permute(1, 2, 0).cpu().numpy()

        # Composite mouth onto original frame
        if self.compositor is not None and rendered_np is not None and mask_256 is not None:
            composited = self.compositor.blend(orifra, rendered_np, mask_256, prepared_crop)
        else:
            composited = orifra

        # Build metadata for the consumer
        frame_meta = None
        with self._pending_lock:
            if self._pending_audio_start:
                frame_meta = {'audio_start': True}
                self._pending_audio_start = False

        # Push into the output buffer (blocking when full — backpressure)
        t_put_0 = time.perf_counter()
        if self.frame_buffer is not None:
            self.frame_buffer.put(composited, meta=frame_meta)
        timings['put'] = (time.perf_counter() - t_put_0) * 1000.0

        timings['total'] = (time.perf_counter() - t_total_0) * 1000.0
        self._record_timings(timings)

        self._frame_index += 1

    def _resolve_mask(self, facemark, precomputed_mask):
        """Resolve mask via compositor or return precomputed."""
        if precomputed_mask is not None:
            return precomputed_mask
        if self.compositor is not None:
            return self.compositor._resolve_mask(facemark, None)
        return None

    def _repair_allowed(self) -> bool:
        """Whether face repair may run on this frame.

        'on': every frame. 'adaptive': only while the output buffer has
        headroom (production comfortably ahead of consumption) — the
        ~9ms synchronous GPU wait is unaffordable otherwise. 'off' never
        reaches here (face_repair is not constructed).
        """
        if self.face_repair_mode != 'adaptive':
            return True
        if self.frame_buffer is None:
            return True
        return self.frame_buffer.depth >= self.REPAIR_MIN_BUFFER

    def _gaze_allowed(self) -> bool:
        """Whether gaze correction may run on this frame (adaptive mode).

        Gaze costs ~2-3ms/frame of GPU work; adaptive shares the repair
        headroom threshold (REPAIR_MIN_BUFFER) — at that depth production
        is comfortably ahead of the 25fps output line.
        """
        if self.gaze_mode != 'adaptive':
            return True
        if self.frame_buffer is None:
            return True
        return self.frame_buffer.depth >= self.REPAIR_MIN_BUFFER

    def _record_timings(self, timings: dict):
        """Append timings to rolling histogram."""
        with self._timing_lock:
            for k, v in timings.items():
                self._timings[k].append(v)
                if len(self._timings[k]) > self._timing_window:
                    self._timings[k].pop(0)

    def _timing_summary(self) -> str:
        """Return p50/p95/p99 summary string for the last frames."""
        import numpy as np
        with self._timing_lock:
            if not self._timings['total']:
                return ""
            parts = []
            for key in ['gen', 'gaze', 'comp', 'repair', 'put', 'total']:
                vals = self._timings[key]
                if not vals:
                    continue
                p50, p95, p99 = np.percentile(vals, [50, 95, 99])
                parts.append(f"{key}=p50:{p50:.1f}/p95:{p95:.1f}/p99:{p99:.1f}ms")
            return " ".join(parts)

    def _flush_buffer(self) -> None:
        """Flush the output buffer, keeping the configured reserve."""
        if self.frame_buffer is not None:
            dropped = self.frame_buffer.flush()
            if dropped > 0:
                print(f"[RenderLoop] Flushed {dropped} buffered frames")

    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        return self._frame_index
