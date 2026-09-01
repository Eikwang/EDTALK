"""Render loop for 25fps streaming inference.

Key design: Audio-driven synchronization with wall-clock timing
----------------------------------------------------------------
The render loop uses wall-clock time since audio started to determine
which lip coefficient to render. This is fundamentally different from
the previous approaches:

1. Previous "elapsed_seconds" approach: Used sounddevice callback's
   sample consumption count. Had bugs: measures callback consumption
   not DAC output, didn't reset between segments, pre-buffer offset.

2. Previous "frame-locked" approach: Each video frame pushed
   corresponding audio chunk. Required render speed == audio speed.
   Failed because render speed is slightly faster than real-time
   (Generator ~35ms + send ~3ms = 38ms < 40ms), causing audio
   buffer to accumulate and lip movement to finish before audio.

3. Current "wall-clock audio-driven" approach: Record the timestamp
   when audio starts playing, then use (current_time - start_time)
   to determine which lip coefficient to render. Push ALL PCM to
   AudioPlayer at once (independent playback), but use wall-clock
   time for lip coefficient selection.

   This works because:
   - Wall-clock time advances at exactly 1 second per second
   - Audio playback also advances at exactly 1 second per second
   - Lip coefficients are indexed at 25fps = 1 frame per 40ms
   - frame_idx = int((now - start_time) * 25)
   - If render is fast, some lip coefficients are repeated (mouth stays)
   - If render is slow, some lip coefficients are skipped (mouth catches up)
   - Either way, lip position matches audio position

Audio push strategy:
  The full PCM is pushed to AudioPlayer at once when process_full_audio()
  is called. AudioPlayer plays it independently via sounddevice callback.
  This ensures audio plays at exactly real-time speed regardless of GPU
  render rate. The render loop only needs to track which lip coefficient
  corresponds to the current audio playback time.

Idle mode design:
  When no audio is available, the render loop uses a cached "closed-mouth"
  lip coefficient to drive Generator inference. No audio is played.
"""
import time
import threading
from typing import Optional
import cv2
import numpy as np
import torch


class RenderLoop:
    """Main rendering loop that runs at target FPS.

    The loop processes frames in a continuous cycle:
    1. Read frame from current character segment
    2. Determine which lip coefficient corresponds to current audio time
    3. Run Generator inference to get rendered mouth frame
    4. Composite mouth region onto original frame
    5. Send frame to virtual camera

    Thread model: Runs in a separate thread, independent of API server.
    Audio-video sync: Wall-clock audio-driven — lip position follows audio time.
    """

    def __init__(self, target_fps: int = 25):
        """Initialize RenderLoop.

        Args:
            target_fps: Target frame rate (default: 25)
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
        self.unity_capture = None
        self.audio_buffer = None
        self.audio_processor = None
        self.compositor = None
        self.face_repair = None
        self.audio_player = None

        # Audio-driven lip coefficient state
        # _lip_coeffs: [T, 20] tensor — all lip coefficients for current segment
        # _audio_start_time: wall-clock timestamp when audio started playing
        # _total_lip_frames: number of lip coefficient frames (T)
        self._lip_coeffs: Optional[torch.Tensor] = None
        self._audio_start_time: float = 0.0
        self._total_lip_frames: int = 0
        self._audio_lock = threading.Lock()

        # Idle lip coefficient cache
        self._idle_lip_cache: Optional[torch.Tensor] = None
        self._idle_lip_cache_valid: bool = False

        # Debug
        self._debug_audio_chunks = 0
        self._debug_idle_frames = 0
        self._debug_real_frames = 0
        self._debug_last_print = 0.0
        self._last_rendered_lip_idx: int = -1

    def _debug_print_stats(self):
        """Print debug statistics periodically."""
        now = time.perf_counter()
        if now - self._debug_last_print < 5.0:
            return
        self._debug_last_print = now
        total = self._debug_real_frames + self._debug_idle_frames
        if total > 0:
            real_pct = self._debug_real_frames / total * 100
            print(f"[RenderLoop DEBUG] Real audio frames: {self._debug_real_frames} "
                  f"({real_pct:.0f}%), Idle frames: {self._debug_idle_frames}, "
                  f"Audio chunks consumed: {self._debug_audio_chunks}")

    def set_character_data(self, character_data):
        """Set or replace the character data pack."""
        with self._segment_lock:
            self._character_data = character_data

    def get_current_pack(self):
        """Get the current character data pack (thread-safe)."""
        with self._segment_lock:
            return self._character_data

    def switch_segment(self, segment: str) -> bool:
        """Switch to a different segment within the current character."""
        with self._segment_lock:
            if self._character_data is None:
                raise ValueError("No character data loaded")
            return self._character_data.switch_segment(segment)

    def start(self):
        """Start the rendering loop in a separate thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the rendering loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        """Main loop running in background thread.

        Frame rate is NOT explicitly controlled here. The actual frame
        rate is determined by:
        1. Generator inference time (~35ms)
        2. send_frame() time (~3-5ms)
        3. Any remaining time to reach 40ms interval

        Since sleep_until_next_frame() doesn't work with UnityCapture
        backend, we add our own frame interval control using
        time.perf_counter() for precise timing.
        """
        frame_count = 0
        _fps_timer_start = time.perf_counter()
        _fps_frame_count = 0
        next_frame_time = time.perf_counter()

        while self._running:
            # Wait until next frame time for steady frame rate
            now = time.perf_counter()
            wait = next_frame_time - now
            if wait > 0.002:  # Only sleep if > 2ms to avoid busy-wait
                time.sleep(wait * 0.8)  # Sleep 80%, then busy-wait the rest

            t_start = time.perf_counter()

            frames_rendered = self._process_batch()
            frame_count += frames_rendered
            _fps_frame_count += frames_rendered

            # Calculate FPS over 1-second windows
            fps_elapsed = time.perf_counter() - _fps_timer_start
            if fps_elapsed >= 1.0:
                self.actual_fps = _fps_frame_count / fps_elapsed
                _fps_frame_count = 0
                _fps_timer_start = time.perf_counter()

            # Schedule next frame at exact interval from anchor
            next_frame_time = max(next_frame_time + self.frame_interval,
                                  time.perf_counter())

            # Print status periodically
            if frame_count <= 10 or frame_count % 30 == 0:
                audio_info = ""
                buffer_info = ""
                with self._audio_lock:
                    if self._lip_coeffs is not None:
                        elapsed = self._get_audio_elapsed()
                        lip_idx = min(int(elapsed * self.target_fps),
                                      self._total_lip_frames - 1)
                        audio_info = (f", lip_idx={lip_idx}/{self._total_lip_frames}"
                                     f", audio_t={elapsed:.2f}s")
                if self.audio_player is not None and self.audio_player.enabled:
                    # Compare wall-clock vs AudioPlayer's sample-counted progress
                    ap_elapsed = self.audio_player.elapsed_seconds
                    wc_elapsed = self._get_audio_elapsed()
                    drift_ms = (wc_elapsed - ap_elapsed) * 1000
                    buffer_info = (f", audio_buf={self.audio_player.buffered_ms:.0f}ms"
                                  f", ap_t={ap_elapsed:.2f}s"
                                  f", drift={drift_ms:.0f}ms")
                frame_time_ms = (time.perf_counter() - t_start) * 1000
                print(f"[RenderLoop] Frame {frame_count}, FPS: {self.actual_fps:.1f}"
                      f", frame_time={frame_time_ms:.1f}ms"
                      f"{audio_info}{buffer_info}")
                self._debug_print_stats()

    def process_full_audio(self, pcm_float: np.ndarray):
        """Run Audio2Lip on the full audio segment and start audio playback.

        The entire audio is processed at once with full conv_feat smoothing.
        The resulting lip coefficients are stored for audio-driven lookup.
        The full PCM is pushed to AudioPlayer for independent playback.

        Args:
            pcm_float: Full audio PCM data, float32 in [-1, 1] range
        """
        if self.audio_processor is None or self.model_manager is None:
            return

        self._debug_audio_chunks += 1
        self._idle_lip_cache_valid = False

        # Pad audio to match video segment length to prevent lip exhaustion
        with self._segment_lock:
            video_total = self._character_data.total_frames if self._character_data else 0
        if video_total > 0:
            expected_samples = int(video_total / self.target_fps *16000)
            if len(pcm_float) < expected_samples:
                pcm_float = np.pad(pcm_float, (0, expected_samples - len(pcm_float)),
                                   mode='constant', constant_values=0.0)

        # Run Audio2Lip on full audio (same as demo)
        mel_input, bs, T = self.audio_processor.pcm_to_mel(pcm_float)

        pcm_duration = len(pcm_float) / 16000.0
        lip_duration = T / self.target_fps
        print(f"[RenderLoop] Full audio: pcm={pcm_duration:.3f}s, T={T} frames, "
              f"lip_dur={lip_duration:.3f}s, "
              f"pcm_samples={len(pcm_float)}, "
              f"expected_T={int(pcm_duration * self.target_fps)}")

        if mel_input is None:
            return

        mel_input_flat = mel_input.view(-1, 1, 80, 16)
        lip_coeffs = self.model_manager.infer_lip(mel_input_flat, bs, T)

        # Reset AudioPlayer state and push full PCM at once
        # AudioPlayer plays independently via sounddevice callback
        if self.audio_player is not None:
            self.audio_player.reset()
            self.audio_player.push_pcm(pcm_float)

        # Store lip coefficients and record audio start time
        # IMPORTANT: Set _audio_start_time FIRST, then _lip_coeffs.
        # This prevents a race where the render loop sees _lip_coeffs
        # is not None but uses a stale _audio_start_time from a
        # previous segment, which would cause it to immediately
        # clear the lip coefficients (elapsed too large).
        now = time.perf_counter()
        with self._audio_lock:
            self._audio_start_time = now
            self._total_lip_frames = lip_coeffs.shape[0]
            self._lip_coeffs = lip_coeffs
            self._last_rendered_lip_idx = -1

        duration_s = self._total_lip_frames / self.target_fps
        print(f"[RenderLoop] Ready: {self._total_lip_frames} lip coefficients, "
              f"audio duration: {len(pcm_float)/16000:.2f}s, "
              f"lip duration: {duration_s:.2f}s")

    def _get_audio_elapsed(self) -> float:
        """Get elapsed audio playback time in seconds using wall-clock.

        Uses time.perf_counter() relative to _audio_start_time, which
        is set when process_full_audio() completes. This is precise
        because:
        - Wall-clock time advances at exactly 1s/s
        - Audio playback also advances at exactly 1s/s
        - No dependency on sounddevice callback timing or buffer state
        """
        if self._audio_start_time <= 0:
            return 0.0
        return time.perf_counter() - self._audio_start_time

    def _get_lip_coeff_for_current_audio(self) -> Optional[tuple]:
        """Get the lip coefficient corresponding to current audio time.

        Uses wall-clock time since audio started to determine which
        lip coefficient to render. This ensures lip movement stays
        synchronized with audio regardless of GPU render rate.

        Returns:
            Tuple of (lip_coeff, lip_idx) or None if audio finished.
        """
        with self._audio_lock:
            if self._lip_coeffs is None or self._total_lip_frames == 0:
                return None

            elapsed = self._get_audio_elapsed()
            frame_idx = int(elapsed * self.target_fps)

            if frame_idx < 0:
                frame_idx = 0
            if frame_idx >= self._total_lip_frames:
                # Audio finished — hold last lip position instead of switching to idle
                last_idx = self._total_lip_frames - 1
                return (self._lip_coeffs[last_idx:last_idx+1], last_idx)

            return (self._lip_coeffs[frame_idx:frame_idx+1], frame_idx)

    def _get_idle_lip_coeff(self) -> Optional[torch.Tensor]:
        """Get a closed-mouth lip coefficient for idle mode."""
        if self._idle_lip_cache_valid and self._idle_lip_cache is not None:
            return self._idle_lip_cache

        if (self.audio_processor is not None
                and self.model_manager is not None):
            idle_pcm = self.audio_processor.consume_idle_pcm()
            mel_input, bs, mel_T = self.audio_processor.pcm_to_mel(idle_pcm)

            if mel_input is not None:
                mel_input_flat = mel_input.view(-1, 1, 80, 16)
                lip_coeffs = self.model_manager.infer_lip(mel_input_flat, bs, mel_T)
                lip_coeffs = lip_coeffs[:1]
                self._idle_lip_cache = lip_coeffs.detach().clone()
                self._idle_lip_cache_valid = True
                return self._idle_lip_cache

        return None

    def _process_batch(self) -> int:
        """Process one frame with audio-driven lip synchronization.

        Audio-driven design:
        - Audio plays independently via AudioPlayer at exact real-time speed
        - Lip coefficient is selected based on wall-clock time since audio start
        - If GPU is fast, some lip coefficients are rendered multiple times
        - If GPU is slow, some lip coefficients are skipped
        - Either way, lip position always matches audio position

        Returns:
            Number of frames rendered (always 1 or 0)
        """
        with self._segment_lock:
            character_data = self._character_data

        if character_data is None:
            print("[RenderLoop] ERROR: No character data!")
            return 0

        # --- Get lip coefficient for current audio time ---
        lip_result = self._get_lip_coeff_for_current_audio()

        if lip_result is None:
            # No audio data — idle mode
            t0 = time.perf_counter()
            character_data.ensure_loaded(self._frame_index)
            orifra, cropped, meta, facemark, pre_mask = character_data.get_frame(self._frame_index)
            t1 = time.perf_counter()
            alpha_D_lip = self._get_idle_lip_coeff()
            t2 = time.perf_counter()

            rendered_np = None
            if self.model_manager is not None and cropped is not None and alpha_D_lip is not None:
                rendered = self.model_manager.infer_frame(cropped, alpha_D_lip, cropped)
                t3 = time.perf_counter()
                # Do ALL processing on GPU before .cpu() to minimize CPU work after sync
                rendered_np = rendered.clamp(-1, 1)
                rendered_np = (rendered_np[0] * 127.5 + 127.5).to(torch.uint8)
                rendered_np = rendered_np[[2, 1, 0], :, :]  # RGB→BGR on GPU
                rendered_np = rendered_np.permute(1, 2, 0).cpu().numpy()
            else:
                t3 = time.perf_counter()

            if self.face_repair is not None and rendered_np is not None:
                rendered_np = self.face_repair.repair(rendered_np)

            t4 = time.perf_counter()
            if self.compositor is not None and rendered_np is not None and facemark is not None:
                composited = self.compositor.composite(orifra, rendered_np, facemark, meta,
                                                       precomputed_mask=pre_mask)
            else:
                composited = orifra
            t5 = time.perf_counter()

            if self.unity_capture is not None:
                success = self.unity_capture.send_frame(composited)
                if not success:
                    print(f"[RenderLoop] Failed to send frame {self._frame_index}")
            t6 = time.perf_counter()

            # Print timing breakdown every30 frames
            if self._frame_index <= 5 or self._frame_index % 30 == 0:
                print(f"[TIMING] load={ (t1-t0)*1000:.1f}ms lip={ (t2-t1)*1000:.1f}ms "
                      f"gen={ (t3-t2)*1000:.1f}ms post={ (t4-t3)*1000:.1f}ms "
                      f"comp={ (t5-t4)*1000:.1f}ms send={ (t6-t5)*1000:.1f}ms")

            self._frame_index += 1
            self._debug_idle_frames += 1
            return 1

        # --- Render frame with audio-synced lip coefficient ---
        alpha_D_lip, lip_idx = lip_result
        self._last_rendered_lip_idx = lip_idx

        # Clamp lip_idx to valid segment range to prevent segment wrap
        with self._segment_lock:
            total = self._character_data.total_frames if self._character_data else 0
        if total > 0 and lip_idx >= total:
            lip_idx = lip_idx % total

        t0 = time.perf_counter()
        character_data.ensure_loaded(lip_idx)
        orifra, cropped, meta, facemark, pre_mask = character_data.get_frame(lip_idx)
        t1 = time.perf_counter()

        self._debug_real_frames += 1

        rendered_np = None
        if self.model_manager is not None and cropped is not None and alpha_D_lip is not None:
            img_source = cropped
            pose_img_drive = cropped

            rendered = self.model_manager.infer_frame(img_source, alpha_D_lip, pose_img_drive)
            t2 = time.perf_counter()

            rendered_np = rendered.clamp(-1, 1)
            rendered_np = (rendered_np[0] * 127.5 + 127.5).to(torch.uint8)
            rendered_np = rendered_np[[2, 1, 0], :, :]  # RGB→BGR on GPU
            rendered_np = rendered_np.permute(1, 2, 0).cpu().numpy()
        else:
            t2 = time.perf_counter()

        if self.face_repair is not None and rendered_np is not None:
            rendered_np = self.face_repair.repair(rendered_np)

        t3 = time.perf_counter()
        if self.compositor is not None and rendered_np is not None and facemark is not None:
            composited = self.compositor.composite(orifra, rendered_np, facemark, meta,
                                                   precomputed_mask=pre_mask)
        else:
            composited = orifra
        t4 = time.perf_counter()

        if self.unity_capture is not None:
            success = self.unity_capture.send_frame(composited)
            if not success:
                print(f"[RenderLoop] Failed to send frame {self._frame_index}")
        t5 = time.perf_counter()

        if self._frame_index <= 5 or self._frame_index % 30 == 0:
            print(f"[TIMING:audio] load={ (t1-t0)*1000:.1f}ms "
                  f"gen={ (t2-t1)*1000:.1f}ms post={ (t3-t2)*1000:.1f}ms "
                  f"comp={ (t4-t3)*1000:.1f}ms send={ (t5-t4)*1000:.1f}ms")

        self._frame_index += 1
        return 1

    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        return self._frame_index
