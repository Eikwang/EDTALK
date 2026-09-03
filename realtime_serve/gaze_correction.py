"""Gaze correction for the realtime pipeline (Route C', probe-verified).

Corrects the eye gaze on the Generator's rendered 256x256 output BEFORE the
compositor pastes it back. The compositor mask must include the eye regions
for the correction to reach the final frame (see compositor.py eyes_enabled).

Pipeline position (probe-verified 2026-09-03):
    count crop -> Generator -> rendered(256,GPU) -> [HERE] -> repair -> composite
Correction applied on the Generator INPUT (count crop) was measured to be
erased by the generator redraw (eye diff 1.8x noise vs 29x here).

Design notes:
- All tensor work stays on the GPU; the caller passes the rendered [-1,1]
  RGB CHW tensor and receives a new tensor (never in-place: the preloader
  buffers are shared and frames repeat).
- Eye positions come from the precomputed per-frame facemark (68 points in
  256 space) — dlib never runs in the realtime loop.
- Blink gate: EAR state machine ported from the offline reference
  (pytorch_gaze_redirection/src/inference.py). Detection-failure frames
  HOLD the previous eye centers and keep correcting; blink frames BYPASS.
  eye_size and the movement threshold live in 256-space so the tracker's
  zoom does not drift the locked size (ENG review A-6).
- Dynamic convergence angle: set_convergence() re-builds the angle tensor;
  effective from the next frame.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from pytorch_gaze_redirection.src.archs import Generator

# EAR blink state machine (ported verbatim from the offline reference).
EAR_CLOSE = 0.12
EAR_OPEN = 0.16
OPEN_CONFIRM_FRAMES = 3
MAX_BLINK_FRAMES = 26


class GazeCorrector:
    """GPU-inline gaze correction on the rendered 256 crop.

    Errors never kill the render loop: correct() catches its own failures
    and returns the input unchanged (counted in error_skips).
    """

    L_EYE = list(range(36, 42))
    R_EYE = list(range(42, 48))

    def __init__(self, ckpt_path: str, device: str = 'cuda',
                 convergence_h: float = 0.05, smooth_alpha: float = 0.4,
                 image_size: int = 64):
        self.device = torch.device(device)
        self.smooth_alpha = smooth_alpha
        self.image_size = image_size

        self.generator = Generator(style_dim=3)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        if 'current_generator' not in ckpt:
            raise KeyError("not a gaze model weight: missing key 'current_generator'")
        self.generator.load_state_dict(ckpt['current_generator'])
        self.generator = self.generator.to(self.device)
        self.generator.eval()

        self._convergence_h = float(convergence_h)
        self._angles = self._build_angles(convergence_h)

        # 64x64 circular blend mask (GPU constant, matches the offline reference).
        size = image_size
        m = np.zeros((size, size), dtype=np.float32)
        cv2.circle(m, (size // 2, size // 2), size // 2 - 2, 1.0, thickness=-1)
        m = cv2.GaussianBlur(m, (7, 7), 0)
        self._mask64 = torch.from_numpy(m[..., np.newaxis]).to(self.device)

        # Smoothed eye centers + locked eye size, both in 256-space.
        self._prev_left = None
        self._prev_right = None
        self._fixed_eye_size = None

        # Blink state machine counters.
        self._low_ear_count = 0
        self._open_confirm = 0

        # Temporal smoothing of the generated eyes (GAN outputs jitter
        # frame-to-frame even for identical inputs). New-frame weight.
        self._temporal_alpha = 0.7
        self._prev_gen = None
        self._skip_run = 0

        # Observability counters (reset per segment).
        self.corrected_frames = 0
        self.blink_skips = 0
        self.nan_skips = 0
        self.error_skips = 0

        self._graph = None
        self._static_in = None
        self._static_angles = None
        self._static_out = None
        self._graph_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_convergence(self, convergence_h: float) -> None:
        """Change the convergence angle at runtime (takes effect next frame)."""
        self._convergence_h = float(convergence_h)
        self._angles = self._build_angles(self._convergence_h)
        if self._graph_ready:
            self._static_angles.copy_(self._angles)

    def reset(self) -> None:
        """Reset per-segment state (segment switch / character switch)."""
        self._prev_left = None
        self._prev_right = None
        self._fixed_eye_size = None
        self._low_ear_count = 0
        self._open_confirm = 0
        self._prev_gen = None
        self._skip_run = 0
        self.corrected_frames = 0
        self.blink_skips = 0
        self.nan_skips = 0
        self.error_skips = 0

    @torch.inference_mode()
    def correct(self, rendered: torch.Tensor, facemark_256: np.ndarray,
                gaze_frame: Optional[dict] = None) -> torch.Tensor:
        """Correct both eyes of a rendered 256 crop.

        Args:
            rendered: (3,256,256) RGB float [-1,1] on GPU — the Generator output.
            facemark_256: (68,2) 256-space landmarks for this frame (numpy),
                          possibly containing NaN for detection-failure frames.
                          Only used when gaze_frame is None (legacy realtime path).
            gaze_frame: optional per-frame dict from precomputed gaze_data
                        (prepare_gaze_data.py step 11) with keys
                        state/l_center/r_center/eye_size. When provided the
                        blink decision and eye trajectory come from the
                        offline time-series (far more stable than per-frame
                        realtime landmarks) — the recommended path.

        Returns:
            New (3,256,256) tensor. Same device/domain. Never mutates the input.
        """
        try:
            if gaze_frame is not None:
                state = int(gaze_frame['state'])
                if state == 1:
                    self.blink_skips += 1
                    self._note_skip()
                    return rendered
                if state == 2:
                    self.nan_skips += 1
                    self._note_skip()
                    return rendered
                eye_size = int(gaze_frame['eye_size'])
                lx, ly = gaze_frame['l_center']
                rx, ry = gaze_frame['r_center']
                boxes = [(float(lx), float(ly), eye_size),
                         (float(rx), float(ry), eye_size)]
            else:
                if facemark_256 is None or np.any(np.isnan(facemark_256)):
                    self.nan_skips += 1
                    self._note_skip()
                    return rendered

                l_info, r_info = self._eye_state(facemark_256)
                if l_info is None:
                    # Blink in progress: bypass entirely.
                    self.blink_skips += 1
                    self._note_skip()
                    return rendered

                boxes = []
                for (cx, cy), eye_size in ((l_info, self._fixed_eye_size),
                                           (r_info, self._fixed_eye_size)):
                    boxes.append((cx, cy, eye_size))

            out = self._correct_eyes(rendered, boxes)
            self.corrected_frames += 1
            self._skip_run = 0
            return out
        except Exception as e:
            self.error_skips += 1
            # Rate-limited: a silent bypass loop would hide programming
            # errors (measured to have happened during development).
            if self.error_skips == 1:
                import traceback
                print(f"[GazeCorrector] correction failed ({e}); "
                      f"bypassing. Traceback (once):")
                traceback.print_exc()
            return rendered

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _note_skip(self) -> None:
        """Track consecutive skip frames; invalidate the temporal-smoothing
        history after a long gap so resuming correction doesn't ghost."""
        self._skip_run += 1
        if self._skip_run > 3:
            self._prev_gen = None

    def _build_angles(self, convergence_h: float) -> torch.Tensor:
        return torch.tensor([[-convergence_h, 0.0, -1.0],
                             [convergence_h, 0.0, 1.0]],
                            device=self.device)

    @staticmethod
    def _ear(pts: np.ndarray) -> float:
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3]) + 1e-6
        return (v1 + v2) / (2.0 * h)

    def _smooth(self, raw: float, prev, alpha: float) -> float:
        """EMA with reduced gain on large jumps (threshold is 256-space)."""
        if prev is None:
            return raw
        a = alpha if abs(raw - prev) < 20 else alpha * 0.3
        return a * raw + (1.0 - a) * prev

    def _eye_state(self, facemark_256: np.ndarray):
        """EAR gate + smoothed centers. Returns (l_info, r_info); both None
        while a blink is in progress."""
        l_pts = facemark_256[self.L_EYE]
        r_pts = facemark_256[self.R_EYE]
        min_ear = min(self._ear(l_pts), self._ear(r_pts))

        infos = []
        for pts, prev in ((l_pts, self._prev_left), (r_pts, self._prev_right)):
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            cx = self._smooth((x1 + x2) / 2.0, prev[0] if prev else None,
                              self.smooth_alpha)
            cy = self._smooth((y1 + y2) / 2.0, prev[1] if prev else None,
                              self.smooth_alpha)
            infos.append((cx, cy))
            if self._fixed_eye_size is None:
                size = int(1.5 * max(x2 - x1, y2 - y1))
                self._fixed_eye_size = max(16, min(size, 128))
        self._prev_left, self._prev_right = infos[0], infos[1]

        # Blink state machine (detection failure keeps correcting via the
        # hold-above; only actual blinks gate).
        if min_ear < EAR_CLOSE:
            self._low_ear_count += 1
            self._open_confirm = 0
        elif self._low_ear_count > 0:
            if min_ear < EAR_OPEN:
                self._open_confirm = 0
            else:
                self._open_confirm += 1
                if self._open_confirm >= OPEN_CONFIRM_FRAMES:
                    self._low_ear_count = 0
                    self._open_confirm = 0
        else:
            self._open_confirm = 0

        if 0 < self._low_ear_count <= MAX_BLINK_FRAMES:
            return None, None
        return infos[0], infos[1]

    def _correct_eyes(self, rendered: torch.Tensor,
                      boxes) -> torch.Tensor:
        out = rendered.clone()
        tensors = []
        coords = []
        for cx, cy, eye_size in boxes:
            half = eye_size / 2.0
            x1 = int(round(cx - half))
            y1 = int(round(cy - half))
            x1c, y1c = max(0, x1), max(0, y1)
            x2c = min(256, x1 + eye_size)
            y2c = min(256, y1 + eye_size)
            if x2c - x1c < 8 or y2c - y1c < 8:
                continue
            region = rendered[:, y1c:y2c, x1c:x2c].unsqueeze(0)
            t = F.interpolate(region, size=(self.image_size, self.image_size),
                              mode='bilinear', antialias=True)
            tensors.append(t)
            coords.append((x1c, y1c, x2c - x1c, y2c - y1c))

        if not tensors:
            return rendered

        batch = torch.cat(tensors, dim=0)  # [2,3,64,64]
        gen = self._forward(batch)

        # Temporal smoothing: lerp with the previous generated eyes. GAN
        # outputs jitter frame-to-frame even for near-identical inputs;
        # blink/skip frames don't touch the history (see _note_skip).
        if self._prev_gen is not None and self._prev_gen.shape == gen.shape:
            gen = self._temporal_alpha * gen + (1.0 - self._temporal_alpha) * self._prev_gen
        self._prev_gen = gen.detach().clone()

        for i, (x1c, y1c, w, h) in enumerate(coords):
            eye = gen[i]
            target = rendered[:, y1c:y1c + h, x1c:x1c + w]
            for c in range(3):
                s_mean = eye[c].mean(); s_std = eye[c].std() + 1e-6
                d_mean = target[c].mean(); d_std = target[c].std() + 1e-6
                eye[c] = (eye[c] - s_mean) * (d_std / s_std) + d_mean
            eye = eye.clamp(-1, 1)

            eye_resized = F.interpolate(eye.unsqueeze(0), size=(h, w),
                                        mode='bilinear', antialias=True)[0]
            mk = F.interpolate(self._mask64.permute(2, 0, 1).unsqueeze(0),
                               size=(h, w), mode='bilinear', antialias=True)[0]
            region_out = out[:, y1c:y1c + h, x1c:x1c + w]
            out[:, y1c:y1c + h, x1c:x1c + w] = eye_resized * mk + region_out * (1.0 - mk)

        return out

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Generator forward: CUDA graph when captured, eager otherwise."""
        if self._graph_ready:
            self._static_in.copy_(batch)
            self._static_angles.copy_(self._angles)
            self._graph.replay()
            return self._static_out
        with torch.inference_mode():
            return self.generator(batch, self._angles)

    def try_capture_graph(self) -> bool:
        """Best-effort CUDA graph capture. Returns True when active."""
        if self.device.type != 'cuda':
            return False
        try:
            self._static_in = torch.zeros(2, 3, self.image_size, self.image_size,
                                          device=self.device)
            self._static_angles = self._angles.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3):
                    _ = self.generator(self._static_in, self._static_angles)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.inference_mode():
                out = self.generator(self._static_in, self._static_angles)
            if not isinstance(out, torch.Tensor) or out.shape != (2, 3, 64, 64):
                raise RuntimeError(f"unexpected graph output {type(out)}")
            self._graph = g
            self._static_out = out
            self._graph_ready = True
            return True
        except Exception:
            self._graph = None
            self._graph_ready = False
            return False
