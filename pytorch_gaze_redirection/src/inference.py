import os
import time
import warnings
warnings.filterwarnings('ignore')

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.exposure import match_histograms

from .archs import Generator

CONVERGENCE_H = 0.05
LANDMARK_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'utils', 'eyes_catch_model',
                              'shape_predictor_68_face_landmarks.dat')
EAR_BLINK = 0.12


class GazeRedirectInference:

    def __init__(self, checkpoint_path, device='cuda', convergence_h=0.05,
                 smooth_alpha=0.4, image_size=64):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.convergence_h = convergence_h
        self.smooth_alpha = smooth_alpha
        self.image_size = image_size

        self.generator = Generator(style_dim=3)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(ckpt['current_generator'])
        self.generator = self.generator.to(self.device)
        self.generator.eval()

        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(LANDMARK_MODEL)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

        self._gaze_angles = torch.tensor(
            [[-convergence_h, 0.0, -1.0],
             [convergence_h, 0.0, 1.0]],
            device=self.device
        )

        self._prev_left_center = None
        self._prev_right_center = None
        self._fixed_eye_size = None
        self._cached_mask = None
        self._frame_count = 0
        self._prev_ear_l = None
        self._prev_ear_r = None
        self._blink_frames_l = 0
        self._blink_frames_r = 0
        self._is_blinking_l = False
        self._is_blinking_r = False

    def reset(self):
        self._prev_left_center = None
        self._prev_right_center = None
        self._fixed_eye_size = None
        self._cached_mask = None
        self._frame_count = 0
        self._prev_ear_l = None
        self._prev_ear_r = None
        self._blink_frames_l = 0
        self._blink_frames_r = 0
        self._is_blinking_l = False
        self._is_blinking_r = False

    def _detect_landmarks(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = self.face_detector(rgb, 0)
        if not dets:
            return None
        d = max(dets, key=lambda r: r.area())
        lm = self.shape_predictor(rgb, d)
        left_pts = [(lm.part(n).x, lm.part(n).y) for n in range(36, 42)]
        right_pts = [(lm.part(n).x, lm.part(n).y) for n in range(42, 48)]
        l_ear = self._ear(left_pts)
        r_ear = self._ear(right_pts)
        return left_pts, right_pts, l_ear, r_ear

    @staticmethod
    def _ear(pts):
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _compute_eye_info(self, left_pts, right_pts):
        l_x1 = min(p[0] for p in left_pts)
        l_y1 = min(p[1] for p in left_pts)
        l_x2 = max(p[0] for p in left_pts)
        l_y2 = max(p[1] for p in left_pts)
        r_x1 = min(p[0] for p in right_pts)
        r_y1 = min(p[1] for p in right_pts)
        r_x2 = max(p[0] for p in right_pts)
        r_y2 = max(p[1] for p in right_pts)

        l_cx = (l_x1 + l_x2) / 2.0
        l_cy = (l_y1 + l_y2) / 2.0
        r_cx = (r_x1 + r_x2) / 2.0
        r_cy = (r_y1 + r_y2) / 2.0

        l_size = int(1.5 * max(l_x2 - l_x1, l_y2 - l_y1))
        r_size = int(1.5 * max(r_x2 - r_x1, r_y2 - r_y1))

        return (l_cx, l_cy, l_size), (r_cx, r_cy, r_size)

    def _smooth(self, raw, prev, alpha):
        if prev is None:
            return raw
        diff = abs(raw - prev)
        a = alpha if diff < 20 else alpha * 0.3
        return a * raw + (1.0 - a) * prev

    def _get_mask(self, eye_size):
        if self._cached_mask is not None and self._cached_mask.shape[0] == eye_size:
            return self._cached_mask
        mask = np.zeros((eye_size, eye_size), dtype=np.float32)
        center = (eye_size // 2, eye_size // 2)
        radius = eye_size // 2 - 2
        cv2.circle(mask, center, radius, 1.0, thickness=-1)
        ksize = 7
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        self._cached_mask = mask[..., np.newaxis]
        return self._cached_mask

    def _crop_eye(self, frame_rgb, cx, cy, eye_size):
        h, w = frame_rgb.shape[:2]
        x1 = int(cx - eye_size / 2)
        y1 = int(cy - eye_size / 2)
        x2 = x1 + eye_size
        y2 = y1 + eye_size

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        crop = frame_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                      cv2.BORDER_REFLECT_101)
        pil_img = Image.fromarray(crop)
        tensor = self.transform(pil_img)
        return tensor, x1, y1

    def _paste_eye(self, original_bgr, gen_tensor, x, y, eye_size):
        overlay = ((gen_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 0.5 * 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(overlay_bgr, (eye_size, eye_size), interpolation=cv2.INTER_LINEAR)

        h, w = original_bgr.shape[:2]
        if y < 0 or x < 0 or y + eye_size > h or x + eye_size > w:
            return

        target = original_bgr[y:y + eye_size, x:x + eye_size]
        matched = match_histograms(resized, target, channel_axis=-1)

        mask = self._get_mask(eye_size)
        blended = (matched.astype(np.float32) * mask +
                   target.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
        original_bgr[y:y + eye_size, x:x + eye_size] = blended

    def process_frame(self, frame_bgr):
        if frame_bgr is None:
            return None

        result = frame_bgr.copy()
        self._frame_count += 1

        lm_result = self._detect_landmarks(frame_bgr)

        blink_l = False
        blink_r = False

        if lm_result is None:
            if self._prev_left_center is not None and self._fixed_eye_size is not None:
                l_cx, l_cy = self._prev_left_center
                r_cx, r_cy = self._prev_right_center
                eye_size = self._fixed_eye_size
            else:
                return result
        else:
            left_pts, right_pts, l_ear, r_ear = lm_result

            l_ear = self._smooth(l_ear, self._prev_ear_l, 0.5)
            r_ear = self._smooth(r_ear, self._prev_ear_r, 0.5)
            self._prev_ear_l = l_ear
            self._prev_ear_r = r_ear

            if not self._is_blinking_l:
                if l_ear < EAR_BLINK:
                    self._blink_frames_l += 1
                else:
                    self._blink_frames_l = 0
                self._is_blinking_l = self._blink_frames_l >= 2
            else:
                if l_ear >= EAR_BLINK:
                    self._is_blinking_l = False
                    self._blink_frames_l = 0

            if not self._is_blinking_r:
                if r_ear < EAR_BLINK:
                    self._blink_frames_r += 1
                else:
                    self._blink_frames_r = 0
                self._is_blinking_r = self._blink_frames_r >= 2
            else:
                if r_ear >= EAR_BLINK:
                    self._is_blinking_r = False
                    self._blink_frames_r = 0

            blink_l = self._is_blinking_l
            blink_r = self._is_blinking_r

            if blink_l and blink_r:
                return result

            l_info, r_info = self._compute_eye_info(left_pts, right_pts)
            l_cx_raw, l_cy_raw, l_size = l_info
            r_cx_raw, r_cy_raw, r_size = r_info

            l_cx = self._smooth(l_cx_raw, self._prev_left_center[0] if self._prev_left_center else None, self.smooth_alpha)
            l_cy = self._smooth(l_cy_raw, self._prev_left_center[1] if self._prev_left_center else None, self.smooth_alpha)
            r_cx = self._smooth(r_cx_raw, self._prev_right_center[0] if self._prev_right_center else None, self.smooth_alpha)
            r_cy = self._smooth(r_cy_raw, self._prev_right_center[1] if self._prev_right_center else None, self.smooth_alpha)

            self._prev_left_center = (l_cx, l_cy)
            self._prev_right_center = (r_cx, r_cy)

            if self._fixed_eye_size is None:
                self._fixed_eye_size = max(l_size, r_size)

            eye_size = self._fixed_eye_size

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not blink_l and not blink_r:
            l_tensor, l_x, l_y = self._crop_eye(frame_rgb, l_cx, l_cy, eye_size)
            r_tensor, r_x, r_y = self._crop_eye(frame_rgb, r_cx, r_cy, eye_size)

            batch = torch.stack([l_tensor, r_tensor]).to(self.device)

            with torch.no_grad():
                generated = self.generator(batch, self._gaze_angles)

            self._paste_eye(result, generated[0], l_x, l_y, eye_size)
            self._paste_eye(result, generated[1], r_x, r_y, eye_size)
        elif not blink_l:
            l_tensor, l_x, l_y = self._crop_eye(frame_rgb, l_cx, l_cy, eye_size)
            batch = l_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                generated = self.generator(batch, self._gaze_angles[:1])
            self._paste_eye(result, generated[0], l_x, l_y, eye_size)
        elif not blink_r:
            r_tensor, r_x, r_y = self._crop_eye(frame_rgb, r_cx, r_cy, eye_size)
            batch = r_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                generated = self.generator(batch, self._gaze_angles[1:])
            self._paste_eye(result, generated[0], r_x, r_y, eye_size)

        return result

    def process_image_sequence(self, input_dir, output_dir, pattern=None):
        os.makedirs(output_dir, exist_ok=True)
        self.reset()

        def _sort_key(name):
            stem = os.path.splitext(name)[0]
            return int(stem) if stem.isdigit() else stem

        files = sorted(
            (f for f in os.listdir(input_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
             and os.path.isfile(os.path.join(input_dir, f))),
            key=_sort_key
        )

        if not files:
            print(f"No images found in {input_dir}")
            return

        print(f"Processing {len(files)} frames from {input_dir} -> {output_dir}")

        total_time = 0.0
        for i, fname in enumerate(files):
            fpath = os.path.join(input_dir, fname)
            frame = cv2.imread(fpath)
            if frame is None:
                print(f"Warning: cannot read {fpath}, skipping")
                continue

            t0 = time.time()
            result = self.process_frame(frame)
            dt = time.time() - t0
            total_time += dt

            fname = os.path.basename(fpath)
            cv2.imwrite(os.path.join(output_dir, fname), result)

            if (i + 1) % 100 == 0 or i == 0:
                fps = 1.0 / dt if dt > 0 else 0
                print(f"  Frame {i+1}/{len(files)}: {dt*1000:.1f}ms ({fps:.1f} FPS)")

        avg_ms = total_time / max(len(files), 1) * 1000
        print(f"Done. Avg: {avg_ms:.1f}ms/frame ({1000/avg_ms:.1f} FPS)")

    def process_video(self, input_path, output_path, fps=30):
        self.reset()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Cannot open video: {input_path}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

        print(f"Processing video: {input_path} ({w}x{h}, {src_fps:.1f} FPS, {total} frames)")

        total_time = 0.0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            result = self.process_frame(frame)
            dt = time.time() - t0
            total_time += dt

            writer.write(result)
            frame_idx += 1

            if frame_idx % 100 == 0:
                fps_now = 1.0 / dt if dt > 0 else 0
                print(f"  Frame {frame_idx}/{total}: {dt*1000:.1f}ms ({fps_now:.1f} FPS)")

        cap.release()
        writer.release()

        avg_ms = total_time / max(frame_idx, 1) * 1000
        print(f"Done. Avg: {avg_ms:.1f}ms/frame ({1000/avg_ms:.1f} FPS)")
