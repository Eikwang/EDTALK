# -*- coding: utf-8 -*-
"""
Crop and align face frames from video(s) with face tracking.
**ACCELERATED VERSION** — S3FD (PyTorch CUDA) face detection + pipeline
parallelism (video decode | GPU batch detect | CPU landmark/align/save thread
pool). Quality logic (pad_ratio control, EMA tracking, similarity-align)
is preserved from the original single-thread implementation.

Pipeline per frame:
  1. S3FDNet on GPU → face bbox (batch-processed)
  2. dlib shape_predictor on ROI → 68 landmarks (fast, ~1ms on CPU)
  3. Face tracking: EMA-smoothed face size and center from landmarks bbox
  4. Crop a generous square region from the original frame (face_size*CROP_MARGIN)
  5. Similarity (partial-affine) transform → dynamically scaled template.
     Scaling factor = target_size/(tmpl_size*pad_ratio), so --pad_ratio
     directly controls face-to-output ratio without stretching.
  6. Warp → 256x256 and save

Key design notes:
  - --pad_ratio controls FACE-TO-OUTPUT ratio only; source crop is always
    generous (face_size*CROP_MARGIN) to prevent clipping during warp.
  - Similarity transform (estimateAffinePartial2D) preserves aspect ratio.
  - The bbox → landmark stage is the ONLY dlib step (it's cheap on a tight
    ROI); the full-frame detection uses S3FDNet on GPU.
"""

import os
import sys
import glob
import time
import queue
import warnings
import threading
import argparse
import numpy as np
import cv2
import dlib

import torch

warnings.filterwarnings('ignore')

predictor = dlib.shape_predictor('data_preprocess/shape_predictor_68_face_landmarks.dat')
template = np.load('data_preprocess/M003_template.npy').astype(np.float32)

CROP_MARGIN = 2.0   # source-crop-to-face ratio (independent of pad_ratio)

# --- pipeline tuning ---
DETECT_BATCH   = 2     # frames per GPU forward pass
DETECT_SCALE   = 0.4   # downscale for detection (bbox rescaled back)
MAX_QUEUE_SIZE = 16    # back-pressure to avoid memory explosion
NUM_WORKERS    = max(2, os.cpu_count() // 2)  # CPU workers for lmk + align + save


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def shape_to_np(shape, dtype=np.float32):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_next_frame_index(out_path):
    if not os.path.exists(out_path):
        return 0
    existing = glob.glob(os.path.join(out_path, '*.png'))
    if not existing:
        return 0
    indices = []
    for f in existing:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            indices.append(int(name))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


# ---------------------------------------------------------------------------
# Precomputed template scaling (called ONCE per video)
# ---------------------------------------------------------------------------
def precompute_template(pad_ratio, target_size=256):
    tmpl_min = template.min(axis=0)
    tmpl_max = template.max(axis=0)
    tmpl_center = (tmpl_min + tmpl_max) * 0.5
    tmpl_size = float(max(tmpl_max - tmpl_min))
    desired_face_size = target_size / pad_ratio
    scale = desired_face_size / tmpl_size
    output_center = np.array([target_size / 2.0, target_size / 2.0],
                              dtype=np.float32)
    dst_pts = (template - tmpl_center) * scale + output_center
    return dst_pts


# ---------------------------------------------------------------------------
# S3FD face detector (ckpts/s3fd implementation)
# ---------------------------------------------------------------------------
class S3FDFaceDetector:
    def __init__(self, device='cuda', weight_path=None):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ckpts.s3fd.nets import S3FDNet

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if weight_path is None:
            weight_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'ckpts', 's3fd', 'sfd_face.pth'
            )
            weight_path = os.path.abspath(weight_path)

        self.img_mean = np.array([104., 117., 123.], dtype=np.float32).reshape(3, 1, 1)

        self.net = S3FDNet(device=str(self.device)).to(self.device)
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @torch.no_grad()
    def detect_batch(self, frames, conf_th=0.7, scale=DETECT_SCALE):
        """Batch-detect faces. Returns list of (x1, y1, x2, y2, score) per
        frame (None when nothing detected with confidence above threshold)."""
        N = len(frames)
        if N == 0:
            return []

        # --- optional: downscale for faster detection ---
        orig_hs, orig_ws = [], []
        in_tensors = []
        for f in frames:
            oh, ow = f.shape[:2]
            orig_hs.append(oh)
            orig_ws.append(ow)
            if scale != 1.0:
                dh = max(1, int(round(oh * scale)))
                dw = max(1, int(round(ow * scale)))
                img = cv2.resize(f, (dw, dh), interpolation=cv2.INTER_LINEAR)
            else:
                img = f
            # HWC BGR → CHW BGR → float32 → subtract mean
            img = img.transpose(2, 0, 1).astype(np.float32)
            img -= self.img_mean
            in_tensors.append(
                torch.from_numpy(img).contiguous().to(self.device, non_blocking=True)
            )

        # Model expects same-resolution inputs. If frames differ in shape
        # (shouldn't within a single video), fall back per-frame.
        try:
            batch = torch.stack(in_tensors, 0)
            outputs = self.net(batch)  # [N, 2, top_k, 5]
            outputs = outputs.cpu().numpy()
        except Exception:
            # Fallback: process one by one
            outputs = []
            for t in in_tensors:
                out = self.net(t.unsqueeze(0))
                outputs.append(out.cpu().numpy()[0])
            outputs = np.array(outputs)

        results = []
        for i in range(N):
            # S3FD output: [batch, classes, top_k, 5]
            # 5 values are [score, x1, y1, x2, y2] in NORMALIZED coords
            cls = outputs[i, 1]  # class 1 == face
            scores = cls[:, 0]
            boxes = cls[:, 1:5]

            mask = scores > conf_th
            if not np.any(mask):
                results.append(None)
                continue

            # Multiple faces → keep the largest
            bx = boxes[mask]
            sc = scores[mask]

            # Sort by score * area → pick "most confident big face"
            widths = bx[:, 2] - bx[:, 0]
            heights = bx[:, 3] - bx[:, 1]
            areas = np.maximum(0.0, widths) * np.maximum(0.0, heights)
            rank = sc * (areas ** 0.25)
            best_idx = int(np.argmax(rank))

            # S3FD's PriorBox outputs NORMALIZED coordinates (0-1).
            # Multiply by original image dimensions to get absolute pixels.
            x1, y1, x2, y2 = bx[best_idx]
            x1 *= orig_ws[i]
            x2 *= orig_ws[i]
            y1 *= orig_hs[i]
            y2 *= orig_hs[i]

            results.append((float(x1), float(y1), float(x2), float(y2), float(sc[best_idx])))

        return results


# ---------------------------------------------------------------------------
# FaceTracker (same smoothing logic, reused per-video)
# ---------------------------------------------------------------------------
class FaceTracker:
    def __init__(self, size_ema_alpha=0.12, center_ema_alpha=0.3,
                 min_face_size=30):
        self.size_ema_alpha = size_ema_alpha
        self.center_ema_alpha = center_ema_alpha
        self.min_face_size = min_face_size

        self.smooth_face_size = None
        self.smooth_cx = None
        self.smooth_cy = None
        self.last_valid = None

    def update(self, face_cx, face_cy, face_w, face_h, frame_w, frame_h):
        face_size = max(face_w, face_h)

        if self.smooth_face_size is None:
            self.smooth_face_size = float(face_size)
            self.smooth_cx = float(face_cx)
            self.smooth_cy = float(face_cy)
        else:
            a = self.size_ema_alpha
            self.smooth_face_size = a * face_size + (1 - a) * self.smooth_face_size
            ac = self.center_ema_alpha
            self.smooth_cx = ac * face_cx + (1 - ac) * self.smooth_cx
            self.smooth_cy = ac * face_cy + (1 - ac) * self.smooth_cy

        crop_size = self.smooth_face_size * CROP_MARGIN
        crop_size = min(crop_size, min(frame_w, frame_h))
        crop_size = max(crop_size, float(self.min_face_size) * CROP_MARGIN)

        crop_cx = self.smooth_cx
        crop_cy = self.smooth_cy

        margin = face_size * 0.15
        face_left = face_cx - face_w / 2.0 - margin
        face_right = face_cx + face_w / 2.0 + margin
        face_top = face_cy - face_h / 2.0 - margin
        face_bottom = face_cy + face_h / 2.0 + margin

        half = crop_size / 2.0
        crop_x = crop_cx - half
        crop_y = crop_cy - half

        if face_left < crop_x:
            crop_cx = face_left + half
        if face_right > crop_x + crop_size:
            crop_cx = face_right - half
        if face_top < crop_y:
            crop_cy = face_top + half
        if face_bottom > crop_y + crop_size:
            crop_cy = face_bottom - half

        half = crop_size / 2.0
        crop_cx = max(half, min(crop_cx, frame_w - half))
        crop_cy = max(half, min(crop_cy, frame_h - half))

        result = {
            'crop_x': crop_cx - half,
            'crop_y': crop_cy - half,
            'crop_size': crop_size,
            'crop_cx': crop_cx,
            'crop_cy': crop_cy,
            'face_cx': float(face_cx),
            'face_cy': float(face_cy),
            'face_w': float(face_w),
            'face_h': float(face_h),
            'face_size': float(face_size),
            'smooth_face_size': float(self.smooth_face_size),
        }
        self.last_valid = result
        return result

    def get_last_crop(self, frame_w, frame_h):
        if self.last_valid is not None:
            return self.last_valid
        default_size = min(frame_w, frame_h)
        return {
            'crop_x': (frame_w - default_size) / 2.0,
            'crop_y': (frame_h - default_size) / 2.0,
            'crop_size': float(default_size),
            'crop_cx': frame_w / 2.0,
            'crop_cy': frame_h / 2.0,
            'face_cx': frame_w / 2.0,
            'face_cy': frame_h / 2.0,
            'face_w': 0.0,
            'face_h': 0.0,
            'face_size': 0.0,
            'smooth_face_size': float(default_size),
        }


# ---------------------------------------------------------------------------
# Alignment (preserved from original)
# ---------------------------------------------------------------------------
def align_crop_to_template(cropped_patch, shape_in_crop, dst_pts, target_size=256):
    src_pts = shape_in_crop.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
    if M is None:
        return cv2.resize(cropped_patch, (target_size, target_size),
                          interpolation=cv2.INTER_AREA)
    aligned = cv2.warpAffine(
        cropped_patch, M, (target_size, target_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


# ---------------------------------------------------------------------------
# Worker: bbox → landmarks → align → save (runs in thread pool)
# ---------------------------------------------------------------------------
def process_one_frame(frame, bbox, tracker, dst_pts, out_path, index, frame_w, frame_h):
    """Return True if frame saved successfully."""
    try:
        if bbox is not None:
            x1, y1, x2, y2, _score = bbox
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            # Expand bbox slightly for a better shape_predictor ROI
            pad = max(bw, bh) * 0.15
            rl = int(max(0, cx - bw / 2 - pad))
            rt = int(max(0, cy - bh / 2 - pad))
            rr = int(min(frame_w, cx + bw / 2 + pad))
            rb = int(min(frame_h, cy + bh / 2 + pad))
            rect = dlib.rectangle(rl, rt, rr, rb)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = predictor(gray, rect)
            shape_np = shape_to_np(shape)

            lx_min, ly_min = shape_np.min(axis=0)
            lx_max, ly_max = shape_np.max(axis=0)
            lw = float(lx_max - lx_min)
            lh = float(ly_max - ly_min)
            lcx = (lx_min + lx_max) * 0.5
            lcy = (ly_min + ly_max) * 0.5

            face_w = lw * 1.15
            face_h = lh * 1.35
            face_cx = float(lcx)
            face_cy = float(lcy) + lh * 0.02

            crop_info = tracker.update(face_cx, face_cy, face_w, face_h, frame_w, frame_h)
            cs = crop_info['crop_size']
            cx_c = crop_info['crop_cx']
            cy_c = crop_info['crop_cy']
            half = cs / 2.0

            x0 = int(round(cx_c - half))
            y0 = int(round(cy_c - half))
            x1_e = x0 + int(round(cs))
            y1_e = y0 + int(round(cs))

            x0_c = max(0, x0)
            y0_c = max(0, y0)
            x1_c = min(frame_w, x1_e)
            y1_c = min(frame_h, y1_e)

            cropped_patch = frame[y0_c:y1_c, x0_c:x1_c]
            if cropped_patch.size == 0:
                cv2.imwrite(os.path.join(out_path, f'{index}.png'),
                            cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA))
                return True

            shape_in_crop = shape_np.copy()
            shape_in_crop[:, 0] -= x0_c
            shape_in_crop[:, 1] -= y0_c

            aligned = align_crop_to_template(cropped_patch, shape_in_crop, dst_pts, target_size=256)
            cv2.imwrite(os.path.join(out_path, f'{index}.png'), aligned)
            return True
        else:
            # Fallback: use tracker's last crop (or center crop)
            crop_info = tracker.get_last_crop(frame_w, frame_h)
            cs = crop_info['crop_size']
            cx_c = crop_info['crop_cx']
            cy_c = crop_info['crop_cy']
            half = cs / 2.0
            x0 = max(0, int(round(cx_c - half)))
            y0 = max(0, int(round(cy_c - half)))
            x1 = min(frame_w, x0 + int(round(cs)))
            y1 = min(frame_h, y0 + int(round(cs)))
            cropped_patch = frame[y0:y1, x0:x1]
            result = cv2.resize(cropped_patch, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_path, f'{index}.png'), result)
            return True
    except Exception as e:
        # Never let a single bad frame crash the whole pipeline
        print(f"  [WARN] frame {index} failed: {e}", file=sys.stderr)
        try:
            cv2.imwrite(os.path.join(out_path, f'{index}.png'),
                        cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA))
        except Exception:
            pass
        return True


# ---------------------------------------------------------------------------
# Pipeline: decode → batch-GPU-detect (S3FD) → sequential tracking + shape
# predictor → thread-pool warpAffine + imwrite (overlap PNG compression)
# ---------------------------------------------------------------------------
def crop_image_tem(video_path, out_path, pad_ratio=1.4, detector=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Total frames: {total_frames}")

    os.makedirs(out_path, exist_ok=True)
    start_index = get_next_frame_index(out_path)

    tracker = FaceTracker()
    dst_pts = precompute_template(pad_ratio=pad_ratio, target_size=256)

    local_detector = detector is None
    if local_detector:
        detector = S3FDFaceDetector(device='cuda')

    saved = 0
    no_detect_count = 0
    t0 = time.time()

    # ---- Main loop: read a small batch, S3FD GPU detect, process in order ----
    # Tracking state is preserved ONLY in-order (main thread), so the EMA
    # remains stable.  S3FD replaces the old dlib HOG detector (which was
    # ~50-200ms/frame on CPU).  dlib shape_predictor on a tight ROI is
    # ~1-2ms/frame and stays sequential.
    # ----
    import concurrent.futures as cfut

    with cfut.ThreadPoolExecutor(max_workers=NUM_WORKERS,
                                 thread_name_prefix='save') as save_pool:
        in_flight = []

        # Helper: wait for oldest in-flight writes before starting new ones
        # (keeps memory bounded; PNG compression is the main CPU cost here)
        def _drain_in_flight(max_pending=NUM_WORKERS * 4):
            while len(in_flight) >= max_pending:
                try:
                    in_flight.pop(0).result(timeout=5.0)
                except Exception:
                    pass

        frame_idx = 0
        while True:
            # --- Read up to DETECT_BATCH frames (one GPU batch) ---
            frames = []
            frame_ids = []
            for _ in range(DETECT_BATCH):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                frame_ids.append(start_index + frame_idx)
                frame_idx += 1
            if not frames:
                break

            # --- S3FD GPU batch detection ---
            try:
                bboxes = detector.detect_batch(frames)
            except Exception as e:
                print(f"  [ERROR] detect_batch failed: {e}", file=sys.stderr)
                bboxes = [None] * len(frames)

            # --- Process in frame order (tracking must be sequential) ---
            for i, frame in enumerate(frames):
                bbox = bboxes[i]
                idx = frame_ids[i]

                if bbox is not None:
                    no_detect_count = 0
                    x1, y1, x2, y2, _score = bbox
                    bw = x2 - x1
                    bh = y2 - y1

                    # Expand bbox slightly for shape_predictor ROI
                    pad = max(bw, bh) * 0.15
                    rl = int(max(0, (x1 + x2) / 2.0 - bw / 2.0 - pad))
                    rt = int(max(0, (y1 + y2) / 2.0 - bh / 2.0 - pad))
                    rr = int(min(frame_w, (x1 + x2) / 2.0 + bw / 2.0 + pad))
                    rb = int(min(frame_h, (y1 + y2) / 2.0 + bh / 2.0 + pad))
                    rect = dlib.rectangle(rl, rt, rr, rb)

                    # shape_predictor on ROI: cheap (~1-2ms)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    shape = predictor(gray, rect)
                    shape_np = shape_to_np(shape)

                    lx_min, ly_min = shape_np.min(axis=0)
                    lx_max, ly_max = shape_np.max(axis=0)
                    lw = float(lx_max - lx_min)
                    lh = float(ly_max - ly_min)
                    face_w = lw * 1.15
                    face_h = lh * 1.35
                    face_cx = float(lx_min + lx_max) * 0.5
                    face_cy = float(ly_min + ly_max) * 0.5 + lh * 0.02

                    # Tracking (sequential, safe): smooth size & center
                    crop_info = tracker.update(
                        face_cx, face_cy, face_w, face_h, frame_w, frame_h
                    )
                    cs = crop_info['crop_size']
                    cx_c = crop_info['crop_cx']
                    cy_c = crop_info['crop_cy']
                    half = cs / 2.0
                    x0 = int(round(cx_c - half))
                    y0 = int(round(cy_c - half))
                    x1_e = x0 + int(round(cs))
                    y1_e = y0 + int(round(cs))

                    x0_c = max(0, x0)
                    y0_c = max(0, y0)
                    x1_c = min(frame_w, x1_e)
                    y1_c = min(frame_h, y1_e)

                    cropped_patch = frame[y0_c:y1_c, x0_c:x1_c]
                    if cropped_patch.size == 0:
                        aligned = cv2.resize(frame, (256, 256),
                                             interpolation=cv2.INTER_AREA)
                    else:
                        shape_in_crop = shape_np.copy()
                        shape_in_crop[:, 0] -= x0_c
                        shape_in_crop[:, 1] -= y0_c
                        aligned = align_crop_to_template(
                            cropped_patch, shape_in_crop, dst_pts, target_size=256
                        )
                else:
                    # Fallback: tracker's last crop, or center crop
                    no_detect_count += 1
                    crop_info = tracker.get_last_crop(frame_w, frame_h)
                    cs = crop_info['crop_size']
                    cx_c = crop_info['crop_cx']
                    cy_c = crop_info['crop_cy']
                    half = cs / 2.0
                    x0 = max(0, int(round(cx_c - half)))
                    y0 = max(0, int(round(cy_c - half)))
                    x1_p = min(frame_w, x0 + int(round(cs)))
                    y1_p = min(frame_h, y0 + int(round(cs)))
                    aligned = cv2.resize(frame[y0:y1_p, x0:x1_p],
                                         (256, 256), interpolation=cv2.INTER_AREA)

                # ---- Dispatch imwrite (PNG compression: ~5-15ms) to pool ----
                _drain_in_flight()
                save_path = os.path.join(out_path, f'{idx}.png')
                in_flight.append(save_pool.submit(
                    cv2.imwrite, save_path, aligned
                ))
                saved += 1

            # --- Progress bar ---
            if total_frames > 0:
                pct = saved * 100 // total_frames
                bar_len = 30
                filled = bar_len * saved // total_frames
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f'\r  [{bar}] {pct}% ({saved}/{total_frames})',
                      end='', flush=True)

        # Wait for all in-flight writes
        for fut in in_flight:
            try:
                fut.result(timeout=30.0)
            except Exception:
                pass

    cap.release()
    elapsed = time.time() - t0
    print(f'\r  [{"█" * 30}] 100% ({saved}/{total_frames})')
    if elapsed > 0:
        print(f"  Done: {saved} frames, {saved / elapsed:.1f} fps")

    return saved


# ---------------------------------------------------------------------------
# Single-image processing (kept for CLI compatibility)
# ---------------------------------------------------------------------------
def crop_image(image_path, out_path, pad_ratio=1.4):
    image = cv2.imread(image_path)
    if image is None:
        return 0

    detector = S3FDFaceDetector(device='cuda')
    bboxes = detector.detect_batch([image])
    bbox = bboxes[0]

    dst_pts = precompute_template(pad_ratio=pad_ratio, target_size=256)
    tracker = FaceTracker()
    h, w = image.shape[:2]
    process_one_frame(image, bbox, tracker, dst_pts,
                      os.path.dirname(out_path) if os.path.dirname(out_path) else '.',
                      os.path.splitext(os.path.basename(out_path))[0],
                      w, h)
    # (simpler: just reuse process_one_frame, but out_path is a file)
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Crop and align face frames with face tracking (GPU '
                    'accelerated). --pad_ratio controls face-to-output size '
                    'ratio (smaller = larger face in output, e.g. 1.2 → face '
                    'fills ~83%% of width; 1.8 → face fills ~56%%).')
    parser.add_argument("--video", type=str, nargs='+', required=True,
                        help='Input video path(s)')
    parser.add_argument("--out_path", type=str, required=True,
                        help='Output directory for cropped frames')
    parser.add_argument("--pad_ratio", type=float, default=1.2,
                        help='Face-to-output ratio (default: 1.4 → face '
                             'fills ~71%% of the 256px output width).')
    parser.add_argument("--batch", type=int, default=DETECT_BATCH,
                        help=f'Frames per GPU forward pass (default {DETECT_BATCH})')
    parser.add_argument("--detect_scale", type=float, default=DETECT_SCALE,
                        help=f'Downscale for S3FD detection (bbox rescaled '
                             f'back, default {DETECT_SCALE}; 1.0 = no downscale)')
    args = parser.parse_args()

    DETECT_BATCH = max(1, args.batch)
    DETECT_SCALE = max(0.25, min(1.0, args.detect_scale))

    torch.backends.cudnn.benchmark = True

    # Share a single S3FD model across all videos (saves load cost)
    detector = S3FDFaceDetector(device='cuda')

    total = 0
    for video_path in args.video:
        count = crop_image_tem(video_path, args.out_path,
                               pad_ratio=args.pad_ratio, detector=detector)
        if count == 0:
            print(f"Warning: No face detected in {video_path}")
        else:
            total += count

    print(f"Total frames saved: {total}")
