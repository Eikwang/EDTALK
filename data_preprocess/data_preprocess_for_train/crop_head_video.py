import os
import sys
import argparse
import warnings
import cv2
import numpy as np
from tqdm import tqdm
import threading
import queue
import dlib
import subprocess
warnings.filterwarnings("ignore")

# 使用独立版 FFMPEG（更稳定，适合大规模数据处理）
# 优先使用 runtime312/ffmpeg/bin/ffmpeg.exe
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_FFMPEG_PATH = os.path.join(_PROJECT_ROOT, 'runtime312', 'ffmpeg', 'bin', 'ffmpeg.exe')
_FFPROBE_PATH = os.path.join(_PROJECT_ROOT, 'runtime312', 'ffmpeg', 'bin', 'ffprobe.exe')

if not os.path.exists(_FFMPEG_PATH):
    # 回退到系统 ffmpeg
    _FFMPEG_PATH = 'ffmpeg'
    _FFPROBE_PATH = 'ffprobe'

print(f"[INFO] Using FFMPEG: {_FFMPEG_PATH}")


# 当前脚本位于 data_preprocess/data_preprocess_for_train/，向上两级到项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sort_frame_files(files):
    def _key(f):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)
    return sorted(files, key=_key)


# ============================================================
# S3FD 人脸检测器 (PyTorch, GPU 加速)
# 输出格式: (x1, y1, w, h, conf) 或 None
# ============================================================

def _get_face_detector(device='cuda', s3fd_conf_th=0.5):
    """加载 S3FD 人脸检测器"""
    try:
        import torch
        sys.path.insert(0, _PROJECT_ROOT)
        from ckpts.s3fd import S3FD
    except Exception as e:
        raise RuntimeError(f'Failed to import S3FD: {e}. '
                           f'Make sure ckpts/s3fd/sfd_face.pth exists.')

    # 自动选择设备
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    detector = S3FD(device=device)
    return ('s3fd', detector, device, float(s3fd_conf_th))


def detect_face(detector_info, frame, min_face_size=30, max_detect_size=640):
    """检测单帧中的人脸，返回最大置信度的一个

    Args:
        detector_info: 检测器元组
        frame: BGR 图像
        min_face_size: 最小人脸尺寸（像素）
        max_detect_size: 检测时图像最大短边（超过则缩小，加速 10x+）
    """
    det_type = detector_info[0]

    if det_type == 's3fd':
        _, detector, _, conf_th = detector_info

        # 缩小图像加速检测（4K → 640 短边，速度提升 10-20 倍）
        h, w = frame.shape[:2]
        scale = 1.0
        if max_detect_size > 0 and min(h, w) > max_detect_size:
            scale = max_detect_size / min(h, w)
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            small_frame = frame

        bboxes = detector.detect_faces(small_frame, conf_th=conf_th)
        if len(bboxes) == 0:
            return None

        # 选面积最大的
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best_idx = int(np.argmax(areas))
        x1, y1, x2, y2, score = bboxes[best_idx]

        # 映射回原始分辨率
        if scale != 1.0:
            x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale

        w_face = float(x2 - x1)
        h_face = float(y2 - y1)
        if max(w_face, h_face) < min_face_size:
            return None
        return (float(x1), float(y1), w_face, h_face, float(score))

    return None


def detect_faces_batch(detector_info, frames, min_face_size=30, max_detect_size=640):
    """批量检测人脸 (逐帧调用，PriorBox 已缓存)"""
    return [detect_face(detector_info, f, min_face_size, max_detect_size) for f in frames]


# ============================================================
# dlib 面部关键点检测 (68 landmarks)
# ============================================================

def _get_landmark_predictor(model_path=None):
    """加载 dlib 68 点面部关键点检测器"""
    if model_path is None:
        model_path = os.path.join(_PROJECT_ROOT, 'ckpts', 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(model_path):
        print(f'[WARN] dlib landmark model not found: {model_path}, landmark detection disabled.')
        return None
    try:
        predictor = dlib.shape_predictor(str(model_path))
        print(f'[INFO] dlib landmark predictor loaded: {model_path}')
        return predictor
    except Exception as e:
        print(f'[WARN] Failed to load dlib landmark predictor: {e}')
        return None


def detect_landmarks(predictor, frame, face_bbox):
    """检测单帧 68 个面部关键点

    Args:
        predictor: dlib shape_predictor 对象
        frame: BGR 图像（原始分辨率）
        face_bbox: (x1, y1, w, h) 人脸 bbox（原始帧坐标）

    Returns:
        (68, 2) float32 数组，每行为 (x, y)；失败返回 None
    """
    if predictor is None:
        return None
    x1, y1, w, h = face_bbox
    rect = dlib.rectangle(int(x1), int(y1), int(x1 + w), int(y1 + h))
    try:
        shape = predictor(frame, rect)
        pts = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            pts[i, 0] = shape.part(i).x
            pts[i, 1] = shape.part(i).y
        return pts
    except Exception:
        return None


# ============================================================
# 简化的头部跟踪器（2025 重写版）
# 核心原则：
# 1. 裁切框中心 = 鼻尖位置（面部几何中心近似）
# 2. 裁切框大小 = 面部尺寸 × pad_ratio（线性，无复杂平滑）
# 3. 无论人物如何移动/转头，始终保持面部在画面中心
# 4. 无论靠近或远离镜头，都在设定的 pad_ratio 范围内
# ============================================================

class HeadTracker:
    """极简跟踪器：鼻尖居中 + EMA 平滑 + 线性 pad_ratio"""

    def __init__(self, pan_alpha=0.3, zoom_alpha=0.3, dead_zone_ratio=0.12,
                 pad_ratio_range=(1.05, 1.3)):
        # 平滑系数 (0-1，越小越平滑，越大越响应迅速)
        self.pan_alpha = pan_alpha  # 位置平滑
        self.zoom_alpha = zoom_alpha  # 尺寸平滑

        # 固定 pad_ratio，使用默认值（中间值）
        self.pad_ratio = sum(pad_ratio_range) / 2.0
        self.default_pad_ratio = self.pad_ratio

        # 裁切框状态
        self.crop_cx = None  # 裁切框中心 X
        self.crop_cy = None  # 裁切框中心 Y
        self.crop_size = None  # 裁切框尺寸

        self.first_detection = False

    def update(self, face_cx=None, face_cy=None, face_size=0.0,
               face_w=0.0, face_h=0.0, frame_w=1920, frame_h=1080,
               face_detected=True):
        fh, fw = float(frame_h), float(frame_w)

        # 未检测到面部：返回画面中心 + 默认尺寸
        if not face_detected or face_size <= 0:
            default_size = min(fh, fw)
            if self.crop_size is not None:
                # 未检测时保持上一帧尺寸
                crop_size = self.crop_size
            else:
                crop_size = default_size

            # 裁切框中心保持在最后检测到的位置，或默认画面中心
            crop_cx = self.crop_cx if self.crop_cx is not None else fw / 2.0
            crop_cy = self.crop_cy if self.crop_cy is not None else fh / 2.0

            crop_x = crop_cx - crop_size / 2.0
            crop_y = crop_cy - crop_size / 2.0

            return {
                'crop_x': float(crop_x),
                'crop_y': float(crop_y),
                'crop_size': float(crop_size),
                'face_cx': float(crop_cx),
                'face_cy': float(crop_cy),
                'face_w': float(face_w),
                'face_h': float(face_h),
                'smooth_crop_size': float(crop_size),
                'pad_ratio': self.pad_ratio,
            }

        # 鼻尖近似位置：面部中心偏上 1/6（鼻子在面部上方 2/3 处）
        nose_cx = face_cx if face_cx is not None else 0.0
        nose_cy = (face_cy if face_cy is not None else 0.0) - (face_h if face_h is not None else 0.0) / 6.0

        # 目标裁切框：鼻尖居中 + 向上偏移35像素
        target_cx = nose_cx
        target_cy = nose_cy + 60.0
        target_size = (face_size if face_size is not None else 0.0) * self.pad_ratio

        # 首次检测到面部：直接初始化
        if not self.first_detection:
            self.crop_cx = target_cx
            self.crop_cy = target_cy
            self.crop_size = target_size
            self.first_detection = True
        else:
            # EMA 平滑：当前位置向目标位置插值
            # crop_cx = crop_cx * (1 - alpha) + target_cx * alpha
            self.crop_cx = (self.crop_cx if self.crop_cx is not None else 0.0) + self.pan_alpha * (target_cx - (self.crop_cx if self.crop_cx is not None else 0.0))
            self.crop_cy = (self.crop_cy if self.crop_cy is not None else 0.0) + self.pan_alpha * (target_cy - (self.crop_cy if self.crop_cy is not None else 0.0))
            self.crop_size = (self.crop_size if self.crop_size is not None else 0.0) + self.zoom_alpha * (target_size - (self.crop_size if self.crop_size is not None else 0.0))

        # ========== 画面边界约束 ==========
        crop_size = self.crop_size if self.crop_size is not None else 0.0
        # 确保不超出画面
        self.crop_cx = max(crop_size / 2.0, min(self.crop_cx if self.crop_cx is not None else 0.0, fw - crop_size / 2.0))
        self.crop_cy = max(crop_size / 2.0, min(self.crop_cy if self.crop_cy is not None else 0.0, fh - crop_size / 2.0))

        # 确保裁切框尺寸不小于画面
        crop_size = min(crop_size, min(fh, fw))

        crop_x = self.crop_cx - crop_size / 2.0
        crop_y = self.crop_cy - crop_size / 2.0

        return {
            'crop_x': float(crop_x),
            'crop_y': float(crop_y),
            'crop_size': float(crop_size),
            'face_cx': float(nose_cx),
            'face_cy': float(nose_cy),
            'face_w': float(face_w),
            'face_h': float(face_h),
            'smooth_crop_size': float(crop_size),
            'pad_ratio': float(self.pad_ratio),
        }


# ============================# 裁切 + resize============================================

def crop_and_resize(frame, crop_info, output_size=256, do_resize=True):
    fh, fw = frame.shape[:2]
    x1 = max(0, int(round(crop_info['crop_x'])))
    y1 = max(0, int(round(crop_info['crop_y'])))
    x2 = min(fw, int(round(crop_info['crop_x'] + crop_info['crop_size'])))
    y2 = min(fh, int(round(crop_info['crop_y'] + crop_info['crop_size'])))

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = frame

    if not do_resize:
        scale_factor = 1.0
        return cropped, scale_factor

    # 缩放到目标尺寸
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
    scale_factor = output_size / float(max(1, crop_info['crop_size']))
    return resized, scale_factor


# ============================================================
# 帧迭代器
# ============================================================

def iter_frames(input_path):
    """统一的帧迭代器：输入可以是视频文件或序列帧目录
    返回 (generator, total_frames) - total_frames 用于进度条显示
    """
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        files = _sort_frame_files(files)
        total = len(files)

        def _gen():
            for fname in files:
                img = cv2.imread(os.path.join(input_path, fname))
                if img is not None:
                    yield img
        return _gen(), total
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            total = 0
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def _gen():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()
        return _gen(), total


def extract_frames(video_path, output_dir, longest_edge=1920, quality=95):
    """从视频拆帧并缩放最长边到指定长度，保存为 JPEG
    
    使用独立版 FFMPEG 进行拆帧，避免 OpenCV 内置 FFmpeg 的稳定性问题。
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（序列帧保存位置）
        longest_edge: 最长边目标长度（默认 1920）
        quality: JPEG 保存质量 1-100（默认 95，视觉无损）
    Returns:
        实际处理的帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 续传：扫描已有帧
    start_idx = 0
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    idx = int(os.path.splitext(f)[0])
                    if idx >= start_idx:
                        start_idx = idx + 1
                except ValueError:
                    pass
    
    if start_idx > 0:
        print(f'[INFO] 检测到已有帧，从序号 {start_idx} 开始拆帧')
    
    # 获取视频总帧数（用于进度显示）
    try:
        probe_cmd = [
            _FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        total_frames = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    except Exception as e:
        print(f'[WARN] 无法获取视频帧数: {e}')
        total_frames = 0
    
    # 构建缩放滤镜（保持宽高比）
    # scale='min(1920,iw)':-2 表示：宽度最大1920，高度自动计算
    scale_filter = f"scale='min({longest_edge},iw)':-2"
    
    # FFMPEG 质量参数：1-31（数字越小质量越高）
    # quality=95 对应 ffmpeg 的 -q:v 2
    ffmpeg_quality = max(1, min(31, 32 - int(quality * 31 / 100)))
    
    # 使用 FFMPEG 拆帧
    # -start_number: 从指定帧开始（支持断点续传）
    cmd = [
        _FFMPEG_PATH,
        '-y',  # 覆盖已存在的文件
        '-i', video_path,
        '-vf', scale_filter,
        '-q:v', str(ffmpeg_quality),
        '-start_number', str(start_idx),
        os.path.join(output_dir, '%d.jpg'),
    ]
    
    print(f"[INFO] 拆帧: {video_path} -> {output_dir}/")
    print(f"       总帧数: {total_frames}, 起始序号: {start_idx}, 最长边: {longest_edge}")
    
    try:
        # 执行 FFMPEG，实时显示进度
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 解析 FFMPEG 输出，提取帧计数
        pbar = tqdm(
            total=total_frames if total_frames > 0 else None,
            desc='拆帧', unit='帧',
            mininterval=2.0, miniters=50, ncols=90,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        )
        
        frame_count = start_idx
        for line in process.stderr:
            # FFMPEG 输出格式: frame= 123 fps= 30 q= 2.0 ...
            if 'frame=' in line:
                try:
                    # 提取当前帧数
                    parts = line.split('frame=')
                    if len(parts) > 1:
                        frame_str = parts[1].strip().split()[0]
                        current_frame = int(frame_str)
                        # 更新进度条
                        pbar.update(current_frame - frame_count)
                        frame_count = current_frame
                except (ValueError, IndexError):
                    pass
        
        process.wait()
        pbar.close()
        
        if process.returncode != 0:
            print(f'[ERROR] FFMPEG 拆帧失败 (返回码 {process.returncode})')
            # 打印错误信息
            stderr_output = process.stderr.read() if process.stderr else ''
            if stderr_output:
                print(f'       错误信息: {stderr_output[:500]}')
            return 0
        
        # 统计实际生成的帧数
        generated_count = 0
        if os.path.isdir(output_dir):
            generated_count = len([f for f in os.listdir(output_dir) 
                                   if f.lower().endswith(('.jpg', '.jpeg'))])
        
        print(f'[OK] 拆帧完成: 生成 {generated_count} 帧')
        return generated_count
        
    except subprocess.TimeoutExpired:
        print('[ERROR] FFMPEG 执行超时')
        return 0
    except FileNotFoundError:
        print(f'[ERROR] FFMPEG 未找到: {_FFMPEG_PATH}')
        return 0
    except Exception as e:
        print(f'[ERROR] 拆帧异常: {e}')
        return 0


def process(input_path, output_dir, output_size=256, output_format='jpg',
            pad_ratio_range=(1.05, 1.3), min_face_size=30,
            num_workers=4, device='cuda', s3fd_conf_th=0.5,
            max_detect_size=640, detect_interval=1,
            landmark_model=None, enable_landmark=True):

    os.makedirs(output_dir, exist_ok=True)
    output_ext = output_format.lower()
    if output_ext not in ('png', 'jpg', 'jpeg'):
        output_ext = 'jpg'

    # 扫描输出目录中已有的图像文件，获取最大序号
    start_idx = 0
    existing_files = []
    if os.path.isdir(output_dir):
        existing_files = [f for f in os.listdir(output_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for fname in existing_files:
            stem = os.path.splitext(fname)[0]
            try:
                idx = int(stem)
                if idx >= start_idx:
                    start_idx = idx + 1
            except ValueError:
                pass
    if start_idx > 0:
        print(f'[INFO] 检测到已有 {len(existing_files)} 张图像，新帧从 {start_idx} 开始编号')

    # PNG 无压缩 = 写入速度快 5-10 倍；JPG 用适度质量
    if output_ext == 'png':
        imwrite_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    else:
        imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

    # 加载人脸检测器
    detector_info = _get_face_detector(device=device, s3fd_conf_th=s3fd_conf_th)
    det_device = detector_info[2]

    # 加载 dlib 关键点检测器
    landmark_predictor = _get_landmark_predictor(landmark_model) if enable_landmark else None

    # 初始化跟踪器
    tracker = HeadTracker(
        pad_ratio_range=pad_ratio_range,
    )

    # metadata
    metadata = {
        'input_path': input_path,
        'output_size': output_size,
        'output_format': output_ext,
        'pad_ratio_range': list(pad_ratio_range),
        'min_face_size': min_face_size,
        'num_workers': num_workers,
        'device': det_device,
        'frames': [],
    }

    # ========== 核心优化：检测与写入解耦 ==========
    # 写入队列：主线程放入裁切好的数据，写入线程消费
    write_queue = queue.Queue(maxsize=256)
    write_error_count = 0

    def _writer_loop():
        """写入线程：从队列取数据并写磁盘"""
        nonlocal write_error_count
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                break
            resized_img, out_path, params = item
            try:
                cv2.imwrite(out_path, resized_img, params)
            except Exception as e:
                write_error_count += 1
            write_queue.task_done()

    # 启动写入线程
    writer_thread = threading.Thread(target=_writer_loop, name='disk_writer', daemon=True)
    writer_thread.start()

    # 获取帧生成器 + 总数
    frame_gen, total_frames = iter_frames(input_path)

    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc='裁切', unit='帧',
        mininterval=2.0,
        miniters=50,
        ncols=90,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )

    frame_idx = start_idx
    face_hit = 0
    last_face_result = None
    landmarks_list = []

    import torch

    try:
        # 预热 GPU：让 CUDA context 和模型权重常驻 GPU
        warmup_frame = None
        for frame in frame_gen:
            if frame is not None:
                warmup_frame = frame
                break
        
        if warmup_frame is not None:
            warmup_h, warmup_w = warmup_frame.shape[:2]
            if 'source_resolution' not in metadata:
                metadata['source_resolution'] = [warmup_w, warmup_h]
            
            # 预热检测
            _warmup_result = detect_face(detector_info, warmup_frame,
                                          min_face_size=min_face_size,
                                          max_detect_size=max_detect_size)
            
            # 确保 CUDA stream 同步完毕
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 处理预热帧
            face_result = _warmup_result
            last_face_result = face_result
            
            if face_result is not None:
                x1, y1, w, h, conf = face_result
                face_size = max(w, h)
                face_cx = x1 + w / 2.0
                face_cy = y1 + h / 2.0
                crop_info = tracker.update(
                    face_cx=face_cx, face_cy=face_cy, face_size=face_size,
                    face_w=w, face_h=h, frame_w=warmup_w, frame_h=warmup_h,
                )
                face_hit += 1
            else:
                crop_info = tracker.update(
                    face_detected=False, frame_w=warmup_w, frame_h=warmup_h,
                )
            
            crop_info['frame_idx'] = frame_idx

            # Crop and resize first — we need the 256x256 image for landmark detection
            resized, _ = crop_and_resize(warmup_frame, crop_info, output_size)

            if face_result is not None:
                # Detect landmarks on the 256x256 cropped face image
                # This gives coordinates directly in 256x256 space, eliminating
                # the need for coordinate mapping during compositing
                fx1, fy1, fw, fh, _ = face_result
                # Map face bbox to 256x256 space for landmark detection
                scale = output_size / float(max(1, crop_info['crop_size']))
                bbox_256 = (
                    (fx1 - crop_info['crop_x']) * scale,
                    (fy1 - crop_info['crop_y']) * scale,
                    fw * scale,
                    fh * scale,
                )
                lms = detect_landmarks(landmark_predictor, resized, bbox_256)
            else:
                lms = None
            landmarks_list.append(lms if lms is not None else np.full((68, 2), np.nan, dtype=np.float32))

            out_path = os.path.join(output_dir, f'{frame_idx}.{output_ext}')
            write_queue.put((resized, out_path, imwrite_params))
            
            _append_meta(metadata, crop_info, face_result, frame_idx)
            frame_idx += 1
            pbar.update(1)

        # 主循环
        for frame in frame_gen:
            if frame is None:
                continue
            frame_h, frame_w = frame.shape[:2]
            if 'source_resolution' not in metadata:
                metadata['source_resolution'] = [frame_w, frame_h]

            # 跳帧检测
            if detect_interval <= 1 or frame_idx % detect_interval == 0:
                face_result = detect_face(
                    detector_info, frame,
                    min_face_size=min_face_size,
                    max_detect_size=max_detect_size,
                )
                last_face_result = face_result
            else:
                face_result = last_face_result

            if face_result is not None:
                x1, y1, w, h, conf = face_result
                face_size = max(w, h)
                face_cx = x1 + w / 2.0
                face_cy = y1 + h / 2.0
                crop_info = tracker.update(
                    face_cx=face_cx, face_cy=face_cy, face_size=face_size,
                    face_w=w, face_h=h, frame_w=frame_w, frame_h=frame_h,
                )
                face_hit += 1
            else:
                crop_info = tracker.update(
                    face_detected=False, frame_w=frame_w, frame_h=frame_h,
                )
            
            crop_info['frame_idx'] = frame_idx

            # Crop and resize first — we need the 256x256 image for landmark detection
            resized, _ = crop_and_resize(frame, crop_info, output_size)

            if face_result is not None:
                # Detect landmarks on the 256x256 cropped face image
                # This gives coordinates directly in 256x256 space
                fx1, fy1, fw, fh, _ = face_result
                scale = output_size / float(max(1, crop_info['crop_size']))
                bbox_256 = (
                    (fx1 - crop_info['crop_x']) * scale,
                    (fy1 - crop_info['crop_y']) * scale,
                    fw * scale,
                    fh * scale,
                )
                lms = detect_landmarks(landmark_predictor, resized, bbox_256)
            else:
                lms = None
            landmarks_list.append(lms if lms is not None else np.full((68, 2), np.nan, dtype=np.float32))

            out_path = os.path.join(output_dir, f'{frame_idx}.{output_ext}')
            write_queue.put((resized, out_path, imwrite_params))
            
            _append_meta(metadata, crop_info, face_result, frame_idx)
            frame_idx += 1
            pbar.update(1)

    finally:
        write_queue.put(None)
        write_queue.join()
        writer_thread.join()
        pbar.close()

    if write_error_count > 0:
        print(f'[WARN] {write_error_count} 帧写入失败')

    metadata['frame_count'] = len(metadata['frames'])

    # 保存 metadata 为 npy 到 <dataset_dir>/metadata/<video_name>.npy
    video_stem = os.path.splitext(os.path.basename(input_path))[0]
    dataset_dir = os.path.dirname(os.path.dirname(input_path))
    meta_dir = os.path.join(dataset_dir, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f'{video_stem}.npy')

    dtype = np.dtype([
        ('frame_idx', 'i4'), ('crop_x', 'f4'), ('crop_y', 'f4'),
        ('crop_size', 'f4'), ('scale_factor', 'f4'),
        ('face_detected', '?'), ('pad_ratio', 'f4'),
        ('face_bbox_x', 'f4'), ('face_bbox_y', 'f4'),
        ('face_bbox_w', 'f4'), ('face_bbox_h', 'f4'),
    ])
    records = []
    for m in metadata['frames']:
        fb = m.get('face_bbox', {})
        records.append((
            m['frame_idx'], m['crop_x'], m['crop_y'],
            m['crop_size'], m['scale_factor'],
            m['face_detected'], m['pad_ratio'],
            fb.get('x', np.nan), fb.get('y', np.nan),
            fb.get('w', np.nan), fb.get('h', np.nan),
        ))
    arr = np.array(records, dtype=dtype)
    np.save(meta_path, arr)

    # 保存 facemark 为 npy 到 <dataset_dir>/facemark/<video_name>.npy
    if len(landmarks_list) > 0:
        landmarks_arr = np.stack(landmarks_list, axis=0).astype(np.float32)
        facemark_dir = os.path.join(dataset_dir, 'facemark')
        os.makedirs(facemark_dir, exist_ok=True)
        facemark_path = os.path.join(facemark_dir, f'{video_stem}.npy')
        np.save(facemark_path, landmarks_arr)

    # 总结输出
    summary = (
        f'完成: {output_dir} | '
        f'{len(metadata["frames"])} 帧 '
        f'({output_size}×{output_size} {output_ext.upper()}) | '
        f'人脸检测率: {face_hit}/{len(metadata["frames"])} '
        f'({100 * face_hit / max(1, len(metadata["frames"])):.1f}%)'
    )
    print(summary)
    print(f'元数据: {meta_path}')

    if len(metadata['frames']) == 0:
        print('[WARN] 没有处理任何帧，请检查输入')

    return metadata, meta_path


def _append_meta(metadata, crop_info, face_result, frame_idx):
    """追加帧元数据"""
    meta = {
        'frame_idx': frame_idx,
        'crop_x': crop_info['crop_x'],
        'crop_y': crop_info['crop_y'],
        'crop_size': crop_info['crop_size'],
        'scale_factor': 1.0,
        'face_detected': face_result is not None,
        'pad_ratio': crop_info.get('pad_ratio', 0.0),
    }
    if face_result is not None:
        meta['face_bbox'] = {
            'x': crop_info['face_cx'] - crop_info['face_w'] / 2,
            'y': crop_info['face_cy'] - crop_info['face_h'] / 2,
            'w': crop_info['face_w'],
            'h': crop_info['face_h'],
        }
    metadata['frames'].append(meta)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Face detection + crop to sequence frames'
    )
    parser.add_argument('input', type=str,
                        help='Input dataset directory (must contain original_videos/ subdirectory)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <dataset_dir>/count/)')
    parser.add_argument('--output_size', type=int, default=256,
                        help='Output image size N*N (default: 256)')
    parser.add_argument('--output_format', type=str, default='jpg',
                        choices=['png', 'jpg'],
                        help='Output image format (default: jpg)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for I/O (default: 4)')

    # S3FD 检测参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for S3FD detection (default: cuda, auto-falls-back to cpu)')
    parser.add_argument('--s3fd_conf_th', type=float, default=0.5,
                        help='S3FD confidence threshold (default: 0.5)')
    parser.add_argument('--max_detect_size', type=int, default=480,
                        help='Max image short side for detection (4K->1280 = 10x faster, default: 1280)')
    parser.add_argument('--detect_interval', type=int, default=1,
                        help='Detect every N frames (1=all frames, 5=every 5th frame, default: 1)')

    parser.add_argument('--pad_ratio_min', type=float, default=1.2,
                        help='Minimum pad ratio (tight crop, default: 1.2)')
    parser.add_argument('--pad_ratio_max', type=float, default=1.3,
                        help='Maximum pad ratio (default: 1.3)')
    parser.add_argument('--min_face_size', type=int, default=30,
                        help='Minimum face size in pixels (default: 30)')

    # 关键点检测参数
    parser.add_argument('--landmark_model', type=str, default=None,
                        help='Path to dlib shape_predictor model (default: <project_root>/ckpts/shape_predictor_68_face_landmarks.dat)')
    parser.add_argument('--no-landmark', action='store_true', default=False,
                        help='Disable facial landmark detection')

    # 拆帧参数
    parser.add_argument('--longest_edge', type=int, default=1920,
                        help='Target longest edge length for frame extraction (default: 1920)')
    parser.add_argument('--extract_quality', type=int, default=95,
                        help='JPEG quality for extracted frames 1-100 (default: 95, visually lossless)')

    opt = parser.parse_args()

    # 验证输入目录存在
    dataset_dir = os.path.abspath(opt.input)
    if not os.path.isdir(dataset_dir):
        print(f'[ERROR] Input directory does not exist: {dataset_dir}')
        sys.exit(1)

    # 在数据集目录下寻找 original_videos 子目录
    original_videos_dir = os.path.join(dataset_dir, 'original_videos')
    if not os.path.isdir(original_videos_dir):
        print(f'[ERROR] original_videos/ subdirectory not found in: {dataset_dir}')
        sys.exit(1)

    # 遍历 original_videos 目录中的视频文件
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v')
    video_files = [f for f in os.listdir(original_videos_dir)
                   if f.lower().endswith(video_extensions)]
    video_files.sort()

    if len(video_files) == 0:
        print(f'[ERROR] No video files found in: {original_videos_dir}')
        sys.exit(1)

    # 默认输出目录: <dataset_dir>/count/
    if opt.output_dir is None:
        opt.output_dir = os.path.join(dataset_dir, 'count')

    # 确保输出目录是绝对路径
    if not os.path.isabs(opt.output_dir):
        opt.output_dir = os.path.join(_PROJECT_ROOT, opt.output_dir)

    # 启动消息
    print(f'Dataset: {os.path.basename(dataset_dir)} | '
          f'Videos: {len(video_files)} | '
          f'Output: {os.path.basename(opt.output_dir)}'
          f' | {opt.output_size}x{opt.output_size} {opt.output_format.upper()}'
          f' | device={opt.device} | workers={opt.num_workers}'
          f' | longest_edge={opt.longest_edge}')

    # 遍历每个视频文件进行处理
    for video_file in video_files:
        video_path = os.path.join(original_videos_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        # 第一步：拆帧并缩放最长边到 longest_edge
        orifra_dir = os.path.join(dataset_dir, 'orifra', video_name)
        print(f'\n{"="*60}')
        print(f'[Step 1] 拆帧: {video_file} -> orifra/{video_name}/ (longest_edge={opt.longest_edge})')
        print(f'{"="*60}')
        n_extracted = extract_frames(video_path, orifra_dir, longest_edge=opt.longest_edge, quality=opt.extract_quality)
        print(f'拆帧完成: {n_extracted} 帧')

        # 第二步：面部裁切 + 关键点
        video_output_dir = os.path.join(opt.output_dir, video_name)
        print(f'\n{"="*60}')
        print(f'[Step 2] 裁切 + 关键点: orifra/{video_name}/ -> {video_name}/')
        print(f'{"="*60}')

        process(
            input_path=orifra_dir,
            output_dir=video_output_dir,
            output_size=opt.output_size,
            output_format=opt.output_format,
            pad_ratio_range=(opt.pad_ratio_min, opt.pad_ratio_max),
            min_face_size=opt.min_face_size,
            num_workers=opt.num_workers,
            device=opt.device,
            s3fd_conf_th=opt.s3fd_conf_th,
            max_detect_size=opt.max_detect_size,
            detect_interval=opt.detect_interval,
            landmark_model=opt.landmark_model,
            enable_landmark=not opt.no_landmark,
        )

    print(f'\n{"="*60}')
    print(f'All done. {len(video_files)} videos processed.')
    print(f'Output directory: {opt.output_dir}')
    print(f'{"="*60}')
