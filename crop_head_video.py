import os
import sys
import json
import argparse
import math
import warnings
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing as mp
from functools import partial
import onnxruntime as ort

# 检查 onnxruntime 能否正确加载 CUDA
_available_providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in _available_providers:
    print(f"[INFO] ONNX Runtime CUDA 可用，版本: {ort.__version__}")
else:
    print(f"[WARN] ONNX Runtime CUDA 不可用，可用: {_available_providers}")

warnings.filterwarnings("ignore")


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


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

    def update(self, face_cx=None, face_cy=None, face_size=0,
               face_w=0, face_h=0, frame_w=1920, frame_h=1080,
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
        nose_cx = face_cx
        nose_cy = face_cy - face_h / 6.0

        # 目标裁切框：鼻尖居中 + 向上偏移35像素
        target_cx = nose_cx
        target_cy = nose_cy + 60.0
        target_size = face_size * self.pad_ratio

        # 首次检测到面部：直接初始化
        if not self.first_detection:
            self.crop_cx = target_cx
            self.crop_cy = target_cy
            self.crop_size = target_size
            self.first_detection = True
        else:
            # EMA 平滑：当前位置向目标位置插值
            # crop_cx = crop_cx * (1 - alpha) + target_cx * alpha
            self.crop_cx = self.crop_cx + self.pan_alpha * (target_cx - self.crop_cx)
            self.crop_cy = self.crop_cy + self.pan_alpha * (target_cy - self.crop_cy)
            self.crop_size = self.crop_size + self.zoom_alpha * (target_size - self.crop_size)

        # ========== 画面边界约束 ==========
        crop_size = self.crop_size
        # 确保不超出画面
        self.crop_cx = max(crop_size / 2.0, min(self.crop_cx, fw - crop_size / 2.0))
        self.crop_cy = max(crop_size / 2.0, min(self.crop_cy, fh - crop_size / 2.0))

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
# GFPGAN 面部增强
# ============================================================

class CropFaceEnhancer:
    """局部面部区域增强：仅增强人脸所在区域，保留背景"""

    def __init__(self, model_path=None, strength=0.4):
        self.strength = max(0.0, min(1.0, float(strength)))
        self.session = None
        providers = ['CPUExecutionProvider']  # 防御性编程：提前初始化默认值

        if model_path is None:
            model_path = os.path.join(_PROJECT_ROOT, 'gfpgan-1024.onnx')

        if not os.path.exists(model_path):
            print(f'[WARN] GFPGAN model not found: {model_path}, face enhancement disabled.')
            return

        try:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
        except Exception as e:
            print(f'[WARN] onnxruntime not available: {e}, face enhancement disabled.')
            return

        try:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f'[INFO] GFPGAN loaded: {model_path}, strength={self.strength}, providers={providers}')
        except Exception as e:
            print(f'[WARN] Failed to load GFPGAN: {e}, face enhancement disabled.')
            self.session = None

    def enhance(self, img, face_bbox=None):
        """
        Args:
            img: 256×256 BGR ndarray
            face_bbox: [x, y, w, h] 在 img 坐标系中的人脸 bbox
        """
        if self.session is None or self.strength == 0.0:
            return img
        if face_bbox is None:
            return img

        h, w = img.shape[:2]
        fx, fy, fw, fh = [float(v) for v in face_bbox]

        expand = 0.5
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        new_fw = fw * (1 + expand)
        new_fh = fh * (1 + expand)
        ex = int(max(0, cx - new_fw / 2))
        ey = int(max(0, cy - new_fh / 2))
        ex2 = int(min(w, cx + new_fw / 2))
        ey2 = int(min(h, cy + new_fh / 2))

        face_region = img[ey:ey2, ex:ex2]
        if face_region.size == 0:
            return img
        region_h, region_w = face_region.shape[:2]

        face_512 = cv2.resize(face_region, (512, 512), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        rgb = rgb.transpose(2, 0, 1)[np.newaxis]

        output = self.session.run(None, {self.input_name: rgb})[0]
        output = (output[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = cv2.resize(output, (region_w, region_h), interpolation=cv2.INTER_AREA)

        mask = np.zeros((region_h, region_w), dtype=np.float32)
        center = (region_w // 2, region_h // 2)
        axes = (region_w // 2, region_h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask = mask[:, :, np.newaxis]

        blended = (face_region.astype(np.float32) * (1 - self.strength * mask) +
                   output.astype(np.float32) * (self.strength * mask))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        result = img.copy()
        result[ey:ey2, ex:ex2] = blended
        return result


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


_thread_local = threading.local()

def _get_worker_session(model_dir):
    """获取当前线程专属的 rembg session (绕过 rembg 的 new_session Bug)"""
    if not hasattr(_thread_local, "session"):
        os.environ["U2NET_HOME"] = model_dir
        thread_name = threading.current_thread().name  # 提前定义，避免 except 中未定义

        # 1. 让 U2netSession 自动下载模型（如果不存在）
        from rembg.sessions.u2net import U2netSession
        model_path = U2netSession.download_models()

        # 2. 手动配置 SessionOptions
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        # 3. 智能选择 Providers (GPU 优先)
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        try:
            # 4. 直接使用原生 ort.InferenceSession 加载模型
            inner_session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=providers
            )

            # 5. 实例化 rembg 的 U2netSession 并注入我们自定义的 inner_session
            _thread_local.session = U2netSession("u2net", ort.SessionOptions())
            _thread_local.session.inner_session = inner_session

            # 仅在线程 0 打印一次加载成功信息
            if "worker_0" in thread_name or thread_name == "MainThread":
                print(f"\n[INFO] 工作线程 {thread_name} 成功加载 U2Net，使用: {providers}")

        except Exception as e:
            print(f"\n[ERROR] 工作线程 {thread_name} 加载 U2Net 失败: {type(e).__name__}: {e}")
            _thread_local.session = None

    return _thread_local.session

# ============================================================
# 全局函数：用于多线程处理（必须放在全局作用域）
# ============================================================
def _process_single_frame_global(args):
    """
    在工作进程中执行：裁切 + 抠图 + 缩放 + 写入
    必须是全局函数才能被 ProcessPoolExecutor pickle
    """
    frame, crop_info, frame_idx, output_dir, output_size, output_ext, imwrite_params, model_dir, face_result, enable_rembg = args
    
    import os
    import cv2
    import numpy as np
    from PIL import Image
    try:
        # === 步骤 1: 裁切面部（不缩放，用于高分辨率抠图）===
        fh, fw = frame.shape[:2]
        x1 = max(0, int(round(crop_info['crop_x'])))
        y1 = max(0, int(round(crop_info['crop_y'])))
        x2 = min(fw, int(round(crop_info['crop_x'] + crop_info['crop_size'])))
        y2 = min(fh, int(round(crop_info['crop_y'] + crop_info['crop_size'])))
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = frame
        
        # === 步骤 2: 抠图（在原始裁切尺寸上进行，使用GPU）===
        # 使用线程本地 session，避免多线程共享导致的 CUDA 冲突
        if enable_rembg:
            try:
                session = _get_worker_session(model_dir)
                if session is not None:
                    frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)

                    # 调用 rembg 抠图
                    pil_img_no_bg = remove(
                        pil_img,
                        session=session,
                        alpha_matting=False,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=3,
                    )

                    # 替换为纯白背景
                    white_bg = Image.new("RGBA", pil_img_no_bg.size, (255, 255, 255, 255))
                    final_img = Image.alpha_composite(white_bg, pil_img_no_bg).convert("RGB")
                    cropped = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
                else:
                    # 如果 session 为 None，说明模型加载彻底失败了
                    if frame_idx % 100 == 0:  # 避免刷屏，每100帧提示一次
                        print(f"\n[WARN] 帧 {frame_idx}: rembg session 加载失败，跳过抠图。请检查 onnxruntime-gpu 是否正确安装。")
            except Exception as e:
                # 【关键修改】：打印出真实的错误原因，不再静默 pass！
                print(f"\n[ERROR] 帧 {frame_idx} 抠图失败: {type(e).__name__}: {e}")
                # 发生错误时，cropped 保持原样，继续往下走（保证程序不中断）
        
        # === 步骤 3: 缩放到目标尺寸（使用 INTER_CUBIC）===
        resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        
        # === 步骤 4: 写入 ===
        out_path = os.path.join(output_dir, f'{frame_idx}.{output_ext}')
        cv2.imwrite(out_path, resized, imwrite_params)
        
        return frame_idx, crop_info, face_result
    except Exception as e:
        return None, None, None


def process(input_path, output_dir, output_size=256, output_format='png',
            pad_ratio_range=(1.05, 1.3), min_face_size=30,
            enhance=False, gfpgan_model_path=None, enhance_strength=0.4,
            num_workers=4, device='cuda', s3fd_conf_th=0.5,
            max_detect_size=640, detect_interval=1, enable_rembg=True):

    os.makedirs(output_dir, exist_ok=True)
    output_ext = output_format.lower()
    if output_ext not in ('png', 'jpg', 'jpeg'):
        output_ext = 'png'

    # 扫描输出目录中已有的图像文件，获取最大序号，新帧从 max_idx+1 开始编号
    start_idx = 0
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

    # === 初始化 Rembg 抠图模型（GPU 加速）===
    print("正在加载 Rembg 抠图模型 (首次运行可能需要下载)...")

    # 设置模型保存目录为项目根目录的 ckpts/rembg
    global _model_dir
    _model_dir = os.path.join(_PROJECT_ROOT, "ckpts", "rembg")
    os.makedirs(_model_dir, exist_ok=True)
    os.environ["U2NET_HOME"] = _model_dir

    # 初始化跟踪器
    tracker = HeadTracker(
        pad_ratio_range=pad_ratio_range,
    )

    # 初始化增强器
    enhancer = None
    if enhance:
        enhancer = CropFaceEnhancer(gfpgan_model_path, strength=enhance_strength)
        enhancer_loaded = enhancer is not None and enhancer.session is not None
    else:
        enhancer_loaded = False

    # metadata
    metadata = {
        'input_path': input_path,
        'output_size': output_size,
        'output_format': output_ext,
        'pad_ratio_range': list(pad_ratio_range),
        'min_face_size': min_face_size,
        'enhanced': enhancer_loaded,
        'enhance_strength': enhance_strength if enhance else 0.0,
        'num_workers': num_workers,
        'device': det_device,
        'frames': [],
    }

    # ========== 多进程并行处理 ==========
    # 注意：Windows 上 ProcessPoolExecutor 有兼容性问题（需要 pickle 序列化）
    # 使用 ThreadPoolExecutor + u2net 模型（比 u2netp 更好的发丝保留）
    BATCH_SIZE = 32  # 减小批次以增加并行度
    
    # 使用 ThreadPoolExecutor（Windows 兼容性更好）
    write_executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
    pending_futures = []

    # 获取帧生成器 + 总数
    frame_gen, total_frames = iter_frames(input_path)

    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc='裁切', unit='帧',
        mininterval=2.0,        # 最少 1 秒才刷新
        miniters=100,            # 最少 100 帧才刷新
        ncols=90,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )

    frame_idx = start_idx
    face_hit = 0
    last_face_result = None  # 用于跳帧检测时的缓存

    try:
        for frame in frame_gen:
            if frame is None:
                continue
            frame_h, frame_w = frame.shape[:2]
            if 'source_resolution' not in metadata:
                metadata['source_resolution'] = [frame_w, frame_h]

            # 跳帧检测优化：每 N 帧实际检测一次，中间帧使用上次检测结果
            # 注意：默认 detect_interval=1，每帧都检测
            if detect_interval <= 1 or frame_idx % detect_interval == 0:
                face_result = detect_face(
                    detector_info, frame,
                    min_face_size=min_face_size,
                    max_detect_size=max_detect_size,
                )
                last_face_result = face_result
            else:
                # 跳帧时：使用上次的检测结果，让跟踪器进行平滑处理
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
            
            # 添加 frame_idx 到 crop_info 用于输出文件命名
            crop_info['frame_idx'] = frame_idx
            
            # 将任务提交到线程池（裁切+抠图+缩放+写入全部并行）
            task = (frame, crop_info, frame_idx, output_dir, output_size, output_ext, imwrite_params, _model_dir, face_result, enable_rembg)
            fut = write_executor.submit(_process_single_frame_global, task)
            pending_futures.append((fut, frame_idx, face_result, crop_info))
            
            # 收集元数据（基于检测结果）
            scale_factor = 1.0  # 不再在主进程计算，因为不再需要
            meta = {
                'frame_idx': frame_idx,
                'crop_x': crop_info['crop_x'],
                'crop_y': crop_info['crop_y'],
                'crop_size': crop_info['crop_size'],
                'scale_factor': scale_factor,
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
            
            # 控制未完成的批次数量（避免内存爆炸）
            if len(pending_futures) >= BATCH_SIZE:
                # 等待最早的一个完成
                oldest = pending_futures.pop(0)
                oldest[0].result()
            
            frame_idx += 1
            pbar.update(1)

    finally:
        # 等待所有写入完成
        pbar.set_description('写入中')
        for fut, idx, _, _ in pending_futures:
            try:
                fut.result()
            except Exception as e:
                print(f"\n[ERROR] 后台线程处理帧 {idx} 时发生致命错误: {e}")
        write_executor.shutdown(wait=True)
        pbar.close()

    metadata['frame_count'] = len(metadata['frames'])

    # 保存 metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

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


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Face detection + crop to sequence frames (with optional GFPGAN enhancement)'
    )
    parser.add_argument('input', type=str,
                        help='Input video file OR directory containing image sequence')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: avantar/<video_name>/)')
    parser.add_argument('--output_size', type=int, default=256,
                        help='Output image size N×N (default: 256)')
    parser.add_argument('--output_format', type=str, default='png',
                        choices=['png', 'jpg'],
                        help='Output image format (default: png)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for I/O (default: 4)')

    # S3FD 检测参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for S3FD detection (default: cuda, auto-falls-back to cpu)')
    parser.add_argument('--s3fd_conf_th', type=float, default=0.5,
                        help='S3FD confidence threshold (default: 0.5)')
    parser.add_argument('--max_detect_size', type=int, default=640,
                        help='Max image short side for detection (4K→1280 = 10x faster, default: 1280)')
    parser.add_argument('--detect_interval', type=int, default=1,
                        help='Detect every N frames (1=all frames, 5=every 5th frame, default: 1)')

    parser.add_argument('--pad_ratio_min', type=float, default=1.05,
                        help='Minimum pad ratio (tight crop, default: 1.05)')
    parser.add_argument('--pad_ratio_max', type=float, default=1.2,
                        help='Maximum pad ratio (default: 1.2)')
    parser.add_argument('--min_face_size', type=int, default=30,
                        help='Minimum face size in pixels (default: 30)')

    # 面部增强（默认关闭）
    parser.add_argument('--enhance', action='store_true', default=False,
                        help='Enable GFPGAN face enhancement (default: off)')
    parser.add_argument('--gfpgan_model', type=str, default=None,
                        help='Path to GFPGAN ONNX model (default: <project_root>/gfpgan-1024.onnx)')
    parser.add_argument('--enhance_strength', type=float, default=0.4,
                        help='Enhancement strength 0.0~1.0 (default: 0.4, only used with --enhance)')

    # 抠图参数
    parser.add_argument('--rembg', action='store_true', default=True,
                        help='Enable background removal with rembg (default: on)')
    parser.add_argument('--no-rembg', action='store_false', dest='rembg',
                        help='Disable background removal')

    opt = parser.parse_args()

    # 默认输出目录: avantar/<视频名称>/  (位于项目根目录)
    if opt.output_dir is None:
        if os.path.isdir(opt.input):
            # 输入是目录: 用目录名作为子目录
            name = os.path.basename(os.path.normpath(opt.input))
        else:
            # 输入是视频文件: 用文件名(不含扩展名)作为子目录
            name = os.path.splitext(os.path.basename(opt.input))[0]
        opt.output_dir = os.path.join(_PROJECT_ROOT, 'avantar', name)

    # 确保输出目录是绝对路径
    if not os.path.isabs(opt.output_dir):
        opt.output_dir = os.path.join(_PROJECT_ROOT, opt.output_dir)

    # 启动消息
    input_display = os.path.basename(opt.input.rstrip(os.sep)) if os.path.isdir(opt.input) else os.path.basename(opt.input)
    print(f'输入: {input_display} → 输出: {os.path.basename(opt.output_dir)}'
          f' | {opt.output_size}×{opt.output_size} {opt.output_format.upper()}'
          f' | device={opt.device} | workers={opt.num_workers}'
          f' | rembg={"ON" if opt.rembg else "OFF"}'
          f' | enhance={"ON" if opt.enhance else "off"}')

    process(
        input_path=opt.input,
        output_dir=opt.output_dir,
        output_size=opt.output_size,
        output_format=opt.output_format,
        pad_ratio_range=(opt.pad_ratio_min, opt.pad_ratio_max),
        min_face_size=opt.min_face_size,
        enhance=opt.enhance,
        gfpgan_model_path=opt.gfpgan_model,
        enhance_strength=opt.enhance_strength,
        num_workers=opt.num_workers,
        device=opt.device,
        s3fd_conf_th=opt.s3fd_conf_th,
        max_detect_size=opt.max_detect_size,
        detect_interval=opt.detect_interval,
        enable_rembg=opt.rembg,
    )
