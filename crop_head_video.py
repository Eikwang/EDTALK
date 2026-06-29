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
import queue
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


def detect_faces_batch(detector_info, frames, min_face_size=30, max_detect_size=640):
    """批量检测人脸 (逐帧调用，PriorBox 已缓存)"""
    return [detect_face(detector_info, f, min_face_size, max_detect_size) for f in frames]


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
        cv2.ellipse(mask, center, axes, 0, 0, 360, (1.0,), -1)
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
                    if isinstance(pil_img_no_bg, Image.Image):
                        img_size = pil_img_no_bg.size
                    else:
                        img_size = (cropped.shape[1], cropped.shape[0])
                    white_bg = Image.new("RGBA", img_size, (255, 255, 255, 255))
                    if isinstance(pil_img_no_bg, Image.Image) and pil_img_no_bg.mode == 'RGBA':
                        final_img = Image.alpha_composite(white_bg, pil_img_no_bg).convert("RGB")
                    else:
                        if isinstance(pil_img_no_bg, Image.Image):
                            final_img = pil_img_no_bg.convert("RGB")
                        elif isinstance(pil_img_no_bg, np.ndarray):
                            final_img = Image.fromarray(pil_img_no_bg).convert("RGB")
                        else:
                            final_img = Image.fromarray(np.array(pil_img_no_bg)).convert("RGB")
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
        print(f"\n[ERROR] 帧 {frame_idx} 处理失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
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

    # 初始化 Rembg 抠图模型（仅在启用时）
    global _model_dir
    _model_dir = os.path.join(_PROJECT_ROOT, "ckpts", "rembg")
    os.makedirs(_model_dir, exist_ok=True)
    os.environ["U2NET_HOME"] = _model_dir
    if enable_rembg:
        print("正在加载 Rembg 抠图模型 (首次运行可能需要下载)...")
        # 预加载一个 session 确保模型可用
        _get_worker_session(_model_dir)
    else:
        print("[INFO] 抠图已禁用，使用快速裁切模式")

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

    # ========== 核心优化：检测与写入解耦 ==========
    # 关闭抠图时：裁切+resize<2ms，瓶颈100%在S3FD检测
    # 策略：主线程只做检测+裁切，写入线程池只做I/O
    
    # 当 enable_rembg=True 时，抠图耗时(50-200ms/帧)，需要更多写入线程
    # 当 enable_rembg=False 时，裁切极快，只需1-2个写入线程做磁盘I/O
    effective_workers = max(1, num_workers) if enable_rembg else 2

    write_executor = ThreadPoolExecutor(max_workers=effective_workers)
    
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
            
            # 裁切 + resize（极轻量，<2ms）
            if enable_rembg:
                # 开启抠图：提交到线程池做裁切+抠图+resize
                task = (warmup_frame, crop_info, frame_idx, output_dir, output_size,
                        output_ext, imwrite_params, _model_dir, face_result, enable_rembg)
                write_executor.submit(_process_single_frame_global, task)
            else:
                # 关闭抠图：直接裁切+resize，丢给写入线程
                resized, _ = crop_and_resize(warmup_frame, crop_info, output_size)
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
            
            # 裁切 + 写入
            if enable_rembg:
                task = (frame, crop_info, frame_idx, output_dir, output_size,
                        output_ext, imwrite_params, _model_dir, face_result, enable_rembg)
                write_executor.submit(_process_single_frame_global, task)
            else:
                resized, _ = crop_and_resize(frame, crop_info, output_size)
                out_path = os.path.join(output_dir, f'{frame_idx}.{output_ext}')
                write_queue.put((resized, out_path, imwrite_params))
            
            _append_meta(metadata, crop_info, face_result, frame_idx)
            frame_idx += 1
            pbar.update(1)

    finally:
        # 通知写入线程结束
        write_queue.put(None)
        write_queue.join()
        writer_thread.join()
        
        # 关闭写入线程池（抠图模式用）
        if enable_rembg:
            pbar.set_description('写入中')
            write_executor.shutdown(wait=True)
        
        pbar.close()

    if write_error_count > 0:
        print(f'[WARN] {write_error_count} 帧写入失败')

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
        description='Face detection + crop to sequence frames (with optional GFPGAN enhancement)'
    )
    parser.add_argument('input', type=str,
                        help='Input dataset directory (must contain original_videos/ subdirectory)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <dataset_dir>/count/)')
    parser.add_argument('--output_size', type=int, default=256,
                        help='Output image size N*N (default: 256)')
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
    parser.add_argument('--max_detect_size', type=int, default=480,
                        help='Max image short side for detection (4K->1280 = 10x faster, default: 1280)')
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
          f' | rembg={"ON" if opt.rembg else "OFF"}'
          f' | enhance={"ON" if opt.enhance else "off"}')

    # 遍历每个视频文件进行处理
    for video_file in video_files:
        video_path = os.path.join(original_videos_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        # 每个视频的输出子目录: <output_dir>/<video_name>/
        video_output_dir = os.path.join(opt.output_dir, video_name)

        print(f'\n{"="*60}')
        print(f'Processing: {video_file} -> {video_name}/')
        print(f'{"="*60}')

        process(
            input_path=video_path,
            output_dir=video_output_dir,
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

    print(f'\n{"="*60}')
    print(f'All done. {len(video_files)} videos processed.')
    print(f'Output directory: {opt.output_dir}')
    print(f'{"="*60}')
