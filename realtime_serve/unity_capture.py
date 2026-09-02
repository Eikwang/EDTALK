"""UnityCapture synchronous sink for virtual camera output.

This module is a thin synchronous wrapper around pyvirtualcam. Pacing and
frame buffering are owned by OutputStreamer (realtime_serve/output_buffer.py);
UnityCaptureSender only knows how to send one frame at a time to the camera.
"""
import numpy as np
import cv2

try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


class UnityCaptureSender:
    """Synchronous video sink for pyvirtualcam.

    Usage:
        sender = UnityCaptureSender(width=1440, height=2560, fps=25)
        sender.send(bgr_frame)   # blocking camera send
        sender.close()
    """

    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 25,
                 backend: str = 'unitycapture', cap_num: int = 0):
        """Initialize the virtual camera.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frame rate (reported to the camera backend)
            backend: Virtual camera backend ('obs', 'unitycapture', or None for default)
            cap_num: Camera number (for multi-camera setups)
        """
        if not PYVIRTUALCAM_AVAILABLE:
            raise ImportError("pyvirtualcam is required. Install with: pip install pyvirtualcam")

        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.cap_num = cap_num

        backends_to_try = [backend] if backend else ['obs', 'unitycapture', None]
        print(f"Initializing virtual camera: {width}x{height}@{fps}fps")

        self.virtual_cam = None
        for try_backend in backends_to_try:
            try:
                print(f"尝试使用后端: {try_backend if try_backend else 'default'}")
                kwargs = {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'fmt': pyvirtualcam.PixelFormat.BGR,
                }
                if try_backend:
                    kwargs['backend'] = try_backend
                self.virtual_cam = pyvirtualcam.Camera(**kwargs)

                print(f"虚拟摄像头初始化成功 (backend: {try_backend if try_backend else 'default'})")

                test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                test_frame[:] = [0, 255, 0]
                cv2.putText(test_frame, "EDTalk Ready", (50, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                self.virtual_cam.send(np.ascontiguousarray(test_frame))
                print("测试帧发送成功")
                break

            except Exception as e:
                print(f"后端 {try_backend if try_backend else 'default'} 初始化失败: {e}")
                if self.virtual_cam:
                    try:
                        self.virtual_cam.close()
                    except Exception:
                        pass
                    self.virtual_cam = None
                continue

        if self.virtual_cam is None:
            raise RuntimeError("所有虚拟摄像头后端都初始化失败")

    def send(self, bgr_frame: np.ndarray) -> bool:
        """Send one frame synchronously to the virtual camera.

        Args:
            bgr_frame: BGR format frame (H, W, 3) uint8

        Returns:
            True on success, False on failure.
        """
        try:
            if self.virtual_cam is None:
                return False

            if bgr_frame.dtype != np.uint8:
                bgr_frame = bgr_frame.astype(np.uint8)
            if not bgr_frame.flags['C_CONTIGUOUS']:
                bgr_frame = np.ascontiguousarray(bgr_frame)

            h, w = bgr_frame.shape[:2]
            if w != self.width or h != self.height:
                bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

            self.virtual_cam.send(bgr_frame)
            return True

        except Exception as e:
            print(f"[UnityCaptureSender] Error sending frame: {e}")
            import traceback
            traceback.print_exc()
            return False

    def close(self):
        """Release the virtual camera."""
        if self.virtual_cam is not None:
            try:
                self.virtual_cam.close()
            except Exception as e:
                print(f"[UnityCaptureSender] Error closing virtual camera: {e}")
            self.virtual_cam = None
