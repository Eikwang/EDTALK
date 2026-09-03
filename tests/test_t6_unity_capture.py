"""Tests for UnityCaptureSender synchronous sink."""
import numpy as np
from unittest.mock import MagicMock, patch

from realtime_serve.unity_capture import UnityCaptureSender


def _make_sender(fps: int = 25) -> UnityCaptureSender:
    """Create a sender with a mocked pyvirtualcam.Camera (no real device)."""
    with patch('realtime_serve.unity_capture.pyvirtualcam') as mock_pvc:
        mock_cam = MagicMock()
        mock_pvc.Camera.return_value = mock_cam
        mock_pvc.PixelFormat.BGR = 'BGR'
        sender = UnityCaptureSender(width=100, height=80, fps=fps,
                                    backend='unitycapture')
        sender.virtual_cam = mock_cam
    return sender


class TestUnityCaptureSender:
    """Tests for UnityCaptureSender synchronous sink."""

    def test_initialization(self):
        """Initialization succeeds and sends a test frame."""
        with patch('realtime_serve.unity_capture.pyvirtualcam') as mock_pvc:
            mock_cam = MagicMock()
            mock_pvc.Camera.return_value = mock_cam
            mock_pvc.PixelFormat.BGR = 'BGR'

            sender = UnityCaptureSender(width=320, height=240, fps=25,
                                        backend='unitycapture')

            assert sender.virtual_cam is mock_cam
            assert sender.width == 320
            assert sender.height == 240
            assert sender.fps == 25
            assert mock_cam.send.called

    def test_send_synchronous(self):
        """send() copies the frame into the virtual camera synchronously."""
        sender = _make_sender()
        frame = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)

        result = sender.send(frame)

        assert result is True
        sender.virtual_cam.send.assert_called_with(frame)

    def test_send_resizes_mismatched_frame(self):
        """Resolution mismatch is handled by resizing (fallback path)."""
        sender = _make_sender()
        wrong_size = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)

        result = sender.send(wrong_size)

        assert result is True
        sent = sender.virtual_cam.send.call_args[0][0]
        assert sent.shape[:2] == (80, 100)

    def test_send_native_resolution_no_resize(self):
        """Frames matching the camera size are passed through untouched."""
        sender = _make_sender()
        frame = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)

        assert sender.send(frame) is True
        sent = sender.virtual_cam.send.call_args[0][0]
        assert sent is frame

    def test_update_resolution_recreates_camera(self):
        """update_resolution re-opens the camera at the new native size."""
        with patch('realtime_serve.unity_capture.pyvirtualcam') as mock_pvc:
            cam1, cam2 = MagicMock(), MagicMock()
            mock_pvc.Camera.return_value = cam1
            mock_pvc.PixelFormat.BGR = 'BGR'
            sender = UnityCaptureSender(width=100, height=80, fps=25,
                                        backend='unitycapture')
            mock_pvc.Camera.reset_mock()
            mock_pvc.Camera.return_value = cam2

            ok = sender.update_resolution(200, 150)

            assert ok is True
            assert (sender.width, sender.height) == (200, 150)
            mock_pvc.Camera.assert_called_with(
                width=200, height=150, fps=25, fmt='BGR', backend='unitycapture')
            cam1.close.assert_called_once()
            assert sender.virtual_cam is cam2

            # New-size frames pass through untouched.
            frame = np.zeros((150, 200, 3), dtype=np.uint8)
            assert sender.send(frame) is True
            cam2.send.assert_called_once_with(frame)

    def test_update_resolution_same_dims_noop(self):
        """Same-resolution update must not re-open the camera."""
        sender = _make_sender()
        with patch('realtime_serve.unity_capture.pyvirtualcam') as mock_pvc:
            ok = sender.update_resolution(100, 80)
            assert ok is True
            assert not mock_pvc.Camera.called
            assert sender.virtual_cam is not None

    def test_update_resolution_failure_restores_previous(self):
        """Re-open failure at the new size falls back to the old size."""
        with patch('realtime_serve.unity_capture.pyvirtualcam') as mock_pvc:
            cam1, cam_restore = MagicMock(), MagicMock()
            mock_pvc.Camera.return_value = cam1
            mock_pvc.PixelFormat.BGR = 'BGR'
            sender = UnityCaptureSender(width=100, height=80, fps=25,
                                        backend='unitycapture')
            mock_pvc.Camera.reset_mock()
            mock_pvc.Camera.side_effect = [Exception("reopen failed"), cam_restore]

            ok = sender.update_resolution(200, 150)

            assert ok is False
            assert (sender.width, sender.height) == (100, 80)
            assert sender.virtual_cam is cam_restore

    def test_update_resolution_total_failure_disables_camera(self):
        """Both new and old re-opens failing disables the sender (send False)."""
        with patch('realtime_serve.unity_capture.pyvirtualcam') as mock_pvc:
            mock_pvc.Camera.return_value = MagicMock()
            mock_pvc.PixelFormat.BGR = 'BGR'
            sender = UnityCaptureSender(width=100, height=80, fps=25,
                                        backend='unitycapture')
            mock_pvc.Camera.reset_mock()
            mock_pvc.Camera.side_effect = Exception("device gone")

            ok = sender.update_resolution(200, 150)

            assert ok is False
            assert sender.virtual_cam is None
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            assert sender.send(frame) is False

    def test_send_after_close(self):
        """After close(), send() returns False without raising."""
        sender = _make_sender()
        sender.close()

        frame = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
        assert sender.send(frame) is False

    def test_close_releases_camera(self):
        """close() releases the virtual camera."""
        sender = _make_sender()
        sender.close()
        assert sender.virtual_cam is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
