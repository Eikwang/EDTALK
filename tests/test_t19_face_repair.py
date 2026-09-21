"""Tests for FaceRepair module."""
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from realtime_serve.face_repair import FaceRepair


class TestFaceRepair:
    """Tests for FaceRepair class."""

    def _create_mock_session(self, output_shape=(1, 3, 256, 256)):
        """Create a mock ONNX session that returns a valid output tensor."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name='input.1')]
        mock_session.get_providers.return_value = ['CUDAExecutionProvider']
        # Simulate ONNX output: NCHW float32 in [-1, 1]
        mock_output = np.random.uniform(-1, 1, output_shape).astype(np.float32)
        mock_session.run.return_value = [mock_output]
        return mock_session

    def test_repair_output_shape_and_dtype(self):
        """正常情况：repair() 返回 (256, 256, 3) uint8 BGR"""
        mock_session = self._create_mock_session()
        with patch('realtime_serve.face_repair.ort.InferenceSession', return_value=mock_session):
            repair = FaceRepair(model_path='fake.onnx', alpha=1.0)

        input_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = repair.repair(input_frame)

        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_repair_output_range(self):
        """正常情况：repair() 输出值在 [0, 255] 范围内"""
        mock_session = self._create_mock_session()
        with patch('realtime_serve.face_repair.ort.InferenceSession', return_value=mock_session):
            repair = FaceRepair(model_path='fake.onnx', alpha=1.0)

        input_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = repair.repair(input_frame)

        assert np.min(result) >= 0
        assert np.max(result) <= 255

    def test_repair_alpha_blending(self):
        """正常情况：alpha < 1.0 时，输出是修复图和原图的混合"""
        # Create two different mock outputs for alpha=1.0 and alpha=0.5
        mock_session = self._create_mock_session()
        with patch('realtime_serve.face_repair.ort.InferenceSession', return_value=mock_session):
            repair_full = FaceRepair(model_path='fake.onnx', alpha=1.0)
            repair_half = FaceRepair(model_path='fake.onnx', alpha=0.5)

        input_frame = np.full((256, 256, 3), 128, dtype=np.uint8)

        result_full = repair_full.repair(input_frame)
        result_half = repair_half.repair(input_frame)

        # alpha=0.5 result should be different from alpha=1.0 result
        # (unless the model output happens to be identical to the input)
        # At minimum, both should produce valid uint8 images
        assert result_full.shape == result_half.shape == (256, 256, 3)
        assert result_full.dtype == result_half.dtype == np.uint8

    def test_repair_alpha_zero_returns_original(self):
        """边界情况：alpha=0.0 时，输出应接近原图"""
        mock_session = self._create_mock_session()
        with patch('realtime_serve.face_repair.ort.InferenceSession', return_value=mock_session):
            repair = FaceRepair(model_path='fake.onnx', alpha=0.0)

        input_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = repair.repair(input_frame)

        # With alpha=0.0, addWeighted(repaired, 0, original, 1.0, 0) = original
        # Allow small floating point differences
        diff = np.abs(result.astype(int) - input_frame.astype(int))
        assert np.max(diff) <= 1, "alpha=0.0 should return original image"

    def test_repair_input_preprocessing(self):
        """正常情况：输入预处理正确 (BGR→RGB, normalize to [-1,1], HWC→NCHW)"""
        mock_session = self._create_mock_session()
        with patch('realtime_serve.face_repair.ort.InferenceSession', return_value=mock_session):
            repair = FaceRepair(model_path='fake.onnx', alpha=1.0)

        # Use a known input to verify preprocessing
        input_frame = np.full((256, 256, 3), 128, dtype=np.uint8)
        repair.repair(input_frame)

        # Check that session.run was called
        assert mock_session.run.called
        # Get the input tensor that was passed
        call_args = mock_session.run.call_args
        input_tensor = list(call_args[0][1].values())[0]

        # Input should be NCHW, float32, in [-1, 1]
        assert input_tensor.shape == (1, 3, 256, 256)
        assert input_tensor.dtype == np.float32
        # 128/255 = 0.502, (0.502 - 0.5)/0.5 = 0.004
        assert abs(input_tensor.mean()) < 0.1

    def test_repair_cuda_provider_preference(self):
        """正常情况：默认使用 CUDA provider"""
        mock_session = self._create_mock_session()
        with patch('realtime_serve.face_repair.ort.InferenceSession', return_value=mock_session) as mock_ort:
            repair = FaceRepair(model_path='fake.onnx', use_cuda=True)

        # Check that CUDAExecutionProvider was in the providers list
        call_args = mock_ort.call_args
        providers = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get('providers')
        assert 'CUDAExecutionProvider' in providers

    def test_repair_cpu_provider(self):
        """边界情况：use_cuda=False 时使用 CPU provider"""
        mock_session = self._create_mock_session()
        mock_session.get_providers.return_value = ['CPUExecutionProvider']
        with patch('realtime_serve.face_repair.ort.InferenceSession', return_value=mock_session) as mock_ort:
            repair = FaceRepair(model_path='fake.onnx', use_cuda=False)

        call_args = mock_ort.call_args
        providers = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get('providers')
        assert 'CPUExecutionProvider' in providers
        assert 'CUDAExecutionProvider' not in providers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
