"""CHAOS 长跑（/autoplan E6）：连续随机切换下的稳定性下限。

真实 BatchPreloader（tmp 小资产）+ 真 RenderLoop/OutputStreamer + 假发送端，
N 次随机切换（immediate/segment_end 混合 + 一次调度替换 + 音频推送），
断言：渲染零帧错误、输出帧数下限、underruns 上限——2am 事故模式
（连续切换下 production 落后 → underruns 爬升 → 相机冻结）的回归网。
"""
import random
import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from realtime_serve.audio_buffer import AudioBuffer
from realtime_serve.output_buffer import FrameBuffer, OutputStreamer
from realtime_serve.render_loop import RenderLoop
from realtime_serve.sequence_frame_pack import CharacterDataPack
from realtime_serve.service import RealtimeService

_RUN_SECONDS = 4.0
_SEGMENTS = {"1": 120, "22": 120, "55": 120, "33": 120}


@pytest.fixture(autouse=True)
def _fake_imread(monkeypatch):
    monkeypatch.setattr(
        'realtime_serve.sequence_frame_pack.cv2.imread',
        lambda *a, **k: np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


def _build_pack(tmp_path) -> CharacterDataPack:
    character_dir = tmp_path / "chaos_char"
    for sub in ("orifra", "count", "metadata", "facemark"):
        (character_dir / sub).mkdir(parents=True)
    meta_dtype = np.dtype([('frame_idx', 'i4'), ('crop_x', 'f4'),
                           ('crop_y', 'f4'), ('crop_size', 'f4')])
    for name, n in _SEGMENTS.items():
        d1 = character_dir / "orifra" / name
        d2 = character_dir / "count" / name
        d1.mkdir()
        d2.mkdir()
        for i in range(n):
            (d1 / f"{i}.jpg").write_bytes(b"f")
            (d2 / f"{i}.jpg").write_bytes(b"f")
        np.save(character_dir / "metadata" / f"{name}.npy",
                np.zeros(n, dtype=meta_dtype))
        np.save(character_dir / "facemark" / f"{name}.npy",
                np.random.rand(n, 68, 2).astype(np.float32))
    return CharacterDataPack(str(character_dir))


def _build_service(pack) -> RealtimeService:
    frame_buffer = FrameBuffer(capacity=16)
    sender = MagicMock(return_value=True)
    render_loop = RenderLoop(target_fps=25)
    render_loop.set_character_data(pack)  # 真循环需自行接线（facade 不复制逻辑）
    return RealtimeService(
        None,
        components={
            'model_manager': None,
            'render_loop': render_loop,
            'output_streamer': OutputStreamer(
                frame_buffer, target_fps=25, sender=sender, preroll_frames=8),
            'frame_buffer': frame_buffer,
            'unity_capture': None,
            'audio_buffer': AudioBuffer(),
            'audio_processor': None,
            'compositor': None,
            'face_repair': None,
            'audio_player': MagicMock(),
            'character_data': pack,
            'gaze_corrector': None,
        },
        assemble=False)


def test_chaos_random_switches_stability(tmp_path):
    rng = random.Random(42)  # 固定种子：切换序列可复现
    pack = _build_pack(tmp_path)
    service = _build_service(pack)
    service.start()

    segs = pack.segments
    deadline = time.perf_counter() + _RUN_SECONDS
    actions = 0
    try:
        while time.perf_counter() < deadline:
            seg = rng.choice(segs)
            at = rng.choice(["immediate", "segment_end"])
            resp = service.switch_segment(seg, at=at)
            assert resp["ok"] is True
            actions += 1
            if actions == 3:
                order = segs[:]
                rng.shuffle(order)
                r = service.set_schedule(order)  # 运行中替换调度（align=False）
                assert r["ok"] is True
            time.sleep(0.35)
    finally:
        service.stop()

    rl = service.render_loop
    os_ = service.output_streamer
    # 渲染线程零帧错误（producer 保护不吞异常——last_error 必须干净）
    assert rl._frame_errors == 0, f"producer errors: {rl.last_error}"
    assert rl.last_error in ("", None)
    # 输出帧数下限（25fps × 4s × 0.6 宽容系数，preroll/切换停顿计入余量）
    assert os_.output_frames >= _RUN_SECONDS * 25 * 0.6, os_.output_frames
    # underruns 上限（快产能下应很少；切换 churn 允许少量）
    assert os_.underruns <= max(10, os_.output_frames // 4), \
        f"underruns={os_.underruns} output={os_.output_frames}"
    # 状态一致：当前段始终合法
    assert pack.current_segment in segs
    # 面部修复/gaze 未构造但状态字段如实
    status = service.get_status()
    assert status["gaze_capable"] is False and status["repair_capable"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
