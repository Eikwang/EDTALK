"""PrefetchLoader 无缝切换预取测试（/autoplan E1，M15 设计验收）。

覆盖：预取就绪→发布→切换后 batch0 全命中缓存（零磁盘回退）；令牌取消与
新请求 supersede；降级路径（无预取/作废 → 既有冷路径）；mask 有序补算；
/status 快照。
"""
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from realtime_serve.sequence_frame_pack import (
    BatchPreloader, CharacterDataPack, PrefetchLoader)


_META_DTYPE = np.dtype([('frame_idx', 'i4'), ('crop_x', 'f4'),
                        ('crop_y', 'f4'), ('crop_size', 'f4')])


def _make_character(tmp_path, segments: dict) -> str:
    character_dir = tmp_path / "char"
    for sub in ("orifra", "count", "metadata", "facemark"):
        (character_dir / sub).mkdir(parents=True)
    for name, n in segments.items():
        d1 = character_dir / "orifra" / name
        d2 = character_dir / "count" / name
        d1.mkdir()
        d2.mkdir()
        for i in range(n):
            (d1 / f"{i}.jpg").write_bytes(b"f")
            (d2 / f"{i}.jpg").write_bytes(b"f")
        np.save(character_dir / "metadata" / f"{name}.npy",
                np.zeros(n, dtype=_META_DTYPE))
        np.save(character_dir / "facemark" / f"{name}.npy",
                np.random.rand(n, 68, 2).astype(np.float32))
    return str(character_dir)


@pytest.fixture(autouse=True)
def _fake_imread(monkeypatch):
    """假 jpg 字节全程可解码——预取/预载 worker 跑在测试体外，patch 必须整测
    生效（construct 期 patch 会在 worker 读盘时失败，即 t4 测试模式的坑）。"""
    monkeypatch.setattr(
        'realtime_serve.sequence_frame_pack.cv2.imread',
        lambda *a, **k: np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


def _make_pack(tmp_path, segments: dict) -> CharacterDataPack:
    return CharacterDataPack(_make_character(tmp_path, segments))


def _wait_status(loader: PrefetchLoader, states, timeout: float = 5.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = loader.status['state']
        if s in states:
            return s
        time.sleep(0.01)
    return loader.status['state']


def _wait_idle(pack: CharacterDataPack, timeout: float = 5.0) -> None:
    """等 init 期的 batch1 后台预载结束，避免 mask 补算竞态的测试抖动。"""
    deadline = time.time() + timeout
    while pack._preloader.is_preloading and time.time() < deadline:
        time.sleep(0.01)


class TestPrefetchSeamlessSwitch:
    def test_prefetch_publish_switch_hits_cache(self, tmp_path):
        """预取就绪 → 发布 → 切换后 batch0 全命中缓存，零磁盘回退。"""
        pack = _make_pack(tmp_path, {"1": 3, "happy": 5})
        _wait_idle(pack)

        pack.prefetch_segment("happy")
        assert _wait_status(pack._prefetch, {"ready"}) == "ready"

        data = pack.pop_prefetch("happy")
        assert data is not None and data.segment == "happy"

        # 磁盘回退探针：切换后任何 batch0 帧都不得触盘
        pack._read_orifra_frame = MagicMock(
            side_effect=AssertionError("cold-start disk read after prefetch"))
        with patch('realtime_serve.sequence_frame_pack.cv2.imread') as m:
            m.side_effect = AssertionError("cold crop read after prefetch")
            pack.switch_segment("happy", prefetched=data)

        assert pack.current_segment == "happy"
        assert pack._preloader._batch_starts[0] == 0
        for i in range(5):
            assert pack._preloader.get_orifra(i) is not None
            assert pack._preloader.get_crop(i) is not None

    def test_switch_without_prefetch_degrades(self, tmp_path):
        """无预取 → 既有冷路径照常工作（降级不失败）。"""
        pack = _make_pack(tmp_path, {"1": 3, "happy": 5})
        _wait_idle(pack)

        pack.switch_segment("happy")

        assert pack.current_segment == "happy"
        deadline = time.time() + 5.0
        while pack._preloader.is_preloading and time.time() < deadline:
            time.sleep(0.01)
        assert pack._preloader.get_orifra(0) is not None

    def test_pop_clears_state(self, tmp_path):
        pack = _make_pack(tmp_path, {"1": 3, "happy": 5})
        pack.prefetch_segment("happy")
        assert _wait_status(pack._prefetch, {"ready"}) == "ready"
        assert pack.pop_prefetch("happy") is not None
        assert pack.prefetch_status == {"state": "idle", "segment": None}
        assert pack.pop_prefetch("happy") is None  # 只能消费一次


class TestPrefetchCancellation:
    def test_switch_cancels_inflight(self, tmp_path):
        """切换即刻取消在飞预取并释放引用（1.1GB 悬挂防护）。"""
        pack = _make_pack(tmp_path, {"1": 3, "happy": 5, "sad": 4})
        _wait_idle(pack)

        pack.prefetch_segment("happy")
        _wait_status(pack._prefetch, {"loading", "ready"})
        pack.switch_segment("sad")  # 无 prefetched → cancel()

        assert pack.prefetch_status == {"state": "idle", "segment": None}
        assert pack.pop_prefetch("happy") is None

    def test_newer_request_supersedes(self, tmp_path, monkeypatch):
        """monotonic 令牌：只有最新预取有效，旧的按块检查退出并丢弃。"""
        gate = threading.Event()
        real_decode = BatchPreloader._decode_frame

        def gated_decode(job):
            if "happy" in job[0]:
                gate.wait(timeout=5.0)
            return real_decode(job)

        monkeypatch.setattr(BatchPreloader, "_decode_frame",
                            staticmethod(gated_decode))
        pack = _make_pack(tmp_path, {"1": 3, "happy": 5, "sad": 4})
        _wait_idle(pack)

        pack.prefetch_segment("happy")
        time.sleep(0.1)  # 让第一个 worker 进入 gate
        pack.prefetch_segment("sad")  # bump token → happy 作废
        gate.set()

        assert _wait_status(pack._prefetch, {"ready", "discarded"}) == "ready"
        assert pack.pop_prefetch("happy") is None
        data = pack.pop_prefetch("sad")
        assert data is not None and data.segment == "sad"

    def test_broken_segment_discards(self, tmp_path):
        """段数据坏（0 帧）→ 预取标记 discarded，永不阻塞、永不崩溃。"""
        pack = _make_pack(tmp_path, {"1": 3, "bad": 0})
        pack.prefetch_segment("bad")
        assert _wait_status(pack._prefetch, {"discarded"}) == "discarded"

    def test_prefetch_invalid_segment_raises(self, tmp_path):
        """请求时校验白名单（M16：不到切换时才炸）。"""
        pack = _make_pack(tmp_path, {"1": 3})
        with pytest.raises(ValueError, match="not found"):
            pack.prefetch_segment("nope")


class TestPrefetchPublish:
    def test_publish_stale_returns_false(self):
        """preloader 已 reset 到别的段 → 发布拒绝（调用方降级冷路径）。"""
        preloader = BatchPreloader({}, device='cpu')
        preloader.reset("sad")
        ok = preloader.publish_prefetch(
            "happy", [np.zeros((2, 2, 3), np.uint8)],
            torch.zeros(1, 3, 8, 8), [None])
        assert ok is False

    def test_masks_generated_in_order_with_compositor(self, tmp_path):
        """发布后按帧序补算 mask（EMA 顺序敏感，H2）；无 compositor 则保持 None。"""
        pack = _make_pack(tmp_path, {"1": 3, "happy": 5})
        _wait_idle(pack)

        compositor = MagicMock()
        order = []
        compositor.generate_mask.side_effect = \
            lambda fm: order.append(fm.tobytes()) or np.zeros((8, 8), np.float32)
        pack._preloader._compositor = compositor

        pack.prefetch_segment("happy")
        assert _wait_status(pack._prefetch, {"ready"}) == "ready"
        data = pack.pop_prefetch("happy")
        pack.switch_segment("happy", prefetched=data)

        masks = pack._preloader._mask_bufs[0]
        assert len(masks) == 5 and all(m is not None for m in masks)
        assert order == [fm.tobytes() for fm in data.facemark_list]  # 帧序

    def test_masks_none_without_compositor(self, tmp_path):
        """无 compositor（mask=None）→ 渲染路径逐帧兜底，发布仍成功。"""
        pack = _make_pack(tmp_path, {"1": 3, "happy": 5})
        _wait_idle(pack)
        assert pack._preloader._compositor is None

        pack.prefetch_segment("happy")
        assert _wait_status(pack._prefetch, {"ready"}) == "ready"
        pack.switch_segment("happy", prefetched=pack.pop_prefetch("happy"))

        assert pack._preloader._mask_bufs[0] == [None] * 5
        orifra, cropped, meta, facemark, mask = pack.get_frame(0)
        assert mask is None  # 渲染侧 _resolve_mask 走逐帧兜底


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
