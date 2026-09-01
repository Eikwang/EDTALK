"""Character data pack for loading and managing preprocessed video frames.

Batched sliding-window preload design:
    Instead of loading frames on demand (per-frame disk I/O), frames are
    preloaded in batches of BATCH_SIZE. A sliding window of NUM_BATCHES
    batches is maintained: the current batch is being rendered while the
    next batch is loaded in a background thread.

    This eliminates per-frame disk I/O during rendering — all frame reads
    happen in batch during preloading, keeping the render loop free of
    blocking I/O.

Directory structure:
    <character_dir>/
    ├── orifra/<segment_name>/     # Original frames 1080p (JPEG)
    ├── count/<segment_name>/      # Cropped frames 256x256 (JPEG)
    ├── metadata/<segment_name>.npy  # Crop coordinates (structured array)
    ├── facemark/<segment_name>.npy  # 68 landmark points (N, 68, 2) float32
    └── checkpoint/
        ├── audio2lip.pt          # Audio2Lip model weights
        └── new.pt               # Generator model weights
"""
import os
import threading
from collections import OrderedDict
import numpy as np
import cv2
import torch
from typing import Tuple, List, Optional, Dict


class BatchPreloader:
    """Batched sliding-window preloader for video frames.

    Maintains a window of NUM_BATCHES batches, each containing BATCH_SIZE frames.
    The current batch is used for rendering while the next batch is loaded
    in a background thread. When the render crosses a batch boundary, the
    preloaded batch becomes current and a new preload starts.

    Memory layout:
        orifra frames: CPU numpy arrays (BATCH_SIZE × ~125KB = ~12.5MB per batch)
        crop tensors: GPU float32 tensors (BATCH_SIZE × ~786KB = ~78.6MB per batch)
        Total window (2 batches): ~182MB — trivial for 16GB GPU
    """

    BATCH_SIZE = 100   # frames per batch
    NUM_BATCHES = 2    # window = current + next

    def __init__(self, segment_cache: dict, device: str = 'cuda',
                 compositor=None):
        """Initialize BatchPreloader.

        Args:
            segment_cache: Reference to CharacterDataPack._segment_cache
            device: Device for crop tensors
            compositor: MouthCompositor instance for mask precomputation
        """
        self._segment_cache = segment_cache
        self._device = device
        self._compositor = compositor

        # Batch buffers: [batch_slot][frame_offset] = data
        # Two slots for double-buffering
        self._orifra_bufs: List[Optional[List[np.ndarray]]] = [None, None]
        self._crop_bufs: List[Optional[torch.Tensor]] = [None, None]  # [B, 3, 256, 256]
        self._mask_bufs: List[Optional[List[np.ndarray]]] = [None, None]  # precomputed masks
        self._facemark_bufs: List[Optional[List[np.ndarray]]] = [None, None]  # stored facemark for mask recompute

        # Each batch's start frame index
        self._batch_starts: List[int] = [-1, -1]

        # Which slot is "current" (0 or 1)
        self._current_slot: int = 0

        # Current segment name
        self._segment: Optional[str] = None

        # Preload thread control
        self._preload_thread: Optional[threading.Thread] = None
        self._preload_lock = threading.Lock()
        self._preload_done = threading.Event()
        self._preload_target_start: int = -1

    def reset(self, segment: str):
        """Reset preloader for a new segment.

        Args:
            segment: Segment name
        """
        # Wait for any in-progress preload
        if self._preload_thread is not None and self._preload_thread.is_alive():
            self._preload_done.wait(timeout=5.0)

        self._segment = segment
        self._orifra_bufs = [None, None]
        self._crop_bufs = [None, None]
        self._mask_bufs = [None, None]
        self._facemark_bufs = [None, None]
        self._batch_starts = [-1, -1]
        self._current_slot = 0
        self._preload_target_start = -1

    def _load_batch_sync(self, slot: int, start_idx: int, segment: str):
        """Load a batch of frames into the specified slot (blocking).

        Also precomputes mouth masks from facemark data if compositor is available.

        Args:
            slot: Buffer slot (0 or 1)
            start_idx: First frame index to load
            segment: Segment name
        """
        data = self._segment_cache[segment]
        total = data['total_frames']
        count = min(self.BATCH_SIZE, total - start_idx)

        orifra_list = []
        crop_list = []
        mask_list = []
        facemark_list = []

        facemark_data = data.get('facemark')  # (total_frames, 68, 2)

        for i in range(count):
            idx = (start_idx + i) % total

            # Load orifra
            img = cv2.imread(data['orifra_files'][idx])
            if img is None:
                raise ValueError(f"Failed to read orifra: {data['orifra_files'][idx]}")
            orifra_list.append(img)

            # Load crop → tensor
            crop_img = cv2.imread(data['crop_files'][idx])
            if crop_img is None:
                raise ValueError(f"Failed to read crop: {data['crop_files'][idx]}")
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
            crop_tensor = (crop_tensor - 0.5) * 2.0  # normalize to [-1, 1]
            crop_list.append(crop_tensor)

            # Precompute mask from facemark
            if facemark_data is not None and self._compositor is not None:
                fm = facemark_data[idx]  # (68, 2)
                mask = self._compositor.generate_mask(fm)
                mask_list.append(mask)
            else:
                mask_list.append(None)

            # Store facemark for later mask recomputation
            if facemark_data is not None:
                facemark_list.append(facemark_data[idx])
            else:
                facemark_list.append(None)

        # Stack crop tensors and move to GPU in one transfer
        crop_batch = torch.stack(crop_list, dim=0).to(self._device)  # [B, 3, 256, 256]

        with self._preload_lock:
            self._orifra_bufs[slot] = orifra_list
            self._crop_bufs[slot] = crop_batch
            self._mask_bufs[slot] = mask_list
            self._facemark_bufs[slot] = facemark_list
            self._batch_starts[slot] = start_idx

    def _preload_worker(self, slot: int, start_idx: int, segment: str):
        """Background thread: load next batch."""
        try:
            self._load_batch_sync(slot, start_idx, segment)
        except Exception as e:
            print(f"[BatchPreloader] Preload error: {e}")
        finally:
            self._preload_done.set()

    def _start_preload(self, slot: int, start_idx: int, segment: str):
        """Start async preloading of a batch."""
        self._preload_done.clear()
        self._preload_target_start = start_idx
        self._preload_thread = threading.Thread(
            target=self._preload_worker,
            args=(slot, start_idx, segment),
            daemon=True
        )
        self._preload_thread.start()

    def _is_in_batch(self, frame_idx: int, slot: int) -> bool:
        """Check if frame_idx falls within the batch in the given slot."""
        start = self._batch_starts[slot]
        if start < 0:
            return False
        return start <= frame_idx < start + self.BATCH_SIZE

    def ensure_loaded(self, frame_idx: int, segment: str):
        """Ensure frame_idx is available in the preloaded window.

        Call this BEFORE get_frame() for each render iteration.

        Logic:
        1. frame_idx in current batch → do nothing (fast path)
        2. frame_idx in next (preloading) batch → wait for preload, swap slots
        3. frame_idx not in window → synchronous load + start preload

        Args:
            frame_idx: Frame index needed for rendering
            segment: Current segment name
        """
        # Fast path: frame_idx is in current batch
        if self._is_in_batch(frame_idx, self._current_slot):
            return

        # Check if segment changed
        if segment != self._segment:
            self.reset(segment)

        other_slot = 1 - self._current_slot

        # frame_idx is in the other (preloading) batch — wait and swap
        if self._is_in_batch(frame_idx, other_slot):
            # Wait for preload to finish
            if self._preload_thread is not None and self._preload_thread.is_alive():
                self._preload_done.wait(timeout=5.0)
            self._current_slot = other_slot
            # Start preloading the next batch
            next_start = self._batch_starts[other_slot] + self.BATCH_SIZE
            data = self._segment_cache[segment]
            if next_start < data['total_frames']:
                self._start_preload(other_slot, next_start, segment)
            return

        # frame_idx not in any batch — synchronous load + start preload
        batch_start = (frame_idx // self.BATCH_SIZE) * self.BATCH_SIZE
        self._load_batch_sync(self._current_slot, batch_start, segment)

        # Start preloading next batch
        next_start = batch_start + self.BATCH_SIZE
        data = self._segment_cache[segment]
        if next_start < data['total_frames']:
            self._start_preload(other_slot, next_start, segment)

    def get_orifra(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get orifra frame from preloaded buffer.

        Args:
            frame_idx: Frame index

        Returns:
            orifra frame as BGR uint8 numpy array, or None if not loaded
        """
        slot = self._current_slot
        start = self._batch_starts[slot]
        if start < 0 or not (start <= frame_idx < start + self.BATCH_SIZE):
            return None
        offset = frame_idx - start
        bufs = self._orifra_bufs[slot]
        if bufs is None or offset >= len(bufs):
            return None
        return bufs[offset]

    def get_crop(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Get crop tensor from preloaded buffer (already on GPU).

        Args:
            frame_idx: Frame index

        Returns:
            crop tensor [1, 3, 256, 256] on GPU, or None if not loaded
        """
        slot = self._current_slot
        start = self._batch_starts[slot]
        if start < 0 or not (start <= frame_idx < start + self.BATCH_SIZE):
            return None
        offset = frame_idx - start
        batch = self._crop_bufs[slot]
        if batch is None or offset >= batch.shape[0]:
            return None
        return batch[offset:offset+1]  # [1, 3, 256, 256]

    def get_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get precomputed mouth mask from preloaded buffer.

        Args:
            frame_idx: Frame index

        Returns:
            mask (256, 256) float32, or None if not available
        """
        slot = self._current_slot
        start = self._batch_starts[slot]
        if start < 0 or not (start <= frame_idx < start + self.BATCH_SIZE):
            return None
        offset = frame_idx - start
        masks = self._mask_bufs[slot]
        if masks is None or offset >= len(masks):
            return None
        return masks[offset]

    def recompute_current_masks(self):
        """Recompute masks for the currently loaded batch using the compositor.

        Call this after setting the compositor to generate masks for
        frames that were loaded before the compositor was available.
        """
        if self._compositor is None:
            return
        slot = self._current_slot
        facemark_data = self._facemark_bufs[slot]
        if facemark_data is None:
            return
        mask_list = []
        for fm in facemark_data:
            if fm is not None:
                mask = self._compositor.generate_mask(fm)
                mask_list.append(mask)
            else:
                mask_list.append(None)
        self._mask_bufs[slot] = mask_list

    @property
    def is_preloading(self) -> bool:
        """Whether a background preload is in progress."""
        return self._preload_thread is not None and self._preload_thread.is_alive()


class CharacterDataPack:
    """Loads and manages preprocessed character data with multiple segments.

    Batched preload: Only metadata/facemark are loaded at startup. Orifra and
    crop frames are preloaded in batches via BatchPreloader — a sliding window
    of 2×100 frames is maintained, with background preloading of the next batch.

    Each segment contains:
    - orifra: Original frames at full resolution (batch-preloaded to CPU)
    - count: Cropped frames at 256x256 (batch-preloaded to GPU)
    - metadata: Crop coordinates for each frame (loaded at startup)
    - facemark: 68 facial landmark points per frame (loaded at startup)
    """

    def __init__(self, character_dir: str, device: str = 'cuda',
                 batch_size: int = 100):
        """Initialize CharacterDataPack.

        Args:
            character_dir: Path to character directory containing all data
            device: Device to load cropped frames on ('cuda' or 'cpu')
            batch_size: Number of frames per preload batch (default: 100)

        Raises:
            FileNotFoundError: If required paths don't exist
        """
        self.character_dir = character_dir
        self.device = device

        # Discover available segments
        self.orifra_base = os.path.join(character_dir, "orifra")
        self.count_base = os.path.join(character_dir, "count")
        self.metadata_base = os.path.join(character_dir, "metadata")
        self.facemark_base = os.path.join(character_dir, "facemark")
        self.checkpoint_dir = os.path.join(character_dir, "checkpoint")

        # Find all segments by listing orifra subdirectories
        if not os.path.exists(self.orifra_base):
            raise FileNotFoundError(f"orifra directory not found: {self.orifra_base}")

        self.segments: List[str] = sorted([
            d for d in os.listdir(self.orifra_base)
            if os.path.isdir(os.path.join(self.orifra_base, d))
        ])

        if not self.segments:
            raise ValueError(f"No segments found in {self.orifra_base}")

        # Default to first segment
        self.current_segment = self.segments[0]

        # Cache for loaded segment data
        self._segment_cache: dict = {}

        # Batch preloader (replaces LRU cache)
        self._preloader = BatchPreloader(self._segment_cache, device)
        BatchPreloader.BATCH_SIZE = batch_size

        # Load current segment (only metadata + facemark + file lists)
        self._load_segment(self.current_segment)

        # Preload first batch for current segment
        self._preloader.reset(self.current_segment)
        self._preloader.ensure_loaded(0, self.current_segment)

        print(f'==> loaded character: {character_dir}')
        print(f'  -> Available segments: {self.segments}')
        print(f'  -> Current segment: {self.current_segment}')
        print(f'  -> Batch preload enabled (batch_size={batch_size}, window={batch_size*2})')

    def set_compositor(self, compositor):
        """Set the compositor for mask precomputation.

        Call this after MouthCompositor is created to enable mask precomputation
        during batch loading.
        """
        self._preloader._compositor = compositor
        # Recompute masks for already-loaded frames
        self._preloader.recompute_current_masks()

    def _get_segment_paths(self, segment: str) -> dict:
        """Get paths for a specific segment."""
        return {
            'orifra': os.path.join(self.orifra_base, segment),
            'count': os.path.join(self.count_base, segment),
            'metadata': os.path.join(self.metadata_base, f"{segment}.npy"),
            'facemark': os.path.join(self.facemark_base, f"{segment}.npy"),
        }

    def _validate_segment(self, segment: str) -> bool:
        """Validate that a segment has all required data."""
        paths = self._get_segment_paths(segment)
        return (
            os.path.exists(paths['orifra']) and
            os.path.exists(paths['count']) and
            os.path.exists(paths['metadata']) and
            os.path.exists(paths['facemark'])
        )

    def _load_segment(self, segment: str):
        """Load a segment's metadata and file lists (NOT actual frame data).

        Frames are loaded lazily on demand via get_frame().
        Only metadata, facemark, and file path lists are loaded at startup.
        """
        if segment in self._segment_cache:
            return

        paths = self._get_segment_paths(segment)

        # Validate paths
        for key in ['orifra', 'count', 'metadata', 'facemark']:
            if not os.path.exists(paths[key]):
                raise FileNotFoundError(f"{key} not found: {paths[key]}")

        # Load metadata and facemark (small files, fast)
        metadata = np.load(paths['metadata'])
        facemark = np.load(paths['facemark'])

        # Build sorted file lists (just paths, no image data)
        orifra_files = self._build_file_list(paths['orifra'])
        crop_files = self._build_file_list(paths['count'])

        # Validate consistency
        num_orifra = len(orifra_files)
        num_crop = len(crop_files)
        num_meta = len(metadata)
        num_facemark = facemark.shape[0]

        if num_orifra != num_crop or num_orifra != num_meta or num_orifra != num_facemark:
            raise ValueError(
                f"Data mismatch in segment '{segment}': "
                f"orifra={num_orifra}, crop={num_crop}, "
                f"metadata={num_meta}, facemark={num_facemark}"
            )

        self._segment_cache[segment] = {
            'orifra_files': orifra_files,   # list of file paths (lazy)
            'crop_files': crop_files,       # list of file paths (lazy)
            'metadata': metadata,           # numpy array (loaded)
            'facemark': facemark,           # numpy array (loaded)
            'total_frames': num_orifra,
        }

        print(f'  -> Segment "{segment}": {num_orifra} frames (lazy), '
              f'resolution from metadata')

    def _build_file_list(self, directory: str) -> List[str]:
        """Build a numerically sorted list of JPEG file paths."""
        files = sorted([
            f for f in os.listdir(directory) if f.endswith('.jpg')
        ], key=self._numeric_sort_key)
        return [os.path.join(directory, f) for f in files]

    @staticmethod
    def _numeric_sort_key(filename: str):
        """Sort key that extracts the numeric part from filename."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    def _read_orifra_frame(self, path: str) -> np.ndarray:
        """Read a single orifra frame from disk (fallback only)."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img

    def ensure_loaded(self, idx: int):
        """Ensure frame idx is available in the preloaded window.

        Call this BEFORE get_frame() for each render iteration.
        Triggers batch preloading when crossing batch boundaries.

        Args:
            idx: Frame index needed for rendering
        """
        actual = idx % self._segment_cache[self.current_segment]['total_frames']
        self._preloader.ensure_loaded(actual, self.current_segment)

    def switch_segment(self, segment: str) -> bool:
        """Switch to a different segment.

        Resets the batch preloader and preloads the first batch of the new segment.

        Args:
            segment: Segment name to switch to

        Returns:
            True if switch was successful
        """
        if segment not in self.segments:
            raise ValueError(f"Segment '{segment}' not found. Available: {self.segments}")

        self.current_segment = segment
        self._load_segment(segment)

        # Reset preloader and preload first batch of new segment
        self._preloader.reset(segment)
        self._preloader.ensure_loaded(0, segment)

        return True

    def get_frame(self, idx: int) -> Tuple:
        """Get frame at index from current segment (batch-preloaded).

        IMPORTANT: Call ensure_loaded(idx) before this method to ensure
        the frame data is in the preloaded buffer.

        Args:
            idx: Frame index (can be any integer, will be modulo total_frames)

        Returns:
            Tuple of (orifra, cropped, meta, facemark, mask)
                - orifra: (H, W, 3) BGR uint8 numpy array
                - cropped: [1, 3, 256, 256] float tensor in [-1, 1]
                - meta: Single metadata entry (structured array)
                - facemark: (68, 2) float32 numpy array
                - mask: (256, 256) float32 precomputed mouth mask, or None
        """
        data = self._segment_cache[self.current_segment]
        actual = idx % data['total_frames']

        # Get orifra from preloaded buffer (fallback to disk if not loaded)
        orifra = self._preloader.get_orifra(actual)
        if orifra is None:
            orifra = self._read_orifra_frame(data['orifra_files'][actual])

        # Get crop tensor from preloaded buffer (already on GPU, [1,3,256,256])
        cropped = self._preloader.get_crop(actual)
        if cropped is None:
            # Fallback: load single frame (should not happen if ensure_loaded called)
            img = cv2.imread(data['crop_files'][actual])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - 0.5) * 2.0
            cropped = tensor.unsqueeze(0).to(self.device)

        # Get precomputed mask from buffer
        mask = self._preloader.get_mask(actual)

        # Already in memory
        meta = data['metadata'][actual]
        facemark = data['facemark'][actual]

        return (orifra, cropped, meta, facemark, mask)

    @property
    def total_frames(self) -> int:
        """Get total number of frames in current segment."""
        data = self._segment_cache[self.current_segment]
        return data['total_frames']

    @property
    def resolution(self) -> tuple:
        """Get resolution of current segment frames.

        Uses metadata to determine resolution without loading a frame.
        Falls back to loading the first frame if metadata doesn't contain it.
        """
        data = self._segment_cache[self.current_segment]
        # Try to get resolution from first orifra frame
        first_frame = self._read_orifra_frame(data['orifra_files'][0])
        return first_frame.shape[:2]

    def get_model_paths(self) -> dict:
        """Get paths to model checkpoints."""
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        audio2lip_path = os.path.join(self.checkpoint_dir, "audio2lip.pt")
        generator_path = os.path.join(self.checkpoint_dir, "new.pt")

        if not os.path.exists(audio2lip_path):
            raise FileNotFoundError(f"Audio2Lip model not found: {audio2lip_path}")
        if not os.path.exists(generator_path):
            raise FileNotFoundError(f"Generator model not found: {generator_path}")

        return {
            'audio2lip': audio2lip_path,
            'generator': generator_path,
        }

    def close(self):
        """Release GPU memory."""
        # Clear preloader buffers
        self._preloader.reset('')
        for segment_data in self._segment_cache.values():
            segment_data.clear()
        self._segment_cache.clear()
