"""
Synthetic frames (--event_dataset synthetic): an end-to-end test that needs
no dataset download.

Each place is a random binary image_size x image_size frame with
pixels_per_image active pixels, and the queries are the same frames again,
so the method should reach R@1 = 100%. It says nothing about accuracy on
real data, nor about preprocessing latency (there is no event window to bin).
"""

import numpy as np

# Fixed so train.py and inference scripts regenerate identical frames.
SYNTHETIC_SEED = 42


def generate_non_overlapping_single_event_images(num_images, size, pixels_per_image, rng):
    """Frames whose active pixels never overlap across frames (needs
    num_images * pixels_per_image <= size**2)."""
    total_pixels = size * size
    if num_images * pixels_per_image > total_pixels:
        raise SystemExit(f"--event_dataset synthetic with --pixels_per_image {pixels_per_image}: "
                         f"{num_images} non-overlapping frames need {num_images * pixels_per_image} pixels, "
                         f"but a {size}x{size} frame has {total_pixels}. Lower --num_images or raise --image_size.")
    all_indices = np.arange(total_pixels)
    rng.shuffle(all_indices)

    images = []
    for img_idx in range(num_images):
        flat = np.zeros(total_pixels, dtype=np.float32)
        flat[all_indices[img_idx * pixels_per_image:(img_idx + 1) * pixels_per_image]] = 1.0
        images.append(flat)
    return images


def generate_multi_pixel_event_images(num_images, size, pixels_per_image, rng):
    """Frames with pixels_per_image distinct random active pixels each;
    different frames may share pixels."""
    total_pixels = size * size
    if pixels_per_image > total_pixels:
        raise SystemExit(f"--pixels_per_image {pixels_per_image} exceeds the {total_pixels} pixels of a "
                         f"{size}x{size} frame.")
    images = []
    for _ in range(num_images):
        flat = np.zeros(total_pixels, dtype=np.float32)
        flat[rng.choice(total_pixels, size=pixels_per_image, replace=False)] = 1.0
        images.append(flat)
    return images


def generate_synthetic_images(args):
    """The num_images reference frames for this run (deterministic)."""
    rng = np.random.RandomState(SYNTHETIC_SEED)
    if args.pixels_per_image == 1:
        return generate_non_overlapping_single_event_images(args.num_images, args.image_size, 1, rng)
    return generate_multi_pixel_event_images(args.num_images, args.image_size, args.pixels_per_image, rng)


def make_self_recall_gt(n_ref, n_query, tolerance):
    """gt[r, q] = 1 iff |r - q| <= tolerance: the query traverse is the
    reference traverse again."""
    r = np.arange(n_ref, dtype=np.int64)[:, None]
    q = np.arange(n_query, dtype=np.int64)[None, :]
    return (np.abs(r - q) <= int(tolerance)).astype(np.uint8)


class ArrayLoader:
    """EventLoader stand-in over a fixed list of flat binary frames."""

    def __init__(self, images, k_active):
        self._images = [np.asarray(img, dtype=np.float32) for img in images]
        self._active = [np.flatnonzero(img).astype(np.int32) for img in self._images]
        self.n_frames = len(self._images)
        self.k_active = int(k_active)
        self.max_active_pixels = int(k_active)
        # no event stream: a frame's "raw events" are its active pixels
        counts = np.array([a.size for a in self._active], dtype=np.int64)
        self.frame_idx = np.concatenate(([0], np.cumsum(counts)))
        self.window_size_ms = None

    def make_frame_active_pixels(self, frame_id, out_buf=None):
        """int32 view of frame `frame_id`'s active pixel indices."""
        active = self._active[frame_id]
        if out_buf is None:
            out_buf = np.empty(self.max_active_pixels, dtype=np.int32)
        out_buf[:active.size] = active
        return out_buf[:active.size]

    def make_frame_flat(self, frame_id, out=None):
        if out is None:
            return self._images[frame_id].copy()
        out[:] = self._images[frame_id]
        return out

    def make_all_frames_flat(self):
        return [img.copy() for img in self._images]
