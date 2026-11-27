"""
Event-camera data loading and ground truth.

EventLoader reads one traverse from its HDF5 file and turns each fixed-length
time window into a binary frame: events are binned onto an
image_size x image_size grid, and the `pixels_per_image` most frequently hit
pixels become the frame's active pixels. The per-query part
(make_frame_active_pixels, make_frame_flat) can run without allocating,
since it sits inside the timed inference loops.

Ground truth (reference x query, dilated by --gt_tolerance frames):
  qcr_event       hand-annotated route-percentage time knots
  brisbane_event  NMEA GPS, nearest reference frame in metres (absolute time)
  nsavp           Applanix pose log, route-percentage nearest match
"""

import os
import calendar

import h5py
import numba
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ────────────────────────────────────────────────────────────────────────────
#   sensor_w, sensor_h  : sensor resolution
#   sequences           : sequence name -> file stem (<data_dir>/<seq>/<stem>.hdf5|.h5)
#   default_ref/query   : traverses used in the paper
#   gt_files            : sequence name -> GPS/pose file (NMEA, or Applanix for nsavp)
#   gt_times            : sequence name -> seconds at fixed route knots (qcr_event)

_BRISBANE_SEQS = ('sunset1', 'sunset2', 'daytime', 'morning', 'sunrise')

DATASETS = {
    'brisbane_event': {
        'sensor_w': 346, 'sensor_h': 260,
        'default_ref': 'sunset1', 'default_query': 'sunset2',
        'sequences': {s: s for s in _BRISBANE_SEQS},
        'gt_files': {s: f'{s}_ground_truth.nmea' for s in _BRISBANE_SEQS},
        'gt_format': 'nmea',
    },
    'qcr_event': {
        'sensor_w': 346, 'sensor_h': 260,
        'default_ref': 'normal2', 'default_query': 'normal3',
        'sequences': {s: s for s in ('normal1', 'normal2', 'normal3', 'normal4', 'fast1', 'fast2', 'slow1')},
        'gt_times': {
            'normal1': [0, 6, 13, 31, 58, 77, 101.5, 124, 148, 157.7, 163.0],
            'normal2': [0, 6, 13, 31.5, 58, 76, 100.5, 123, 148.5, 156.5, 162.0],
            'normal3': [0, 6.5, 13.5, 31.5, 58.5, 77.5, 102.5, 126, 150, 158.5, 165.0],
            'normal4': [0, 6.7, 13.2, 31, 57, 74, 97, 119, 141, 148, 154],
            'fast1':   [0, 2.9, 5.8, 12.9, 23.5, 32, 41.5, 50, 59.3, 62, 64.5],
            'fast2':   [0, 2.5, 5.0, 12.7, 23.7, 33.6, 43.0, 52.4, 62.3, 65.4, 68.0],
            'slow1':   [0, 15.5, 30.0, 63.5, 110.0, 141.0, 185.0, 217.0, 246.5, 256.0, 263.0],
        },
        'gt_format': 'knots',
    },
    # DVXplorer (640x480), epoch-nanosecond timestamps. GT is an Applanix pose
    # log (ECEF positions + epoch-ns timestamps) on the same clock as the events.
    'nsavp': {
        'sensor_w': 640, 'sensor_h': 480,
        'default_ref': 'R0_FS0', 'default_query': 'R0_FA0',
        'sequences': {
            'R0_FA0': 'R0_FA0_dvxplorer_left',
            'R0_FS0': 'R0_FS0_dvxplorer_left',
        },
        'gt_files': {
            'R0_FA0': 'R0_FA0_applanix.h5',
            'R0_FS0': 'R0_FS0_applanix.h5',
        },
        'gt_format': 'applanix',
    },
}


# ────────────────────────────────────────────────────────────────────────────
# HDF5 access
# ────────────────────────────────────────────────────────────────────────────

_EVENT_KEYS = {'x': ('x', 'x_coordinates'), 'y': ('y', 'y_coordinates'), 't': ('t', 'timestamps')}


def _event_datasets(h5f):
    """(x, y, t) datasets of an event file (all supported files keep them under /events)."""
    grp = h5f['events']
    out = []
    for axis, names in _EVENT_KEYS.items():
        name = next((n for n in names if n in grp), None)
        if name is None:
            raise ValueError(f"{h5f.filename}: no '{axis}' dataset under /events (tried {names})")
        out.append(grp[name])
    return tuple(out)


def _time_scale(dt_raw, n_events):
    """Timestamp ticks per second, inferred from the recording: the unit (ns, us,
    ms or s) that gives an event rate between 10 Hz and 80 MHz."""
    if dt_raw <= 0 or n_events <= 0:
        return 1e6
    for sc in (1e9, 1e6, 1e3, 1.0):
        dur = dt_raw / sc
        rate = n_events / dur if dur > 0 else float('inf')
        if 10.0 <= rate <= 80_000_000.0:
            return sc
    return 1e6


def _time_unit_name(scale):
    return {1e9: 'ns', 1e6: 'us', 1e3: 'ms', 1.0: 's'}.get(scale, f'{scale:g} ticks/s')


def _hdf5_searchsorted(dset, value, lo=0, hi=None, side='left'):
    """np.searchsorted on a sorted h5py Dataset with single-element reads.
    np.searchsorted(dset, ...) would first read the whole dataset into RAM
    (~8 GB for a 500M-event file)."""
    if hi is None:
        hi = dset.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        v = float(dset[mid])
        if (v <= value) if side == 'right' else (v < value):
            lo = mid + 1
        else:
            hi = mid
    return lo


def resolve_event_path(dataset, seq, data_dir):
    """Path of `seq`'s event file. Returns (path, dataset registry entry)."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {sorted(DATASETS)}")
    entry = DATASETS[dataset]
    if seq not in entry['sequences']:
        raise ValueError(f"Unknown sequence '{seq}' for dataset '{dataset}'. Available: {sorted(entry['sequences'])}")
    stem = entry['sequences'][seq]
    candidates = [os.path.join(data_dir, sub, stem + ext) for sub in (seq, '') for ext in ('.hdf5', '.h5')]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand), entry
    raise FileNotFoundError(f"No event file for {dataset}/{seq}; looked for {candidates}")


def resolve_gt_path(dataset_entry, seq, data_dir):
    """Path of `seq`'s GPS/pose ground-truth file, or None if the dataset has none."""
    fname = dataset_entry.get('gt_files', {}).get(seq)
    if not fname:
        return None
    for cand in (os.path.join(data_dir, fname), os.path.join(data_dir, seq, fname),
                 os.path.join(data_dir, 'ground_truth', fname)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def resolve_full_n_frames(h5_path, window_size_ms):
    """Number of whole windows in the recording (resolves --num_images -1)."""
    with h5py.File(h5_path, 'r') as f:
        _, _, t_dset = _event_datasets(f)
        n_total = int(t_dset.shape[0])
        t_first, t_last = float(t_dset[0]), float(t_dset[-1])
    window_ticks = float(window_size_ms) * 1e-3 * _time_scale(t_last - t_first, n_total)
    return max(0, int((t_last - t_first) // window_ticks)) if window_ticks > 0 else 0


# ────────────────────────────────────────────────────────────────────────────
# Per-query kernel: count events per pixel, keep the top-K pixels
# ────────────────────────────────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True, boundscheck=False, inline='always')
def _heap_sift_down(heap_v, heap_i, root, size):
    # min-heap over parallel value/index arrays: O(log k) per replacement
    while True:
        left = 2 * root + 1
        if left >= size:
            return
        smallest = left
        right = left + 1
        if right < size and heap_v[right] < heap_v[left]:
            smallest = right
        if heap_v[smallest] >= heap_v[root]:
            return
        heap_v[root], heap_v[smallest] = heap_v[smallest], heap_v[root]
        heap_i[root], heap_i[smallest] = heap_i[smallest], heap_i[root]
        root = smallest


@numba.njit(cache=True, fastmath=True, boundscheck=False, inline='always')
def _select_topk_by_value(values, touched, n_touched, k_active, out, out_vals):
    """Write the k_active touched pixels with the highest values[idx] to out."""
    if k_active >= n_touched:
        for j in range(n_touched):
            idx = touched[j]
            out[j] = idx
            out_vals[j] = values[idx]
        return n_touched
    topk_v = np.empty(k_active, dtype=np.int32)
    topk_i = np.empty(k_active, dtype=np.int32)
    for m in range(k_active):
        idx = touched[m]
        topk_v[m] = values[idx]
        topk_i[m] = idx
    for m in range(k_active // 2 - 1, -1, -1):
        _heap_sift_down(topk_v, topk_i, m, k_active)
    for t in range(k_active, n_touched):
        idx = touched[t]
        v = values[idx]
        if v > topk_v[0]:
            topk_v[0] = v
            topk_i[0] = idx
            _heap_sift_down(topk_v, topk_i, 0, k_active)
    for j in range(k_active):
        out[j] = topk_i[j]
        out_vals[j] = topk_v[j]
    return k_active


@numba.njit(cache=True, fastmath=True, boundscheck=False)
def _topk_kernel(flat_idx, counts, touched, k_active, out, out_vals):
    """Top-K most frequently hit pixels of one window.

    flat_idx : (n_events,) uint16 grid index of every event in the window
    counts   : (grid,) int32 scratch, all zero on entry and on exit
    touched  : (grid,) int32 scratch
    out, out_vals : receive the selected pixel indices and their counts
    Returns the number of active pixels written to out (<= k_active).

    A window with fewer events than grid cells records each pixel the first
    time it is hit; a busier window counts branch-free, then scans the grid.
    """
    n_events = flat_idx.shape[0]
    if n_events == 0:
        return 0
    grid_size = counts.shape[0]

    if n_events < grid_size:
        n_touched = 0
        for i in range(n_events):
            idx = flat_idx[i]
            if counts[idx] == 0:
                touched[n_touched] = idx
                n_touched += 1
            counts[idx] += 1
        if n_touched == 0:
            return 0
        k = _select_topk_by_value(counts, touched, n_touched, k_active, out, out_vals)
        for t in range(n_touched):
            counts[touched[t]] = 0
        return k

    for i in range(n_events):
        counts[flat_idx[i]] += 1
    n_touched = 0
    for i in range(grid_size):
        if counts[i] > 0:
            touched[n_touched] = i
            n_touched += 1
    k = _select_topk_by_value(counts, touched, n_touched, k_active, out, out_vals)
    for i in range(grid_size):
        counts[i] = 0
    return k


@numba.njit(cache=True, boundscheck=False)
def _gather_blocks(arr, s, step, block_len, n_blocks, out):
    """Copy n_blocks contiguous runs of block_len events, one every `step` events from s."""
    for b in range(n_blocks):
        base = s + b * step
        out[b * block_len:(b + 1) * block_len] = arr[base:base + block_len]


# ────────────────────────────────────────────────────────────────────────────
# Event loader
# ────────────────────────────────────────────────────────────────────────────

class EventLoader:
    """One traverse, cut into n_frames back-to-back windows of window_size_ms.

    At construction every event's grid index (uint16) is computed once and
    kept in flat_idx_all; frame_idx[i]:frame_idx[i+1] is frame i's slice.
    """

    # events read per chunk during ingest (bounds transient memory)
    _STREAM_CHUNK_EVENTS = 20_000_000

    def __init__(self, h5_path, n_frames, window_size_ms, sensor_w, sensor_h,
                 target_w=32, target_h=32, k_active=100, mem_cap_mb=20000,
                 max_events_per_window=0, event_sample_blocks=256):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Event HDF5 not found: {h5_path}")
        self.h5_path = h5_path
        self.n_frames = int(n_frames)
        self.window_size_ms = float(window_size_ms)
        self.sensor_w, self.sensor_h = int(sensor_w), int(sensor_h)
        self.target_w, self.target_h = int(target_w), int(target_h)
        self.k_active = int(k_active)
        self.max_active_pixels = self.k_active
        self.flat_size = self.target_h * self.target_w
        if self.flat_size > 65535:
            raise ValueError(f"image grid of {self.flat_size} pixels does not fit the uint16 event index")

        # windows with more events than this are subsampled (_capped_window); 0 = no cap
        self.max_events_per_window = int(max_events_per_window)
        self.event_sample_blocks = max(1, int(event_sample_blocks))
        self._sample_scratch = (np.empty(self.max_events_per_window, dtype=np.uint16)
                                if self.max_events_per_window else None)

        self._load_events(mem_cap_mb)

        # scratch reused by every query, so the timed path never allocates
        self._counts_scratch = np.zeros(self.flat_size, dtype=np.int32)
        self._touched_scratch = np.empty(self.flat_size, dtype=np.int32)
        self._topk_vals_scratch = np.empty(self.k_active, dtype=np.int32)
        self._active_scratch = np.empty(self.k_active, dtype=np.int32)

        # compile now, outside every timed loop
        if self.flat_idx_all.size:
            _topk_kernel(self.flat_idx_all[:1], self._counts_scratch, self._touched_scratch, self.k_active,
                         np.empty(self.k_active, dtype=np.int32), self._topk_vals_scratch)

    def _load_events(self, mem_cap_mb):
        """Map every event to its grid index (flat_idx_all, uint16, the only
        array kept) and find the frame boundaries by binary search on time."""
        with h5py.File(self.h5_path, 'r') as f:
            x_dset, y_dset, t_dset = _event_datasets(f)
            n_total = int(t_dset.shape[0])
            if n_total < 2:
                raise ValueError(f"HDF5 {self.h5_path} has < 2 events.")

            t_first_raw = float(t_dset[0])
            t_last_raw = float(t_dset[-1])
            self.time_scale = _time_scale(t_last_raw - t_first_raw, n_total)
            window_ticks = self.window_size_ms * 1e-3 * self.time_scale

            start_raw = t_first_raw
            end_raw = start_raw + self.n_frames * window_ticks
            if end_raw > t_last_raw:
                raise ValueError(
                    f"Requested {self.n_frames} frames * {self.window_size_ms:.2f}ms = "
                    f"{(end_raw - start_raw) / self.time_scale:.1f}s exceeds the "
                    f"{(t_last_raw - start_raw) / self.time_scale:.1f}s of events in "
                    f"{os.path.basename(self.h5_path)}.")
            first_idx = 0
            last_idx = _hdf5_searchsorted(t_dset, end_raw, lo=first_idx)
            n_events = last_idx - first_idx

            est_mb = (n_events * 2) / (1024 * 1024)
            print(f"  Event slice: {n_events:,} events spanning {(end_raw - start_raw) / self.time_scale:.2f}s "
                  f"(~{est_mb:.1f} MB resident, time unit {_time_unit_name(self.time_scale)})")
            if est_mb > mem_cap_mb:
                raise MemoryError(f"Event slice estimated at {est_mb:.0f} MB exceeds mem_cap={mem_cap_mb} MB. "
                                  f"Reduce --num_images or --window_size_ms, or raise --event_mem_cap_mb.")

            self.start_t_s = start_raw / self.time_scale

            # sensor pixel -> grid bin, per axis
            x_lut = np.minimum((np.arange(self.sensor_w) / (self.sensor_w / self.target_w)).astype(np.int32),
                               self.target_w - 1)
            y_lut = np.minimum((np.arange(self.sensor_h) / (self.sensor_h / self.target_h)).astype(np.int32),
                               self.target_h - 1)
            self.flat_idx_all = np.empty(n_events, dtype=np.uint16)
            for c0 in range(first_idx, last_idx, self._STREAM_CHUNK_EVENTS):
                c1 = min(c0 + self._STREAM_CHUNK_EVENTS, last_idx)
                ex = np.clip(x_dset[c0:c1].astype(np.int32), 0, self.sensor_w - 1)
                ey = np.clip(y_dset[c0:c1].astype(np.int32), 0, self.sensor_h - 1)
                flat = y_lut[ey] * self.target_w + x_lut[ex]
                self.flat_idx_all[c0 - first_idx:c1 - first_idx] = flat.astype(np.uint16, copy=False)

            timers = start_raw + np.arange(self.n_frames + 1) * window_ticks
            boundaries = np.empty(self.n_frames + 1, dtype=np.int64)
            lo = first_idx
            for i in range(self.n_frames + 1):
                idx = _hdf5_searchsorted(t_dset, timers[i], lo=lo, hi=last_idx, side='right')
                boundaries[i] = idx - first_idx
                lo = idx
            boundaries[0] = 0
            boundaries[-1] = n_events
            self.frame_idx = boundaries

    def _capped_window(self, arr, s, e):
        """Events s:e of `arr`, subsampled to ~max_events_per_window when the
        window holds more. The subsample is event_sample_blocks contiguous runs
        spread evenly over the window: a strided subsample would still touch
        every cache line, while contiguous runs use every byte they fetch."""
        n = e - s
        cap = self.max_events_per_window
        if not cap or n <= cap:
            return arr[s:e]
        n_blocks = max(1, min(self.event_sample_blocks, cap))
        block_len = max(1, cap // n_blocks)
        n_blocks = max(1, min(n_blocks, n // block_len))
        out = self._sample_scratch[:n_blocks * block_len]
        _gather_blocks(arr, s, n // n_blocks, block_len, n_blocks, out)
        return out

    def make_frame_active_pixels(self, frame_id, out_buf=None):
        """int32 view of frame `frame_id`'s active pixel indices (<= k_active)."""
        s = int(self.frame_idx[frame_id])
        e = int(self.frame_idx[frame_id + 1])
        if out_buf is None:
            out_buf = np.empty(self.max_active_pixels, dtype=np.int32)
        if e <= s:
            return out_buf[:0]
        idx_slice = self._capped_window(self.flat_idx_all, s, e)
        k = _topk_kernel(idx_slice, self._counts_scratch, self._touched_scratch,
                         self.k_active, out_buf, self._topk_vals_scratch)
        return out_buf[:k]

    def make_frame_flat(self, frame_id, out=None):
        """Flat float32 binary frame (1.0 at the active pixels), written into
        `out` if given (reused across queries), else into a new array."""
        if out is None:
            out = np.empty(self.flat_size, dtype=np.float32)
        out[:] = 0.0
        out[self.make_frame_active_pixels(frame_id, out_buf=self._active_scratch)] = 1.0
        return out

    def make_all_frames_flat(self):
        return [self.make_frame_flat(i) for i in range(self.n_frames)]

    def frame_centres_s(self):
        """Centre time of each frame (s), relative to the first event."""
        dt = self.window_size_ms / 1000.0
        return np.arange(self.n_frames, dtype=np.float64) * dt + dt / 2.0

    def frame_centres_abs_s(self):
        """Centre time of each frame in the recording's own clock (epoch s for
        Brisbane and NSAVP), for GPS/pose ground truth on the same clock."""
        return self.start_t_s + self.frame_centres_s()


# ────────────────────────────────────────────────────────────────────────────
# Ground truth
# ────────────────────────────────────────────────────────────────────────────

def _dilate_rows(GT, distance):
    if distance <= 0:
        return GT
    out = np.zeros_like(GT)
    R = GT.shape[0]
    rows, cols = np.where(GT > 0)
    for r, c in zip(rows.tolist(), cols.tolist()):
        out[max(0, r - distance):min(R, r + distance + 1), c] = 1
    return out


def _nearest_by_percent(ref_pct, qry_pct, tolerance):
    """GT[R, Q]: each query matches the reference frame closest in route
    percentage, dilated by `tolerance` reference frames."""
    idx = np.searchsorted(ref_pct, qry_pct, side='left')
    left = np.clip(idx - 1, 0, len(ref_pct) - 1)
    right = np.clip(idx, 0, len(ref_pct) - 1)
    pick_right = np.abs(ref_pct[right] - qry_pct) <= np.abs(ref_pct[left] - qry_pct)
    best = np.where(pick_right, right, left).astype(np.int64)

    GT = np.zeros((len(ref_pct), len(qry_pct)), dtype=np.uint8)
    GT[best, np.arange(len(qry_pct))] = 1
    return _dilate_rows(GT, distance=tolerance)


def build_gt_matrix_from_knots(ref_knots, qry_knots, ref_centres_s, qry_centres_s, tolerance):
    """qcr_event: route percentage interpolated from per-traverse time knots."""
    from scipy import interpolate
    ref_knots = np.asarray(ref_knots, dtype=np.float64)
    qry_knots = np.asarray(qry_knots, dtype=np.float64)
    f_ref = interpolate.interp1d(ref_knots, ref_knots / ref_knots[-1] * 100.0,
                                 bounds_error=False, fill_value=(0.0, 100.0))
    f_qry = interpolate.interp1d(qry_knots, qry_knots / qry_knots[-1] * 100.0,
                                 bounds_error=False, fill_value=(0.0, 100.0))
    ref_pct = np.maximum.accumulate(np.clip(f_ref(ref_centres_s), 0.0, 100.0))
    qry_pct = np.maximum.accumulate(np.clip(f_qry(qry_centres_s), 0.0, 100.0))
    return _nearest_by_percent(ref_pct, qry_pct, tolerance)


def parse_applanix_track(h5_path):
    """NSAVP Applanix pose log -> (epoch seconds, cumulative route percentage)."""
    with h5py.File(h5_path, 'r') as f:
        positions = f['pose_base_link/positions'][:].astype(np.float64)
        timestamps = f['pose_base_link/timestamps'][:].astype(np.float64)
    if timestamps.shape[0] < 2:
        raise ValueError(f"Applanix pose log {h5_path} has < 2 samples.")
    t_s = timestamps / 1e9
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    return t_s, np.clip(cum / cum[-1] * 100.0, 0.0, 100.0)


def build_gt_matrix_from_applanix(ref_h5, qry_h5, ref_abs_centres_s, qry_abs_centres_s, tolerance):
    """nsavp: route percentage from the Applanix pose logs, on absolute time."""
    rt, rpct = parse_applanix_track(ref_h5)
    qt, qpct = parse_applanix_track(qry_h5)
    ref_pct = np.maximum.accumulate(np.interp(ref_abs_centres_s, rt, rpct, left=rpct[0], right=rpct[-1]))
    qry_pct = np.maximum.accumulate(np.interp(qry_abs_centres_s, qt, qpct, left=qpct[0], right=qpct[-1]))
    return _nearest_by_percent(ref_pct, qry_pct, tolerance)


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def parse_nmea_track(nmea_path):
    """(UTC epoch s, lat, lon) of every valid $G[PN]RMC fix in an NMEA log."""
    def _ddmm_to_deg(token, hemi):
        try:
            v = float(token)
        except ValueError:
            return None
        deg = int(v / 100.0)
        out = deg + (v - deg * 100.0) / 60.0
        return -out if hemi in ('S', 'W') else out

    def _t_sec(hhmmss):
        try:
            return int(hhmmss[0:2]) * 3600.0 + int(hhmmss[2:4]) * 60.0 + float(hhmmss[4:])
        except (ValueError, IndexError):
            return None

    times, lats, lons = [], [], []
    first_date = None
    with open(nmea_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) < 10 or line[0] != '$':
                continue
            if '*' in line:
                line = line[:line.index('*')]
            parts = line.split(',')
            if not parts[0].endswith('RMC') or len(parts) < 7:
                continue
            t = _t_sec(parts[1])
            lat = _ddmm_to_deg(parts[3], parts[4])
            lon = _ddmm_to_deg(parts[5], parts[6])
            if t is None or lat is None or lon is None or parts[2] != 'A' or (lat == 0.0 and lon == 0.0):
                continue
            if first_date is None and len(parts) > 9:
                first_date = parts[9]
            times.append(t)
            lats.append(lat)
            lons.append(lon)
    if not times:
        raise ValueError(f"No valid RMC fixes parsed from {nmea_path}")

    t = np.asarray(times, dtype=np.float64)
    for i in range(1, len(t)):  # UTC midnight wrap
        if t[i] < t[i - 1]:
            t[i] += 86400.0
    try:
        day, month, year = int(first_date[0:2]), int(first_date[2:4]), 2000 + int(first_date[4:6])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"No usable RMC date field in {nmea_path}")
    t += calendar.timegm((year, month, day, 0, 0, 0))
    return t, np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


def build_gt_matrix_from_nmea_positions(ref_nmea, qry_nmea, ref_centres_abs_s, qry_centres_abs_s,
                                        tolerance, route_window_m=300.0, chunk=512):
    """brisbane_event: GT[R, Q] from GPS positions, events and GPS on the same clock.

    A frame's position is looked up at the frame's absolute time, and a query's
    GT reference frame is the physically nearest one in metres. To keep a route
    that passes the same spot twice from aliasing, candidates are limited to
    reference frames within `route_window_m` of the query's own distance along
    the route (after removing the median along-route offset between the two
    traverses). Frames outside their traverse's GPS coverage have no position:
    such a query gets an all-zero GT column, such a reference is never chosen.

    Returns (GT uint8 (R, Q), info dict).
    """
    rt, rlat, rlon = parse_nmea_track(ref_nmea)
    qt, qlat, qlon = parse_nmea_track(qry_nmea)
    ref_c = np.asarray(ref_centres_abs_s, dtype=np.float64)
    qry_c = np.asarray(qry_centres_abs_s, dtype=np.float64)
    ref_cov = (ref_c >= rt[0]) & (ref_c <= rt[-1])
    qry_cov = (qry_c >= qt[0]) & (qry_c <= qt[-1])
    if not ref_cov.any() or not qry_cov.any():
        raise ValueError(
            f"GPS and event time ranges do not overlap (ref frames {ref_c[0]:.1f}..{ref_c[-1]:.1f} vs GPS "
            f"{rt[0]:.1f}..{rt[-1]:.1f}; query frames {qry_c[0]:.1f}..{qry_c[-1]:.1f} vs GPS "
            f"{qt[0]:.1f}..{qt[-1]:.1f}, epoch s) -- are the event timestamps absolute?")

    def _frame_track(t, lat, lon, centres):
        cum = np.concatenate([[0.0], np.cumsum(_haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
        return np.interp(centres, t, lat), np.interp(centres, t, lon), np.interp(centres, t, cum)

    r_lat, r_lon, r_cum = _frame_track(rt, rlat, rlon, ref_c)
    q_lat, q_lon, q_cum = _frame_track(qt, qlat, qlon, qry_c)
    R, Q = len(ref_c), len(qry_c)

    def _nearest(route_offset_m=None):
        best = np.zeros(Q, dtype=np.int64)
        dist = np.full(Q, np.inf)
        for a in range(0, Q, chunk):
            b = min(Q, a + chunk)
            d = _haversine_m(q_lat[a:b, None], q_lon[a:b, None], r_lat[None, :], r_lon[None, :])
            d[:, ~ref_cov] = np.inf
            if route_offset_m is not None:
                d[np.abs(r_cum[None, :] - (q_cum[a:b, None] + route_offset_m)) > route_window_m] = np.inf
            best[a:b] = d.argmin(axis=1)
            dist[a:b] = d[np.arange(b - a), best[a:b]]
        return best, dist

    free_best, _ = _nearest()
    route_offset_m = float(np.median(r_cum[free_best[qry_cov]] - q_cum[qry_cov]))
    best, dist = _nearest(route_offset_m)
    valid = qry_cov & np.isfinite(dist)

    GT = np.zeros((R, Q), dtype=np.uint8)
    GT[best[valid], np.flatnonzero(valid)] = 1
    GT = _dilate_rows(GT, distance=tolerance)

    info = dict(
        n_ref=R, n_query=Q, n_ref_no_gps=int((~ref_cov).sum()), n_query_no_gt=int((~valid).sum()),
        match_dist_m_median=float(np.median(dist[valid])) if valid.any() else float('nan'),
        match_dist_m_max=float(dist[valid].max()) if valid.any() else float('nan'),
        route_offset_m=route_offset_m,
        ref_first_fix_minus_first_frame_s=float(rt[0] - ref_c[0]),
        qry_first_fix_minus_first_frame_s=float(qt[0] - qry_c[0]),
    )
    return GT, info


def build_gt_matrix(dataset, ref_seq, qry_seq, data_dir, ref_loader, qry_loader, tolerance):
    """(R, Q) uint8 ground truth between two loaded traverses of `dataset`."""
    entry = DATASETS[dataset]
    fmt = entry['gt_format']
    if fmt == 'knots':
        print("  Building route-% GT from time knots")
        knots = entry['gt_times']
        return build_gt_matrix_from_knots(knots[ref_seq], knots[qry_seq], ref_loader.frame_centres_s(),
                                          qry_loader.frame_centres_s(), tolerance)

    ref_gt = resolve_gt_path(entry, ref_seq, data_dir)
    qry_gt = resolve_gt_path(entry, qry_seq, data_dir)
    if not (ref_gt and qry_gt):
        raise FileNotFoundError(f"GT files for {dataset} {ref_seq}/{qry_seq} not found under {data_dir}")

    if fmt == 'applanix':
        print(f"  Building route-% GT from Applanix pose: {os.path.basename(ref_gt)} vs {os.path.basename(qry_gt)}")
        return build_gt_matrix_from_applanix(ref_gt, qry_gt, ref_loader.frame_centres_abs_s(),
                                             qry_loader.frame_centres_abs_s(), tolerance)

    print(f"  Building position GT from NMEA (absolute time): {os.path.basename(ref_gt)} vs {os.path.basename(qry_gt)}")
    GT, info = build_gt_matrix_from_nmea_positions(ref_gt, qry_gt, ref_loader.frame_centres_abs_s(),
                                                   qry_loader.frame_centres_abs_s(), tolerance)
    print(f"    first GPS fix minus first frame centre: ref {info['ref_first_fix_minus_first_frame_s']:+.2f} s, "
          f"query {info['qry_first_fix_minus_first_frame_s']:+.2f} s")
    print(f"    query frame to its GT reference frame: median {info['match_dist_m_median']:.1f} m, "
          f"max {info['match_dist_m_max']:.1f} m (along-route offset between traverses {info['route_offset_m']:+.0f} m)")
    if info['n_query_no_gt'] or info['n_ref_no_gps']:
        print(f"    {info['n_query_no_gt']}/{info['n_query']} query frames lie outside the query GPS log and have "
              f"no ground truth (all-zero GT column, scored as misses); "
              f"{info['n_ref_no_gps']}/{info['n_ref']} reference frames lie outside the reference GPS log")
    if info['match_dist_m_max'] > 50.0:
        print(f"    WARNING: some queries are {info['match_dist_m_max']:.0f} m from every reference frame")
    return GT
