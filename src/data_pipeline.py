"""
Builds the reference frames, the query EventLoader and the ground truth from
the command-line arguments. train.py, infer.py, sad_inference.py and
lens_event_inference.py all go through here, so every method sees the same
frames. --event_dataset synthetic swaps the event files for generated frames
(src/synthetic_data.py).
"""

import os
from dataclasses import dataclass

import numpy as np

from .event_data_loader import DATASETS, EventLoader, resolve_event_path, resolve_full_n_frames, build_gt_matrix
from .plottings import plot_gt_matrix
from .synthetic_data import ArrayLoader, generate_synthetic_images, make_self_recall_gt

# --event_data_dir defaults to <data root>/<event_dataset>, with the data root
# taken from this environment variable, or Event-LAB next to this repository.
DATA_ROOT_ENV = 'EVENT_LAB_DATASETS'
DEFAULT_DATA_ROOT = os.path.join('..', 'Event-LAB', 'datasets')


@dataclass
class Dataset:
    images: list                    # reference frames, flat binary float32 arrays of length input_size
    event_loader: EventLoader       # reference traverse
    query_loader: EventLoader = None
    gt_matrix: np.ndarray = None    # (n_ref, n_query) uint8, tolerance-dilated


def resolve_full_sequence_args(args, ref_only=False):
    """Fill in dataset defaults and resolve the -1 ("whole sequence") frame counts.

    Must run before setup_experiment/build_dataset so folder names and the
    network shape see real counts. output_size is always num_images (one
    neuron per reference place).

    ref_only=True stops before the query count: infer.py resolves the
    reference first, then the query once the checkpoint's window size is in
    place, so a -1 query count uses the training-time window.
    """
    if args.event_dataset == 'synthetic':
        _resolve_synthetic_args(args)
        return

    entry = DATASETS[args.event_dataset]
    if not args.event_data_dir:
        args.event_data_dir = os.path.join(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT), args.event_dataset)
    if not args.event_seq:
        args.event_seq = entry['default_ref']
    if not args.event_query_seq:
        args.event_query_seq = entry['default_query']

    raw_num_images = args.num_images
    if raw_num_images == -1:
        ref_path, _ = resolve_event_path(args.event_dataset, args.event_seq, args.event_data_dir)
        args.num_images = resolve_full_n_frames(ref_path, args.window_size_ms)
        print(f"  --num_images -1: resolved to {args.num_images} frames (whole {args.event_seq} at "
              f"{args.window_size_ms} ms windows)")
    args.output_size = args.num_images

    if ref_only:
        return

    query_sentinel = args.event_query_num_images if args.event_query_num_images != 0 else raw_num_images
    if query_sentinel == -1:
        qry_path, _ = resolve_event_path(args.event_dataset, args.event_query_seq, args.event_data_dir)
        args.event_query_num_images = resolve_full_n_frames(qry_path, args.window_size_ms)
        print(f"  --event_query_num_images -1: resolved to {args.event_query_num_images} frames "
              f"(whole {args.event_query_seq} at {args.window_size_ms} ms windows)")
    else:
        args.event_query_num_images = query_sentinel


def _resolve_synthetic_args(args):
    """The query traverse is the reference frames again, so it has
    num_images frames whatever --event_query_num_images says."""
    if args.num_images <= 0:
        raise SystemExit("--event_dataset synthetic needs an explicit --num_images > 0")
    args.event_seq = args.event_query_seq = 'synthetic'
    if args.event_query_num_images not in (0, args.num_images):
        print(f"  synthetic data: queries are the {args.num_images} reference frames, "
              f"ignoring --event_query_num_images {args.event_query_num_images}")
    args.event_query_num_images = args.num_images
    args.output_size = args.num_images


def query_event_cap(args):
    """Events-per-window cap of the QUERY loader: --query_max_events_per_window,
    or the (checkpoint-locked) reference cap when that is -1."""
    cap = args.query_max_events_per_window
    return cap if cap >= 0 else args.max_events_per_window


def make_event_loader(args, seq, n_frames, max_events_per_window):
    """EventLoader for one traverse of args.event_dataset with the run's frame settings."""
    if args.event_dataset == 'synthetic':
        print(f"\nGenerating {n_frames} synthetic {args.image_size}x{args.image_size} frames "
              f"({args.pixels_per_image} active px each)")
        return ArrayLoader(generate_synthetic_images(args), args.pixels_per_image)
    path, entry = resolve_event_path(args.event_dataset, seq, args.event_data_dir)
    print(f"\nLoading event data ({args.event_dataset}/{seq}): {path}")
    print(f"  {n_frames} frames | window={args.window_size_ms}ms | {args.image_size}x{args.image_size} | "
          f"top-K={args.pixels_per_image} | events/window cap={max_events_per_window:,}")
    return EventLoader(
        h5_path=path, n_frames=n_frames, window_size_ms=args.window_size_ms,
        sensor_w=entry['sensor_w'], sensor_h=entry['sensor_h'],
        target_w=args.image_size, target_h=args.image_size, k_active=args.pixels_per_image,
        mem_cap_mb=args.event_mem_cap_mb, max_events_per_window=max_events_per_window,
        event_sample_blocks=args.event_sample_blocks,
    )


def make_query_loader(args):
    return make_event_loader(args, args.event_query_seq, args.event_query_num_images, query_event_cap(args))


def build_dataset(args, data_path=None, need_query=True):
    """Reference frames (built once, as training sees them), plus the query
    loader and ground truth when need_query. The query frames are not built
    here: inference scripts build each one inside their timed loop."""
    event_loader = make_event_loader(args, args.event_seq, args.num_images, args.max_events_per_window)
    images = event_loader.make_all_frames_flat()

    if not need_query:
        return Dataset(images=images, event_loader=event_loader)

    query_loader = make_query_loader(args)
    if args.event_dataset == 'synthetic':
        gt_matrix = make_self_recall_gt(event_loader.n_frames, query_loader.n_frames, args.gt_tolerance)
    else:
        gt_matrix = build_gt_matrix(args.event_dataset, args.event_seq, args.event_query_seq, args.event_data_dir,
                                    event_loader, query_loader, args.gt_tolerance)
    R, Q = gt_matrix.shape
    print(f"  GT matrix built: shape=({R}, {Q}) tolerance={args.gt_tolerance} | "
          f"mean per-query matches={np.mean(gt_matrix.sum(axis=0)):.2f}")
    if data_path is not None:
        title = f'Dilated GT ({args.event_seq} vs {args.event_query_seq}, R={R} Q={Q}, tolerance={args.gt_tolerance})'
        plot_gt_matrix(gt_matrix, data_path, fname='gt_matrix_built.png', title=title)

    return Dataset(images=images, event_loader=event_loader, query_loader=query_loader, gt_matrix=gt_matrix)
