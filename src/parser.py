"""Command-line arguments shared by train.py, infer.py, sad_inference.py and
lens_event_inference.py.

Defaults are the paper's QCR-Event configuration. Arguments marked as
checkpoint-locked define the network shape or the reference gallery: infer.py
replaces them with the values stored in the checkpoint's args.json.
"""
import argparse


# Filled by parse_arguments(): every checkpoint-locked dest, plus the two
# derived sizes (input_size = image_size**2, output_size = num_images).
CHECKPOINT_LOCKED_ARGS = []
_DERIVED_LOCKED_ARGS = ['input_size', 'output_size']


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    locked = []

    def locked_arg(*a, **kw):
        locked.append(parser.add_argument(*a, **kw).dest)

    # ── Data: event windows -> top-K binary frames ──
    locked_arg('--event_dataset', type=str, default='qcr_event',
               choices=['qcr_event', 'brisbane_event', 'nsavp', 'synthetic'],
               help='Event dataset (sensor size, sequence files and GT source come from '
                    'event_data_loader.DATASETS). "synthetic" generates random top-K frames instead and '
                    'queries them against themselves (src/synthetic_data.py): an end-to-end test that needs '
                    'no data.')
    locked_arg('--event_data_dir', type=str, default='',
               help="Directory with the dataset's event files. Empty = $EVENT_LAB_DATASETS/<event_dataset>, "
                    "or ../Event-LAB/datasets/<event_dataset> if that variable is not set.")
    locked_arg('--event_seq', type=str, default='',
               help="Reference sequence. Empty = the dataset's default reference traverse.")
    parser.add_argument('--event_query_seq', type=str, default='',
                        help="Query sequence. Empty = the dataset's default query traverse.")
    locked_arg('--num_images', type=int, default=1600,
               help='Number of reference frames (= learned places). -1 = every whole window in the '
                    'reference sequence.')
    parser.add_argument('--event_query_num_images', type=int, default=1600,
                        help='Number of query frames. -1 = every whole window in the query sequence, '
                             '0 = same value as --num_images.')
    locked_arg('--window_size_ms', type=float, default=100.0, help='Event integration window per frame (ms).')
    locked_arg('--image_size', type=int, default=32, help='Frames are image_size x image_size pixels.')
    locked_arg('--pixels_per_image', type=int, default=100,
               help='Top-K: number of active (most frequently hit) pixels kept per frame.')
    locked_arg('--max_events_per_window', type=int, default=20000,
               help='Cap on events read per reference window; a window holding more is subsampled '
                    '(see --event_sample_blocks). 0 = no cap.')
    parser.add_argument('--query_max_events_per_window', type=int, default=20000,
                        help='Same cap for the query windows, which is where the per-query preprocessing '
                             'latency is measured. -1 = follow --max_events_per_window.')
    locked_arg('--event_sample_blocks', type=int, default=256,
               help='When the cap binds, read the subsample as this many contiguous runs spread evenly '
                    'across the window (see EventLoader._capped_window).')
    locked_arg('--event_mem_cap_mb', type=int, default=20000, help='Memory cap (MB) for one loaded event slice.')
    parser.add_argument('--gt_tolerance', type=int, default=10,
                        help='A query matches every reference frame within this many frames of its '
                             'ground-truth frame.')

    # ── LIF neuron ──
    locked_arg('--dt_ms', type=float, default=0.001, help='Simulation time step.')
    locked_arg('--v_rest', type=float, default=0.0, help='Resting potential.')
    locked_arg('--v_th', type=float, default=0.5, help='Spike threshold.')
    locked_arg('--v_reset', type=float, default=0.0, help='Reset potential.')
    locked_arg('--tau', type=float, default=0.01, help='Membrane time constant.')
    locked_arg('--r', type=float, default=1.0, help='Membrane resistance.')
    locked_arg('--tau_ref', type=float, default=0.005, help='Refractory period.')

    # ── STDP training (train.py) ──
    parser.add_argument('--n_epochs', type=int, default=30, help='Training epochs.')
    parser.add_argument('--n_train_steps', type=int, default=125, help='Time steps per image presentation.')
    parser.add_argument('--p_spike_train', type=float, default=0.1,
                        help='Per-step Poisson spike probability of an active pixel during training.')
    parser.add_argument('--tau_pre', type=float, default=0.02, help='Pre-synaptic trace time constant.')
    parser.add_argument('--tau_post', type=float, default=0.02, help='Post-synaptic trace time constant.')
    parser.add_argument('--A_plus_frac', type=float, default=0.015, help='LTP amplitude as a fraction of W_max.')
    parser.add_argument('--A_minus_frac', type=float, default=0.003, help='LTD amplitude as a fraction of W_max.')
    parser.add_argument('--theta_plus_frac', type=float, default=0.05,
                        help='Homeostatic threshold increase per output spike, as a fraction of v_th.')
    parser.add_argument('--theta_init_max_frac', type=float, default=0.05,
                        help='Initial threshold offsets ~ U(0, this * v_th).')
    parser.add_argument('--theta_decay', type=float, default=0.99,
                        help='Multiplicative decay of the threshold offsets after every image.')
    parser.add_argument('--silence_threshold', type=int, default=3,
                        help='A neuron silent for more than this many consecutive images gets its '
                             'threshold lowered (force-fire).')
    parser.add_argument('--silence_bonus_frac', type=float, default=0.4,
                        help='Force-fire threshold reduction, as a fraction of v_th.')
    parser.add_argument('--silence_decay_end_epoch', type=int, default=10,
                        help='Epoch by which the force-fire bonus has decayed linearly to 0.')
    parser.add_argument('--target_bonus_frac', type=float, default=0.5,
                        help="Supervised WTA: threshold reduction for the presented image's own neuron, "
                             "as a fraction of v_th (image i owns neuron i).")

    # ── Inference / timing ──
    parser.add_argument('--n_timing_repeats', type=int, default=10,
                        help="Each query's timed call is repeated this many times back to back and the "
                             "per-query latency is the median of the repeats (see common.repeat_timed_call).")
    parser.add_argument('--query_plot_indices', type=int, default=list(range(20)), nargs='+',
                        help='Query indices rendered in query_matches/ (infer.py).')

    # ── Output ──
    parser.add_argument('--output_path', type=str, default='', help='Output directory (empty = auto-named).')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='infer.py: directory holding a train.py checkpoint (model.npz + args.json).')

    args = parser.parse_args()
    args.input_size = args.image_size * args.image_size
    args.output_size = args.num_images
    CHECKPOINT_LOCKED_ARGS[:] = locked + _DERIVED_LOCKED_ARGS
    return args
