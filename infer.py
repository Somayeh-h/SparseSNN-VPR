"""
Inference with a trained STDP checkpoint: Dense SNN and Sparse SNN.

Queries are processed one at a time. Each call is timed in two parts:
preprocessing (raw event window -> the K active pixels, or the full binary
frame for the Dense SNN) and the kernel. Each call is repeated
--n_timing_repeats times and a query's latency is the median of its repeats.

Two more kernels are timed only for the speedup decomposition
(plots/make_latency_decomposition_table.py): the gather-sum kernel (the
Sparse SNN's K-row walk over float32 weights) and the dense kernel on one
thread.

Outputs (<output_path>/infer/):
  results.json       R@1, R@N and mean preprocessing / kernel latency of "dense" and "sparse"
  stdp_latency.npz   per-query latencies and predictions of every kernel
  vpr_similarity_matrix_SparseSNN.png, gt_matrix_built.png, query_matches/

    python train.py --output_path outputs/run1 [dataset args]
    python infer.py --checkpoint outputs/run1 --output_path outputs/run1 [dataset args]
"""

import json
import os
import time

import numba
import numpy as np

from src.parser import parse_arguments, CHECKPOINT_LOCKED_ARGS
from src.common import (print_argparse_args, setup_experiment, physical_core_count, repeat_timed_call,
                        print_weight_cache_fit)
from src.checkpoint import load_checkpoint
from src.data_pipeline import build_dataset, resolve_full_sequence_args
from src.eval_metrics import compute_accuracy_and_recall, debug_gt_and_predictions
from src.inference_kernels import prepare_inference_state
from src.plottings import plot_query_matches, plot_similarity_matrix


RECALL_NS = (1, 5, 10, 20, 50, 100)


def main(args):
    resolve_full_sequence_args(args, ref_only=True)
    data_path = setup_experiment(args, subdir="infer")

    print("=" * 70)
    print("STDP Inference for Visual Place Recognition")
    print("=" * 70)
    print_argparse_args(args)

    model = get_model(args)
    resolve_full_sequence_args(args)  # now with the checkpoint's window size
    dataset = build_dataset(args, data_path)

    print_avg_events_per_frame(dataset.event_loader, "ref")
    print_avg_events_per_frame(dataset.query_loader, "qry")

    results = run_query_network(args, dataset.query_loader, model)
    evaluate(args, dataset.query_loader, model, results, data_path, dataset.gt_matrix, dataset.event_loader)
    print(f"\nDone. All results saved to {data_path}/")


def print_avg_events_per_frame(loader, label):
    """Mean raw event count per window, before the event cap and top-K."""
    counts = np.diff(loader.frame_idx)
    print(f"  {label}: mean raw events/frame = {counts.mean():.1f} "
          f"(window_size_ms={loader.window_size_ms}, n_frames={loader.n_frames})")


def apply_checkpoint_args(args, train_args):
    """Overwrite the checkpoint-locked args (network shape, reference
    gallery) with the values the checkpoint was trained with."""
    for key in CHECKPOINT_LOCKED_ARGS:
        if key not in train_args:
            continue
        cli_val = getattr(args, key, None)
        ckpt_val = train_args[key]
        if cli_val != ckpt_val:
            print(f"  (using checkpoint's {key}={ckpt_val!r}, ignoring CLI value {cli_val!r})")
        setattr(args, key, ckpt_val)


def get_model(args):
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    apply_checkpoint_args(args, ckpt['train_args'])
    print(f"  W shape: {ckpt['W'].shape}, {len(ckpt['neuron_to_image'])}/{args.output_size} neurons assigned")
    return ckpt


def print_weight_cache_report(args, state):
    """Whether each SNN's weights fit in the CPU caches."""
    print(f"\n  Weight matrix CPU cache fit:")
    print_weight_cache_fit("Dense SNN weights (float32, input_size x output_size)",
                           args.input_size * args.output_size * 4.0)
    print_weight_cache_fit("Sparse SNN synapse mask (uint8, input_size x n_cols)",
                           float(state['synapse_mask_info']['bytes']))


def _time_kernel(n_query, n_repeats, call_once):
    """Time call_once(i) -> (pred, total_us, preprocess_us, kernel_us) on every
    query. Keeps the per-query medians, plus the mean, std and IQR of the
    kernel time over the repeats as a record of the timing noise."""
    out = {'pred': np.full(n_query, -1, dtype=np.int64)}
    for key in ('total', 'preprocess', 'kernel', 'kernel_mean', 'kernel_std', 'kernel_iqr'):
        out[key] = np.full(n_query, np.nan, dtype=np.float64)
    for i in range(n_query):
        pred, (total_us, pre_us, kernel_us), stats = repeat_timed_call(n_repeats, lambda i=i: call_once(i))
        out['pred'][i] = pred
        out['total'][i], out['preprocess'][i], out['kernel'][i] = total_us, pre_us, kernel_us
        out['kernel_mean'][i], out['kernel_std'][i], out['kernel_iqr'][i] = (
            stats['mean'][2], stats['std'][2], stats['iqr'][2])
    return out


def _print_kernel_timing(label, t, n_repeats):
    print(f"\n  {label}: mean over {len(t['kernel'])} queries of the median of {n_repeats} repeats")
    print(f"    preprocessing {np.mean(t['preprocess']):8.3f} us")
    print(f"    kernel        {np.mean(t['kernel']):8.3f} us  (std across queries {np.std(t['kernel']):.3f} us)")
    print(f"    total         {np.mean(t['total']):8.3f} us")
    if n_repeats > 1:
        print(f"    kernel time over the repeats of one query, averaged over queries: "
              f"mean {np.mean(t['kernel_mean']):.3f} us, std {np.mean(t['kernel_std']):.3f} us, "
              f"IQR {np.mean(t['kernel_iqr']):.3f} us")


def run_query_network(args, query_loader, model):
    """Time each kernel over the whole query set, one kernel at a time, with
    each query's preprocessing timed in the same call. Returns the timings,
    predictions and per-query scores."""
    n_query = query_loader.n_frames
    state = prepare_inference_state(args, model['W'], model['image_to_neurons'], n_query,
                                    prune_thresholds=model['prune_thresholds'])
    print_weight_cache_report(args, state)

    n_repeats = max(1, args.n_timing_repeats)
    kernels = {name: state[f'fused_infer_{name}_closure'] for name in ('gathersum', 'sparse', 'dense')}
    scores = {name: np.zeros_like(state['per_img_spikes_arr']) for name in kernels}
    scratch = {'gathersum': np.zeros_like(state['scratch_current']), 'sparse': state['sparse_cnt_scratch'],
               'dense': np.zeros_like(state['scratch_current'])}
    # query buffers, reused by every call
    active_buf = np.empty(query_loader.max_active_pixels, dtype=np.int32)
    frame_buf = np.empty(args.input_size, dtype=np.float32)

    def sparse_call(name):
        kernel = kernels[name]

        def _call_once(i):
            t0 = time.perf_counter()
            active = query_loader.make_frame_active_pixels(i, out_buf=active_buf)
            t_pre = time.perf_counter()
            pred = int(kernel(active, active.size, i, scratch[name], scores[name]))
            t1 = time.perf_counter()
            return pred, (t1 - t0) * 1e6, (t_pre - t0) * 1e6, (t1 - t_pre) * 1e6
        return _call_once

    def dense_call(i):
        t0 = time.perf_counter()
        q = query_loader.make_frame_flat(i, out=frame_buf)
        t_pre = time.perf_counter()
        pred = int(kernels['dense'](q, i, scratch['dense'], scores['dense']))
        t1 = time.perf_counter()
        return pred, (t1 - t0) * 1e6, (t_pre - t0) * 1e6, (t1 - t_pre) * 1e6

    # The dense kernel is memory-bandwidth bound: one thread per physical core.
    prev_num_threads = numba.get_num_threads()
    numba.set_num_threads(physical_core_count())
    try:
        timings = {}
        for name in ('gathersum', 'sparse'):
            call_once = sparse_call(name)
            call_once(0)  # JIT warmup
            timings[name] = _time_kernel(n_query, n_repeats, call_once)

        dense_call(0)  # JIT warmup
        timings['dense'] = _time_kernel(n_query, n_repeats, dense_call)
        n_threads_dense = numba.get_num_threads()

        # the dense kernel alone on one thread, for its speedup from threading
        dense_1t_kernel_us = np.empty(n_query, dtype=np.float64)
        numba.set_num_threads(1)
        dense_call(0)  # warmup at 1 thread
        for i in range(n_query):
            q = query_loader.make_frame_flat(i, out=frame_buf)

            def _call_once(i=i):
                t0 = time.perf_counter()
                pred = int(kernels['dense'](q, i, scratch['dense'], scores['dense']))
                return pred, (time.perf_counter() - t0) * 1e6
            _, (dense_1t_kernel_us[i],), _ = repeat_timed_call(n_repeats, _call_once)
    finally:
        numba.set_num_threads(prev_num_threads)

    _print_kernel_timing("Sparse SNN (uint8 mask rows of the K active pixels, 1 thread)", timings['sparse'], n_repeats)
    _print_kernel_timing(f"Dense SNN ({n_threads_dense} threads)", timings['dense'], n_repeats)
    k_dense_1t = np.mean(dense_1t_kernel_us)
    print(f"    kernel on 1 thread {k_dense_1t:8.3f} us  "
          f"({k_dense_1t / np.mean(timings['dense']['kernel']):.2f}x slower than {n_threads_dense} threads)")

    return {'state': state, 'timings': timings, 'scores': scores,
            'dense_1t_kernel_us': dense_1t_kernel_us, 'n_threads_dense': n_threads_dense}


def _build_similarity_matrix(per_img_spikes, image_to_neurons, n_assigned, n_query, num_images):
    """(num_images, n_query) similarity from (n_query, n_assigned) scores;
    images with no assigned neuron get a zero row."""
    similarity_matrix = np.zeros((num_images, n_query), dtype=np.float32)
    if n_assigned > 0:
        assigned_imgs = [i for i, ns in enumerate(image_to_neurons) if ns]
        similarity_matrix[assigned_imgs, :] = per_img_spikes[:, :n_assigned].T.astype(np.float32)
    return similarity_matrix


def _score(label, args, per_img_spikes, assigned_imgs_arr, n_query, gt):
    """R@1 and R@N of one kernel's scores. Returns (R@1, hits per N, top-1 predictions)."""
    recall_Ns = [n for n in RECALL_NS if n <= args.output_size]
    top1, top1_pred, hits = compute_accuracy_and_recall(per_img_spikes, assigned_imgs_arr, gt, recall_Ns)
    acc = float(np.mean(top1))
    print(f"\n  Accuracy ({label}):")
    print(f"    R@1: {int(top1.sum())}/{n_query} ({100*acc:.1f}%)")
    for N in recall_Ns:
        print(f"    Recall@{N:<3d}: {hits[N]}/{n_query} ({100*hits[N]/n_query:.1f}%)")
    return acc, hits, top1_pred


def _plot_sparse_snn(args, query_loader, event_loader, model, state, scores, gt, predicted, data_path):
    """Sparse SNN similarity matrix, and one figure per query (query, true
    ref, predicted ref, learned receptive field)."""
    gt_arr = np.asarray(gt)
    n_query = gt_arr.shape[1]
    fname = 'vpr_similarity_matrix_SparseSNN.png'
    similarity_matrix = _build_similarity_matrix(scores, model['image_to_neurons'], state['n_assigned'], n_query,
                                                 args.num_images)
    plot_similarity_matrix(similarity_matrix, gt, predicted, data_path, fname=fname,
                           title=f'Similarity Matrix (Sparse SNN)\n({args.num_images} refs x {n_query} queries)',
                           value_label='Summed current')
    print(f"  Similarity matrix plot (Sparse SNN): {data_path}/{fname}")

    query_plot_indices = [q for q in args.query_plot_indices if 0 <= q < n_query]
    query_imgs_plot = [query_loader.make_frame_flat(i) for i in query_plot_indices]
    needed_refs = set()
    for q in query_plot_indices:
        if 0 <= predicted[q] < gt_arr.shape[0]:
            needed_refs.add(int(predicted[q]))
        positives = np.where(gt_arr[:, q] > 0)[0]
        if positives.size > 0:
            needed_refs.add(int(positives[positives.size // 2]))
    ref_imgs_plot = [None] * gt_arr.shape[0]
    for r in needed_refs:
        if 0 <= r < event_loader.n_frames:
            ref_imgs_plot[r] = event_loader.make_frame_flat(r)

    plot_query_matches(query_imgs_plot, ref_imgs_plot, predicted[query_plot_indices], gt_arr,
                       args.image_size, data_path, W=np.asarray(model['W']), image_to_neurons=model['image_to_neurons'],
                       n_show=len(query_plot_indices), query_indices=query_plot_indices)
    print(f"  Per-query match plots: {data_path}/query_matches/ ({len(query_plot_indices)} figures)")


def evaluate(args, query_loader, model, results, data_path, gt, event_loader):
    """Score both SNNs, plot the Sparse SNN, print the latency summary and
    save results.json and stdp_latency.npz."""
    state, timings, scores = results['state'], results['timings'], results['scores']
    assigned_imgs_arr = state['assigned_imgs_arr']
    n_query = query_loader.n_frames

    acc, hits = {}, {}
    acc['dense'], hits['dense'], _ = _score('Dense SNN', args, scores['dense'], assigned_imgs_arr, n_query, gt)
    acc['sparse'], hits['sparse'], top1_pred = _score('Sparse SNN', args, scores['sparse'], assigned_imgs_arr,
                                                      n_query, gt)

    debug_gt_and_predictions(gt, scores['sparse'], assigned_imgs_arr, predicted=timings['sparse']['pred'],
                             top1_pred=top1_pred, label='Sparse SNN', tolerance=args.gt_tolerance,
                             max_show=min(20, n_query))
    _plot_sparse_snn(args, query_loader, event_loader, model, state, scores['sparse'], gt,
                     timings['sparse']['pred'], data_path)

    mean = {name: {k: float(np.mean(t[k])) for k in ('total', 'preprocess', 'kernel')} for name, t in timings.items()}
    mi = state['synapse_mask_info']
    print(f"\n{'=' * 70}")
    print("INFERENCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sparse SNN: uint8 synapse mask ({mi['n_cols']} places, {mi['bytes'] / 1e6:.3f} MB), 1 thread")
    print(f"  Dense SNN:  float32 weights, {results['n_threads_dense']} threads")
    print(f"\n  Mean latency per query (us)   preprocessing    kernel     total")
    for name, label in (('dense', 'Dense SNN'), ('sparse', 'Sparse SNN')):
        m = mean[name]
        print(f"  {label:<28s}{m['preprocess']:12.2f}{m['kernel']:10.2f}{m['total']:10.2f}")
    print(f"\n  Sparse over Dense SNN speedup: {mean['dense']['kernel'] / mean['sparse']['kernel']:.2f}x (kernel), "
          f"{mean['dense']['total'] / mean['sparse']['total']:.2f}x (total)")
    print(f"  R@1: Dense SNN {100 * acc['dense']:.1f}%, Sparse SNN {100 * acc['sparse']:.1f}%")

    summary = {name: dict(r_at_1=acc[name], recall={str(k): v / n_query for k, v in hits[name].items()},
                          prep_us=mean[name]['preprocess'], net_us=mean[name]['kernel'])
               for name in ('dense', 'sparse')}
    summary['n_query'] = n_query
    with open(os.path.join(data_path, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fields = {}
    for name, t in timings.items():
        fields[f'per_image_us_{name}'] = t['total']
        fields[f'per_image_us_{name}_preprocess'] = t['preprocess']
        fields[f'per_image_us_{name}_kernel'] = t['kernel']
        fields[f'per_image_us_{name}_kernel_mean'] = t['kernel_mean']
        fields[f'per_image_us_{name}_kernel_std'] = t['kernel_std']
        fields[f'per_image_us_{name}_kernel_iqr'] = t['kernel_iqr']
        fields[f'predictions_{name}'] = t['pred']
    np.savez(os.path.join(data_path, "stdp_latency.npz"), **fields,
             per_image_us_dense_1t_kernel=results['dense_1t_kernel_us'], n_threads_dense=results['n_threads_dense'])


if __name__ == "__main__":
    main(parse_arguments())
