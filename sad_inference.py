"""
Sum of Absolute Differences (SAD) baseline: each query is matched to the
reference frame with the smallest SAD.

On the binary top-K frames the SNN also sees, SAD is the Hamming distance
popcount(q XOR r), so every frame is bit-packed into ceil(D/64) uint64 words
(16 words at D=1024 instead of 1024 subtractions).

Timing follows infer.py, per query on one CPU core:
  preprocessing  raw event window -> top-K active pixels -> bit-packed query
  network        the packed SAD kernel, including its argmin
each call repeated --n_timing_repeats times, latency = median. R@N ties keep
the lowest reference index, as in the kernel.

Outputs (<output_path>/): results.json, sad_latency.npz, args.json and plots.
"""

import json
import os
import time

import numba
import numpy as np

from src.parser import parse_arguments
from src.common import redirect_output, default_run_name, repeat_timed_call
from src.data_pipeline import build_dataset, resolve_full_sequence_args
from src.eval_metrics import debug_gt_and_predictions
from src.plottings import plot_query_matches, plot_similarity_matrix


RECALL_NS = (1, 5, 10, 20, 50, 100)


def main(args):
    resolve_full_sequence_args(args)
    data_path = args.output_path or os.path.join("outputs", default_run_name(args, prefix="sad"))
    redirect_output(data_path)
    np.random.seed(42)

    print("=" * 70)
    print("Sum-of-Absolute-Differences (SAD) inference for VPR")
    print("=" * 70)
    print(f"Images: {args.num_images} x {args.image_size}x{args.image_size}, {args.pixels_per_image} active px/image")
    print(f"Time clock info: {time.get_clock_info('perf_counter')}")

    dataset = build_dataset(args, data_path)
    query_loader, gt = dataset.query_loader, dataset.gt_matrix
    ref_images = dataset.images
    print(f"  Building {query_loader.n_frames} query frames (for the SAD matrix and plots)")
    qry_images = query_loader.make_all_frames_flat()

    with open(os.path.join(data_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    n_refs, n_query = len(ref_images), len(qry_images)
    D = args.input_size
    n_words = (D + 63) // 64
    R_dense = np.stack([np.asarray(im, dtype=np.float32) for im in ref_images])
    Q_dense = np.stack([np.asarray(im, dtype=np.float32) for im in qry_images])

    # bit-packed reference database, built once
    R_packed = np.zeros((n_refs, n_words), dtype=np.uint64)
    pack_scratch = np.zeros(n_words, dtype=np.uint64)
    for i in range(n_refs):
        ap = np.where(R_dense[i] > 0)[0].astype(np.int32)
        pack_active_pixels(ap, len(ap), pack_scratch)
        R_packed[i] = pack_scratch
    print(f"Bit-packed reference matrix: {R_packed.shape} ({R_packed.nbytes/1024:.1f} KB uint64, "
          f"{D}->{n_words*64} bits/row)")
    sad_scratch = np.empty(n_refs, dtype=np.int64)
    active_buf = np.empty(query_loader.max_active_pixels, dtype=np.int32)

    def _call_once(i):
        t0 = time.perf_counter()
        ap = query_loader.make_frame_active_pixels(i, out_buf=active_buf)
        k = len(ap)
        t1 = time.perf_counter()
        pack_active_pixels(ap, k, pack_scratch)
        t1b = time.perf_counter()
        pred = int(sad_dense_packed(pack_scratch, R_packed, sad_scratch))
        t2 = time.perf_counter()
        return pred, (t1 - t0) * 1e6, (t1b - t1) * 1e6, (t2 - t1b) * 1e6

    for i in range(min(100, n_query)):  # JIT + warmup
        _call_once(i)

    n_repeats = max(1, args.n_timing_repeats)
    pred = np.empty(n_query, dtype=np.int64)
    t_topk, t_pack, t_kernel = (np.empty(n_query) for _ in range(3))
    kernel_mean, kernel_std, kernel_iqr = (np.empty(n_query) for _ in range(3))
    for i in range(n_query):
        pred[i], (t_topk[i], t_pack[i], t_kernel[i]), stats = repeat_timed_call(n_repeats, lambda i=i: _call_once(i))
        kernel_mean[i], kernel_std[i], kernel_iqr[i] = stats['mean'][2], stats['std'][2], stats['iqr'][2]
    t_prep = t_topk + t_pack

    print(f"\nAverage latency over {n_query} queries (median of {n_repeats} repeat(s) per query):")
    print(f"  preprocessing (raw event window -> active pixels, then bit-pack = "
          f"{np.mean(t_topk):.3f}us + {np.mean(t_pack):.3f}us) = {np.mean(t_prep):.3f}us")
    print(f"  network       (sad_dense_packed kernel)                = {np.mean(t_kernel):.3f}us")
    print(f"  total                                                  = {np.mean(t_prep + t_kernel):.3f}us")
    if n_repeats > 1:
        print(f"  repeat noise (kernel): mean={np.mean(kernel_mean):.3f}us  median={np.mean(t_kernel):.3f}us  "
              f"std={np.mean(kernel_std):.3f}us  iqr={np.mean(kernel_iqr):.3f}us")

    # full (Q, R) SAD matrix for R@N, untimed: SAD = k_q + k_r - 2 |q AND r|
    k_q = (Q_dense > 0).sum(axis=1).astype(np.int64)
    k_r = (R_dense > 0).sum(axis=1).astype(np.int64)
    intersection = (Q_dense > 0).astype(np.float32) @ (R_dense > 0).astype(np.float32).T
    sad_matrix = np.rint(k_q[:, None] + k_r[None, :] - 2.0 * intersection).astype(np.int64)

    recall_Ns = [n for n in RECALL_NS if n <= n_refs]
    top1, hits = _recall_from_sad_distances(sad_matrix, gt, recall_Ns)
    correct = int(top1.sum())
    accuracy = correct / n_query
    eval_label = f"ref/query GT, R={gt.shape[0]} refs x Q={gt.shape[1]} queries, tolerance={args.gt_tolerance}"
    print(f"\nAccuracy (top-1 vs {eval_label}): {correct}/{n_query} ({100*accuracy:.1f}%)")
    for N in recall_Ns:
        print(f"  Recall@{N:<3d}: {hits[N]:>5d} ({100*hits[N]/n_query:5.1f}%)")

    with open(os.path.join(data_path, "results.json"), "w") as f:
        json.dump(dict(r_at_1=accuracy, recall={str(k): v / n_query for k, v in hits.items()},
                       prep_us=float(np.mean(t_prep)), net_us=float(np.mean(t_kernel)), n_query=n_query), f, indent=2)
    np.savez(os.path.join(data_path, "sad_latency.npz"),
             per_image_us_topk=t_topk, per_image_us_pack=t_pack, per_image_us_preprocess=t_prep,
             per_image_us_kernel=t_kernel, per_image_us_kernel_mean=kernel_mean,
             per_image_us_kernel_std=kernel_std, per_image_us_kernel_iqr=kernel_iqr,
             predictions=pred, gt=gt, sad_matrix=sad_matrix)

    debug_gt_and_predictions(gt, None, np.arange(n_refs, dtype=np.int64), predicted=pred, top1_pred=pred,
                             label='real', tolerance=args.gt_tolerance, max_show=min(20, n_query))

    plot_query_matches(qry_images, ref_images, pred, gt, args.image_size, data_path, n_show=min(20, n_query))
    sim_matrix = 1.0 - sad_matrix.astype(np.float64) / np.max(sad_matrix)
    plot_similarity_matrix(sim_matrix.T, gt, pred, data_path, fname='vpr_similarity_matrix.png',
                           title=f"SAD similarity matrix ({eval_label})",
                           value_label='Similarity (1 - SAD / max SAD)')
    print(f"\nResults saved to {data_path}/")


# ──────────────────────────────────────────────────────────────────────
# SAD kernels
# ──────────────────────────────────────────────────────────────────────

@numba.njit(inline='always')
def _popcount64(x):
    # branch-free popcount; every mask has bit 63 clear, so signed shifts are fine
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    return (x * 0x0101010101010101) >> 56


@numba.njit(cache=True, fastmath=True, boundscheck=False)
def pack_active_pixels(active_pixels, k_active, out_words):
    """Bit-pack an active-pixel index list into out_words (uint64)."""
    for w in range(out_words.shape[0]):
        out_words[w] = 0
    for i in range(k_active):
        p = active_pixels[i]
        out_words[p >> 6] |= np.uint64(1) << np.uint64(p & 63)


@numba.njit(cache=True, fastmath=True, boundscheck=False)
def sad_dense_packed(query_packed, R_packed, sad_out):
    """Hamming distance of the packed query to every packed reference; returns
    the argmin (lowest index on ties)."""
    n_refs = R_packed.shape[0]
    n_words = R_packed.shape[1]
    best = -1
    best_val = np.int64(1 << 62)
    for i in range(n_refs):
        s = np.int64(0)
        for w in range(n_words):
            s += np.int64(_popcount64(query_packed[w] ^ R_packed[i, w]))
        sad_out[i] = s
        if s < best_val:
            best_val = s
            best = i
    return best


def _recall_from_sad_distances(sad_matrix, gt, recall_ks):
    """Top-1 hits and recall@K hit counts from a (Q, R) distance matrix."""
    Q, R = sad_matrix.shape
    hit_counts = {k: 0 for k in recall_ks}
    top1_correct = np.zeros(Q, dtype=bool)
    for q in range(Q):
        order = np.argsort(sad_matrix[q], kind='stable')
        gt_col = gt[:, q]
        top1_correct[q] = bool(gt_col[order[0]])
        for k in recall_ks:
            if gt_col[order[:k]].any():
                hit_counts[k] += 1
    return top1_correct, hit_counts


if __name__ == "__main__":
    main(parse_arguments())
