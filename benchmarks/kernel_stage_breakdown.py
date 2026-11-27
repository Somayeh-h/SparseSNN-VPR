"""
Per-stage latency of the Dense and Sparse SNN kernels.

Timing each stage as its own numba call would charge every stage the call
overhead that the real kernel pays once. Instead, this script builds
truncated copies of each kernel, the same body cut off after stage 1, 2, ...,
and each stage's cost is the difference between two neighbouring copies:

    stage s = t(copy up to s) - t(copy up to s-1),    t(copy up to 0) = empty body

The longest copy is identical to the real kernel, which is timed alongside
as a check (same predictions, same latency up to noise). Every copy takes
the real kernel's arguments, used or not, so that each one pays the same
call and unboxing cost and the stages add up to the real kernel.

Sparse SNN kernel (build_fused_infer_sparse_closure):
    call      empty body: call overhead and argument unboxing
    zero      cnt_scratch[:] = 0
    count     cnt[p] += T[active[i], p] over the k active rows (uint8)
    score     out_per_img[p] = cnt[p] * scale
    argmax    running best over the scores (fused into the score loop)

Dense SNN kernel (build_fused_infer_dense_closure, with the contiguous-block
readout of _build_readout_argmax_fn):
    call      empty body
    threads   prange fork/join over the n_out rows with a trivial body
    matvec    the N_out x N_in float32 dot products inside that prange
    readout   per-image sum of scratch_current into out_per_img
    argmax    running best over the per-image sums

Each copy is compiled on its own, so LLVM may optimise the score loop
differently without its argmax branch; the figure therefore merges readout
and argmax into one stage.

Preprocessing is timed per query as in infer.py; the kernels then receive
queries prepared beforehand.

Writes <run-dir>/infer/kernel_stage_breakdown.npz, read by
plots/make_latency_decomposition_table.py.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import numba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_arguments
from src.checkpoint import load_checkpoint
from src.common import physical_core_count, repeat_timed_call
from src.data_pipeline import make_query_loader, resolve_full_sequence_args
from src.inference_kernels import _contiguous_block_width, prepare_inference_state


SPARSE_STAGES = ["call", "zero", "count", "score", "argmax"]
DENSE_STAGES = ["call", "threads", "matvec", "readout", "argmax"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="outputs/main_results")
    ap.add_argument("--dataset", default="qcr_event")
    ap.add_argument("--n-repeats", type=int, default=None, help="default: the run's n_timing_repeats")
    ap.add_argument("--n-queries", type=int, default=0, help="0 = every query frame")
    cli = ap.parse_args()

    matches = sorted(glob.glob(os.path.join(cli.results_dir, f"stdp_*{cli.dataset}*")))
    if len(matches) != 1:
        raise RuntimeError(f"expected one stdp_*{cli.dataset}* run in {cli.results_dir}, got {matches}")
    run_dir = matches[0]
    args = load_args(run_dir)
    n_repeats = cli.n_repeats or args.n_timing_repeats

    query_loader = make_query_loader(args)
    n_query = query_loader.n_frames if cli.n_queries <= 0 else min(cli.n_queries, query_loader.n_frames)
    ckpt = load_checkpoint(run_dir)
    state = prepare_inference_state(args, ckpt['W'], ckpt['image_to_neurons'], query_loader.n_frames,
                                    prune_thresholds=ckpt['prune_thresholds'])

    flat_neurons, flat_img_pos = state['flat_neurons_i32'], state['flat_img_pos_i32']
    assigned_imgs, n_assigned = state['assigned_imgs_arr'], state['n_assigned']
    block_w = _contiguous_block_width(flat_neurons, flat_img_pos, n_assigned)
    if block_w <= 0:
        raise RuntimeError("the dense copies assume one block of neurons per image, which this checkpoint lacks")
    sparse_fns = build_sparse_prefixes(state['synapse_mask'], state['synapse_mask_scale'], assigned_imgs) + [
        state['fused_infer_sparse_closure']]
    dense_fns = build_dense_prefixes(state['W_dense_pruned'], assigned_imgs, block_w) + [
        state['fused_infer_dense_closure']]

    numba.set_num_threads(physical_core_count())
    active_buf = np.empty(query_loader.max_active_pixels, dtype=np.int32)
    frame_buf = np.empty(args.input_size, dtype=np.float32)
    scratch_d = np.zeros_like(state['scratch_current'])
    spikes_s = np.zeros_like(state['per_img_spikes_arr'])
    spikes_d = np.zeros_like(state['per_img_spikes_arr'])
    cnt = state['sparse_cnt_scratch']

    # preprocessing into reused buffers, as in infer.py; the copies kept for
    # the kernel timing below are made outside the timed calls
    prep_sparse = np.empty(n_query)
    prep_dense = np.empty(n_query)
    sparse_args, dense_args = [], []
    for i in range(n_query):
        def _prep_sparse(i=i):
            t0 = time.perf_counter()
            a = query_loader.make_frame_active_pixels(i, out_buf=active_buf)
            return a, (time.perf_counter() - t0) * 1e6

        def _prep_dense(i=i):
            t0 = time.perf_counter()
            q = query_loader.make_frame_flat(i, out=frame_buf)
            return q, (time.perf_counter() - t0) * 1e6
        a, (prep_sparse[i],), _ = repeat_timed_call(n_repeats, _prep_sparse)
        q, (prep_dense[i],), _ = repeat_timed_call(n_repeats, _prep_dense)
        sparse_args.append((a.copy(), a.size, i, cnt, spikes_s))
        dense_args.append((q.copy(), i, scratch_d, spikes_d))

    for fn in sparse_fns:
        fn(*sparse_args[0])  # JIT warmup
    for fn in dense_fns:
        fn(*dense_args[0])

    print(f"\ntiming {n_query} queries x {len(sparse_fns)} sparse + {len(dense_fns)} dense variants, "
          f"median of {n_repeats} repeats each, numba threads={numba.get_num_threads()}")
    t_sparse, p_sparse = time_variants(sparse_fns, sparse_args, n_repeats)
    t_dense, p_dense = time_variants(dense_fns, dense_args, n_repeats)

    for label, stages, t, p in (("Sparse SNN", SPARSE_STAGES, t_sparse, p_sparse),
                                ("Dense SNN", DENSE_STAGES, t_dense, p_dense)):
        mean = t.mean(axis=0)
        agree = int(np.sum(p[:, -2] == p[:, -1]))
        print(f"\n{label}: real kernel {mean[-1]:.3f}us, full copy {mean[-2]:.3f}us, "
              f"predictions agree {agree}/{n_query}")
        prev = 0.0
        for s, m in zip(stages, mean[:-1]):
            print(f"  {s:8s} copy {m:7.3f}us   stage {m - prev:+7.3f}us")
            prev = m

    out = os.path.join(run_dir, "infer", "kernel_stage_breakdown.npz")
    np.savez(out, sparse_stages=np.array(SPARSE_STAGES), dense_stages=np.array(DENSE_STAGES),
             sparse_prefix_us=t_sparse[:, :-1], sparse_production_us=t_sparse[:, -1],
             dense_prefix_us=t_dense[:, :-1], dense_production_us=t_dense[:, -1],
             prep_sparse_us=prep_sparse, prep_dense_us=prep_dense, n_repeats=n_repeats,
             n_threads=numba.get_num_threads())
    print(f"\nmean preprocessing: sparse {prep_sparse.mean():.3f}us  dense {prep_dense.mean():.3f}us")
    print(f"wrote {out}")


def load_args(run_dir):
    """The run's arguments: parser defaults overridden by its args.json."""
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        args = parse_arguments()
    finally:
        sys.argv = argv
    with open(os.path.join(run_dir, "args.json")) as f:
        for key, val in json.load(f).items():
            setattr(args, key, val)
    resolve_full_sequence_args(args)
    return args


def build_sparse_prefixes(T, scale, assigned_imgs):
    """Truncated copies of build_fused_infer_sparse_closure's body."""
    n_cols_const = int(T.shape[1])
    n_assigned_const = int(assigned_imgs.shape[0])
    scale_const = np.float32(scale)
    njit = numba.njit(cache=False, fastmath=True, boundscheck=False, error_model='numpy')

    @njit
    def call(active_pixels, k_active, image_idx, cnt_scratch, per_img_spikes):
        return np.int64(0)

    @njit
    def zero(active_pixels, k_active, image_idx, cnt_scratch, per_img_spikes):
        for p in range(n_cols_const):
            cnt_scratch[p] = 0
        return np.int64(0)

    @njit
    def count(active_pixels, k_active, image_idx, cnt_scratch, per_img_spikes):
        for p in range(n_cols_const):
            cnt_scratch[p] = 0
        for i in range(k_active):
            row = T[active_pixels[i]]
            for p in range(n_cols_const):
                cnt_scratch[p] += row[p]
        return np.int64(0)

    @njit
    def score(active_pixels, k_active, image_idx, cnt_scratch, per_img_spikes):
        for p in range(n_cols_const):
            cnt_scratch[p] = 0
        for i in range(k_active):
            row = T[active_pixels[i]]
            for p in range(n_cols_const):
                cnt_scratch[p] += row[p]
        out_per_img = per_img_spikes[image_idx]
        for p in range(n_assigned_const):
            out_per_img[p] = cnt_scratch[p] * scale_const
        return np.int64(0)

    @njit
    def argmax(active_pixels, k_active, image_idx, cnt_scratch, per_img_spikes):
        for p in range(n_cols_const):
            cnt_scratch[p] = 0
        for i in range(k_active):
            row = T[active_pixels[i]]
            for p in range(n_cols_const):
                cnt_scratch[p] += row[p]
        out_per_img = per_img_spikes[image_idx]
        best_pos = -1
        best_cnt = 0
        for p in range(n_assigned_const):
            c = cnt_scratch[p]
            out_per_img[p] = c * scale_const
            if c > best_cnt:
                best_cnt = c
                best_pos = p
        if best_pos < 0:
            return np.int64(-1)
        return np.int64(assigned_imgs[best_pos])

    return [call, zero, count, score, argmax]


def build_dense_prefixes(W_dense, assigned_imgs, block_w):
    """Truncated copies of build_fused_infer_dense_closure's body, with the
    contiguous-block readout inlined."""
    W_T = np.ascontiguousarray(W_dense.T, dtype=np.float32)
    n_in_const = int(W_dense.shape[0])
    n_out_const = int(W_dense.shape[1])
    n_assigned_const = int(assigned_imgs.shape[0])
    w_const = int(block_w)
    njit_par = numba.njit(parallel=True, cache=False, fastmath=True, boundscheck=False, error_model='numpy')

    @njit_par
    def call(query_dense, image_idx, scratch_current, per_img_spikes):
        return np.int64(0)

    @njit_par
    def threads(query_dense, image_idx, scratch_current, per_img_spikes):
        for j in numba.prange(n_out_const):
            scratch_current[j] = np.float32(0.0)
        return np.int64(0)

    @njit_par
    def matvec(query_dense, image_idx, scratch_current, per_img_spikes):
        for j in numba.prange(n_out_const):
            row = W_T[j]
            s = np.float32(0.0)
            for p in range(n_in_const):
                s += query_dense[p] * row[p]
            scratch_current[j] = s
        return np.int64(0)

    @njit_par
    def readout(query_dense, image_idx, scratch_current, per_img_spikes):
        for j in numba.prange(n_out_const):
            row = W_T[j]
            s = np.float32(0.0)
            for p in range(n_in_const):
                s += query_dense[p] * row[p]
            scratch_current[j] = s
        out_per_img = per_img_spikes[image_idx]
        for j in range(n_assigned_const):
            base = j * w_const
            acc = np.float32(0.0)
            for w in range(w_const):
                acc += scratch_current[base + w]
            out_per_img[j] = acc
        return np.int64(0)

    @njit_par
    def argmax(query_dense, image_idx, scratch_current, per_img_spikes):
        for j in numba.prange(n_out_const):
            row = W_T[j]
            s = np.float32(0.0)
            for p in range(n_in_const):
                s += query_dense[p] * row[p]
            scratch_current[j] = s
        out_per_img = per_img_spikes[image_idx]
        best_pos = -1
        best_val = 0
        for j in range(n_assigned_const):
            base = j * w_const
            acc = np.float32(0.0)
            for w in range(w_const):
                acc += scratch_current[base + w]
            out_per_img[j] = acc
            if acc > best_val:
                best_val = acc
                best_pos = j
        if best_pos < 0:
            return np.int64(-1)
        return np.int64(assigned_imgs[best_pos])

    return [call, threads, matvec, readout, argmax]


def time_variants(fns, arg_tuples, n_repeats):
    """Median-of-repeats latency per (query, variant). The variant order is
    rotated per query so that no variant always runs first."""
    n_q, n_v = len(arg_tuples), len(fns)
    out = np.empty((n_q, n_v), dtype=np.float64)
    preds = np.empty((n_q, n_v), dtype=np.int64)
    for i, a in enumerate(arg_tuples):
        for r in range(n_v):
            v = (i + r) % n_v
            fn = fns[v]

            def _call_once(fn=fn, a=a):
                t0 = time.perf_counter()
                pred = int(fn(*a))
                return pred, (time.perf_counter() - t0) * 1e6
            pred, (us,), _ = repeat_timed_call(n_repeats, _call_once)
            out[i, v] = us
            preds[i, v] = pred
    return out, preds


if __name__ == "__main__":
    main()
