"""
Fused numba inference kernels: query -> per-image scores -> argmax in one call.

Each reference image owns one output neuron, and its score is that neuron's
synaptic current over the pruned weights W (pruned with the checkpoint's
global threshold, see eval_metrics.compute_global_prune_threshold).

  Dense SNN    float32 matrix-vector product over the full binary frame,
               output rows split across threads.
  Sparse SNN   sums the uint8 synapse-mask rows of the k active pixels,
               one thread.
  Gather-sum   sums the float32 weight rows of the k active pixels, one
               thread. Only timed, for the speedup decomposition
               (plots/make_latency_decomposition_table.py).

Call signatures:
    sparse, gather-sum   fn(active_pixels, k_active, image_idx, scratch, per_img_spikes)
    dense                fn(query_dense, image_idx, scratch, per_img_spikes)
Frames are binary, so the sparse-input kernels take only the active pixel
indices. per_img_spikes[image_idx] receives the scores; the return value is
the predicted reference image (lowest index on ties, -1 if every score is 0).
"""

import numba
import numpy as np

_NJIT = dict(cache=False, fastmath=True, boundscheck=False, error_model='numpy')


def _contiguous_block_width(flat_neurons, flat_img_pos, n_assigned):
    """w if image j owns exactly neurons [j*w, (j+1)*w) for every assigned
    image, else 0. Holds with one neuron per image when every image is
    assigned, and lets the readout use a strided sum instead of a gather."""
    n_links = int(flat_neurons.shape[0])
    if n_assigned <= 0 or n_links <= 0 or n_links % n_assigned != 0:
        return 0
    w = n_links // n_assigned
    if not np.array_equal(flat_neurons, np.arange(n_links, dtype=flat_neurons.dtype)):
        return 0
    if not np.array_equal(flat_img_pos, (np.arange(n_links) // w).astype(flat_img_pos.dtype)):
        return 0
    return w


def _build_readout_argmax_fn(flat_neurons, flat_img_pos, assigned_imgs, n_assigned):
    """Shared tail of the dense and gather-sum kernels: sum each image's
    neuron currents into out_per_img and return the argmax image."""
    n_links_const = int(flat_neurons.shape[0])
    n_assigned_const = int(n_assigned)
    njit_inline = numba.njit(inline='always', **_NJIT)

    blocked_w = _contiguous_block_width(flat_neurons, flat_img_pos, n_assigned)
    if blocked_w > 0:
        w_const = int(blocked_w)

        @njit_inline
        def _readout_argmax(scratch_current, out_per_img):
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

        return _readout_argmax

    @njit_inline
    def _readout_argmax(scratch_current, out_per_img):
        out_per_img[:n_assigned_const] = 0
        for m in range(n_links_const):
            out_per_img[flat_img_pos[m]] += scratch_current[flat_neurons[m]]
        best_pos = -1
        best_val = 0
        for j in range(n_assigned_const):
            v = out_per_img[j]
            if v > best_val:
                best_val = v
                best_pos = j
        if best_pos < 0:
            return np.int64(-1)
        return np.int64(assigned_imgs[best_pos])

    return _readout_argmax


def build_binary_synapse_mask(W_dense_pruned, flat_neurons, flat_img_pos, n_assigned, k_active_max=None):
    """Re-encode the pruned weights as a uint8 synapse mask.

    Additive STDP with hard clipping saturates the surviving synapses at
    W_max, so they barely differ in magnitude, and the float32 current is a
    constant times the number of active pixels each neuron kept. The mask
    stores that: T[i, p] is the number of image p's neurons that keep pixel i.
    It reads 1 byte per weight instead of 4 and needs no multiply.

    Returns (T, scale, info): T is C-contiguous uint8 (input_size, n_cols);
    scale * T[active].sum(0) approximates the float current, with relative
    error bounded by info['cv'], the coefficient of variation of the
    surviving weights.
    """
    W = np.asarray(W_dense_pruned, dtype=np.float32)
    mask = (W != 0.0)
    surviving = W[mask]
    if surviving.size:
        w_mean = float(surviving.mean())
        cv = float(surviving.std() / abs(w_mean)) if w_mean else float('inf')
    else:
        w_mean, cv = 0.0, 0.0

    n_cols = max(n_assigned, 1)
    T = np.zeros((W.shape[0], n_cols), dtype=np.uint8)
    counts = np.zeros(n_cols, dtype=np.int64)
    for m in range(flat_neurons.shape[0]):
        T[:, flat_img_pos[m]] += mask[:, flat_neurons[m]]
        counts[flat_img_pos[m]] += 1
    max_per_col = int(counts.max()) if counts.size else 1

    # narrowest counter that cannot overflow: a column's count is bounded by
    # both its kept synapses and the number of active query pixels
    col_sum_max = int(T.sum(axis=0).max()) if T.size else 0
    k_max = int(k_active_max) if k_active_max else W.shape[0]
    max_count = min(k_max * max_per_col, col_sum_max)
    cnt_dtype = np.uint8 if max_count <= 255 else (np.uint16 if max_count <= 65535 else np.uint32)

    info = {
        'cv': cv, 'w_mean': w_mean, 'n_cols': n_cols, 'max_count': max_count,
        'cnt_dtype': np.dtype(cnt_dtype).name, 'nnz': int(mask.sum()),
        'bytes': int(T.nbytes), 'bytes_f32': int(W.nbytes),
    }
    return np.ascontiguousarray(T), np.float32(w_mean), info


def build_fused_infer_sparse_closure(T, scale, assigned_imgs, n_assigned):
    """Sparse SNN kernel: cnt[p] += T[active[i], p] over the k active rows
    (unit-stride uint8 adds), then scores = cnt * scale and the argmax in one
    pass. cnt_scratch is passed in because numba makes captured arrays
    read-only."""
    n_cols_const = int(T.shape[1])
    n_assigned_const = int(n_assigned)
    scale_const = np.float32(scale)

    @numba.njit(**_NJIT)
    def _impl(active_pixels, k_active, image_idx, cnt_scratch, per_img_spikes):
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

    return _impl


def build_fused_infer_gathersum_closure(W_dense_pruned, flat_neurons, flat_img_pos, assigned_imgs, n_assigned):
    """Gather-sum kernel: the Sparse SNN's k-row walk over the float32 pruned
    weights, accumulated into scratch_current, then the shared readout.
    Multi-threading it does not pay: the prange fork/join alone costs about
    as much as the whole kernel."""
    readout_argmax = _build_readout_argmax_fn(flat_neurons, flat_img_pos, assigned_imgs, n_assigned)
    W = np.ascontiguousarray(W_dense_pruned, dtype=np.float32)
    n_out_const = int(W.shape[1])

    @numba.njit(**_NJIT)
    def _impl(active_pixels, k_active, image_idx, scratch_current, per_img_spikes):
        scratch_current[:] = 0.0
        for i in range(k_active):
            row = W[active_pixels[i]]
            for j in range(n_out_const):
                scratch_current[j] += row[j]
        return readout_argmax(scratch_current, per_img_spikes[image_idx])

    return _impl


def build_fused_infer_dense_closure(W_dense, flat_neurons, flat_img_pos, assigned_imgs, n_assigned):
    """Dense SNN kernel: scratch_current = query_dense @ W over every pixel,
    whatever the sparsity. W is stored transposed, so each output is a
    unit-stride dot product, and the outputs are split across threads."""
    n_in_const = int(W_dense.shape[0])
    n_out_const = int(W_dense.shape[1])
    readout_argmax = _build_readout_argmax_fn(flat_neurons, flat_img_pos, assigned_imgs, n_assigned)
    W_T = np.ascontiguousarray(W_dense.T, dtype=np.float32)

    @numba.njit(parallel=True, **_NJIT)
    def _impl(query_dense, image_idx, scratch_current, per_img_spikes):
        for j in numba.prange(n_out_const):
            row = W_T[j]
            s = np.float32(0.0)
            for p in range(n_in_const):
                s += query_dense[p] * row[p]
            scratch_current[j] = s
        return readout_argmax(scratch_current, per_img_spikes[image_idx])

    return _impl


def prune_dense_weights(W_dense, threshold):
    """Zero every weight below `threshold` (scalar or (1, output_size) array)."""
    W = np.array(W_dense, dtype=np.float32)
    if np.any(threshold > 0):
        W[W < threshold] = 0.0
    return W


def prepare_inference_state(args, W, image_to_neurons, n_query, prune_thresholds):
    """Prune W and build every kernel with the arrays and scratch buffers they
    share. per_img_spikes_arr is (n_query, n_assigned): row q gets query q's
    scores."""
    print(f"\nPreparing inference state")

    prune_threshold_arr = np.asarray(prune_thresholds, dtype=np.float32).reshape(1, -1)
    W_pruned = prune_dense_weights(W, prune_threshold_arr)
    nnz = int(np.count_nonzero(W_pruned))
    total_els = args.input_size * args.output_size
    print(f"  Pruned weights: {nnz} non-zeros / {total_els} total ({100*nnz/total_els:.1f}% kept, "
          f"global threshold={prune_threshold_arr.max():.4g}), mean {nnz / args.input_size:.2f} per input pixel")

    assigned_imgs = [i for i, ns in enumerate(image_to_neurons) if ns]
    assigned_imgs_arr = np.array(assigned_imgs, dtype=np.int64)
    n_assigned = len(assigned_imgs)
    if n_assigned > 0:
        flat_neurons = np.concatenate([np.asarray(image_to_neurons[i], dtype=np.int32) for i in assigned_imgs])
        flat_img_pos = np.concatenate([np.full(len(image_to_neurons[i]), k, dtype=np.int32)
                                       for k, i in enumerate(assigned_imgs)])
    else:
        flat_neurons = np.array([], dtype=np.int32)
        flat_img_pos = np.array([], dtype=np.int32)

    per_img_spikes_arr = np.zeros((n_query, max(n_assigned, 1)), dtype=np.float32)
    scratch_current = np.zeros(args.output_size, dtype=np.float32)

    fused_infer_dense_closure = build_fused_infer_dense_closure(
        W_pruned, flat_neurons, flat_img_pos, assigned_imgs_arr, n_assigned)
    fused_infer_gathersum_closure = build_fused_infer_gathersum_closure(
        W_pruned, flat_neurons, flat_img_pos, assigned_imgs_arr, n_assigned)

    T_u8, t_scale, mask_info = build_binary_synapse_mask(
        W_pruned, flat_neurons, flat_img_pos, n_assigned, k_active_max=args.pixels_per_image)
    sparse_cnt_scratch = np.zeros(mask_info['n_cols'], dtype=np.dtype(mask_info['cnt_dtype']))
    fused_infer_sparse_closure = build_fused_infer_sparse_closure(T_u8, t_scale, assigned_imgs_arr, n_assigned)
    print(f"  Binary synapse mask: {mask_info['n_cols']} columns x {args.input_size} pixels uint8 "
          f"({mask_info['bytes'] / 1e6:.3f} MB vs {mask_info['bytes_f32'] / 1e6:.3f} MB float32, "
          f"{mask_info['bytes_f32'] / max(mask_info['bytes'], 1):.1f}x smaller), "
          f"accumulator={mask_info['cnt_dtype']} (max count {mask_info['max_count']})")
    print(f"    surviving weights: mean={mask_info['w_mean']:.6g} cv={100 * mask_info['cv']:.2f}% "
          f"(relative error of replacing them by their mean)")
    if mask_info['cv'] > 0.05:
        print(f"    WARNING: cv > 5%, so the Sparse SNN scores only approximate the float32 matvec here.")

    return {
        'fused_infer_dense_closure': fused_infer_dense_closure,
        'fused_infer_gathersum_closure': fused_infer_gathersum_closure,
        'fused_infer_sparse_closure': fused_infer_sparse_closure,
        'sparse_cnt_scratch': sparse_cnt_scratch,
        'synapse_mask': T_u8, 'synapse_mask_scale': t_scale, 'synapse_mask_info': mask_info,
        'assigned_imgs_arr': assigned_imgs_arr, 'n_assigned': n_assigned,
        'flat_neurons_i32': flat_neurons, 'flat_img_pos_i32': flat_img_pos,
        'per_img_spikes_arr': per_img_spikes_arr, 'scratch_current': scratch_current,
        'W_dense_pruned': W_pruned,
    }
