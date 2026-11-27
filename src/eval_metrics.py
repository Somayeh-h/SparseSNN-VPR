"""
Pruning threshold, weight selectivity, and recall scoring.
"""

import numpy as np


def compute_global_prune_threshold(W, images, neuron_to_image):
    """One network-wide prune threshold, calibrated on the reference images only.

    Each assigned neuron's weights split into ACTIVE (pixel on in its image)
    and INACTIVE (pixel off). Pooled over all neurons, the threshold theta
    maximises Youden's J

        J(theta) = kept_active(theta)/n_active - kept_inactive(theta)/n_inactive

    One cut fits every neuron because the L1 column normalisation
    (stdp_kernel.py) gives every neuron the same weight sum. A neuron whose
    active weights never rose above its noise floor (common at very low
    pixels_per_image) loses all its synapses, so its image can never be
    matched (info['n_dead_images']): an abstention rather than a wrong match.

    Returns (thresholds, info): the threshold as an (output_size,) float32
    array (as stored in the checkpoint) and diagnostics for logging.
    """
    W_np = np.asarray(W, dtype=np.float32)
    n_input, output_size = W_np.shape
    neurons = np.array(sorted(neuron_to_image.keys()), dtype=np.int64)

    info = {'threshold': 0.0, 'n_assigned': int(neurons.size), 'n_active': 0, 'n_inactive': 0,
            'kept_active_frac': 1.0, 'kept_inactive_frac': 1.0, 'youden_j': 0.0,
            'n_dead_neurons': 0, 'n_dead_images': 0}
    if neurons.size == 0:
        return np.zeros(output_size, dtype=np.float32), info

    # active-pixel weights of every assigned neuron (unassigned neurons have
    # no image, so their weights cannot be labelled)
    img_cache = {}
    active_blocks = []
    active_counts = np.zeros(neurons.size, dtype=np.int64)
    for k, j in enumerate(neurons):
        img_idx = neuron_to_image[int(j)]
        active_mask = img_cache.get(img_idx)
        if active_mask is None:
            active_mask = np.asarray(images[img_idx]) > 0
            img_cache[img_idx] = active_mask
        w_active = W_np[active_mask, j]
        active_counts[k] = w_active.size
        if w_active.size:
            active_blocks.append(w_active)

    w_active_all = np.concatenate(active_blocks) if active_blocks else np.zeros(0, dtype=np.float32)
    n_active = int(w_active_all.size)
    n_all = int(n_input) * int(neurons.size)
    n_inactive = n_all - n_active
    if n_active == 0 or n_inactive == 0:
        return np.zeros(output_size, dtype=np.float32), info

    # kept_inactive = kept_all - kept_active, so the (large) inactive
    # population is never built. J only changes at active-weight values,
    # so those are the candidate cuts.
    act_sorted = np.sort(w_active_all)
    all_sorted = np.sort(W_np[:, neurons], axis=None)
    candidates = np.unique(act_sorted)

    kept_active = n_active - np.searchsorted(act_sorted, candidates, side='left')
    kept_all = n_all - np.searchsorted(all_sorted, candidates, side='left')
    kept_inactive = kept_all - kept_active
    j_stat = kept_active / n_active - kept_inactive / n_inactive

    best = int(np.argmax(j_stat))   # first max = lowest optimal cut
    threshold = float(candidates[best])

    # a neuron is dead if the cut removes all its synapses; an image is
    # unmatchable if all its neurons are dead
    kept_per_neuron = (W_np[:, neurons] >= threshold).sum(axis=0)
    dead = kept_per_neuron == 0
    dead_images = {neuron_to_image[int(j)] for j in neurons[dead]}
    live_images = {neuron_to_image[int(j)] for j in neurons[~dead]}

    info.update(threshold=threshold, n_active=n_active, n_inactive=n_inactive,
                kept_active_frac=float(kept_active[best]) / n_active,
                kept_inactive_frac=float(kept_inactive[best]) / n_inactive,
                youden_j=float(j_stat[best]), n_dead_neurons=int(dead.sum()),
                n_dead_images=len(dead_images - live_images))

    return np.full(output_size, threshold, dtype=np.float32), info


def compute_weight_selectivity(W, images):
    """Mean neuron selectivity, (best - mean of the others) / best over the
    cosine similarities between a neuron's weights and every image
    (0 = no preference, 1 = responds to one image only)."""
    W_np = np.asarray(W)
    n_neurons = W_np.shape[1]
    w_norms = np.linalg.norm(W_np, axis=0)

    best_sim = np.full(n_neurons, -np.inf, dtype=np.float64)
    sum_sim = np.zeros(n_neurons, dtype=np.float64)
    count = 0
    for img in images:
        ev = np.asarray(img, dtype=np.float32)
        ev_norm = np.linalg.norm(ev)
        if ev_norm < 1e-8:
            continue
        sims = (ev @ W_np) / (ev_norm * np.maximum(w_norms, 1e-8))
        np.maximum(best_sim, sims, out=best_sim)
        sum_sim += sims
        count += 1

    if count <= 1:
        return 0.0
    others_mean = (sum_sim - best_sim) / (count - 1)
    sel = (best_sim - others_mean) / np.maximum(best_sim, 1e-8)
    sel[w_norms < 1e-8] = 0.0
    return np.mean(sel)


def compute_accuracy_and_recall(per_query_scores, assigned_imgs, gt, recall_ks):
    """R@1 and R@N from each query's per-image scores.

    Ties go to the lowest reference index (stable sort), as in the kernels'
    argmax. A query is a hit at N if any of its top-N references is a match.

    Args:
        per_query_scores : (Q, A) scores of the A assigned reference images
        assigned_imgs    : (A,) reference image index of each score column
        gt               : (R, Q) uint8 ground-truth matrix
        recall_ks        : the values of N

    Returns:
        top1_correct : (Q,) bool, R@1 hit per query
        top1_pred    : (Q,) top-1 reference index (-1 if every score is 0)
        hit_counts   : {N: number of queries hit at N}
    """
    n_query = per_query_scores.shape[0]
    n_assigned = len(assigned_imgs)
    recall_ks = list(recall_ks)
    hit_counts = {k: 0 for k in recall_ks}
    top1_correct = np.zeros(n_query, dtype=bool)
    top1_pred = np.full(n_query, -1, dtype=np.int64)
    if n_assigned == 0:
        return top1_correct, top1_pred, hit_counts

    k_max = min(max(recall_ks) if recall_ks else 1, n_assigned)
    for q in range(n_query):
        scores = per_query_scores[q]
        if scores.sum() == 0:
            continue
        order = np.argsort(-scores, kind='stable')[:k_max]
        top_scores = scores[order]
        top_refs = np.where(top_scores > 0, assigned_imgs[order], -1)

        gt_col = gt[:, q]
        if top_refs[0] >= 0:
            top1_pred[q] = top_refs[0]
            if gt_col[top_refs[0]]:
                top1_correct[q] = True

        for k in recall_ks:
            top_k = top_refs[:k]
            valid = top_k >= 0
            if valid.any() and gt_col[top_k[valid]].any():
                hit_counts[k] += 1

    return top1_correct, top1_pred, hit_counts


def debug_gt_and_predictions(gt, per_query_spikes, assigned_imgs, predicted, top1_pred, label, tolerance=None, max_show=20):
    """Print ground-truth statistics and, for the first max_show queries, the
    prediction against the centre of the GT band. Helps tell a misaligned
    ground truth from a model that misses."""
    gt = np.asarray(gt)
    R, Q = gt.shape
    print(f"\n  ── GT debug ({label}) ──")
    tol_str = f', tolerance={tolerance}' if tolerance is not None else ''
    print(f"    gt shape: ({R}, {Q}){tol_str}, dtype={gt.dtype}, "
          f"sum={int(gt.sum())} ({100*(gt.sum())/(R*Q):.2f}% filled)")
    col_sums = gt.sum(axis=0)
    print(f"    per-query positives: min={int(col_sums.min())} "
          f"median={int(np.median(col_sums))} max={int(col_sums.max())} "
          f"empty_cols={int((col_sums == 0).sum())}/{Q}")
    row_sums = gt.sum(axis=1)
    print(f"    per-ref positives:   min={int(row_sums.min())} "
          f"median={int(np.median(row_sums))} max={int(row_sums.max())} "
          f"empty_rows={int((row_sums == 0).sum())}/{R}")

    if assigned_imgs is not None and len(assigned_imgs):
        a = np.asarray(assigned_imgs)
        print(f"    assigned refs: {len(a)} unique, range=[{int(a.min())}, "
              f"{int(a.max())}], head={a[:min(10, len(a))].tolist()}")

    if per_query_spikes is not None:
        ps = np.asarray(per_query_spikes)
        is_int = np.issubdtype(ps.dtype, np.integer)
        fmt = (lambda v: str(int(v))) if is_int else (lambda v: f"{v:.3g}")
        max_spikes = ps.max(axis=1) if ps.size else np.zeros(0)
        total_spikes = ps.sum(axis=1) if ps.size else np.zeros(0)
        print(f"    per-query max score:   min={fmt(max_spikes.min())} "
              f"median={fmt(np.median(max_spikes))} max={fmt(max_spikes.max())}")
        print(f"    per-query total score: min={fmt(total_spikes.min())} "
              f"median={fmt(np.median(total_spikes))} max={fmt(total_spikes.max())} "
              f"silent_queries={int((total_spikes == 0).sum())}/{Q}")

    pred = np.asarray(predicted) if predicted is not None else top1_pred
    pred = np.asarray(pred)
    n_show = min(max_show, Q)
    print(f"    per-query alignment (first {n_show}):")
    print(f"      {'q':>3} {'pred_ref':>8} {'gt_anchor':>9} {'in_band':>7} {'dist_to_pos':>11} {'col_sum':>7}")
    for q in range(n_show):
        pr = int(pred[q]) if pred[q] >= 0 else -1
        gt_col = gt[:, q]
        positives = np.where(gt_col > 0)[0]
        if positives.size == 0:
            anchor = -1
            dist = -1
        else:
            anchor = int(positives[positives.size // 2])  # midpoint of band
            if pr >= 0:
                dist = int(np.min(np.abs(positives - pr)))
            else:
                dist = -1
        in_band = (pr >= 0 and gt_col[pr] > 0)
        print(f"      {q:>3d} {pr:>8d} {anchor:>9d} {str(in_band):>7s} {dist:>11d} {int(col_sums[q]):>7d}")
