"""Figures written by train.py and the inference scripts."""
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_active_vs_inactive_weights(W_before, W_after, images, neuron_to_image, W_max, output_dir, prune_threshold):
    """Histograms of each neuron's weights on the pixels of its own reference
    image vs all other pixels, before and after training, with the prune
    threshold."""
    W_b = np.array(W_before)
    W_a = np.array(W_after)

    n_input, n_output = W_a.shape
    n_images = len(images)
    active_mask = np.zeros_like(W_a, dtype=bool)
    img_cache = {}
    for j in range(n_output):
        img_idx = neuron_to_image.get(j, None)
        if isinstance(img_idx, int) and 0 <= img_idx < n_images:
            mask = img_cache.get(img_idx)
            if mask is None:
                mask = np.asarray(images[img_idx]) > 0
                img_cache[img_idx] = mask
            active_mask[:, j] = mask

    active_before   = W_b[active_mask]
    active_after    = W_a[active_mask]
    inactive_before = W_b[~active_mask]
    inactive_after  = W_a[~active_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    bins = np.linspace(0, max(W_max, W_a.max()) * 1.02, 60)

    ax1.hist(active_before, bins=bins, alpha=0.4, color='gray', label=f'before (n={active_before.size})')
    ax1.hist(active_after,  bins=bins, alpha=0.5, color='red',  label=f'after  (n={active_after.size})')
    ax1.axvline(W_max, color='k', linestyle='--', alpha=0.5, label='W_max')
    ax2.hist(inactive_before, bins=bins, alpha=0.4, color='gray', label=f'before (n={inactive_before.size})')
    ax2.hist(inactive_after,  bins=bins, alpha=0.5, color='blue', label=f'after  (n={inactive_after.size})')
    ax2.axvline(W_max, color='k', linestyle='--', alpha=0.5, label='W_max')

    for ax in (ax1, ax2):
        ax.axvline(prune_threshold, color='magenta', linestyle=':', linewidth=2,
                   label=f'global prune thr ({prune_threshold:.3g})')

    ax1.set_title('Active-pixel weights (pixel on in assigned image)')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Count')
    ax1.legend()

    ax2.set_title('Inactive-pixel weights (pixel off in assigned image)')
    ax2.set_xlabel('Weight')
    ax2.legend()

    fig.suptitle('Active vs Inactive Weight Distributions (before vs after STDP)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stdp_active_vs_inactive_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_matrix(matrix, gt, predicted, output_dir, fname, title, value_label, max_side=3000):
    """Plot an (R, Q) reference-vs-query similarity matrix (higher = better).

    Green: for each correct query, the most similar reference inside its GT
    band. Red: the prediction of each query that misses its band.
    predicted is (Q,) with -1 for no prediction. Rows and columns are strided
    down to at most max_side each.
    """
    matrix = np.asarray(matrix)
    gt = np.asarray(gt)
    predicted = np.asarray(predicted)
    R, Q = matrix.shape

    row_stride = max(1, R // max_side)
    col_stride = max(1, Q // max_side)
    row_idx = np.arange(0, R, row_stride)
    col_idx = np.arange(0, Q, col_stride)
    view = matrix[np.ix_(row_idx, col_idx)]
    if row_stride > 1 or col_stride > 1:
        title += f'\n(strided to {len(row_idx)}x{len(col_idx)}: row={row_stride}, col={col_stride})'

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(view, cmap='Blues_r', aspect='equal', interpolation='nearest', origin='upper')
    ax.set_xlabel('Query frame')
    ax.set_ylabel('Reference frame')
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label=value_label)

    if len(row_idx) > 20:
        tick_pos = np.linspace(0, len(row_idx) - 1, 6).astype(int)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(row_idx[tick_pos])
    if len(col_idx) > 20:
        tick_pos = np.linspace(0, len(col_idx) - 1, 6).astype(int)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(col_idx[tick_pos])

    row_pos = {int(r): k for k, r in enumerate(row_idx)}
    ok_x, ok_y, bad_x, bad_y = [], [], [], []
    for pos_q, q in enumerate(col_idx[col_idx < predicted.shape[0]]):
        gt_col = gt[:, q]
        r = int(predicted[q])
        if 0 <= r < R and not gt_col[r]:
            if r in row_pos:
                bad_x.append(pos_q); bad_y.append(row_pos[r])
            continue
        positives = np.where(gt_col > 0)[0]
        if positives.size > 0:
            best_r = int(positives[np.argmax(matrix[positives, q])])
            if best_r in row_pos:
                ok_x.append(pos_q); ok_y.append(row_pos[best_r])
    if ok_x:
        ax.scatter(ok_x, ok_y, s=18, c='lime', edgecolors='black',
                   linewidths=0.4, label=f'best match in GT tol. ({len(ok_x)})', zorder=3)
    if bad_x:
        ax.scatter(bad_x, bad_y, s=18, c='red', edgecolors='black',
                   linewidths=0.4, label=f'pred miss ({len(bad_x)})', zorder=4)
    if ok_x or bad_x:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_gt_matrix(gt, output_dir, fname, title):
    """Plot the (R, Q) ground-truth matrix: column q marks the references
    that count as a match for query q."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.asarray(gt), cmap='Greys', aspect='equal',
                   interpolation='nearest', origin='upper', vmin=0, vmax=1)
    ax.set_xlabel('Query frame')
    ax.set_ylabel('Reference frame')
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Match (1=GT)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_query_matches(query_images, ref_images, predicted, gt, image_size, output_dir,
                       W=None, image_to_neurons=None, n_show=20, subdir='query_matches',
                       query_indices=None):
    """One figure per query: the query, its true reference (centre of the GT
    band), its matched reference and, when W and image_to_neurons are given,
    the learned weights of the matched reference's neuron.

    query_images[q] and predicted[q] belong to query query_indices[q]
    (default: query q). ref_images is indexed by reference; entries may be
    None, so only the references needed can be passed. Figures go to
    output_dir/subdir.
    """
    out = os.path.join(output_dir, subdir)
    os.makedirs(out, exist_ok=True)
    gt = np.asarray(gt)
    show_rf = W is not None and image_to_neurons is not None
    if show_rf:
        W = np.asarray(W)
    if query_indices is None:
        n_q = min(int(n_show), len(query_images), gt.shape[1])
        query_indices = range(n_q)
    else:
        n_q = min(int(n_show), len(query_images), len(query_indices))
    R = gt.shape[0]
    n_panels = 4 if show_rf else 3

    def _neurons_for(ref):
        if not show_rf or ref is None or ref < 0:
            return []
        if ref < len(image_to_neurons):
            return list(image_to_neurons[ref] or [])
        return []

    for q, real_q in zip(range(n_q), query_indices):
        pred = int(predicted[q]) if predicted[q] >= 0 else -1
        gt_col = gt[:, real_q]
        positives = np.where(gt_col > 0)[0]
        true_ref = int(positives[positives.size // 2]) if positives.size > 0 else -1
        in_band = (pred >= 0 and pred < R and bool(gt_col[pred]))
        match_label = 'correct' if in_band else ('WRONG' if pred >= 0 else 'silent')

        fig, axes = plt.subplots(1, n_panels, figsize=(13 if show_rf else 10, 3.3))

        q_img = np.asarray(query_images[q]).reshape(image_size, image_size)
        axes[0].imshow(q_img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[0].set_title(f'Query q={real_q}', fontsize=10)

        def _get_ref(r):
            if r is None or r < 0:
                return None
            if r < len(ref_images):
                return ref_images[r]
            return None

        t_img_src = _get_ref(true_ref)
        if t_img_src is not None:
            t_img = np.asarray(t_img_src).reshape(image_size, image_size)
            axes[1].imshow(t_img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            axes[1].set_title(f'True ref r={true_ref}', fontsize=10)
        else:
            axes[1].text(0.5, 0.5, 'no GT', ha='center', va='center',
                         transform=axes[1].transAxes)
            axes[1].set_title('True ref (none)', fontsize=10)

        m_img_src = _get_ref(pred)
        if m_img_src is not None:
            m_img = np.asarray(m_img_src).reshape(image_size, image_size)
            axes[2].imshow(m_img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            axes[2].set_title(f'Matched ref r={pred} ({match_label})', fontsize=10)
        else:
            axes[2].text(0.5, 0.5, 'silent', ha='center', va='center',
                         transform=axes[2].transAxes)
            axes[2].set_title('Matched ref (none)', fontsize=10)

        if show_rf:
            neurons = _neurons_for(pred)
            if neurons:
                neuron_idx = int(neurons[0])
                rf = W[:, neuron_idx].reshape(image_size, image_size)
                im = axes[3].imshow(rf, cmap='hot', interpolation='nearest')
                axes[3].set_title(f'Learned W of n={neuron_idx} (r={pred})', fontsize=10)
                plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            else:
                axes[3].text(0.5, 0.5, 'not learned', ha='center', va='center',
                             transform=axes[3].transAxes)
                axes[3].set_title('Learned W (n/a)', fontsize=10)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f'Query {real_q}: true=r{true_ref}, pred=r{pred} ({match_label})',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(out, f'query_{real_q:03d}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
