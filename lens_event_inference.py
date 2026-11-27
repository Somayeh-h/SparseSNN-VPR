"""
LENS baseline: the 2-layer BliTNet SNN of LENS (LENS submodule), trained and
queried on the same top-K binary frames as our SNN and scored against the
same ground truth.

Both layers are trained on the reference frames with LENS' own STDP code and
default hyperparameters. Each query frame then goes through
clamp(W2 . clamp(W1 . q - thr1) - thr2) on the GPU when there is one, as in
LENS. Timing follows infer.py, per query:
  preprocessing  raw event window -> binary frame
  network        copy to the GPU, forward pass, argmax, copy of the prediction back
each call repeated --n_timing_repeats times, latency = median.

Outputs (<output_path>/): results.json, lens_latency.npz, lens_model.npz,
lens_args.json and plots.
"""

import json
import os
import sys
import time

import numpy as np
import torch

_LENS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LENS")
if _LENS_ROOT not in sys.path:
    sys.path.insert(0, _LENS_ROOT)
import lens.src.blitnet as bn

from src.parser import parse_arguments
from src.common import redirect_output, default_run_name, repeat_timed_call
from src.data_pipeline import build_dataset, resolve_full_sequence_args
from src.checkpoint import save_lens_checkpoint
from src.eval_metrics import debug_gt_and_predictions
from src.plottings import plot_query_matches, plot_similarity_matrix


RECALL_NS = (1, 5, 10, 20, 50, 100)

LENS_DEFAULTS = dict(
    feature_multiplier=2.0,
    thr_l_feat=0.0,   thr_h_feat=0.75,
    fire_l_feat=0.4,  fire_h_feat=0.6,
    ip_rate_feat=0.02, stdp_rate_feat=0.01,
    thr_l_out=0.0,    thr_h_out=0.5,
    fire_l_out=0.5,   fire_h_out=0.5,
    ip_rate_out=0.02, stdp_rate_out=0.01,
    f_exc=0.35, f_inh=0.75,
    o_exc=1.0,  o_inh=1.0,
    epoch_feat=128, epoch_out=128,
)


def main(args):
    resolve_full_sequence_args(args)
    data_path = args.output_path or os.path.join("outputs", default_run_name(args, prefix="lens"))
    redirect_output(data_path)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("LENS baseline")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Images: {args.num_images} x {args.image_size}x{args.image_size}, {args.pixels_per_image} active px/image")

    dataset = build_dataset(args, data_path)
    ref_frames = np.stack([np.asarray(im, dtype=np.float32) for im in dataset.images])
    print(f"\n  refs={ref_frames.shape[0]} queries={dataset.query_loader.n_frames} D={args.input_size}")

    feat_layer, out_layer = _train_lens_network(ref_frames, args.input_size, LENS_DEFAULTS, device)
    save_lens_checkpoint(data_path, feat_layer, out_layer, args)

    _run_lens_inference(args, device, data_path, feat_layer, out_layer, dataset.query_loader,
                        dataset.gt_matrix, ref_frames)
    print(f"\nResults saved to {data_path}/")


def _train_lens_network(ref_frames, D, cfg, device):
    """Build and STDP-train the feature and output layers on the reference
    frames, as LENS' own training loop does."""
    n_places = ref_frames.shape[0]
    R_dense = torch.from_numpy(np.ascontiguousarray(ref_frames, dtype=np.float32)).to(device)

    feature_dim = int(D * cfg['feature_multiplier'])
    print(f"\nLENS architecture: input={D} feature={feature_dim} output={n_places}")
    print(f"  feat: thr=[{cfg['thr_l_feat']},{cfg['thr_h_feat']}] fire=[{cfg['fire_l_feat']},{cfg['fire_h_feat']}] "
          f"ip={cfg['ip_rate_feat']} stdp={cfg['stdp_rate_feat']} p_exc/inh=[{cfg['f_exc']},{cfg['f_inh']}]")
    print(f"  out : thr=[{cfg['thr_l_out']},{cfg['thr_h_out']}] fire=[{cfg['fire_l_out']},{cfg['fire_h_out']}] "
          f"ip={cfg['ip_rate_out']} stdp={cfg['stdp_rate_out']} p_exc/inh=[{cfg['o_exc']},{cfg['o_inh']}] (spk_force=True)")
    print(f"  epochs: feat={cfg['epoch_feat']} out={cfg['epoch_out']}")

    feat_layer = bn.SNNLayer(dims=[D, feature_dim], thr_range=[cfg['thr_l_feat'], cfg['thr_h_feat']],
        fire_rate=[cfg['fire_l_feat'], cfg['fire_h_feat']], ip_rate=cfg['ip_rate_feat'], stdp_rate=cfg['stdp_rate_feat'],
        p=[cfg['f_exc'], cfg['f_inh']], device=device).to(device)
    out_layer = bn.SNNLayer(dims=[feature_dim, n_places], thr_range=[cfg['thr_l_out'], cfg['thr_h_out']],
        fire_rate=[cfg['fire_l_out'], cfg['fire_h_out']], ip_rate=cfg['ip_rate_out'], stdp_rate=cfg['stdp_rate_out'],
        p=[cfg['o_exc'], cfg['o_inh']], spk_force=True, device=device).to(device)

    _train_layer(R_dense, feat_layer, prev_layer=None, n_places=n_places, n_epochs=cfg['epoch_feat'],
                 device=device, layer_name="feature_layer")
    _train_layer(R_dense, out_layer, prev_layer=feat_layer, n_places=n_places, n_epochs=cfg['epoch_out'],
                 device=device, layer_name="output_layer")
    feat_layer.eval()
    out_layer.eval()
    return feat_layer, out_layer


def _train_layer(ref_frames_t, layer, prev_layer, n_places, n_epochs, device, layer_name=""):
    """STDP-train one layer on the (n_places, D) reference frames, fed
    through the already-trained prev_layer if given."""
    T = int(n_places * n_epochs)
    init_itp = layer.eta_stdp.detach() * 2
    init_stdp = layer.eta_stdp.detach()
    print(f"\n  training {layer_name}: places={n_places} epochs={n_epochs} steps={T}")

    mod = 0
    t_start = time.perf_counter()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_places, device=device)
        for j in range(n_places):
            idx_i = int(perm[j].item())
            spikes = ref_frames_t[idx_i].to(torch.float32)
            idx = torch.tensor([idx_i], device=device, dtype=torch.float32)

            if prev_layer is not None:
                with torch.no_grad():
                    spikes = prev_layer.w(spikes)
                    spikes = bn.clamp_spikes(spikes, prev_layer)

            pre_spike = spikes.detach()
            spikes_post = layer.w(spikes)
            spikes_noclp = spikes_post.detach()
            spikes_clp = bn.clamp_spikes(spikes_post, layer)
            layer = bn.calc_stdp(pre_spike, spikes_clp, spikes_noclp, layer, idx, prev_layer=prev_layer)

            # learning-rate annealing every 10 steps, as in LENS_Trainer._anneal_learning_rate
            if mod % 10 == 0:
                pt = pow(float(T - mod) / T, 2)
                layer.eta_ip = torch.mul(init_itp, pt)
                layer.eta_stdp = torch.mul(init_stdp, pt)
            mod += 1

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"    epoch {epoch+1}/{n_epochs} thr.mean={float(layer.thr.mean()):.4f}  "
                  f"|w|.mean={float(layer.w.weight.abs().mean()):.4f} ({time.perf_counter()-t_start:.1f}s elapsed)")
    print(f"  done in {time.perf_counter()-t_start:.1f}s")
    return layer


@torch.no_grad()
def lens_forward(q, feat_layer, out_layer):
    """(n_places,) scores clamp(W2 . clamp(W1 . q - thr1) - thr2) of a (D,) query."""
    h = bn.clamp_spikes(feat_layer.w(q), feat_layer)
    return bn.clamp_spikes(out_layer.w(h), out_layer).reshape(-1)


def _recall_from_sim(sim, predicted, gt, recall_ks):
    """Top-1 hits (from `predicted`) and recall@K hit counts from an (R, Q)
    similarity matrix."""
    R, Q = sim.shape
    hit_counts = {k: 0 for k in recall_ks}
    top1_correct = np.zeros(Q, dtype=bool)
    k_eff = min(max(recall_ks), R)
    for q in range(Q):
        s = sim[:, q]
        if k_eff >= R:
            order = np.argsort(-s)
        else:
            part = np.argpartition(-s, k_eff - 1)[:k_eff]
            order = part[np.argsort(-s[part])]
        gt_col = gt[:, q]
        if predicted[q] >= 0 and gt_col[predicted[q]]:
            top1_correct[q] = True
        for k in recall_ks:
            if gt_col[order[:k]].any():
                hit_counts[k] += 1
    return top1_correct, hit_counts


def _run_lens_inference(args, device, data_path, feat_layer, out_layer, query_loader, gt, ref_frames):
    n_query = query_loader.n_frames
    n_refs = ref_frames.shape[0]
    print(f"\n[LENS] Per-query inference over {n_query} queries:")

    frame_buf = np.empty(args.input_size, dtype=np.float32)  # reused by every query

    def _step(i):
        t0 = time.perf_counter()
        flat = query_loader.make_frame_flat(i, out=frame_buf)
        t1 = time.perf_counter()
        out = lens_forward(torch.from_numpy(flat).to(device), feat_layer, out_layer)
        pred = int(out.argmax())  # first max, as in our kernels; int() waits for the GPU
        t2 = time.perf_counter()
        return (pred, out), (t1 - t0) * 1e6, (t2 - t1) * 1e6

    for i in range(min(100, n_query)):  # warmup
        _step(i)

    n_repeats = max(1, args.n_timing_repeats)
    pred = np.empty(n_query, dtype=np.int64)
    sim_cols = torch.empty((n_query, n_refs), dtype=torch.float32, device=device)
    t_prep, t_net, net_mean, net_std, net_iqr = (np.empty(n_query) for _ in range(5))
    for i in range(n_query):
        (pred[i], sim_cols[i]), (t_prep[i], t_net[i]), stats = repeat_timed_call(n_repeats, lambda i=i: _step(i))
        if sim_cols[i, pred[i]] <= 0:
            pred[i] = -1  # every score is 0: no prediction, as in our kernels
        net_mean[i], net_std[i], net_iqr[i] = stats['mean'][1], stats['std'][1], stats['iqr'][1]

    print(f"\n[LENS] Average latency over {n_query} queries (median of {n_repeats} repeat(s) per query): "
          f"preprocess={np.mean(t_prep):.3f}us  network={np.mean(t_net):.3f}us, "
          f"total={np.mean(t_prep) + np.mean(t_net):.3f}us")
    if n_repeats > 1:
        print(f"[LENS] repeat noise (network): mean={np.mean(net_mean):.3f}us  median={np.mean(t_net):.3f}us  "
              f"std={np.mean(net_std):.3f}us  iqr={np.mean(net_iqr):.3f}us")

    sim_matrix = sim_cols.detach().cpu().numpy().T  # (R, Q)
    recall_Ns = [n for n in RECALL_NS if n <= n_refs]
    top1_hit, hits = _recall_from_sim(sim_matrix, pred, gt, recall_Ns)
    correct = int(top1_hit.sum())
    accuracy = correct / n_query
    eval_label = f"ref/query GT, R={gt.shape[0]}x Q={gt.shape[1]}, tolerance={args.gt_tolerance}"
    print(f"\n[LENS] Accuracy (top-1 vs {eval_label}): {correct}/{n_query} ({100*accuracy:.1f}%)")
    for N in recall_Ns:
        print(f"  Recall@{N:<3d}: {hits[N]:>5d} ({100*hits[N]/n_query:5.1f}%)")

    with open(os.path.join(data_path, "results.json"), "w") as f:
        json.dump(dict(r_at_1=accuracy, recall={str(k): v / n_query for k, v in hits.items()},
                       prep_us=float(np.mean(t_prep)), net_us=float(np.mean(t_net)), n_query=n_query), f, indent=2)
    np.savez(os.path.join(data_path, "lens_latency.npz"), per_image_us=t_net, per_image_us_preprocess=t_prep,
             per_image_us_mean=net_mean, per_image_us_std=net_std, per_image_us_iqr=net_iqr,
             predictions=pred, similarity=sim_matrix, gt=gt)

    debug_gt_and_predictions(gt, None, np.arange(n_refs, dtype=np.int64), predicted=pred, top1_pred=pred,
                             label='real (LENS)', tolerance=args.gt_tolerance, max_show=min(20, n_query))

    title = f"{eval_label} -- LENS predictions overlaid ({correct}/{n_query} top-1)"
    n_plots = min(20, n_query)
    plot_query_matches([query_loader.make_frame_flat(i) for i in range(n_plots)], list(ref_frames), pred, gt,
                       args.image_size, data_path, n_show=n_plots)
    plot_similarity_matrix(sim_matrix, gt, pred, data_path, fname='vpr_similarity_matrix.png', title=title,
                           value_label='Activation (similarity)')


if __name__ == "__main__":
    main(parse_arguments())
