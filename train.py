"""
STDP training: learn one output neuron per reference place with additive STDP,
winner-take-all, homeostasis and a supervised WTA bias (see src/stdp_kernel.py),
then save a checkpoint for infer.py.

    python train.py --output_path outputs/run1 [dataset args]
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from src.parser import parse_arguments
from src.common import print_argparse_args, setup_experiment
from src.data_pipeline import build_dataset, resolve_full_sequence_args
from src.checkpoint import save_checkpoint
from src.stdp_kernel import create_stdp_train_fn, encode_image_poisson
from src.eval_metrics import compute_weight_selectivity, compute_global_prune_threshold
from src.plottings import plot_active_vs_inactive_weights


def main(args):
    resolve_full_sequence_args(args)
    data_path = setup_experiment(args)

    print_argparse_args(args)

    dataset = build_dataset(args, data_path, need_query=False)

    hp = init_hyperparams(args, dataset.images)

    W, theta, W_initial, train_results, neuron_to_image, image_to_neurons = train_model(args, dataset, hp)

    # one network-wide prune threshold, applied by infer.py
    prune_thresholds, prune_info = compute_global_prune_threshold(W, dataset.images, neuron_to_image)
    print(f"\nGlobal prune threshold: {prune_info['threshold']:.4g} "
          f"(Youden J={prune_info['youden_j']:.4f} over {prune_info['n_active']:,} active / "
          f"{prune_info['n_inactive']:,} inactive weights on {prune_info['n_assigned']}/"
          f"{args.output_size} assigned neurons)")
    print(f"  keeps {100*prune_info['kept_active_frac']:.2f}% of active-pixel weight, "
          f"{100*prune_info['kept_inactive_frac']:.4f}% of inactive-pixel weight")
    if prune_info['n_dead_neurons']:
        # expected at low --pixels_per_image: such a neuron abstains instead
        # of matching every query
        print(f"  {prune_info['n_dead_neurons']}/{prune_info['n_assigned']} assigned neurons keep no "
              f"synapse at this cut (never saturated); {prune_info['n_dead_images']} reference "
              f"images end up with no live neuron and are unmatchable")

    evaluate_training(args, dataset, W, W_initial, train_results, neuron_to_image, hp,
                       data_path, prune_thresholds=prune_thresholds)

    save_checkpoint(data_path, W, theta, neuron_to_image, image_to_neurons, args, prune_thresholds=prune_thresholds)
    print(f"\nDone. All results saved to {data_path}/")


def init_hyperparams(args, images):
    """Derive the STDP/LIF constants from args and print the run config.
    w_threshold is the per-pixel weight that brings a neuron to v_th in one
    step under the mean per-image input drive."""
    mean_input_drive = float(np.mean([np.sum(img) for img in images]))
    w_threshold = args.v_th * args.tau / (args.r * args.dt_ms * mean_input_drive)
    w_init_max = 1.0 * w_threshold   # Uniform(0, w_threshold) -> mean = 0.5*w_threshold
    W_min = 0.0                      # excitatory only
    W_max = 2.0 * w_threshold
    w_sum_target = args.input_size * w_threshold * 0.5   # per-column L1 target (= E[initial sum])

    A_plus = args.A_plus_frac * W_max
    A_minus = args.A_minus_frac * W_max
    theta_plus = args.theta_plus_frac * args.v_th
    theta_init_max = args.theta_init_max_frac * args.v_th
    silence_bonus = args.silence_bonus_frac * args.v_th
    target_bonus = args.target_bonus_frac * args.v_th

    hp = dict(
        w_threshold=w_threshold, w_init_max=w_init_max, W_min=W_min, W_max=W_max,
        w_sum_target=w_sum_target, A_plus=A_plus, A_minus=A_minus,
        theta_plus=theta_plus, theta_init_max=theta_init_max,
        silence_bonus=silence_bonus, target_bonus=target_bonus,
    )

    print("=" * 70)
    print("STDP Learning for Visual Place Recognition")
    print("=" * 70)
    print(f"Network: {args.input_size} inputs -> {args.output_size} outputs")
    print(f"Images:  {args.num_images} images, {args.image_size}x{args.image_size}, {args.pixels_per_image} px/image")
    print(f"LIF:     dt={args.dt_ms}ms, v_th={args.v_th}, tau={args.tau}, r={args.r}")
    print(f"STDP:    tau_pre={args.tau_pre}, tau_post={args.tau_post}, A+={A_plus:.4g} (frac={args.A_plus_frac}*W_max), "
          f"A-={A_minus:.4g} (frac={args.A_minus_frac}*W_max)")
    print(f"         W_min={W_min}, W_max={W_max:.4f}, w_init~U(0, {w_init_max:.4f})")
    print(f"         n_train_steps={args.n_train_steps}, n_epochs={args.n_epochs}, p_spike_train={args.p_spike_train}")
    print(f"         w_threshold={w_threshold:.4f} (per-pixel weight to reach v_th in 1 step, "
          f"mean per-image input drive={mean_input_drive:.2f})")
    print(f"         theta_plus={theta_plus:.4f} (homeostatic threshold bump per spike, training only), "
          f"theta_decay={args.theta_decay} (per-image multiplicative decay of theta)")
    print(f"         w_sum_target={w_sum_target:.2f} (L1 column sum, conserved per image)")
    print(f"         silence_threshold={args.silence_threshold} images, silence_bonus={silence_bonus:.4f} "
          f"(force-fire, linear decay to 0 by epoch {args.silence_decay_end_epoch})")
    print(f"         target_bonus={target_bonus:.4f} (supervised WTA: image i owns neuron i)")
    print(f"Frames:  window_size_ms={args.window_size_ms}, top-K={args.pixels_per_image} by event count")
    print(f"Time clock info: {time.get_clock_info('perf_counter')}")

    return hp


def train_model(args, dataset, hp):
    """Initialise the weights and run STDP training. Image i owns neuron i
    (supervised WTA), so the neuron<->image assignment is fixed up front."""
    print(f"\nInitializing weight matrix (all-to-all connectivity)")
    key = jax.random.PRNGKey(42)
    W = jax.random.uniform(key, shape=(args.input_size, args.output_size), minval=0.0, maxval=hp['w_init_max'])
    W_initial = jnp.array(W)
    print(f"   Shape: {W.shape}")
    print(f"   Initial weights mean: {W.mean():.4f}")

    print(f"\nBuilding STDP training function")
    train_fn = create_stdp_train_fn(args.dt_ms, args.v_rest, args.v_th, args.v_reset,
        args.tau, args.r, args.tau_ref, args.tau_pre, args.tau_post, hp['A_plus'], hp['A_minus'],
        hp['W_min'], hp['W_max'], hp['theta_plus'], hp['w_sum_target'], args.silence_threshold,
        target_bonus=hp['target_bonus'], theta_decay=args.theta_decay)

    # target_masks_all[i] marks image i's own neuron (the supervised WTA bias)
    target_masks_all = jnp.array(np.eye(args.num_images, args.output_size, dtype=np.float32))

    key_theta = jax.random.PRNGKey(43)
    theta_warm = jax.random.uniform(key_theta, (args.output_size,), minval=0.0, maxval=hp['theta_init_max'])

    warmup_spike_train = encode_image_poisson(dataset.images[0], n_steps=args.n_train_steps, p_spike=args.p_spike_train, seed=999)

    t0 = time.time()
    W_tmp, _, _, _, _, _ = train_fn(W, jnp.zeros(args.input_size), jnp.zeros(args.output_size),
                                    theta_warm, jnp.zeros(args.output_size, dtype=jnp.int32),
                                    jnp.float32(hp['silence_bonus']), target_masks_all[0], warmup_spike_train)
    W_tmp.block_until_ready()
    print(f"   JIT compilation: {time.time() - t0:.3f}s")

    train_results = train(args, dataset.images, train_fn, W, hp['theta_init_max'], hp['silence_bonus'],
                          args.silence_decay_end_epoch, hp['W_max'], key_theta, target_masks_all)

    neuron_to_image = {i: i for i in range(args.num_images)}
    image_to_neurons = [[i] for i in range(args.num_images)]

    return (train_results['W'], train_results['theta'], W_initial, train_results, neuron_to_image, image_to_neurons)


def train(args, images, train_fn, W, theta_init_max, silence_bonus, silence_decay_end_epoch, W_max, key_theta, target_masks_all):
    """Run STDP training for args.n_epochs, logging per-epoch accuracy (from
    each image's most active neuron) and diagnostics."""
    print(f"\nTraining ({args.n_epochs} epochs, {args.num_images} images/epoch)")
    print("-" * 70)

    accuracies = []
    weight_stats = {'mean': [], 'std': [], 'min': [], 'max': [], 'sparsity': [], 'selectivity': []}

    spike_counts_per_epoch = np.zeros((args.n_epochs, args.output_size), dtype=np.int64)
    theta_trajectory = np.zeros((args.n_epochs, args.output_size), dtype=np.float32)
    dW_per_epoch = np.zeros(args.n_epochs, dtype=np.float32)
    silent_images_per_epoch = np.zeros(args.n_epochs, dtype=np.int32)

    # an image is learned only if it has a winning neuron (coverage) that
    # no other image shares (collisions)
    collision_frac_per_epoch = np.zeros(args.n_epochs, dtype=np.float32)
    coverage_per_epoch = np.zeros(args.n_epochs, dtype=np.float32)

    theta = jax.random.uniform(key_theta, (args.output_size,), minval=0.0, maxval=theta_init_max)
    silent_counter = jnp.zeros(args.output_size, dtype=jnp.int32)

    total_train_start = time.time()
    report_interval = max(1, args.n_epochs // 10)

    SELECTIVITY_MAX_IMG = 2000  # skip the selectivity diagnostic above this many images

    neuron_best_img_ep = np.full(args.output_size, -1, dtype=np.int64)
    neuron_best_count_ep = np.zeros(args.output_size, dtype=np.int64)
    winner_per_image_ep = np.full(args.num_images,  -1, dtype=np.int64)

    for epoch in range(args.n_epochs):
        epoch_start = time.time()
        W_epoch_start = W
        epoch_spikes_np = np.zeros(args.output_size, dtype=np.int64)

        neuron_best_img_ep.fill(-1)
        neuron_best_count_ep.fill(0)
        winner_per_image_ep.fill(-1)
        silent_images_in_epoch = 0

        decay_frac = max(0.0, 1.0 - epoch / max(1, silence_decay_end_epoch))
        silence_bonus_eff = jnp.float32(silence_bonus * decay_frac)

        perm = np.random.permutation(args.num_images)
        for idx in perm:
            idx_i = int(idx)
            pre_trace = jnp.zeros(args.input_size)
            post_trace = jnp.zeros(args.output_size)

            # a fresh Poisson draw per (image, epoch), so STDP averages over many samples of each image
            spike_train_idx = encode_image_poisson(images[idx_i], n_steps=args.n_train_steps,
                                                   p_spike=args.p_spike_train, seed=1000 + idx_i * args.n_epochs + epoch)

            W, pre_trace, post_trace, theta, silent_counter, spike_counts = train_fn(
                W, pre_trace, post_trace, theta, silent_counter, silence_bonus_eff,
                target_masks_all[idx_i], spike_train_idx)

            sc_np = np.asarray(spike_counts).astype(np.int64)
            epoch_spikes_np += sc_np

            sc_total = int(sc_np.sum())
            if sc_total == 0:
                silent_images_in_epoch += 1
            else:
                winner_per_image_ep[idx_i] = int(np.argmax(sc_np))

            update = sc_np > neuron_best_count_ep
            np.copyto(neuron_best_img_ep, np.where(update, idx_i, neuron_best_img_ep))
            np.copyto(neuron_best_count_ep, np.where(update, sc_np, neuron_best_count_ep))

        dW_norm = jnp.linalg.norm(W - W_epoch_start)
        W.block_until_ready()
        epoch_time = time.time() - epoch_start

        spike_counts_per_epoch[epoch] = epoch_spikes_np
        theta_trajectory[epoch] = np.array(theta)
        dW_per_epoch[epoch] = dW_norm
        silent_images_per_epoch[epoch] = silent_images_in_epoch

        valid = winner_per_image_ep >= 0
        winners_safe = np.where(valid, winner_per_image_ep, 0)
        mapped_img = neuron_best_img_ep[winners_safe]
        img_idx_arr = np.arange(args.num_images, dtype=np.int64)
        correct = int(((mapped_img == img_idx_arr) & valid).sum())
        acc = correct / args.num_images
        accuracies.append(acc)

        winners_valid = winner_per_image_ep[valid]
        if winners_valid.size:
            _, counts = np.unique(winners_valid, return_counts=True)
            n_collided = int(counts[counts > 1].sum())
        else:
            n_collided = 0
        collision_frac_per_epoch[epoch] = n_collided / args.num_images
        coverage_per_epoch[epoch] = valid.sum() / args.num_images

        W_np = np.array(W)
        weight_stats['mean'].append(np.mean(W_np))
        weight_stats['std'].append(np.std(W_np))
        weight_stats['min'].append(np.min(W_np))
        weight_stats['max'].append(np.max(W_np))
        weight_stats['sparsity'].append((W_np < 0.01 * W_max).sum() / W_np.size)

        is_report = (epoch + 1) % report_interval == 0 or epoch == 0 or epoch == args.n_epochs - 1
        if is_report:
            if args.num_images <= SELECTIVITY_MAX_IMG:
                sel = compute_weight_selectivity(W, images)
            else:
                sel = float('nan')
            weight_stats['selectivity'].append(sel)
            ep_spikes = spike_counts_per_epoch[epoch]
            th = theta_trajectory[epoch]
            sel_str = f"{sel:.3f}" if sel == sel else "skipped"
            print(f"  Epoch {epoch+1:3d}/{args.n_epochs}: acc={acc*100:5.1f}% sel={sel_str} "
                  f"spikes_total={ep_spikes.sum()} (mean/neuron={np.mean(ep_spikes):.1f}) "
                  f"silent_imgs={silent_images_per_epoch[epoch]}/{args.num_images} "
                  f"dW={dW_per_epoch[epoch]:.4f} theta_mean={np.mean(th):.3f} "
                  f"s_bonus={silence_bonus_eff:.3f} "
                  f"sparsity={weight_stats['sparsity'][-1]*100:.1f}% "
                  f"coverage={coverage_per_epoch[epoch]*100:.1f}% "
                  f"collisions={collision_frac_per_epoch[epoch]*100:.1f}% "
                  f"time={epoch_time:.3f}s")
        else:
            weight_stats['selectivity'].append(weight_stats['selectivity'][-1] if weight_stats['selectivity'] else 0.0)

    total_train_time = time.time() - total_train_start
    per_image_train_ms = total_train_time / (args.n_epochs * args.num_images) * 1000

    print("-" * 70)
    print(f"\nTraining complete")
    print(f"  Total training time: {total_train_time:.2f}s")
    print(f"  Per-image training time: {per_image_train_ms:.2f}ms")
    print(f"  Final accuracy (greedy): {accuracies[-1]*100:.1f}%")

    total_spikes_all = int(spike_counts_per_epoch.sum())
    total_presentations = args.n_epochs * args.num_images
    total_silent = int(silent_images_per_epoch.sum())
    print(f"\n  --- Training diagnostics ---")
    print(f"  Total post-spikes over training: {total_spikes_all} "
          f"(image presentations: {total_presentations}, avg spikes/image: {total_spikes_all/total_presentations:.2f})")
    print(f"  Silent images (no post-spike): {total_silent}/{total_presentations} ({100*total_silent/total_presentations:.1f}%)")
    print(f"  Final theta mean: {np.mean(theta_trajectory[-1]):.3f}")
    print(f"  dW-per-epoch: first={dW_per_epoch[0]:.4f}, last={dW_per_epoch[-1]:.6f}, mean={np.mean(dW_per_epoch):.4f}")

    print(f"  Final coverage (images with a winner):  {coverage_per_epoch[-1]*100:.1f}%")
    print(f"  Final collisions (winner shared by 2+ images): {collision_frac_per_epoch[-1]*100:.1f}%")

    return {
        'W': W,
        'theta': theta,
        'accuracies': accuracies,
        'weight_stats': weight_stats,
        'neuron_best_img_ep': neuron_best_img_ep,
        'winner_per_image_ep': winner_per_image_ep,
        'per_image_train_ms': per_image_train_ms,
        'coverage_per_epoch': coverage_per_epoch,
        'collision_frac_per_epoch': collision_frac_per_epoch,
    }


def evaluate_training(args, dataset, W, W_initial, train_results, neuron_to_image, hp,
                      data_path, prune_thresholds):
    """Print assignment/weight diagnostics and save the active vs inactive weight plot."""
    images = dataset.images
    accuracies = train_results['accuracies']
    weight_stats = train_results['weight_stats']
    winner_per_image_ep = train_results['winner_per_image_ep']
    per_image_train_ms = train_results['per_image_train_ms']

    print(f"\n  Final assignment")
    final_acc = accuracies[-1]
    print(f"  Final accuracy (training-activity assignment): {final_acc*100:.1f}%")
    print(f"  Neurons with assigned image: {len(neuron_to_image)}/{args.output_size}")

    n_show_assign = min(len(neuron_to_image), 20)
    print(f"\n  Neuron assignments (first {n_show_assign} by neuron index):")
    for n_idx in sorted(neuron_to_image.keys())[:n_show_assign]:
        print(f"    Neuron {n_idx:3d} -> Image {neuron_to_image[n_idx]:3d}")
    if len(neuron_to_image) > n_show_assign:
        print(f"    ... ({len(neuron_to_image) - n_show_assign} more)")

    n_show_results = min(args.num_images, 20)
    print(f"\n  Per-image results (first {n_show_results}):")
    for i in range(n_show_results):
        winning_neuron = int(winner_per_image_ep[i]) if winner_per_image_ep[i] >= 0 else -1
        assigned_image = neuron_to_image.get(winning_neuron, -1) if winning_neuron >= 0 else -1
        ok = 'correct' if assigned_image == i else 'WRONG'
        print(f"    Image {i:3d}: neuron {winning_neuron:3d} fires (assigned to img {assigned_image})  {ok}")
    if args.num_images > n_show_results:
        print(f"    ... ({args.num_images - n_show_results} more)")

    W_np = np.array(W)
    print(f"\n  Weight analysis:")
    print(f"    Mean: {np.mean(W_np):.4f},  Std: {np.std(W_np):.4f}")
    print(f"    Min:  {np.min(W_np):.4f},  Max: {np.max(W_np):.4f}")

    n_show_neurons = min(args.output_size, 10)
    active_means = []
    inactive_means = []
    for n_idx in range(n_show_neurons):
        if n_idx not in neuron_to_image:
            continue
        img_idx = neuron_to_image[n_idx]
        active_px = np.where(np.array(images[img_idx]) > 0)[0]
        w_active = np.mean(W_np[active_px, n_idx])
        inactive_mask = np.ones(args.input_size, dtype=bool)
        inactive_mask[active_px] = False
        w_inactive = np.mean(W_np[inactive_mask, n_idx])
        active_means.append(w_active)
        inactive_means.append(w_inactive)
        ratio = w_active / max(w_inactive, 1e-8)
        print(f"    Neuron {n_idx:3d} (img {img_idx:3d}): active_w={w_active:.4f}, inactive_w={w_inactive:.4f}, ratio={ratio:.1f}x")
    if active_means:
        print(f"    Average active/inactive ratio: {np.mean(active_means)/max(np.mean(inactive_means), 1e-8):.1f}x")

    print(f"\n{'=' * 70}")
    print("STDP TRAINING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  1. Learning convergence: {accuracies[0]*100:.1f}% -> {accuracies[-1]*100:.1f}% (epoch 1 -> {args.n_epochs})")
    print(f"  2. Weight selectivity:   {weight_stats['selectivity'][-1]:.3f} (1.0 = perfect)")

    w_near_min = (W_np < 0.1 * hp['W_max']).sum() / W_np.size
    w_near_max = (W_np > 0.9 * hp['W_max']).sum() / W_np.size
    w_middle = 1.0 - w_near_min - w_near_max
    print(f"  3. Weight bimodality: near W_min={w_near_min*100:.1f}%, near W_max={w_near_max*100:.1f}%, middle={w_middle*100:.1f}%")

    if active_means and inactive_means:
        print(f"  4. Active/inactive ratio: {np.mean(active_means)/max(np.mean(inactive_means),1e-8):.1f}x")
    print(f"  5. Training accuracy:     {accuracies[-1]*100:.1f}%")
    print(f"  6. Training time/image:   {per_image_train_ms:.2f} ms")

    plot_active_vs_inactive_weights(W_initial, W, images, neuron_to_image, hp['W_max'], data_path,
                                    prune_threshold=float(prune_thresholds[0]))
    print(f"\nSaved {data_path}/stdp_active_vs_inactive_weights.png")


if __name__ == "__main__":
    main(parse_arguments())
