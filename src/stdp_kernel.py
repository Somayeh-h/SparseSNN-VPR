"""
STDP training kernel: pure JAX, one jax.lax.scan over each image's spike train.

Pair-based additive STDP (Song et al. 2000, the rule of BrainPy's STDP_Song2000):
  dA_pre/dt  = -A_pre / tau_pre   (+1 on pre-spike)
  dA_post/dt = -A_post / tau_post (+1 on post-spike)
  on a post-spike: dw = +A_plus  * A_pre    then clip to [W_min, W_max]   (LTP)
  on a pre-spike:  dw = -A_minus * A_post   then clip to [W_min, W_max]   (LTD)
Hard clipping makes the weights bimodal: synapses from the image's active
pixels saturate at W_max, the rest sink towards W_min.

Competition between output neurons:
  * winner-take-all: per step only the neuron furthest above its own
    effective threshold fires, and every membrane is reset (lateral inhibition);
  * homeostasis: each spike raises that neuron's threshold by theta_plus, so a
    neuron that keeps winning prices itself out (theta decays between images);
  * force-fire: a neuron silent for more than silence_threshold images gets
    its threshold lowered by silence_bonus (decayed to 0 over the first epochs);
  * supervised WTA: the presented image's own neuron (image i owns neuron i)
    gets its threshold lowered by target_bonus. It still needs real input
    current to fire -- the bias tilts the competition, it does not force a spike.

After every image each neuron's incoming weights are L1-renormalised to
w_sum_target (Diehl & Cook 2015). LTP grows only the active-pixel synapses,
so the shrink that follows hits the inactive ones with nothing to regrow them:
repeated over many presentations this drives them to W_min.
"""

import jax
import jax.numpy as jnp
import numpy as np


def encode_image_poisson(frame, n_steps, p_spike, seed):
    """(n_steps, input_size) Poisson spike train: each active pixel of the
    binary `frame` spikes with probability p_spike per step, inactive pixels
    never. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    active = np.where(frame > 0)[0]
    train = np.zeros((n_steps, len(frame)), dtype=np.float32)
    if len(active) > 0:
        train[:, active] = (rng.random((n_steps, len(active))) < p_spike).astype(np.float32)
    return jnp.array(train)


def create_stdp_train_fn(dt_ms, v_rest, v_th, v_reset, tau, r, tau_ref, tau_pre, tau_post,
                         A_plus, A_minus, W_min, W_max, theta_plus, w_sum_target,
                         silence_threshold, target_bonus, theta_decay):
    """Build the JIT-compiled per-image training step.

    Returns train_one_image(W, pre_trace, post_trace, theta, silent_counter,
    silence_bonus, target_mask, spike_train)
        -> (W, pre_trace, post_trace, theta, silent_counter, spike_counts)
    """
    decay_pre = jnp.exp(-dt_ms / tau_pre)
    decay_post = jnp.exp(-dt_ms / tau_post)

    def step_fn(state, inp_spike):
        V, t_last, step_i, pre_tr, post_tr, theta, W, wta_offset = state
        t = step_i * dt_ms
        n_out = W.shape[1]

        # LIF Euler step (dense synaptic current), refractory neurons frozen
        I = jnp.dot(inp_spike, W)
        V_new = V + dt_ms * (-V + v_rest + r * I) / tau
        V_new = jnp.where((t - t_last) <= tau_ref, V, V_new)

        # WTA: the neuron furthest above its own effective threshold fires
        v_th_eff = v_th + theta + wta_offset
        above_th = V_new >= v_th_eff
        any_above = jnp.any(above_th)
        winner = jnp.argmax(jnp.where(above_th, V_new - v_th_eff, -jnp.inf))
        spike = jnp.zeros(n_out).at[winner].set(any_above.astype(jnp.float32))

        # global lateral inhibition: reset every neuron when the winner fires
        V_new = jnp.where(any_above, v_reset, V_new)
        t_last_new = jnp.where(any_above, t, t_last)

        # traces + additive STDP (LTD uses the post trace before this step's update)
        pre_tr_new = pre_tr * decay_pre + inp_spike
        dW_ltp = A_plus * jnp.outer(pre_tr_new, spike)
        dW_ltd = -A_minus * jnp.outer(inp_spike, post_tr)
        W_new = jnp.clip(W + dW_ltp + dW_ltd, W_min, W_max)
        post_tr_new = post_tr * decay_post + spike

        theta_new = theta + theta_plus * spike
        return (V_new, t_last_new, step_i + 1, pre_tr_new, post_tr_new, theta_new, W_new, wta_offset), spike

    @jax.jit
    def train_one_image(W, pre_trace, post_trace, theta, silent_counter, silence_bonus, target_mask, spike_train):
        n_out = W.shape[1]
        V = jnp.zeros(n_out)
        t_last = jnp.full(n_out, -1e7)

        # threshold offsets held for this whole image: force-fire for
        # chronically silent neurons, supervised bias for the target neuron
        silence_offset = -silence_bonus * (silent_counter > silence_threshold).astype(jnp.float32)
        target_offset = -target_bonus * target_mask
        wta_offset = silence_offset + target_offset

        state0 = (V, t_last, jnp.array(0, jnp.int32), pre_trace, post_trace, theta, W, wta_offset)
        final, all_spikes = jax.lax.scan(step_fn, state0, spike_train)
        _, _, _, pre_tr, post_tr, theta_new, W_new, _ = final

        # L1 column normalisation
        col_sum = jnp.sum(W_new, axis=0)
        W_new = jnp.clip(W_new * (w_sum_target / jnp.maximum(col_sum, 1e-8))[None, :], W_min, W_max)

        theta_new = theta_new * theta_decay

        spike_counts = jnp.sum(all_spikes, axis=0)
        silent_counter_new = jnp.where(spike_counts > 0, 0, silent_counter + 1)
        return W_new, pre_tr, post_tr, theta_new, silent_counter_new, spike_counts

    return train_one_image
