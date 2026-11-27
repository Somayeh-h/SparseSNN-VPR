"""
Save / load the artifacts produced by a training run so infer.py can run
independently, without retraining.

A checkpoint is a directory containing:
  model.npz   - W, theta, the neuron<->image assignment and the prune threshold
  args.json   - the arguments the model was trained with (infer.py restores the
                checkpoint-locked ones, see parser.CHECKPOINT_LOCKED_ARGS)
"""

import json
import os

import numpy as np


def save_checkpoint(data_path, W, theta, neuron_to_image, image_to_neurons, args, prune_thresholds):
    os.makedirs(data_path, exist_ok=True)

    neuron_ids = np.array(sorted(neuron_to_image.keys()), dtype=np.int64)
    assigned_images = np.array([neuron_to_image[n] for n in neuron_ids], dtype=np.int64)

    np.savez_compressed(
        os.path.join(data_path, "model.npz"),
        W=np.asarray(W, dtype=np.float32),
        theta=np.asarray(theta, dtype=np.float32),
        neuron_ids=neuron_ids,
        assigned_images=assigned_images,
        num_images=np.int64(len(image_to_neurons)),
        prune_thresholds=np.asarray(prune_thresholds, dtype=np.float32),
    )
    with open(os.path.join(data_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"  Checkpoint saved: {data_path}/model.npz (+ args.json)")


def save_lens_checkpoint(data_path, feat_layer, out_layer, args):
    """Save the 2-layer LENS/BliTNet network (lens_event_inference.py): each
    SNNLayer's trained weight matrix and threshold vector, which is all
    lens_forward reads at inference time."""
    os.makedirs(data_path, exist_ok=True)

    np.savez_compressed(
        os.path.join(data_path, "lens_model.npz"),
        W1=feat_layer.w.weight.detach().cpu().numpy().astype(np.float32),
        thr1=feat_layer.thr.detach().cpu().numpy().astype(np.float32),
        W2=out_layer.w.weight.detach().cpu().numpy().astype(np.float32),
        thr2=out_layer.thr.detach().cpu().numpy().astype(np.float32),
    )
    with open(os.path.join(data_path, "lens_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"  LENS checkpoint saved: {data_path}/lens_model.npz (+ lens_args.json)")


def load_checkpoint(data_path):
    npz = np.load(os.path.join(data_path, "model.npz"))
    W = npz["W"]
    theta = npz["theta"]
    neuron_to_image = {int(n): int(img) for n, img in zip(npz["neuron_ids"], npz["assigned_images"])}
    num_images = int(npz["num_images"])
    prune_thresholds = npz["prune_thresholds"]

    image_to_neurons = [[] for _ in range(num_images)]
    for n_idx, img_idx in neuron_to_image.items():
        if 0 <= img_idx < num_images:
            image_to_neurons[img_idx].append(n_idx)

    with open(os.path.join(data_path, "args.json")) as f:
        train_args = json.load(f)

    return {
        'W': W,
        'theta': theta,
        'neuron_to_image': neuron_to_image,
        'image_to_neurons': image_to_neurons,
        'train_args': train_args,
        'prune_thresholds': prune_thresholds,
    }
