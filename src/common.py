"""Helpers shared by the entry points: output folder and log file, platform
report, the timing protocol (repeat_timed_call) and CPU cache sizes."""

import os
import re
import subprocess
import sys

import numpy as np


class Logger:
    """Tee stdout to a log file."""

    def __init__(self, filename="logs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def redirect_output(data_path):
    """Create data_path and tee stdout/stderr into data_path/logs.txt."""
    os.makedirs(data_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(data_path, "logs.txt"))
    sys.stderr = sys.stdout


def default_run_name(args, prefix="stdp"):
    if args.event_dataset == 'synthetic':
        return (f"{prefix}_synthetic_{args.num_images}imgs_{args.image_size}x{args.image_size}_"
                f"{args.pixels_per_image}px")
    return (f"{prefix}_{args.event_dataset}_{args.event_seq}_{args.event_query_seq}_{args.num_images}imgs_"
            f"{args.image_size}x{args.image_size}_{args.pixels_per_image}px_{args.window_size_ms:g}ms")


def report_platform():
    """Print the JAX and PyTorch devices; JAX uses the GPU when there is one."""
    import jax
    import torch
    print("HARDWARE / PLATFORM")
    print(f"  JAX {jax.__version__}, default backend: {jax.default_backend()}")
    for dev in jax.devices():
        print(f"   {dev}  (platform={dev.platform})")
    print(f"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()} "
          f"(CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, {torch.cuda.device_count()} GPU(s))")
    jax.config.update("jax_platform_name", "gpu" if any(d.platform == 'gpu' for d in jax.devices()) else "cpu")


def setup_experiment(args, subdir=None, seed=42):
    """Create the output directory (plus `subdir`), log to its logs.txt,
    report the platform and seed numpy. Returns the directory."""
    data_path = args.output_path or os.path.join("outputs", default_run_name(args))
    if subdir:
        data_path = os.path.join(data_path, subdir)
    redirect_output(data_path)

    report_platform()
    np.random.seed(seed)

    if not args.checkpoint:
        args.checkpoint = os.path.join("outputs", default_run_name(args))
    return data_path


def print_argparse_args(args):
    print("\nCommand-line arguments:")
    print("".join(f"  {key}={value}" for key, value in vars(args).items()))


def repeat_timed_call(n_repeats, call_once):
    """Run one query n_repeats times back to back and take the median time.

    call_once() returns (result, t1_us, t2_us, ...), one duration per timed
    phase (e.g. preprocessing, kernel). Scheduler stalls and interrupts only
    ever add time, so the median is a robust per-query latency.

    Returns (result of the last repeat, per-phase medians, stats), where stats
    holds per-phase 'mean', 'median', 'std' and 'iqr' over the repeats.
    """
    results = [call_once() for _ in range(n_repeats)]
    result = results[-1][0]
    timings = np.array([r[1:] for r in results], dtype=np.float64)  # (n_repeats, n_phases)
    q25, q75 = np.percentile(timings, [25, 75], axis=0)
    stats = {
        'mean': np.mean(timings, axis=0),
        'median': np.median(timings, axis=0),
        'std': np.std(timings, axis=0),
        'iqr': q75 - q25,
    }
    return result, tuple(stats['median']), stats


def physical_core_count():
    """Physical cores this process may run on. The Dense SNN uses one thread
    per core: it is memory-bandwidth bound, so hyper-threads would not help."""
    try:
        cpus = os.sched_getaffinity(0)
    except AttributeError:
        return max(1, (os.cpu_count() or 2) // 2)
    cores = set()
    for cpu in cpus:
        topo = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        try:
            with open(f"{topo}/physical_package_id") as f:
                package = f.read().strip()
            with open(f"{topo}/core_id") as f:
                core = f.read().strip()
        except OSError:
            return max(1, len(cpus) // 2)  # no sysfs topology: assume 2 threads per core
        cores.add((package, core))
    return max(1, len(cores))


def _parse_cache_size(size_str):
    """'32K', '1536K', '32768' or '16 MiB' (sysfs / lscpu) -> bytes, or None."""
    size_str = size_str.strip().replace(",", "")
    m = re.match(r'^([\d.]+)\s*([KMG]?i?B?)$', size_str, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).upper().rstrip("IB")
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}.get(unit)
    if mult is None:
        return None
    return int(value * mult)


def get_cpu_cache_sizes():
    """{1: L1d, 2: L2, 3: L3} data cache sizes in bytes, from sysfs or
    `lscpu`; {} if neither works."""
    sizes = {}
    cache_root = "/sys/devices/system/cpu/cpu0/cache"
    if os.path.isdir(cache_root):
        for entry in sorted(os.listdir(cache_root)):
            idx_dir = os.path.join(cache_root, entry)
            if not entry.startswith("index") or not os.path.isdir(idx_dir):
                continue
            try:
                with open(os.path.join(idx_dir, "level")) as f:
                    level = int(f.read().strip())
                with open(os.path.join(idx_dir, "type")) as f:
                    ctype = f.read().strip()
                with open(os.path.join(idx_dir, "size")) as f:
                    size_str = f.read().strip()
            except (OSError, ValueError):
                continue
            if ctype == "Instruction":
                continue
            size_bytes = _parse_cache_size(size_str)
            if size_bytes is None:
                continue
            # capacity of one cache at this level, not summed across cores
            sizes[level] = max(sizes.get(level, 0), size_bytes)
        if sizes:
            return sizes

    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return sizes
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        for level, prefix in ((1, "l1d cache"), (2, "l2 cache"), (3, "l3 cache")):
            if key == prefix:
                size_bytes = _parse_cache_size(val.strip())
                if size_bytes is not None:
                    sizes[level] = size_bytes
    return sizes


def print_weight_cache_fit(label, size_bytes):
    """Print whether size_bytes of weights fit in each CPU cache level."""
    sizes = get_cpu_cache_sizes()
    if not sizes:
        print(f"  {label}: {size_bytes / 1024:.1f} KiB (CPU cache sizes unavailable on this machine)")
        return
    print(f"  {label}: {size_bytes / 1024:.1f} KiB")
    for level in sorted(sizes):
        cache_bytes = sizes[level]
        ratio = size_bytes / cache_bytes
        verdict = "fits" if ratio <= 1.0 else "does NOT fit"
        label_l = "L1d" if level == 1 else f"L{level}"
        print(f"    {label_l} ({cache_bytes / 1024:.0f} KiB): {verdict} ({100 * ratio:.1f}% of cache)")
