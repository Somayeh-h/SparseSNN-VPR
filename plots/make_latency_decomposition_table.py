"""Per-query cost and speedup decomposition of the Sparse over the Dense SNN
kernel, from one infer.py run (and benchmarks/kernel_stage_breakdown.py for
the stage-share figure).

Cost table (from the run's shapes):
  element operations   dense N_in*N_out     sparse k*N_cols
  weight bytes read    dense 4*N_in*N_out   sparse k*N_cols        (float32 vs uint8 mask)
  vector instructions  dense N_in*N_out/8   sparse k*N_cols/lanes  (256-bit AVX2 registers)

Speedup decomposition (t_* are mean kernel latencies from stdp_latency.npz):
  S_total   = t_dense / t_sparse                   measured
  S_work    = (N_in*N_out) / (k*N_cols)            from the shapes  -> speedup
  S_repr    = t_gathersum / t_sparse               measured         -> speedup
  S_threads = t_dense_1t / t_dense                 measured         -> slowdown
  S_gather  = S_work * t_gathersum / t_dense_1t    measured         -> slowdown
so that S_total = S_work * S_repr / (S_threads * S_gather) exactly.

The gather-sum kernel reads the same k rows as the Sparse SNN but as float32
weights, so gather-sum vs sparse isolates the uint8 mask. t_dense_1t is the
dense kernel on one thread.

Writes the table (.txt) and two figures (.png):
  <prefix>_speedup_budget.png  the four factors on a log axis: speedups
                               stacked right of 1x, slowdowns left of 1x, net bar
  <prefix>_stage_share.png     share of each SNN's per-query latency per stage
                               (needs kernel_stage_breakdown.npz)
"""
import argparse
import glob
import json
import os
import re

import numpy as np

DATASETS = {
    "qcr_event": "QCR-Event",
    "brisbane_event": "Brisbane-Event",
    "nsavp": "NSAVP",
}

REGISTER_BITS = 256          # AVX2
DTYPE_BITS = {"uint8": 8, "uint16": 16, "uint32": 32}
MASK_BYTES = 1               # the synapse mask is uint8; acc_dtype is the width of its counters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="outputs/main_results",
                    help="directory containing the stdp_* run folders")
    ap.add_argument("--dataset", choices=list(DATASETS), default="qcr_event")
    ap.add_argument("--output", default=None,
                    help="path of the .txt file to write (default: "
                         "<results-dir>/latency_decomposition_table_<dataset>.txt)")
    ap.add_argument("--figure-prefix", default=None,
                    help="output prefix for the figures; _speedup_budget.png and _stage_share.png are appended "
                         "(default: <results-dir>/latency_decomposition_<dataset>)")
    args = ap.parse_args()

    r = load_run(args.results_dir, args.dataset)
    d = decompose(r)

    output = args.output or os.path.join(args.results_dir, f"latency_decomposition_table_{args.dataset}.txt")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(build_table(d, DATASETS[args.dataset]))

    print(f"wrote {output}")

    _setup_style()
    prefix = args.figure_prefix or os.path.join(args.results_dir, f"latency_decomposition_{args.dataset}")
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    plot_speedup_budget(d, f"{prefix}_speedup_budget")
    stages = load_stage_breakdown(r["run_dir"])
    if stages is None:
        print(f"  no {STAGE_BREAKDOWN_FILE} in {r['run_dir']}: run benchmarks/kernel_stage_breakdown.py "
              f"to get the stage-share figure")
    else:
        plot_stage_share(stages, f"{prefix}_stage_share")
        for name, segs in stages.items():
            total = sum(v for _, v in segs)
            print(f"  {name}: " + "  ".join(f"{s} {v:.2f}us ({100 * v / total:.0f}%)" for s, v in segs)
                  + f"  = {total:.2f}us")
    print(f"  kernel latency: dense {d['t_dense']:.3f}us  dense 1-thread {d['t_dense_1t']:.3f}us  "
          f"gather-sum {d['t_gather']:.3f}us  sparse {d['t_sparse']:.3f}us")
    print(f"  S_work    {d['S_work']:6.3f}x faster  (analytic)")
    print(f"  S_threads {d['S_threads']:6.3f}x slower  (dense 1-thread / dense)")
    print(f"  S_gather  {d['S_gather']:6.3f}x slower  (S_work * gather-sum / dense 1-thread)")
    print(f"  S_repr    {d['S_repr']:6.3f}x faster  (gather-sum / sparse)")
    print(f"  net       {d['S_work'] * d['S_repr'] / (d['S_threads'] * d['S_gather']):6.3f}x  "
          f"== measured S_total {d['S_total']:.3f}x")


def _find_one(results_dir, pattern):
    matches = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one match for {pattern!r} in {results_dir}, got {matches}")
    return matches[0]


def load_run(results_dir, dataset_key):
    run_dir = _find_one(results_dir, f"stdp_*{dataset_key}*")
    with open(os.path.join(run_dir, "args.json")) as f:
        run_args = json.load(f)
    with open(os.path.join(run_dir, "infer", "logs.txt")) as f:
        log = f.read()
    lat = np.load(os.path.join(run_dir, "infer", "stdp_latency.npz"))

    m = re.search(r"Binary synapse mask: (\d+) columns x (\d+) pixels .*accumulator=(\w+)", log)
    if m is None:
        raise RuntimeError(f"no binary synapse mask line in {run_dir}/infer/logs.txt")
    n_cols, acc_dtype = int(m.group(1)), m.group(3)

    # mean over queries of the per-query medians, as in results.json
    return dict(
        run_dir=run_dir,
        n_in=int(run_args["input_size"]),
        n_out=int(run_args["output_size"]),
        k=int(run_args["pixels_per_image"]),
        n_cols=n_cols,
        acc_dtype=acc_dtype,
        n_threads=int(lat["n_threads_dense"]),
        t_dense=float(lat["per_image_us_dense_kernel"].mean()),
        t_gather=float(lat["per_image_us_gathersum_kernel"].mean()),
        t_sparse=float(lat["per_image_us_sparse_kernel"].mean()),
        t_dense_1t=float(lat["per_image_us_dense_1t_kernel"].mean()),
        prep_dense=float(lat["per_image_us_dense_preprocess"].mean()),
        prep_sparse=float(lat["per_image_us_sparse_preprocess"].mean()),
    )


def decompose(r):
    lanes_dense = REGISTER_BITS // 32
    lanes_sparse = REGISTER_BITS // DTYPE_BITS[r["acc_dtype"]]  # the adds run at the counters' width

    d = dict(r)
    d["elem_dense"] = r["n_in"] * r["n_out"]
    d["elem_sparse"] = r["k"] * r["n_cols"]
    d["bytes_dense"] = 4 * d["elem_dense"]
    d["bytes_sparse"] = MASK_BYTES * d["elem_sparse"]
    d["vec_dense"] = d["elem_dense"] / lanes_dense
    d["vec_sparse"] = d["elem_sparse"] / lanes_sparse
    d["lanes_dense"], d["lanes_sparse"] = lanes_dense, lanes_sparse

    d["S_work"] = d["elem_dense"] / d["elem_sparse"]
    d["S_total"] = r["t_dense"] / r["t_sparse"]
    d["S_repr"] = r["t_gather"] / r["t_sparse"]
    d["S_threads"] = r["t_dense_1t"] / r["t_dense"]
    d["S_gather"] = d["S_work"] * r["t_gather"] / r["t_dense_1t"]
    return d


def fmt_sci(x):
    e = int(np.floor(np.log10(x)))
    return f"{x / 10**e:.2f}e{e:+03d}"


def build_table(d, dataset_label):
    """Plain-text table: per-query cost of both kernels, then the measured
    kernel speedup split into its factors."""
    w0, w1, w2 = 22, 18, 18
    rule = "-" * (w0 + 2 * w1 + 16)

    def row(name, dense, sparse, ratio=""):
        return f"{name:<{w0}}{dense:<{w1}}{sparse:<{w2}}{ratio}".rstrip()

    L = [
        f"Per-query inference cost of the Dense and Sparse SNN on {dataset_label}",
        f"(N_in = {d['n_in']}, N_out = {d['n_out']}, k = {d['k']}; a 256-bit AVX2 register holds "
        f"{d['lanes_dense']} float32 or {d['lanes_sparse']} {d['acc_dtype']} lanes)",
        "",
        row("Per-query cost", "Dense", "Sparse", "Ratio"),
        rule,
        row("Element operations", "N_in*N_out", "k*N_out"),
        row("", fmt_sci(d["elem_dense"]), fmt_sci(d["elem_sparse"]),
            f"{d['elem_dense'] / d['elem_sparse']:.1f}x fewer"),
        row("Weight bytes read", "4*N_in*N_out", "k*N_out"),
        row("", f"{d['bytes_dense'] / 1e6:.2f} MB", f"{d['bytes_sparse'] / 1e6:.2f} MB",
            f"{d['bytes_dense'] / d['bytes_sparse']:.1f}x fewer"),
        row("Vector instructions", f"N_in*N_out/{d['lanes_dense']}", f"k*N_out/{d['lanes_sparse']}"),
        row("", fmt_sci(d["vec_dense"]), fmt_sci(d["vec_sparse"]),
            f"{d['vec_dense'] / d['vec_sparse']:.1f}x fewer"),
        row("Threads", str(d["n_threads"]), "1"),
        row("Kernel latency", f"{d['t_dense']:.2f} us", f"{d['t_sparse']:.2f} us", f"{d['S_total']:.2f}x faster"),
        "",
        f"Decomposition of the {d['S_total']:.2f}x kernel speedup",
        rule,
    ]
    # 3 decimals, so the printed factors multiply to the printed S_total
    factors = [
        ("Fewer weights visited (N_in/k)", f"{d['S_work']:.2f}x faster"),
        (f"Single-threaded (dense gains {d['S_threads']:.3f}x from {d['n_threads']} cores)",
         f"{d['S_threads']:.3f}x slower"),
        ("Accumulator in memory, not registers", f"{d['S_gather']:.3f}x slower"),
        (f"uint8 mask, {d['lanes_sparse']} SIMD lanes, integer add", f"{d['S_repr']:.3f}x faster"),
        (f"Net = {d['S_work']:.2f} x {d['S_repr']:.3f} / ({d['S_threads']:.3f} x {d['S_gather']:.3f})",
         f"{d['S_total']:.2f}x faster"),
    ]
    width = w0 + w1 + w2
    L += [f"{name:<{width}}{value}" for name, value in factors]
    return "\n".join(L) + "\n"


# Same method colours as plot_main_comparison.py
METHOD_COLORS = {
    "Dense SNN (ours)": "tab:blue",
    "Sparse SNN (ours)": "tab:green",
}
PREP_COLOR = "lightgray"
BAR_ALPHA = 0.7
FIG_WIDTH = 3.5


def _setup_style():
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid", font_scale=1)


def _save(fig, out_path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}.png")


def _fmt_factor_tick(v, _pos=None):
    return f"{v:g}×" if v >= 1 else f"1/{1 / v:g}×"


def plot_speedup_budget(d, out_path, fontsize=6.5):
    """The four speedup factors on a log axis, where factors multiply by
    laying segments end to end: speedups stack right of 1x, slowdowns left,
    and the net bar is their ratio."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    # (factor, legend label, colour)
    gains = [
        (d["S_work"], "visit $k$ of $N_\\mathrm{in}$ weights", "#2A9D8F"),
        (d["S_repr"], "uint8 mask, integer add", "#A8E6CF"),
    ]
    losses = [
        (d["S_threads"], f"1 thread (dense uses {d['n_threads']})", "#E76F51"),
        (d["S_gather"], "accumulator in memory", "#FFD3B6"),
    ]
    gain_total = np.prod([f for f, *_ in gains])
    loss_total = np.prod([f for f, *_ in losses])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 1.45))
    y_gain, y_loss, y_net = 2, 1, 0
    h = 0.62
    min_inside_decades = 0.3  # narrower segments get their factor printed outside

    left = 1.0
    for f, _, color in gains:
        ax.barh(y_gain, left * f - left, h, left=left, color=color, edgecolor="0.25", linewidth=0.5)
        ax.text(left * np.sqrt(f), y_gain, f"{f:.2f}×", va="center", ha="center", fontsize=fontsize)
        left *= f

    right = 1.0
    for f, _, color in losses:
        new = right / f
        ax.barh(y_loss, right - new, h, left=new, color=color, edgecolor="0.25", linewidth=0.5)
        if np.log10(f) >= min_inside_decades:
            ax.text(new * np.sqrt(f), y_loss, f"{f:.2f}×", va="center", ha="center", fontsize=fontsize)
        else:
            ax.text(new / 1.08, y_loss, f"{f:.2f}×", va="center", ha="right", fontsize=fontsize)
        right = new

    ax.barh(y_net, d["S_total"] - 1, h, left=1, color=METHOD_COLORS["Sparse SNN (ours)"], alpha=BAR_ALPHA,
            edgecolor="white")
    ax.text(np.sqrt(d["S_total"]), y_net, f"{d['S_total']:.2f}×",
            va="center", ha="center", fontsize=fontsize, fontweight="bold")
    ax.axvline(1, color="0.25", linewidth=0.6)

    ax.set_xscale("log", base=2)
    ax.set_xlim(1 / 10, 40)
    ax.xaxis.set_major_locator(FixedLocator([2.0 ** e for e in range(-2, 6)]))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_factor_tick))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([y_gain, y_loss, y_net])
    ax.set_yticklabels([f"speedups ({gain_total:.1f}×)", f"slowdowns ({loss_total:.2f}×)",
                        f"net ({gain_total:.1f} ÷ {loss_total:.2f})"], fontsize=fontsize + 0.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.grid(False, axis="y")
    ax.set_xlabel("Kernel speedup, sparse over dense SNN (×, log scale)", fontsize=fontsize + 0.5)

    handles = [Patch(facecolor=c, edgecolor="0.25", linewidth=0.5, label=lab) for _, lab, c in gains + losses]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=fontsize - 0.5, frameon=False, handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
    _save(fig, out_path)


STAGE_BREAKDOWN_FILE = os.path.join("infer", "kernel_stage_breakdown.npz")
# stage -> colour
STAGE_COLORS = {
    "preprocessing": PREP_COLOR,
    "call overhead": "#F4A2A2",
    "thread fork/join": "#B3A2C7",
    "accumulate": "#88D4E4",
    "readout + argmax": "#F1C40F",
}


def load_stage_breakdown(run_dir):
    """Mean latency (us) of each stage of both SNNs, from
    benchmarks/kernel_stage_breakdown.py, with readout and argmax merged (see
    that script). None if the benchmark has not been run."""
    path = os.path.join(run_dir, STAGE_BREAKDOWN_FILE)
    if not os.path.exists(path):
        return None
    z = np.load(path)

    def stages(prefix_us, names):
        prefix = prefix_us.mean(axis=0)
        return dict(zip(names, np.diff(np.concatenate([[0.0], prefix]))))

    s = stages(z["sparse_prefix_us"], list(z["sparse_stages"]))
    dn = stages(z["dense_prefix_us"], list(z["dense_stages"]))
    return {
        "Dense SNN (ours)": [("preprocessing", float(z["prep_dense_us"].mean())),
                             ("call overhead", dn["call"]),
                             ("thread fork/join", dn["threads"]),
                             ("accumulate", dn["matvec"]),
                             ("readout + argmax", dn["readout"] + dn["argmax"])],
        "Sparse SNN (ours)": [("preprocessing", float(z["prep_sparse_us"].mean())),
                              ("call overhead", s["call"]),
                              ("accumulate", s["zero"] + s["count"]),
                              ("readout + argmax", s["score"] + s["argmax"])],
    }


def plot_stage_share(stages, out_path, fontsize=6.5):
    """One 100%-stacked bar per SNN: the share of its per-query latency spent
    in each stage. Narrow segments get their label on a leader line."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    min_label_pct = 6
    bar_span = 96  # drawn bar width for a 100%-total stack; leaves a margin at each end
    margin = (100 - bar_span) / 2
    row_spacing = 2.6
    bar_height = 0.7
    callout_gap = 0.55  # extra clearance beyond the bar edge before the leader line's label

    names = list(stages)
    n = len(names)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 1.3 + 0.35 * (row_spacing - 1)))
    ys = np.arange(n)[::-1] * row_spacing
    for row, (y, name) in enumerate(zip(ys, names)):
        segs = stages[name]
        total = sum(max(v, 0.0) for _, v in segs)
        # callouts go below every bar but the last, i.e. into the gap between rows
        callout_below = row < n - 1
        left = margin
        for stage, v in segs:
            pct = 100 * max(v, 0.0) / total
            width = pct * bar_span / 100
            ax.barh(y, width, bar_height, left=left, color=STAGE_COLORS[stage], edgecolor="0.25", linewidth=0.4)

            xc = left + width / 2
            label = f"{pct:.0f}%"
            if pct >= min_label_pct:
                ax.text(xc, y, label, va="center", ha="center", fontsize=fontsize - 0.5)
            else:
                edge_y = y - bar_height / 2 if callout_below else y + bar_height / 2
                label_y = edge_y - callout_gap if callout_below else edge_y + callout_gap
                va = "top" if callout_below else "bottom"
                ax.annotate(label, xy=(xc, edge_y), xytext=(xc, label_y), va=va, ha="center",
                            fontsize=fontsize - 1.5, arrowprops=dict(arrowstyle="-", color="0.3",
                            linewidth=0.4, shrinkA=0, shrinkB=1))
            left += width

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=fontsize + 0.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.grid(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(ys.min() - 0.85, ys.max() + 0.85)
    ax.set_xlabel("Share of per-query latency (%)", fontsize=fontsize + 0.5)

    handles = [Patch(facecolor=c, edgecolor="0.25", linewidth=0.4, label=s) for s, c in STAGE_COLORS.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=3,
              fontsize=fontsize - 0.5, frameon=False, handlelength=1.4, columnspacing=0.8, borderaxespad=0.2)
    _save(fig, out_path)


if __name__ == "__main__":
    main()
