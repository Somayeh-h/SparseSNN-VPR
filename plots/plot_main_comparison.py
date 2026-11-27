"""R@1 and per-query latency of the Dense SNN, SAD, LENS and the Sparse SNN on
the three datasets, with every speedup relative to LENS. Also prints the
numbers behind the figures.

Reads the results.json of each run folder under --results-dir:
  stdp_<dataset>/infer/results.json   {"dense": {...}, "sparse": {...}}      (infer.py)
  sad_<dataset>/results.json          {"r_at_1", "prep_us", "net_us", ...}   (sad_inference.py)
  lens_<dataset>/results.json         same fields                            (lens_event_inference.py)
Latencies are the mean over queries of each query's median over
--n_timing_repeats repeats.

Writes three figures, so they can be placed independently:
  <out>_r1.png       R@1 as grouped bars, one group per dataset
  <out>_latency.png  per-query latency as stacked bars, preprocessing (light
                     gray) under network (method colour), each bar annotated
                     with its total and network speedup over LENS
  <out>_legend.png   the shared legend
"""
import argparse
import glob
import json
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


sns.set_theme(context="paper", style="whitegrid", font_scale=1)


DATASETS = [
    ("qcr_event", "QCR Event"),
    ("brisbane_event", "Brisbane Event"),
    ("nsavp", "NSAVP"),
]

# (row name, legend label, colour); the same colours are used in every figure
METHODS = [
    ("Dense SNN (ours)", "Dense SNN", "tab:blue"),
    ("SAD", "SAD", "tab:orange"),
    ("LENS", "LENS", "tab:red"),
    ("Sparse SNN (ours)", "Sparse SNN (ours)", "tab:green"),
]
ROW_ORDER = [m[0] for m in METHODS]

REFERENCE = "LENS"
PREP_COLOR = "lightgray"
BAR_ALPHA = 0.7
BAR_WIDTH = 0.2
FIG_SIZE = (3.5, 3.5 * 0.52)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="outputs/main_results",
                    help="directory containing the stdp_*/sad_*/lens_* run folders")
    ap.add_argument("--output", default=None,
                    help="output path prefix; _r1.png, _latency.png and _legend.png are appended "
                         "(default: <results-dir>/main_comparison)")
    args = ap.parse_args()

    all_data = {key: load_dataset(args.results_dir, key) for key, _ in DATASETS}
    _print_summary(all_data)

    output = args.output or os.path.join(args.results_dir, "main_comparison")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_r_at_1(ax, all_data, fontsize=8)
    _save(fig, f"{output}_r1")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_latency(ax, all_data, fontsize=6.5)
    _save(fig, f"{output}_latency")

    _save_legend(f"{output}_legend", ncol=6)


def _find_one(results_dir, pattern):
    matches = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one match for {pattern!r} in {results_dir}, got {matches}")
    return matches[0]


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def load_dataset(results_dir, dataset_key):
    """{row name: {'acc', 'prep', 'net', 'total'}} for one dataset."""
    stdp = _read_json(os.path.join(_find_one(results_dir, f"stdp_*{dataset_key}*"), "infer", "results.json"))
    sad = _read_json(os.path.join(_find_one(results_dir, f"sad_*{dataset_key}*"), "results.json"))
    lens = _read_json(os.path.join(_find_one(results_dir, f"lens_*{dataset_key}*"), "results.json"))
    sources = {"Dense SNN (ours)": stdp["dense"], "SAD": sad, "LENS": lens, "Sparse SNN (ours)": stdp["sparse"]}
    return {name: dict(acc=r["r_at_1"], prep=r["prep_us"], net=r["net_us"], total=r["prep_us"] + r["net_us"])
            for name, r in sources.items()}


def _print_summary(all_data):
    for key, label in DATASETS:
        ref = all_data[key][REFERENCE]
        print(f"\n{label}:")
        for row in ROW_ORDER:
            d = all_data[key][row]
            print(f"  {row:20s} R@1={100 * d['acc']:5.1f}%  prep={d['prep']:6.2f} us  net={d['net']:7.2f} us  "
                  f"total={d['total']:7.2f} us  total speedup={ref['total'] / d['total']:5.2f}x  "
                  f"network speedup={ref['net'] / d['net']:6.2f}x")


def _bar_offsets():
    n = len(METHODS)
    return (np.arange(n) - (n - 1) / 2.0) * BAR_WIDTH


def _set_dataset_xticks(ax):
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([label for _, label in DATASETS])
    ax.tick_params(axis="x", length=0)
    ax.grid(False, axis="x")


def plot_r_at_1(ax, all_data, fontsize):
    """Grouped bars of R@1, one group per dataset, value printed on each bar."""

    xs = np.arange(len(DATASETS))
    for offset, (row, _, color) in zip(_bar_offsets(), METHODS):
        vals = np.array([100.0 * all_data[key][row]["acc"] for key, _ in DATASETS])
        bars = ax.bar(xs + offset, vals, BAR_WIDTH, color=color, alpha=BAR_ALPHA, edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=fontsize, rotation=90)

    ax.set_ylabel("R@1 (%)")
    ax.set_ylim(0, 108)
    _set_dataset_xticks(ax)


def plot_latency(ax, all_data, fontsize):
    """Stacked bars of preprocessing (light gray) and network (method colour)
    latency. Above every other bar: its total speedup over LENS (black) on top
    of its network speedup over LENS (in the method's colour). The LENS bar
    is labelled with its total latency."""

    xs = np.arange(len(DATASETS))
    y_max = max(all_data[key][row]["total"] for key, _ in DATASETS for row in ROW_ORDER)
    pad = 0.02 * y_max
    offsets = _bar_offsets()

    for offset, (row, _, color) in zip(offsets, METHODS):
        prep = np.array([all_data[key][row]["prep"] for key, _ in DATASETS])
        net = np.array([all_data[key][row]["net"] for key, _ in DATASETS])
        ax.bar(xs + offset, prep, BAR_WIDTH, color=PREP_COLOR, edgecolor="white", linewidth=0.4)
        ax.bar(xs + offset, net, BAR_WIDTH, bottom=prep, color=color, alpha=BAR_ALPHA, edgecolor="white",
               linewidth=0.4)

        for i, (key, _) in enumerate(DATASETS):
            d = all_data[key][row]
            ref = all_data[key][REFERENCE]
            x = xs[i] + offset
            y = d["total"] + pad
            if row == REFERENCE:
                ax.text(x, y, f"{d['total']:.0f} µs", ha="center", va="bottom", fontsize=fontsize)
            else:
                net_text = f"{ref['net'] / d['net']:.0f}×"
                total_text = f"{ref['total'] / d['total']:.1f}×"
                ax.annotate(net_text, xy=(x, y), xytext=(-1, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=fontsize, fontweight="bold", color=color)
                ax.annotate(total_text, xy=(x, y), xytext=(-1, 2 + fontsize * 1.3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=fontsize,
                            fontweight="bold", color="black")

    ax.text(0.01, 1, "speedup vs LENS: $\\mathbf{total}$ / network (bar colour)", ha="left", va="top",
             fontsize=fontsize, transform=ax.transAxes)

    ax.set_ylabel("Per-query latency (µs)")
    ax.set_ylim(0, 1.2 * y_max)
    _set_dataset_xticks(ax)


def _legend_handles():
    handles = [Patch(facecolor=color, alpha=BAR_ALPHA, label=label) for _, label, color in METHODS]
    return handles + [Patch(facecolor=PREP_COLOR, label="preprocessing")]


def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}.png")


def _save_legend(out_path, ncol):
    fig = plt.figure(figsize=(7.0, 0.35))
    fig.legend(handles=_legend_handles(), loc="center", ncol=ncol, frameon=False, fontsize=7,
               handlelength=1.2, columnspacing=1.0)
    _save(fig, out_path)


if __name__ == "__main__":
    main()
