"""
make_figures.py
===============
Figures drawn from the result files; nothing is fitted here.

    fig_accuracy_vs_ood.png       mean Mahalanobis AUROC against mean probe accuracy,
                                  one point per encoder
    fig_blur.png                  Mahalanobis AUROC of blurred against unblurred test
                                  patches, per encoder and blur sigma
    fig_selective_accuracy.png    accuracy of the retained predictions against
                                  coverage on SICAPv2 and PANDA

Inputs (in --res-dir): ood_table.csv, probe_summary.csv, correlations.csv,
selective_curves.npz. The figures are written to the figures subdirectory of
--out-dir, which defaults to --res-dir.

Run from the repository root:
    uv run python src/figures/make_figures.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.ood_summary import ALL, encoder_tier_stats, load_accuracy, load_table
from src.lib import config
from src.lib.names import task_name

# One colour, marker and line style per encoder, in the order of config.ENCODERS.
# The binding is by encoder key, so it is the same in every figure and for any subset.
COLOURS = ("#2F6DB5", "#E0556A", "#228833", "#A8901A", "#2792C4", "#AA3377", "#DD6A1F",
           "#009988", "#5B45B0", "#7F8C1F", "#882255", "#C76BD0", "#9C5A1C")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*", "d")
LINESTYLES = ("-", "--", "-.")
if not len(COLOURS) == len(MARKERS) == len(config.ENCODERS):
    raise RuntimeError("one colour and one marker per encoder are required")
STYLE = {m: dict(color=COLOURS[i], marker=MARKERS[i], linestyle=LINESTYLES[i % len(LINESTYLES)])
         for i, m in enumerate(config.ENCODERS)}
DPI = 200
# Label placement in the scatter: (offset in points, horizontal alignment). Encoders that
# lie close together are listed here; the others use the default.
LABEL_DEFAULT = ((5, 4), "left")
LABEL_PLACEMENT = {"hibou_b": ((-6, -11), "right"), "phikon": ((0, -13), "center"),
                   "phikon_v2": ((0, 7), "center"), "hoptimus": ((6, -10), "left")}


def finish(fig, path: str) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def clean_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_accuracy_vs_ood(table, accuracy, corr, models, path: str) -> None:
    """Scatter of mean Mahalanobis AUROC against mean probe accuracy per encoder."""
    auroc = encoder_tier_stats(table, "maha", models)["mean"][ALL]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for m in models:
        baseline = m in config.BASELINES
        ax.scatter(accuracy[m], auroc[m], s=70, color=STYLE[m]["color"],
                   marker="s" if baseline else "o", edgecolor="black", linewidth=0.6, zorder=3)
        offset, align = LABEL_PLACEMENT.get(m, LABEL_DEFAULT)
        ax.annotate(config.display(m), (accuracy[m], auroc[m]), xytext=offset,
                    textcoords="offset points", fontsize=8, color="0.15", ha=align)
    handles = [plt.Line2D([], [], linestyle="", marker="o", color="0.5",
                          markeredgecolor="black", label="pathology encoder"),
               plt.Line2D([], [], linestyle="", marker="s", color="0.5",
                          markeredgecolor="black", label="general-vision baseline")]
    notes = []
    for label, text in (("all", "all encoders"), ("pathology", "pathology encoders")):
        row = corr[(corr["x"] == "accuracy") & (corr["tier"] == ALL) & (corr["encoders"] == label)]
        if len(row) != 1:
            raise RuntimeError(f"correlations.csv has no single accuracy/all row for {label}")
        row = row.iloc[0]
        notes.append(f"{text}: Spearman rho = {row['rho']:.2f}, p = {row['p']:.2g}, "
                     f"n = {int(row['n'])}")
    ax.legend(handles=handles, title="\n".join(notes), title_fontsize=8, fontsize=8,
              loc="best", frameon=False, alignment="left")
    tasks = ", ".join(task_name(t) for t in config.ACCURACY_TASKS)
    ax.set_xlabel(f"Mean probe accuracy ({tasks})", fontsize=9)
    ax.set_ylabel("Mean Mahalanobis AUROC over all OOD conditions", fontsize=9)
    ax.set_title("Probe accuracy and OOD detection per encoder", fontsize=10)
    ax.margins(0.12)
    clean_axes(ax)
    finish(fig, path)


def fig_blur(table, models, path: str) -> None:
    """Mahalanobis AUROC against blur sigma per encoder, mean over the categorical tasks."""
    blur = table[(table["tier"] == "blur") & (table["scorer"] == "maha")]
    if set(blur["task"]) != set(config.CATEGORICAL):
        raise RuntimeError("blur results are needed for every categorical task")
    mean = blur.groupby(["model", "condition"])["auroc"].mean()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in models:
        ax.plot(config.BLUR_SIGMAS, [mean[m, f"blur_s{s}"] for s in config.BLUR_SIGMAS],
                linewidth=1.6, markersize=6, label=config.display(m), **STYLE[m])
    ax.set_xticks(config.BLUR_SIGMAS)
    ax.set_xlabel("Gaussian blur sigma (pixels of a 224 x 224 input)", fontsize=9)
    ax.set_ylabel("Mahalanobis AUROC, blurred against unblurred test patches", fontsize=9)
    ax.set_title(f"Blur detection, mean over the {len(config.CATEGORICAL)} categorical tasks",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    clean_axes(ax)
    finish(fig, path)


def fig_selective_accuracy(curves, models, path: str) -> None:
    """Selective accuracy against coverage on the two ordinal tasks."""
    stored = [str(m) for m in curves["models"]]
    missing = [m for m in models if m not in stored]
    if missing:
        raise RuntimeError(f"selective_curves.npz has no curves for {missing}")
    coverage = curves["coverage"] * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, task, labels in zip(axes, config.ORDINAL, ("region-level labels",
                                                       "whole-slide labels")):
        per_model = {m: curves[task][stored.index(m)].mean(axis=0) * 100.0 for m in models}
        for m in models:
            ax.plot(coverage, per_model[m], linewidth=0.9, color=STYLE[m]["color"],
                    linestyle=STYLE[m]["linestyle"], label=config.display(m))
        ax.plot(coverage, np.mean(list(per_model.values()), axis=0), linewidth=2.8,
                color="black", label="mean over encoders")
        ax.set_title(f"{task_name(task)} ({labels})", fontsize=10)
        ax.set_xlabel("Coverage: predictions retained, most confident first (%)", fontsize=9)
        ax.set_xlim(coverage[0], 100.0)
        clean_axes(ax)
    axes[0].set_ylabel("Accuracy of the retained predictions (%)", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    finish(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default=config.RES_DIR, help="directory of the result files")
    ap.add_argument("--out-dir", default=None, help="defaults to --res-dir")
    ap.add_argument("--models", nargs="+", default=None, choices=list(config.ENCODERS))
    args = ap.parse_args()
    fig_dir = config.fig_dir(args.out_dir or args.res_dir)
    os.makedirs(fig_dir, exist_ok=True)

    table = load_table(os.path.join(args.res_dir, "ood_table.csv"))
    found = set(table["model"])
    models = args.models or [m for m in config.ENCODERS if m in found]
    if not set(models) <= found:
        raise RuntimeError(f"no OOD results for {sorted(set(models) - found)}")
    table = table[table["model"].isin(models)]
    accuracy = load_accuracy(os.path.join(args.res_dir, "probe_summary.csv"), models)
    corr = pd.read_csv(os.path.join(args.res_dir, "correlations.csv"))
    n_corr = set(corr.loc[corr["encoders"] == "all", "n"])
    if n_corr != {len(models)}:
        raise RuntimeError("correlations.csv was computed for a different set of encoders")

    fig_accuracy_vs_ood(table, accuracy, corr, models,
                        os.path.join(fig_dir, "fig_accuracy_vs_ood.png"))
    fig_blur(table, models, os.path.join(fig_dir, "fig_blur.png"))
    with np.load(os.path.join(args.res_dir, "selective_curves.npz")) as curves:
        fig_selective_accuracy(curves, models,
                               os.path.join(fig_dir, "fig_selective_accuracy.png"))


if __name__ == "__main__":
    main()
