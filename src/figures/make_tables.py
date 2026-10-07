"""
make_tables.py
==============
Write the LaTeX bodies of the result tables from the result files.

Each output file holds one `tabular` environment (booktabs rules) with one row per
encoder in the order of config.ENCODERS, the general-vision baselines below a rule.

    tab_tiers.tex   probe accuracy and Mahalanobis AUROC per tier
    tab_grading.tex patch-level grading on SICAPv2 and PANDA
    tab_slide.tex   PANDA QWK of the patch-level probe and of the slide aggregators
    tab_gate.tex    the two-threshold abstention rule at its fixed operating point
    tab_blur.tex    Mahalanobis AUROC and change of probe accuracy under blur

Run from the repository root, after the analysis stages:

    uv run python src/figures/make_tables.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from src.analysis.ood_summary import encoder_tier_stats, load_accuracy, load_table
from src.lib import config
from src.lib.tables import write_lines

TABLE_SUBDIR = "tables"
TIERS = ("far", "near_categorical", "near_grading", "blur", "artefact")
AGGREGATORS = ("patch_vote", "mean_pool", "max_pool", "abmil", "transformer")


def num(value: float, digits: int = 3) -> str:
    return f"${value:.{digits}f}$"


def interval(mean: float, lo: float, hi: float) -> str:
    """Mean with its interval, the bounds to two decimals without the leading zero."""
    bounds = ",".join(f"{b:.2f}".replace("0.", ".", 1) for b in (lo, hi))
    return f"${mean:.3f}\\,[{bounds}]$"


def tabular(columns: str, header: list[str], rows: dict[str, list[str]], models) -> list[str]:
    """Lines of a tabular; `rows` maps an encoder key to its cells after the name."""
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule", *header, "\\midrule"]
    for m in models:
        if m == config.BASELINES[0] and m != models[0]:
            lines.append("\\midrule")
        lines.append(" & ".join([config.display(m), *rows[m]]) + " \\\\")
    return lines + ["\\bottomrule", "\\end{tabular}"]


def tab_tiers(res_dir: str, models) -> list[str]:
    table = load_table(os.path.join(res_dir, "ood_table.csv"))
    accuracy = load_accuracy(os.path.join(res_dir, "probe_summary.csv"), models)
    mean = encoder_tier_stats(table[table["model"].isin(models)], "maha", models)["mean"]
    rows = {m: [num(accuracy[m])] + [num(mean[t][m]) for t in TIERS] for m in models}
    header = [" & & & \\multicolumn{2}{c}{near-OOD} & & \\\\", "\\cmidrule(lr){4-5}",
              "Model & accuracy & far-OOD & categorical & grading & blur & artefact \\\\"]
    return tabular("lcccccc", header, rows, models)


def tab_grading(res_dir: str, models) -> list[str]:
    summary = pd.read_csv(os.path.join(res_dir, "probe_summary.csv"))
    summary = summary.set_index(["model", "task", "metric"])
    rows = {}
    for m in models:
        cells = []
        for metric in ("qwk", "aurc", "e_aurc", "ece"):
            for task in config.ORDINAL:
                r = summary.loc[(m, task, metric)]
                cells.append(interval(r["mean"], r["lo"], r["hi"]) if metric == "qwk"
                             else num(r["mean"]))
        rows[m] = cells
    header = ["& \\multicolumn{2}{c}{QWK $\\uparrow$} & \\multicolumn{2}{c}{AURC $\\downarrow$}",
              "& \\multicolumn{2}{c}{E-AURC $\\downarrow$} & \\multicolumn{2}{c}{ECE "
              "$\\downarrow$} \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
              "Model & S & P & S & P & S & P & S & P \\\\"]
    return tabular("lcccccccc", header, rows, models)


def tab_slide(res_dir: str, models) -> list[str]:
    runs = pd.read_csv(os.path.join(res_dir, "slide_aggregation.csv"))
    patch = runs.drop_duplicates(["model", "seed"]).groupby("model")["patch_level_qwk"].mean()
    qwk = runs.groupby(["model", "aggregator"])["qwk"].mean()
    rows = {m: [num(patch[m])] + [num(qwk[(m, a)]) for a in AGGREGATORS] for m in models}
    header = ["Model & patch probe & patch vote & mean pool & max pool & ABMIL & transformer \\\\"]
    return tabular("lcccccc", header, rows, models)


def tab_gate(res_dir: str, models) -> list[str]:
    gate = pd.read_csv(os.path.join(res_dir, "gate_operating_point.csv"))
    gate = gate[gate["model"].isin(models)]
    main = gate[gate["task"].isin(config.TASKS)]
    mean = main.groupby(["model", "quantity"])["value"].mean()
    value = gate.set_index(["model", "task", "quantity"])["value"]
    rows = {}
    for m in models:
        cells = [num(mean[(m, q)], 2) for q in ("reject_id", "reject_far")]
        cells += [num(mean[(m, f"reject_blur_s{s:.1f}")], 2) for s in config.BLUR_SIGMAS]
        cells.append(num(value[(m, "nct_artefact", "reject_artefact")], 2))
        for task in config.ORDINAL:
            cells += [num(value[(m, task, q)], 2) for q in ("qwk_ungated", "qwk_both_gates")]
        rows[m] = cells
    blur = " & ".join(f"$\\sigma{{=}}{s:g}$" for s in config.BLUR_SIGMAS)
    header = ["& \\multicolumn{6}{c}{Rejected by the novelty gate} & "
              "\\multicolumn{2}{c}{SICAPv2 QWK} & \\multicolumn{2}{c}{PANDA QWK} \\\\",
              "\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
              f"Model & ID & far & {blur} & artefact & all & kept & all & kept \\\\"]
    return tabular("lcccccccccc", header, rows, models)


def tab_blur(res_dir: str, models) -> list[str]:
    table = pd.read_csv(os.path.join(res_dir, "ood_table.csv"))
    blur = table[(table["tier"] == "blur") & (table["scorer"] == "maha")]
    auroc = blur.groupby(["model", "condition"])["auroc"].mean()
    acc = pd.read_csv(os.path.join(res_dir, "blur_accuracy.csv"))
    acc = acc.set_index(["model", "task", "sigma"])["accuracy"].unstack("sigma")
    sharp = acc[0.0].groupby("model").mean()
    change = acc.drop(columns=0.0).sub(acc[0.0], axis=0).groupby("model").mean()
    rows = {}
    for m in models:
        cells = [num(sharp[m])]
        cells += [num(auroc[(m, f"blur_s{s}")]) for s in config.BLUR_SIGMAS]
        cells += [num(round(change.loc[m, s], 3) + 0.0) for s in config.BLUR_SIGMAS]
        rows[m] = cells
    sigmas = " & ".join(f"$\\sigma{{=}}{s:g}$" for s in config.BLUR_SIGMAS)
    header = ["& & \\multicolumn{3}{c}{AUROC} & \\multicolumn{3}{c}{change of accuracy} \\\\",
              "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
              f"Model & accuracy & {sigmas} & {sigmas} \\\\"]
    return tabular("lccccccc", header, rows, models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default=config.RES_DIR, help="directory of the result files")
    ap.add_argument("--out-dir", default=None, help="defaults to <res-dir>/tables")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.res_dir, TABLE_SUBDIR)
    models = list(config.ENCODERS)
    for name, build in (("tab_tiers", tab_tiers), ("tab_grading", tab_grading),
                        ("tab_slide", tab_slide), ("tab_gate", tab_gate),
                        ("tab_blur", tab_blur)):
        path = os.path.join(out_dir, f"{name}.tex")
        write_lines(path, build(args.res_dir, models))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
