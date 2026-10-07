"""
ood_summary.py
==============
Summary tables of the out-of-distribution sweep and rank correlations between
probe accuracy and detection AUROC across encoders.

Reads ood_table.csv (src/pipeline/ood_sweep.py) and probe_summary.csv
(src/analysis/probe_metrics.py); nothing is fitted here. The conditions of the
sweep are grouped into five tiers: far, near on the categorical tasks, near on
the grading tasks, blur and artefact.

Outputs:
    <out-dir>/ood_summary.md      summary tables and correlations
    <out-dir>/correlations.csv    x, y, tier, encoders, n, rho, p, lo, hi, n_boot_used

Run from the repository root:
    uv run python src/analysis/ood_summary.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.lib import config
from src.lib.names import task_name
from src.lib.scoring import SCORERS
from src.lib.tables import fmt, md_table, write_csv, write_lines

TIER_NAMES = {
    "far": "Far",
    "near_categorical": "Near (categorical)",
    "near_grading": "Near (grading)",
    "blur": "Blur",
    "artefact": "Artefact",
}
ALL = "all"                     # tier key of the mean over all conditions
SCORER_NAMES = {"maha": "Mahalanobis", "knn": "kNN", "energy": "Energy", "msp": "MSP"}
RHO_BOOT = 2000
RHO_SEED = 0
CORR_FIELDS = ("x", "y", "tier", "encoders", "n", "rho", "p", "lo", "hi", "n_boot_used")


def load_table(path: str) -> pd.DataFrame:
    """The OOD table with the summary tier of each row in column `group`."""
    table = pd.read_csv(path)
    grading = table["task"].isin(config.ORDINAL)
    table["group"] = table["tier"]
    table.loc[(table["tier"] == "near") & grading, "group"] = "near_grading"
    table.loc[(table["tier"] == "near") & ~grading, "group"] = "near_categorical"
    unknown = set(table["group"]) - set(TIER_NAMES)
    if unknown:
        raise RuntimeError(f"unknown tiers in {path}: {sorted(unknown)}")
    if set(table["scorer"]) != set(SCORERS):
        raise RuntimeError(f"{path} does not hold the scorers {SCORERS}")
    return table


def load_accuracy(path: str, models) -> pd.Series:
    """Mean probe accuracy over config.ACCURACY_TASKS per encoder."""
    summary = pd.read_csv(path)
    rows = summary[(summary["task"] == config.MEAN_TASK) & (summary["metric"] == "accuracy")]
    accuracy = rows.set_index("model")["mean"]
    missing = [m for m in models if m not in accuracy.index]
    if missing:
        raise RuntimeError(f"{path} has no mean accuracy for {missing}")
    return accuracy.loc[list(models)]


def encoder_tier_stats(table: pd.DataFrame, scorer: str, models) -> pd.DataFrame:
    """Mean, sample standard deviation (ddof = 1) and count of the AUROC of one scorer,
    per encoder and tier.

    The column level `ALL` holds the same statistics over all conditions.
    """
    rows = table[table["scorer"] == scorer]
    per_tier = rows.groupby(["model", "group"])["auroc"].agg(["mean", "std", "count"]).unstack()
    overall = rows.groupby("model")["auroc"].agg(["mean", "std", "count"])
    for stat in ("mean", "std", "count"):
        per_tier[(stat, ALL)] = overall[stat]
    return per_tier.loc[list(models)]


def tiers_present(table: pd.DataFrame) -> list[str]:
    return [t for t in TIER_NAMES if t in set(table["group"])]


def conditions_per_tier(table: pd.DataFrame) -> dict[str, int]:
    """Number of (task, condition) pairs per tier; it must be the same for every encoder."""
    keyed = table.assign(key=table["task"] + "/" + table["condition"])
    counts = keyed.groupby(["group", "model"])["key"].nunique()
    out = {}
    for tier in tiers_present(table):
        values = set(counts[tier])
        if len(values) != 1:
            raise RuntimeError(f"encoders differ in the number of {tier} conditions")
        out[tier] = int(values.pop())
    return out


def tier_table(table, scorer, models, n_cond) -> list[str]:
    stats = encoder_tier_stats(table, scorer, models)
    tiers = list(n_cond)
    header = (["Encoder"] + [f"{TIER_NAMES[t]} (n={n_cond[t]})" for t in tiers]
              + [f"All (n={sum(n_cond.values())})"])
    rows = []
    for m in models:
        cells = []
        for t in tiers + [ALL]:
            mean, std = stats.loc[m, ("mean", t)], stats.loc[m, ("std", t)]
            cells.append(fmt(mean) if np.isnan(std) else f"{fmt(mean)} +/- {fmt(std)}")
        rows.append([config.display(m)] + cells)
    return md_table(header, rows)


def spearman_with_interval(x, y) -> dict:
    """Spearman rho, its two-sided p-value and a percentile-bootstrap 95% interval.

    The interval resamples the paired observations with replacement; resamples in
    which either variable is constant have no rank correlation and are left out.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise RuntimeError("a rank correlation needs at least three encoders")
    result = spearmanr(x, y)
    rng = np.random.default_rng(RHO_SEED)
    draws = []
    for _ in range(RHO_BOOT):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            continue
        draws.append(np.corrcoef(rankdata(x[idx]), rankdata(y[idx]))[0, 1])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(n=n, rho=float(result.statistic), p=float(result.pvalue),
                lo=float(lo), hi=float(hi), n_boot_used=len(draws))


def correlations(table, accuracy, models) -> list[dict]:
    """All rank correlations across encoders written to correlations.csv."""
    maha = encoder_tier_stats(table, "maha", models)["mean"]
    knn = encoder_tier_stats(table, "knn", models)["mean"]
    tiers = tiers_present(table) + [ALL]
    pathology = [m for m in models if m not in config.BASELINES]
    out = []
    for label, subset in (("all", list(models)), ("pathology", pathology)):
        for t in tiers:
            entry = dict(x="accuracy", y="maha_auroc", tier=t, encoders=label)
            entry.update(spearman_with_interval(accuracy.loc[subset], maha.loc[subset, t]))
            out.append(entry)
    for t in tiers:
        entry = dict(x="maha_auroc", y="knn_auroc", tier=t, encoders="all")
        entry.update(spearman_with_interval(maha[t], knn[t]))
        out.append(entry)
    return out


def correlation_table(rows) -> list[str]:
    header = ["Tier", "Encoders", "n", "rho", "95% interval", "p"]
    body = [[TIER_NAMES.get(r["tier"], "All conditions"), r["encoders"], r["n"], fmt(r["rho"]),
             f"[{fmt(r['lo'])}, {fmt(r['hi'])}]", f"{r['p']:.3g}"] for r in rows]
    return md_table(header, body)


def report(table, accuracy, corr, models) -> list[str]:
    """Lines of ood_summary.md."""
    n_cond = conditions_per_tier(table)
    tiers = list(n_cond)
    near_tasks = {g: ", ".join(task_name(t) for t in config.TASKS
                               if t in set(table.loc[table["group"] == g, "task"]))
                  for g in ("near_categorical", "near_grading")}
    lines = [
        "# Out-of-distribution detection summary",
        "",
        "AUROC of post-hoc novelty scorers fitted on the training split of each task, with the "
        "out-of-distribution set as the positive class. Tiers: Far: test split of another "
        "task; Near (categorical): one class held out, tasks "
        f"{near_tasks['near_categorical']}; Near (grading): one grade held out, tasks "
        f"{near_tasks['near_grading']}; Blur: the test split under Gaussian blur; Artefact: "
        f"the classes {', '.join(config.ARTEFACT_CLASSES)} of {task_name('nct')} held out "
        "jointly. n is the number of conditions (task and OOD set) per encoder.",
        "",
    ]
    for scorer in ("maha", "knn"):
        lines += [f"## {SCORER_NAMES[scorer]} AUROC per encoder and tier", "",
                  "Cells: mean +/- sample standard deviation (ddof = 1) over the n conditions of "
                  "the tier (mean only when n = 1).", ""]
        lines += tier_table(table, scorer, models, n_cond) + [""]

    lines += ["## Scorer comparison", "",
              "Cells: mean AUROC over all encoders and conditions of the tier; n is the number "
              f"of (encoder, condition) pairs ({len(models)} encoders).", ""]
    by_scorer = table.groupby(["scorer", "group"])["auroc"].mean().unstack()
    overall = table.groupby("scorer")["auroc"].mean()
    header = (["Scorer"] + [f"{TIER_NAMES[t]} (n={n_cond[t] * len(models)})" for t in tiers]
              + [f"All (n={sum(n_cond.values()) * len(models)})"])
    lines += md_table(header, [[SCORER_NAMES[s]] + [fmt(by_scorer.loc[s, t]) for t in tiers]
                               + [fmt(overall[s])] for s in SCORERS]) + [""]

    blur = table[table["group"] == "blur"]
    if len(blur):
        blur_tasks = [t for t in config.TASKS if t in set(blur["task"])]
        names = ", ".join(task_name(t) for t in blur_tasks)
        by_sigma = blur[blur["scorer"] == "maha"].groupby(["model", "condition"])["auroc"].mean()
        conditions = [f"blur_s{s}" for s in config.BLUR_SIGMAS]
        lines += ["## Blur: Mahalanobis AUROC per encoder and sigma", "",
                  "Cells: AUROC of the blurred against the unblurred test split, mean over the "
                  f"{len(blur_tasks)} tasks ({names}). Sigma is in pixels of a 224 x 224 input.",
                  ""]
        lines += md_table(["Encoder"] + [f"sigma = {s}" for s in config.BLUR_SIGMAS],
                          [[config.display(m)] + [fmt(by_sigma[m, c]) for c in conditions]
                           for m in models]) + [""]
        per_scorer = blur.groupby(["model", "scorer"])["auroc"].mean()
        lines += ["## Blur: AUROC per encoder and scorer", "",
                  f"Cells: mean AUROC over the {len(conditions)} values of sigma and the "
                  f"{len(blur_tasks)} tasks.", ""]
        lines += md_table(["Encoder"] + [SCORER_NAMES[s] for s in SCORERS],
                          [[config.display(m)] + [fmt(per_scorer[m, s]) for s in SCORERS]
                           for m in models]) + [""]

    near = table[(table["tier"] == "near") & (table["scorer"] == "maha")]
    if len(near):
        near_by_task = near.groupby(["model", "task"])["auroc"].agg(["mean", "count"])
        tasks = [t for t in config.TASKS if t in set(near["task"])]
        counts = {t: int(near_by_task.loc[(models[0], t), "count"]) for t in tasks}
        lines += ["## Near-OOD: Mahalanobis AUROC per encoder and task", "",
                  "Cells: mean AUROC over the held-out classes of the task (n classes).", ""]
        lines += md_table(
            ["Encoder"] + [f"{task_name(t)} (n={counts[t]})" for t in tasks],
            [[config.display(m)] + [fmt(near_by_task.loc[(m, t), "mean"]) for t in tasks]
             for m in models]) + [""]

    acc_tasks = ", ".join(task_name(t) for t in config.ACCURACY_TASKS)
    lines += [
        "## Rank correlations across encoders", "",
        "Spearman rho with its two-sided p-value. The 95% interval is a percentile bootstrap "
        f"over encoders ({RHO_BOOT} resamples with replacement, seed {RHO_SEED}; resamples in "
        "which a variable is constant are left out). Encoders: all, or pathology (without "
        f"{' and '.join(config.display(m) for m in config.BASELINES)}).",
        "",
        "### Probe accuracy and Mahalanobis AUROC", "",
        f"x: mean probe accuracy over {acc_tasks} (mean over the splits of probe_summary.csv); "
        "y: mean Mahalanobis AUROC of the encoder over the conditions of the tier.", "",
    ]
    lines += correlation_table([r for r in corr if r["x"] == "accuracy"]) + [""]
    lines += ["### Mahalanobis AUROC and kNN AUROC", "",
              "x and y: mean AUROC of the encoder over the conditions of the tier under each "
              "scorer.", ""]
    lines += correlation_table([r for r in corr if r["x"] == "maha_auroc"]) + [""]

    lines += ["## Values per encoder", "",
              "Accuracy: mean probe accuracy over the tasks named above; Mahalanobis and kNN: "
              "mean AUROC over all conditions.", ""]
    maha = encoder_tier_stats(table, "maha", models)["mean"][ALL]
    knn = encoder_tier_stats(table, "knn", models)["mean"][ALL]
    lines += md_table(["Encoder", "Accuracy", "Mahalanobis", "kNN"],
                      [[config.display(m), fmt(accuracy[m]), fmt(maha[m]), fmt(knn[m])]
                       for m in models]) + [""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default=config.RES_DIR,
                    help="directory holding ood_table.csv and probe_summary.csv")
    ap.add_argument("--out-dir", default=None, help="defaults to --res-dir")
    ap.add_argument("--models", nargs="+", default=None, choices=list(config.ENCODERS))
    args = ap.parse_args()
    out_dir = args.out_dir or args.res_dir

    table = load_table(os.path.join(args.res_dir, "ood_table.csv"))
    found = set(table["model"])
    models = args.models or [m for m in config.ENCODERS if m in found]
    if not set(models) <= found:
        raise RuntimeError(f"no OOD results for {sorted(set(models) - found)}")
    table = table[table["model"].isin(models)]
    accuracy = load_accuracy(os.path.join(args.res_dir, "probe_summary.csv"), models)

    corr = correlations(table, accuracy, models)
    write_csv(os.path.join(out_dir, "correlations.csv"), CORR_FIELDS, corr)
    write_lines(os.path.join(out_dir, "ood_summary.md"), report(table, accuracy, corr, models))
    print(f"{len(models)} encoders, {len(corr)} correlations -> {out_dir}")


if __name__ == "__main__":
    main()
