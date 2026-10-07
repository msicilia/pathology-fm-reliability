"""
probe_metrics.py
================
Accuracy, calibration and selective-prediction metrics of the linear probe on
the splits of config.SPLIT_SEEDS.

For every model, task and seed the pooled embeddings are split afresh, the probe
is fitted on the training side and the following are computed on the test side:
accuracy, the selected C, the top-label expected calibration error, the area
under the risk-coverage curve (AURC) and its excess over a perfect ranking
(E-AURC) with uncertainty = 1 - maximum predicted probability, the E-AURC
expected under a random ranking at the same error rate, the mean confidence and,
for the ordinal tasks, the quadratic weighted kappa (QWK). For PANDA the QWK and
accuracy are also computed within each data provider, together with the QWK of a
baseline that predicts the most frequent training grade of the provider of each
test patch.

Outputs:
    <out-dir>/probe_splits.csv        one row per model, task and seed
    <out-dir>/probe_summary.csv       model, task, metric, mean, lo, hi
    <out-dir>/probe_metrics.md        summary tables
    <out-dir>/selective_curves.npz    selective accuracy against coverage (ordinal tasks)

Run from the repository root:
    uv run python src/analysis/probe_metrics.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed

from src.lib import config
from src.lib.evaluation import probe_predict, selective_metrics
from src.lib.names import task_name
from src.lib.probe import fit_probe, selected_c
from src.lib.resampling import bootstrap_indices, pooled_interval
from src.lib.scoring import ECE_BINS, aurc, excess_aurc, qwk, selective_accuracy
from src.lib.splits import resplit
from src.lib.tables import fmt, md_table, write_csv, write_lines

COVERAGE_GRID = np.linspace(0.05, 1.0, 101)
BASE_FIELDS = ("model", "task", "seed", "n_train", "n_test", "n_groups_test", "C", "accuracy",
               "error_rate", "mean_confidence", "ece", "aurc", "e_aurc", "e_aurc_random", "qwk",
               "qwk_provider_baseline")
SUMMARY_FIELDS = ("model", "task", "metric", "mean", "lo", "hi")


def random_ranking_excess_aurc(correct, uncertainty) -> float:
    """Expected E-AURC of a ranking that is independent of correctness.

    Under such a ranking the expected selective risk equals the error rate at every
    coverage, so the expected AURC is the error rate; the oracle AURC is recovered
    from the two library metrics as AURC - E-AURC.
    """
    oracle = aurc(correct, uncertainty) - excess_aurc(correct, uncertainty)
    return float(1.0 - np.mean(correct) - oracle)


def provider_baseline(y_train, sites_train, sites_test) -> np.ndarray:
    """Predict, for each test patch, the most frequent training label of its provider.

    Ties go to the lowest label. A test provider without training patches raises.
    """
    y_train, sites_train = np.asarray(y_train), np.asarray(sites_train)
    modal = {s: int(np.bincount(y_train[sites_train == s]).argmax())
             for s in np.unique(sites_train)}
    return np.array([modal[s] for s in np.asarray(sites_test)])


def site_metrics(y, pred, sites, n_classes: int) -> dict:
    """QWK and accuracy within each data provider."""
    out = {}
    for s in np.unique(sites):
        m = sites == s
        out[f"qwk_{s}"] = qwk(y[m], pred[m], n_classes)
        out[f"accuracy_{s}"] = float((pred[m] == y[m]).mean())
    return out


def run_split(model: str, task: str, seed: int, emb_dir: str, n_boot: int):
    """One (model, task, seed): (row of per-split values, bootstrap draws, curve or None)."""
    split = resplit(config.load(model, task, emb_dir), seed)
    n_classes = len(split["classes"])
    y = split["y_test"]
    groups, sites = split["groups_test"], split["sites_test"]
    probe = fit_probe(split["Z_train"], split["y_train"], groups=split["groups_train"])
    pred, conf = probe_predict(probe, split["Z_test"], n_classes)
    correct = (pred == y).astype(float)
    ordinal = task in config.ORDINAL
    k = n_classes if ordinal else None

    row = dict(model=model, task=task, seed=seed, n_train=len(split["y_train"]), n_test=len(y),
               n_groups_test="" if groups is None else len(np.unique(groups)),
               C=selected_c(probe), error_rate=float(1.0 - correct.mean()),
               mean_confidence=float(conf.mean()),
               e_aurc_random=random_ranking_excess_aurc(correct, 1.0 - conf))
    row.update(selective_metrics(y, pred, conf, k))
    if sites is not None:
        row.update(site_metrics(y, pred, sites, n_classes))
        for s in np.unique(sites):
            row[f"n_test_{s}"] = int((sites == s).sum())
        baseline = provider_baseline(split["y_train"], split["sites_train"], sites)
        row["qwk_provider_baseline"] = qwk(y, baseline, n_classes)

    draws: dict[str, list[float]] = {}
    for idx in bootstrap_indices(len(y), groups, n_boot, seed):
        values = selective_metrics(y[idx], pred[idx], conf[idx], k)
        if sites is not None:
            values.update(site_metrics(y[idx], pred[idx], sites[idx], n_classes))
        for name, value in values.items():
            draws.setdefault(name, []).append(value)

    curve = None
    if ordinal:
        curve = np.array([selective_accuracy(correct, 1.0 - conf, c) for c in COVERAGE_GRID])
    return row, {name: np.array(v) for name, v in draws.items()}, curve


def summarise(results, models, tasks) -> list[dict]:
    """Long-form summary: mean over splits and pooled bootstrap interval per metric."""
    out = []
    for m in models:
        for t in tasks:
            mine = [(row, draws) for row, draws, _ in results
                    if row["model"] == m and row["task"] == t]
            if len(mine) != len(config.SPLIT_SEEDS):
                raise RuntimeError(f"{m}/{t}: expected one result per split seed")
            rows = [row for row, _ in mine]
            skip = ("model", "task", "seed", "n_groups_test", "C")
            for name in (k for k in rows[0] if k not in skip):
                entry = dict(model=m, task=t, metric=name,
                             mean=float(np.mean([r[name] for r in rows])), lo="", hi="")
                if name in mine[0][1]:
                    entry["lo"], entry["hi"] = pooled_interval([d[name] for _, d in mine])
                out.append(entry)
            out.append(dict(model=m, task=t, metric="C_median",
                            mean=float(np.median([r["C"] for r in rows])), lo="", hi=""))
        if set(config.ACCURACY_TASKS) <= set(tasks):
            per_task = [e["mean"] for e in out if e["model"] == m and e["metric"] == "accuracy"
                        and e["task"] in config.ACCURACY_TASKS]
            out.append(dict(model=m, task=config.MEAN_TASK, metric="accuracy",
                            mean=float(np.mean(per_task)), lo="", hi=""))
    return out


def with_interval(entry: dict) -> str:
    return f"{fmt(entry['mean'])} [{fmt(entry['lo'])}, {fmt(entry['hi'])}]"


def report(summary, models, tasks, n_boot: int) -> list[str]:
    """Lines of probe_metrics.md."""
    look = {(e["model"], e["task"], e["metric"]): e for e in summary}
    n_splits = len(config.SPLIT_SEEDS)
    lines = [
        "# Linear-probe metrics",
        "",
        f"Each value is the mean over {n_splits} train/test splits of the pooled embeddings "
        f"(seeds {', '.join(str(s) for s in config.SPLIT_SEEDS)}; the test side is one fifth of "
        "the patches, or of the groups for tasks with patient or slide ids). Intervals are 95% "
        "percentile intervals of the bootstrap draws pooled "
        f"over the splits ({n_boot} draws within each split). Tasks with patient or slide ids "
        "use a cluster bootstrap: groups are resampled with replacement and all their patches "
        "are taken. The other tasks resample patches.",
        "",
    ]

    categorical = [t for t in config.CATEGORICAL if t in tasks]
    if categorical:
        has_mean = set(config.ACCURACY_TASKS) <= set(tasks)
        mean_of = " and ".join([", ".join(task_name(t) for t in config.ACCURACY_TASKS[:-1]),
                                task_name(config.ACCURACY_TASKS[-1])])
        lines += ["## Accuracy on the categorical tasks", "",
                  "Columns: accuracy with its 95% interval per task"
                  + (f"; Mean: mean accuracy over {mean_of}." if has_mean else "."), ""]
        header = ["Encoder"] + [task_name(t) for t in categorical] + ["Mean"] * has_mean
        rows = []
        for m in models:
            row = [config.display(m)] + [with_interval(look[m, t, "accuracy"])
                                         for t in categorical]
            if has_mean:
                row.append(fmt(look[m, config.MEAN_TASK, "accuracy"]["mean"]))
            rows.append(row)
        lines += md_table(header, rows) + [""]

    ordinal = [t for t in config.ORDINAL if t in tasks]
    if ordinal:
        lines += [
            "## Ordinal tasks", "",
            "Columns, per task: Acc: accuracy; QWK: quadratic weighted kappa with 95% interval; "
            "AURC: area under the risk-coverage curve with 95% interval, uncertainty = 1 - "
            "maximum predicted probability; E-AURC: AURC minus the AURC of a perfect ranking at "
            "the same error rate; E-AURC random: expected E-AURC of a ranking independent of "
            f"correctness; ECE: top-label expected calibration error ({ECE_BINS} equal-width "
            "bins); C: median over splits of the selected inverse regularisation strength.",
            "",
            "E-AURC random: with n test predictions of which m are errors, a ranking "
            "independent of correctness has expected selective risk m/n at every coverage, so "
            "its expected AURC is m/n. A perfect ranking has selective risk (k - (n - m)) / k "
            "after keeping the k most certain predictions for k > n - m and zero otherwise, so "
            "E-AURC random = m/n - (1/n) * sum_{k = n-m+1}^{n} (k - (n - m)) / k. For large n "
            "this tends to -(1 - e) * ln(1 - e) with e = m/n.",
            "",
        ]
        columns = ("Acc", "QWK", "AURC", "E-AURC", "E-AURC random", "ECE", "C")
        header = ["Encoder"] + [f"{task_name(t)} {c}" for t in ordinal for c in columns]
        rows = []
        for m in models:
            row = [config.display(m)]
            for t in ordinal:
                row += [fmt(look[m, t, "accuracy"]["mean"]),
                        with_interval(look[m, t, "qwk"]),
                        with_interval(look[m, t, "aurc"]),
                        fmt(look[m, t, "e_aurc"]["mean"]),
                        fmt(look[m, t, "e_aurc_random"]["mean"]),
                        fmt(look[m, t, "ece"]["mean"]),
                        f"{look[m, t, 'C_median']['mean']:g}"]
            rows.append(row)
        lines += md_table(header, rows) + [""]

    if "panda" in tasks:
        sites = sorted(k[2][len("n_test_"):] for k in look
                       if k[0] == models[0] and k[1] == "panda" and k[2].startswith("n_test_"))
        lines += [
            "## PANDA by data provider", "",
            "Columns, per provider: n: mean number of test patches; Acc: accuracy; QWK: "
            "quadratic weighted kappa with 95% interval, both computed on the test patches of "
            "that provider. QWK all: QWK on all test patches. QWK provider-only: QWK on all "
            "test patches of a baseline that predicts the most frequent training grade of the "
            "provider of each patch.",
            "",
        ]
        header = (["Encoder"] + [f"{s} {c}" for s in sites for c in ("n", "Acc", "QWK")]
                  + ["QWK all", "QWK provider-only"])
        rows = []
        for m in models:
            row = [config.display(m)]
            for s in sites:
                row += [fmt(look[m, "panda", f"n_test_{s}"]["mean"], 0),
                        fmt(look[m, "panda", f"accuracy_{s}"]["mean"]),
                        with_interval(look[m, "panda", f"qwk_{s}"])]
            row += [with_interval(look[m, "panda", "qwk"]),
                    fmt(look[m, "panda", "qwk_provider_baseline"]["mean"])]
            rows.append(row)
        lines += md_table(header, rows) + [""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--models", nargs="+", default=None, choices=list(config.ENCODERS))
    ap.add_argument("--tasks", nargs="+", default=list(config.TASKS), choices=list(config.TASKS))
    ap.add_argument("--n-boot", type=int, default=config.N_BOOT)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel (model, task, seed) jobs")
    args = ap.parse_args()
    models = args.models or config.list_models(args.emb_dir)
    tasks = [t for t in config.TASKS if t in args.tasks]

    jobs = [(m, t, s) for m in models for t in tasks for s in config.SPLIT_SEEDS]
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_split)(m, t, s, args.emb_dir, args.n_boot) for m, t, s in jobs)

    rows = [row for row, _, _ in results]
    extra = sorted({k for row in rows for k in row} - set(BASE_FIELDS))
    fields = list(BASE_FIELDS) + extra
    write_csv(os.path.join(args.out_dir, "probe_splits.csv"), fields,
              [{k: row.get(k, "") for k in fields} for row in rows])

    summary = summarise(results, models, tasks)
    write_csv(os.path.join(args.out_dir, "probe_summary.csv"), SUMMARY_FIELDS, summary)
    write_lines(os.path.join(args.out_dir, "probe_metrics.md"),
                report(summary, models, tasks, args.n_boot))

    curves = {"coverage": COVERAGE_GRID, "models": np.array(models),
              "seeds": np.array(config.SPLIT_SEEDS)}
    for t in (t for t in config.ORDINAL if t in tasks):
        curves[t] = np.array([[curve for row, _, curve in results
                               if row["model"] == m and row["task"] == t] for m in models])
    np.savez(os.path.join(args.out_dir, "selective_curves.npz"), **curves)
    print(f"{len(rows)} splits of {len(models)} encoders x {len(tasks)} tasks -> {args.out_dir}")


if __name__ == "__main__":
    main()
