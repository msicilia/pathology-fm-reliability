"""
probe_sensitivity.py
====================
Sensitivity of accuracy, grading and calibration to the regularisation of the
linear probe and to the protocol that selects it, on the tasks whose patches
carry a patient or slide id (`config.GROUPED`).

For every encoder, such task and split seed (`config.SPLIT_SEEDS`):

* fixed C: the probe is fitted once for each value of the grid `probe.PROBE_CS`,
  with C fixed at that value;
* selection protocols: the probe is fitted with the full grid under the four
  combinations of fold construction (grouped: all patches of a patient or slide
  in one fold; ungrouped: folds stratified by class only) and selection
  criterion (log-loss; accuracy). Grouped folds with log-loss is the standard
  probe.

Each fit is evaluated on the test split: accuracy, quadratic weighted kappa,
expected calibration error, mean confidence, AURC and excess AURC.

For the standard probe on PANDA the reliability table (15 equal-width confidence
bins) and percentiles of the confidence are computed per encoder on the test
patches of all splits pooled.

Writes probe_sensitivity_fixed_c.csv, probe_sensitivity_protocols.csv,
probe_reliability_panda.csv and probe_sensitivity.md to the output directory.

Run from the repository root:
    uv run python src/analysis/probe_sensitivity.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed

from src.lib import config, scoring
from src.lib.evaluation import probe_predict, selective_metrics
from src.lib.names import task_name
from src.lib.probe import CV_SCORING, PROBE_CS, fit_probe, selected_c
from src.lib.splits import resplit
from src.lib.tables import fmt, md_table, write_csv, write_lines

# (fold construction, selection criterion); the first entry is the standard probe
PROTOCOLS = (("grouped", CV_SCORING), ("grouped", "accuracy"),
             ("ungrouped", CV_SCORING), ("ungrouped", "accuracy"))
CRITERION_NAMES = {"neg_log_loss": "log-loss", "accuracy": "accuracy"}
RELIABILITY_TASK = "panda"
PERCENTILES = (1, 50, 95, 100)

METRICS = ("accuracy", "qwk", "ece", "mean_confidence", "aurc", "e_aurc")
FIXED_FIELDS = ("model", "task", "seed", "C") + METRICS
PROTOCOL_FIELDS = ("model", "task", "seed", "folds", "criterion", "selected_C") + METRICS
RELIABILITY_FIELDS = (("model", "bin", "confidence_low", "confidence_high", "n",
                       "mean_confidence", "accuracy")
                      + tuple(f"confidence_p{p}" for p in PERCENTILES))


def evaluate(probe, split):
    """Test-split metrics of a fitted probe, with its confidence and correctness per patch."""
    n_classes = len(split["classes"])
    y = split["y_test"]
    pred, conf = probe_predict(probe, split["Z_test"], n_classes)
    metrics = selective_metrics(y, pred, conf, n_classes)
    metrics["mean_confidence"] = float(conf.mean())
    return metrics, conf, pred == y


def run_job(model: str, task: str, seed: int, emb_dir: str):
    """All fits of one (encoder, task, seed): one per fixed C and four selection protocols."""
    split = resplit(config.load(model, task, emb_dir), seed)
    groups = split["groups_train"]
    if groups is None:
        raise ValueError(f"{task} has no group ids; the grouped protocols are undefined")
    key = {"model": model, "task": task, "seed": seed}

    fixed = []
    for C in PROBE_CS:
        probe = fit_probe(split["Z_train"], split["y_train"], groups=groups, Cs=C)
        if not np.isclose(selected_c(probe), C):
            raise RuntimeError(f"probe fitted with C={selected_c(probe)}, expected {C}")
        fixed.append({**key, "C": float(C), **evaluate(probe, split)[0]})

    protocols, standard = [], None
    for folds, criterion in PROTOCOLS:
        probe = fit_probe(split["Z_train"], split["y_train"], groups=groups,
                          grouped_cv=(folds == "grouped"), scoring=criterion)
        metrics, conf, correct = evaluate(probe, split)
        protocols.append({**key, "folds": folds, "criterion": CRITERION_NAMES[criterion],
                          "selected_C": selected_c(probe), **metrics})
        if (folds, criterion) == PROTOCOLS[0]:
            standard = (conf, correct)
    return fixed, protocols, standard


def reliability_rows(model: str, conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Reliability table of one encoder; the bins are those of `scoring.ece`."""
    edges = np.linspace(0.0, 1.0, scoring.ECE_BINS + 1)
    pct = {f"confidence_p{p}": float(np.percentile(conf, p)) for p in PERCENTILES}
    rows = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), 1):
        m = (conf > lo) & (conf <= hi)
        rows.append({"model": model, "bin": b, "confidence_low": float(lo),
                     "confidence_high": float(hi), "n": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else float("nan"),
                     "accuracy": float(correct[m].mean()) if m.any() else float("nan"), **pct})
    return rows


def mean_of(rows, metric: str, **where) -> float:
    """Mean of one metric over the rows matching all `where` values (the split seeds)."""
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in where.items())]
    return float(np.mean(vals)) if vals else float("nan")


def summary(models, tasks, seeds, fixed, protocols, reliability) -> list[str]:
    """Lines of probe_sensitivity.md."""
    cs = [float(C) for C in PROBE_CS]
    names = [f"{folds}, {CRITERION_NAMES[crit]}" for folds, crit in PROTOCOLS]
    L = ["# Sensitivity of the linear probe to its regularisation and selection protocol", "",
         f"Splits: seeds {', '.join(str(s) for s in seeds)}. All values are computed on the test "
         "split and, unless stated otherwise, are means over the splits.", "",
         "- C: inverse regularisation strength of the logistic regression.",
         "- ECE: top-label expected calibration error, "
         f"{scoring.ECE_BINS} equal-width confidence bins.",
         "- QWK: quadratic weighted kappa, given for the ordinal tasks; the other tasks give "
         "the accuracy.",
         "- E-AURC: area under the risk-coverage curve minus that of a perfect ranking at the "
         "same error rate; confidence is the maximum softmax probability.",
         "- Protocol: fold construction of the three-fold cross-validation that selects C "
         "(grouped: all patches of a patient or slide in one fold; ungrouped: stratified by "
         "class only) and selection criterion (log-loss or accuracy of the held-out folds). "
         f"The standard probe is \"{names[0]}\".",
         "- Selected C: median over the splits of the C selected by cross-validation."]
    for task in tasks:
        quality = ("qwk", "QWK") if task in config.ORDINAL else ("accuracy", "Accuracy")
        for metric, title in (("ece", "ECE"), quality):
            L += ["", f"## {task_name(task)}: {title} with C fixed", ""]
            L += md_table(["Encoder"] + [f"C={C:g}" for C in cs],
                          [[config.display(m)] + [fmt(mean_of(fixed, metric, model=m, task=task,
                                                              C=C)) for C in cs]
                           for m in models])
        L += ["", f"## {task_name(task)}: selected C by protocol", ""]
        rows = []
        for m in models:
            row = [config.display(m)]
            for folds, crit in PROTOCOLS:
                vals = [r["selected_C"] for r in protocols
                        if (r["model"], r["task"], r["folds"], r["criterion"])
                        == (m, task, folds, CRITERION_NAMES[crit])]
                row.append(f"{np.median(vals):g}")
            rows.append(row)
        L += md_table(["Encoder"] + names, rows)
        for metric, title in (("ece", "ECE"), quality, ("e_aurc", "E-AURC")):
            L += ["", f"## {task_name(task)}: {title} by protocol", ""]
            L += md_table(["Encoder"] + names,
                          [[config.display(m)]
                           + [fmt(mean_of(protocols, metric, model=m, task=task, folds=folds,
                                          criterion=CRITERION_NAMES[crit]))
                              for folds, crit in PROTOCOLS] for m in models])
    if reliability:
        bins = sorted({(r["bin"], r["confidence_low"], r["confidence_high"])
                       for r in reliability})
        header = ["Encoder"] + [f"({lo:.2f}, {hi:.2f}]" for _, lo, hi in bins]
        by = {(r["model"], r["bin"]): r for r in reliability}
        L += ["", f"## {task_name(RELIABILITY_TASK)}: reliability of the standard probe", "",
              "Test patches of all splits pooled. Columns are confidence bins; a hyphen marks "
              "an empty bin."]
        for field, title, digits in (("n", "Number of patches per bin", 0),
                                     ("mean_confidence", "Mean confidence per bin", 3),
                                     ("accuracy", "Accuracy per bin", 3)):
            L += ["", f"### {title}", ""]
            L += md_table(header, [[config.display(m)]
                                   + [fmt(by[(m, b)][field], digits) for b, _, _ in bins]
                                   for m in models])
        L += ["", "### Percentiles of the confidence", ""]
        L += md_table(["Encoder"] + [f"p{p}" for p in PERCENTILES],
                      [[config.display(m)] + [fmt(by[(m, 1)][f"confidence_p{p}"], 4)
                                              for p in PERCENTILES] for m in models])
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=list(config.GROUPED), choices=config.GROUPED)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SPLIT_SEEDS))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers")
    args = ap.parse_args()
    models = args.models if args.models else config.list_models(args.emb_dir)
    models = [m for m in config.ENCODERS if m in models]
    tasks = [t for t in config.TASKS if t in args.tasks]

    jobs = [(m, t, s) for m in models for t in tasks for s in args.seeds]
    results = Parallel(n_jobs=args.jobs, return_as="generator")(
        delayed(run_job)(m, t, s, args.emb_dir) for m, t, s in jobs)
    fixed, protocols, pooled = [], [], {}
    for i, ((m, t, s), (job_fixed, job_protocols, standard)) in enumerate(zip(jobs, results), 1):
        fixed += job_fixed
        protocols += job_protocols
        if t == RELIABILITY_TASK:
            pooled.setdefault(m, []).append(standard)
        print(f"[{i}/{len(jobs)}] {m} {t} seed {s}", flush=True)

    reliability = []
    for m in models:
        if m in pooled:
            conf = np.concatenate([c for c, _ in pooled[m]])
            correct = np.concatenate([k for _, k in pooled[m]])
            reliability += reliability_rows(m, conf, correct)

    out = args.out_dir
    write_csv(os.path.join(out, "probe_sensitivity_fixed_c.csv"), FIXED_FIELDS, fixed)
    write_csv(os.path.join(out, "probe_sensitivity_protocols.csv"), PROTOCOL_FIELDS, protocols)
    if reliability:
        write_csv(os.path.join(out, f"probe_reliability_{RELIABILITY_TASK}.csv"),
                  RELIABILITY_FIELDS, reliability)
    lines = summary(models, tasks, args.seeds, fixed, protocols, reliability)
    write_lines(os.path.join(out, "probe_sensitivity.md"), lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
