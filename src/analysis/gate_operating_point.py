"""
gate_operating_point.py
=======================
The two-threshold abstention rule evaluated at one fixed operating point.

For every encoder and task, on the stored split:

1. The training split is divided into a fit part and a validation part with
   `splits.split_indices`, seed `VALIDATION_SEED`: the validation part is one
   fifth of the patches, or, for tasks with patient or slide ids, all patches of
   one fifth of the patients or slides. The division is stratified by class
   (and by provider for PANDA).
2. The Mahalanobis scorer and the linear probe are fitted on the fit part.
3. Novelty threshold tau_ood: the 95th percentile of the Mahalanobis scores of
   the validation patches (5% nominal in-distribution false-reject rate). A
   patch is rejected when its score exceeds tau_ood.
4. Confidence threshold tau_conf: the quantile of the maximum softmax probability,
   among the validation patches that pass the novelty gate, that retains 80% of
   them. A patch is retained when its confidence is at least tau_conf.
5. Both thresholds are applied unchanged to the test split and to the
   out-of-distribution sets of src/lib/conditions.py: the test split of the far
   donor task, the blurred test split (categorical tasks), and, for NCT-CRC-HE,
   the artefact setting, in which the artefact classes are removed from the fit,
   validation and test parts and their test patches form the out-of-distribution
   set.

Writes gate_operating_point.csv and gate_operating_point.md to the output directory.

Run from the repository root:
    uv run python src/analysis/gate_operating_point.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed

from src.lib import config, scoring
from src.lib.conditions import ARTEFACT, ARTEFACT_TASK, held_out_conditions, hold_out, shifted_sets
from src.lib.evaluation import probe_predict
from src.lib.names import task_name
from src.lib.probe import fit_probe
from src.lib.splits import joint_strata, split_indices, stored_split
from src.lib.tables import fmt, md_table, write_csv, write_lines

VALIDATION_SEED = 0         # seed of the fit/validation division of the training split
ID_REJECT = 0.05            # nominal in-distribution false-reject rate of the novelty gate
RETAIN = 0.80               # share of the validation patches passing the novelty gate
#                             that the confidence gate retains
ARTEFACT_ROW = f"{ARTEFACT_TASK}_{ARTEFACT}"     # task column of the artefact setting
FIELDS = ("model", "task", "quantity", "value")

# quantity -> column header of the summary tables, in display order
COLUMNS = {
    "reject_id": "ID rej.",
    "reject_far": "far rej.",
    **{f"reject_blur_s{s}": f"blur {s} rej." for s in config.BLUR_SIGMAS},
    f"reject_{ARTEFACT}": "artefact rej.",
    "accuracy_ungated": "acc.",
    "accuracy_novelty_gate": "acc. (N)",
    "accuracy_both_gates": "acc. (N+C)",
    "coverage_novelty_gate": "cov. (N)",
    "coverage_both_gates": "cov. (N+C)",
    "qwk_ungated": "QWK",
    "qwk_novelty_gate": "QWK (N)",
    "qwk_both_gates": "QWK (N+C)",
}


def gate(split: dict, ood: dict, ordinal: bool) -> list[tuple[str, float]]:
    """Fit on the fit part, set both thresholds on the validation part and apply them to
    the test split and to the sets in `ood` (name -> embeddings). Returns (quantity, value)."""
    n_classes = len(split["classes"])
    y, groups = split["y_train"], split["groups_train"]
    fit, val = split_indices(y, groups, joint_strata(y, split["sites_train"]),
                             seed=VALIDATION_SEED)
    Z_fit, Z_val = split["Z_train"][fit], split["Z_train"][val]

    detector = scoring.MahalanobisScorer().fit(Z_fit, y[fit])
    probe = fit_probe(Z_fit, y[fit], groups=None if groups is None else groups[fit])

    # thresholds: validation part only
    novelty_val = detector.score(Z_val)
    tau_ood = float(np.quantile(novelty_val, 1.0 - ID_REJECT))
    passed_val = novelty_val <= tau_ood
    _, conf_val = probe_predict(probe, Z_val, n_classes)
    tau_conf = float(np.quantile(conf_val[passed_val], 1.0 - RETAIN))

    # test split and out-of-distribution sets: thresholds applied unchanged
    y_test = split["y_test"]
    pred, conf = probe_predict(probe, split["Z_test"], n_classes)
    correct = pred == y_test
    passed = detector.score(split["Z_test"]) <= tau_ood
    kept = passed & (conf >= tau_conf)

    out = [("n_fit", len(fit)), ("n_validation", len(val)), ("n_test", len(y_test)),
           ("tau_ood", tau_ood), ("tau_conf", tau_conf),
           ("reject_validation", 1.0 - passed_val.mean()),
           ("retain_validation", (conf_val[passed_val] >= tau_conf).mean()),
           ("reject_id", 1.0 - passed.mean())]
    out += [(f"reject_{name}", (detector.score(Z) > tau_ood).mean()) for name, Z in ood.items()]
    stages = (("ungated", np.ones(len(y_test), dtype=bool)), ("novelty_gate", passed),
              ("both_gates", kept))
    out += [(f"accuracy_{name}", correct[m].mean()) for name, m in stages]
    out += [(f"coverage_{name}", m.mean()) for name, m in stages[1:]]
    if ordinal:
        out += [(f"qwk_{name}", scoring.qwk(y_test[m], pred[m], n_classes))
                for name, m in stages]
    return [(q, float(v)) for q, v in out]


def run_job(model: str, task: str, emb_dir: str) -> list[dict]:
    d = config.load(model, task, emb_dir)
    split = stored_split(d)
    # a task has one far condition, which is named by its tier; the blur and artefact
    # sets are named by their condition
    ood = {("far" if tier == "far" else condition): Z
           for tier, condition, Z in shifted_sets(model, task, d, emb_dir)}
    results = {task: gate(split, ood, task in config.ORDINAL)}
    for tier, condition, held_out in held_out_conditions(task, split["classes"]):
        if tier == ARTEFACT:
            reduced, Z_artefact = hold_out(split, held_out)
            results[ARTEFACT_ROW] = gate(reduced, {condition: Z_artefact}, False)
    return [{"model": model, "task": name, "quantity": q, "value": v}
            for name, values in results.items() for q, v in values]


def summary(models, rows) -> list[str]:
    """Lines of gate_operating_point.md."""
    value = {(r["model"], r["task"], r["quantity"]): r["value"] for r in rows}
    present = {r["task"] for r in rows}
    artefact_title = f"{task_name(ARTEFACT_TASK)}, artefact setting"
    L = ["# Two-threshold abstention rule at a fixed operating point", "",
         "Stored split. The training split is divided into a fit part and a validation part "
         f"(seed {VALIDATION_SEED}), stratified by class (and provider for PANDA): the "
         "validation part is one fifth of the patches, or, for tasks with patient or slide "
         "ids, all patches of one fifth of the patients or slides. The Mahalanobis scorer and "
         "the linear probe are fitted on the fit part. The novelty threshold tau_ood is the "
         f"{100 * (1 - ID_REJECT):.0f}th percentile of the Mahalanobis scores of the validation "
         "patches; a patch is rejected by the novelty gate when its score exceeds tau_ood. The "
         "confidence threshold tau_conf is the quantile of the maximum softmax probability, "
         "among the validation patches that pass the novelty gate, that retains "
         f"{100 * RETAIN:.0f}% of them; a patch passes the confidence gate when its confidence "
         "is at least tau_conf. Both thresholds are applied unchanged to the test split and to "
         "the out-of-distribution sets. All values are fractions.", "",
         f"- ID rej.: test patches rejected by the novelty gate (nominal {ID_REJECT:.2f}).",
         "- far rej.: patches of the test split of the far donor task rejected by the "
         "novelty gate.",
         "- blur s rej.: patches of the test split blurred with a Gaussian of standard "
         "deviation s rejected by the novelty gate.",
         "- artefact rej.: test patches of the removed artefact classes "
         f"({', '.join(config.ARTEFACT_CLASSES)}) rejected by the novelty gate.",
         "- acc., acc. (N), acc. (N+C): accuracy of the probe on all test patches, on those "
         "passing the novelty gate, and on those passing both gates.",
         "- cov. (N), cov. (N+C): share of the test patches passing the novelty gate, and "
         "passing both gates.",
         "- QWK, QWK (N), QWK (N+C): quadratic weighted kappa on the same three sets "
         "(ordinal tasks).",
         f"- {artefact_title}: {task_name(ARTEFACT_TASK)} with the artefact classes removed "
         "from the fit, validation and test parts; scorer, probe and thresholds are fitted on "
         "the remaining classes."]
    groups = (("Categorical tasks", [t for c in config.CATEGORICAL
                                     for t in ((c, ARTEFACT_ROW) if c == ARTEFACT_TASK else (c,))]),
              ("Ordinal tasks", list(config.ORDINAL)))
    for title, names in groups:
        names = [t for t in names if t in present]
        if names:
            L += ["", f"## {title}"]
        for task in names:
            with_task = [m for m in models if (m, task, "reject_id") in value]
            cols = [q for q in COLUMNS if (with_task[0], task, q) in value]
            L += ["", f"### {artefact_title if task == ARTEFACT_ROW else task_name(task)}", ""]
            L += md_table(["Encoder"] + [COLUMNS[q] for q in cols],
                          [[config.display(m)] + [fmt(value[(m, task, q)]) for q in cols]
                           for m in with_task])
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=list(config.TASKS), choices=config.TASKS)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers")
    args = ap.parse_args()
    models = args.models if args.models else config.list_models(args.emb_dir)
    models = [m for m in config.ENCODERS if m in models]
    tasks = [t for t in config.TASKS if t in args.tasks]

    jobs = [(m, t) for m in models for t in tasks]
    results = Parallel(n_jobs=args.jobs, return_as="generator")(
        delayed(run_job)(m, t, args.emb_dir) for m, t in jobs)
    rows = []
    for i, ((m, t), job_rows) in enumerate(zip(jobs, results), 1):
        rows += job_rows
        print(f"[{i}/{len(jobs)}] {m} {t}", flush=True)

    write_csv(os.path.join(args.out_dir, "gate_operating_point.csv"), FIELDS, rows)
    lines = summary(models, rows)
    write_lines(os.path.join(args.out_dir, "gate_operating_point.md"), lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
