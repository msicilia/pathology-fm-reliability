"""
ood_sweep.py
============
Out-of-distribution detection with four post-hoc scorers on the stored split.

For every (model, task) the linear probe and the scorers are fitted on the
training split and evaluated on the four tiers of conditions defined in
src/lib/conditions.py. The in-distribution set is the test split of the task
unless stated otherwise.

    far        OOD = test split of another task (config.FAR_OOD), same encoder
    blur       OOD = the test split under Gaussian blur, one condition per sigma
               (categorical tasks)
    near       one class is held out: probe and scorers are refitted on the
               training split without it; ID = test patches of the other
               classes, OOD = test patches of the held-out class (tasks with
               three or more classes)
    artefact   as near, with the classes of config.ARTEFACT_CLASSES held out
               jointly (NCT-CRC-HE)

Outputs:
    <out-dir>/ood_table.csv      model, task, tier, condition, scorer, auroc, n_id, n_ood
    <out-dir>/probe_stored.csv   model, task, n_train, n_test, n_features, C, accuracy

Run from the repository root:
    uv run python src/pipeline/ood_sweep.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed

from src.lib import config
from src.lib.conditions import TIERS, held_out_conditions, hold_out, shifted_sets
from src.lib.probe import fit_probe, probe_logits, selected_c
from src.lib.scoring import (
    SCORERS,
    KNNScorer,
    MahalanobisScorer,
    auroc,
    energy_score,
    msp_score,
)
from src.lib.splits import stored_split
from src.lib.tables import write_csv

TABLE_FIELDS = ("model", "task", "tier", "condition", "scorer", "auroc", "n_id", "n_ood")
PROBE_FIELDS = ("model", "task", "n_train", "n_test", "n_features", "C", "accuracy")


class Detectors:
    """The probe and the two feature-space scorers, fitted on one training set."""

    def __init__(self, Z, y, groups=None):
        self.probe = fit_probe(Z, y, groups=groups)
        self.maha = MahalanobisScorer().fit(Z, y)
        self.knn = KNNScorer().fit(Z)

    def scores(self, Z) -> dict[str, np.ndarray]:
        """Novelty score of every row of `Z` under each scorer of `SCORERS`."""
        logits = probe_logits(self.probe, Z)
        out = {"maha": self.maha.score(Z), "knn": self.knn.score(Z),
               "energy": energy_score(logits), "msp": msp_score(logits)}
        if tuple(out) != SCORERS:
            raise RuntimeError("scorer set differs from scoring.SCORERS")
        return out


def condition_rows(model, task, tier, condition, s_id, s_ood) -> list[dict]:
    """One row per scorer for a condition, from the scores of its ID and OOD sets."""
    return [dict(model=model, task=task, tier=tier, condition=condition, scorer=name,
                 auroc=auroc(s_id[name], s_ood[name]),
                 n_id=len(s_id[name]), n_ood=len(s_ood[name]))
            for name in SCORERS]


def held_out_rows(model, task, tier, condition, split, held_out) -> list[dict]:
    """Refit without the classes `held_out` and score them against the remaining classes."""
    reduced, z_ood = hold_out(split, held_out)
    det = Detectors(reduced["Z_train"], reduced["y_train"], reduced["groups_train"])
    return condition_rows(model, task, tier, condition,
                          det.scores(reduced["Z_test"]), det.scores(z_ood))


def run_task(model: str, task: str, emb_dir: str):
    """All conditions of one (model, task): (rows of the OOD table, probe row)."""
    d = config.load(model, task, emb_dir)
    split = stored_split(d)
    classes = [str(c) for c in split["classes"]]
    y_train, y_test = split["y_train"], split["y_test"]
    if not np.array_equal(np.unique(y_train), np.arange(len(classes))):
        raise RuntimeError(f"{model}/{task}: training labels are not 0 .. n_classes-1")

    det = Detectors(split["Z_train"], y_train, split["groups_train"])
    if not np.array_equal(det.probe.classes_, np.arange(len(classes))):
        raise RuntimeError(f"{model}/{task}: probe classes are not 0 .. n_classes-1")
    probe_row = dict(model=model, task=task, n_train=len(y_train), n_test=len(y_test),
                     n_features=split["Z_train"].shape[1], C=selected_c(det.probe),
                     accuracy=float((det.probe.predict(split["Z_test"]) == y_test).mean()))
    s_id = det.scores(split["Z_test"])

    rows = []
    for tier, condition, z_ood in shifted_sets(model, task, d, emb_dir):
        rows += condition_rows(model, task, tier, condition, s_id, det.scores(z_ood))
    for tier, condition, held_out in held_out_conditions(task, classes):
        rows += held_out_rows(model, task, tier, condition, split, held_out)
    return rows, probe_row


def sort_key(row: dict):
    """Order of the output rows: encoder, task, tier, condition, scorer."""
    key = [list(config.ENCODERS).index(row["model"]), config.TASKS.index(row["task"])]
    if "tier" in row:
        key += [TIERS.index(row["tier"]), row["condition"], SCORERS.index(row["scorer"])]
    return tuple(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--models", nargs="+", default=None, choices=list(config.ENCODERS))
    ap.add_argument("--tasks", nargs="+", default=list(config.TASKS), choices=list(config.TASKS))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel (model, task) jobs")
    args = ap.parse_args()
    models = args.models or config.list_models(args.emb_dir)

    jobs = [(m, t) for m in models for t in args.tasks]
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(m, t, args.emb_dir) for m, t in jobs)

    table = sorted((row for rows, _ in results for row in rows), key=sort_key)
    probes = sorted((probe_row for _, probe_row in results), key=sort_key)
    per_condition = Counter((r["model"], r["task"], r["condition"]) for r in table)
    if set(per_condition.values()) != {len(SCORERS)}:
        raise RuntimeError("a condition does not have one row per scorer")

    write_csv(os.path.join(args.out_dir, "ood_table.csv"), TABLE_FIELDS, table)
    write_csv(os.path.join(args.out_dir, "probe_stored.csv"), PROBE_FIELDS, probes)
    print(f"{len(table)} rows, {len(per_condition)} conditions, {len(jobs)} (model, task) pairs "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
