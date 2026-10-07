"""
scorer_sensitivity.py
=====================
Sensitivity of the detection results to the settings of the feature-space scorers.

On the stored split, for every encoder and task, the scorers are fitted on the
training split and the AUROC is computed between the test split (in-distribution)
and each out-of-distribution set of the far and blur tiers of src/lib/conditions.py:

* far:  the test split of the donor task `config.FAR_OOD[task]`;
* blur: the Gaussian-blurred test split, one condition per sigma (categorical tasks).

Three settings are varied:

* feature normalisation: Mahalanobis and kNN (k = 50) on raw and on
  L2-normalised embeddings;
* neighbourhood size: kNN on L2-normalised embeddings with k in {1, 10, 50, 200};
* covariance estimate: Mahalanobis on raw embeddings with the Ledoit-Wolf
  shrinkage estimate and with the unshrunk empirical pooled within-class covariance.

Writes scorer_sensitivity.csv and scorer_sensitivity.md to the output directory.

Run from the repository root:
    uv run python src/analysis/scorer_sensitivity.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed

from src.lib import config
from src.lib.conditions import SHIFT_TIERS, shifted_sets
from src.lib.names import task_name
from src.lib.scoring import KNN_K, KNNScorer, MahalanobisScorer, auroc
from src.lib.splits import stored_split
from src.lib.tables import fmt, md_table, write_csv, write_lines

KNN_KS = (1, 10, 50, 200)
FIELDS = ("model", "task", "tier", "condition", "scorer", "setting", "auroc")

# sections of the summary: (title, scorer, settings)
SECTIONS = (
    ("Feature normalisation, Mahalanobis", "maha", ("normalise=False", "normalise=True")),
    (f"Feature normalisation, kNN (k = {KNN_K})", "knn", ("normalise=False", "normalise=True")),
    ("Neighbourhood size, kNN on L2-normalised embeddings", "knn",
     tuple(f"k={k}" for k in KNN_KS)),
    ("Covariance estimate, Mahalanobis on raw embeddings", "maha",
     ("covariance=ledoit_wolf", "covariance=empirical")),
)


class EmpiricalMahalanobisScorer(MahalanobisScorer):
    """Mahalanobis scorer whose shared covariance is the unshrunk empirical
    pooled within-class covariance."""

    def fit(self, Z, y) -> "EmpiricalMahalanobisScorer":
        Z, y = self._prepare(Z), np.asarray(y)
        classes = np.unique(y)
        self.means_ = np.stack([Z[y == c].mean(0) for c in classes])
        centred = Z - self.means_[np.searchsorted(classes, y)]
        self.precision_ = np.linalg.pinv(centred.T @ centred / len(centred))
        return self


def scorer_settings():
    """(scorer, setting, instance) triples; a setting that equals a library default
    shares the instance of that default, so that it is fitted and applied once."""
    maha, knn = MahalanobisScorer(), KNNScorer()
    if (maha.normalise, knn.normalise, knn.k) != (False, True, KNN_K) or KNN_K not in KNN_KS:
        raise RuntimeError("the library defaults differ from the settings named here")
    out = [("maha", "normalise=False", maha),
           ("maha", "normalise=True", MahalanobisScorer(normalise=True)),
           ("knn", "normalise=False", KNNScorer(normalise=False)),
           ("knn", "normalise=True", knn)]
    out += [("knn", f"k={k}", knn if k == KNN_K else KNNScorer(k=k)) for k in KNN_KS]
    out += [("maha", "covariance=ledoit_wolf", maha),
            ("maha", "covariance=empirical", EmpiricalMahalanobisScorer())]
    return out


def run_job(model: str, task: str, emb_dir: str) -> list[dict]:
    d = config.load(model, task, emb_dir)
    split = stored_split(d)
    sets = shifted_sets(model, task, d, emb_dir)
    scores = {}         # id(scorer instance) -> (test scores, [scores of each OOD set])
    rows = []
    for name, setting, scorer in scorer_settings():
        if id(scorer) not in scores:
            scorer.fit(split["Z_train"], split["y_train"])
            scores[id(scorer)] = (scorer.score(split["Z_test"]),
                                  [scorer.score(Z) for _, _, Z in sets])
        s_id, s_ood = scores[id(scorer)]
        rows += [{"model": model, "task": task, "tier": tier, "condition": condition,
                  "scorer": name, "setting": setting, "auroc": auroc(s_id, s)}
                 for (tier, condition, _), s in zip(sets, s_ood)]
    return rows


def summary(models, tasks, rows) -> list[str]:
    """Lines of scorer_sensitivity.md."""
    n_cond = {tier: len({(r["task"], r["condition"]) for r in rows if r["tier"] == tier})
              for tier in SHIFT_TIERS}
    tiers = [tier for tier in SHIFT_TIERS if n_cond[tier]]
    L = ["# Sensitivity of the detection AUROC to scorer settings", "",
         "Stored split; scorers fitted on the training split; in-distribution set: the test "
         "split; the out-of-distribution set is the positive class.", "",
         f"- Tasks: {', '.join(task_name(t) for t in tasks)}.",
         "- far: the out-of-distribution set is the test split of the donor task "
         f"({n_cond['far']} conditions per encoder).",
         "- blur: the out-of-distribution set is the Gaussian-blurred test split, sigma in "
         f"{{{', '.join(str(s) for s in config.BLUR_SIGMAS)}}}, categorical tasks "
         f"({n_cond['blur']} conditions per encoder).",
         "- Each cell is the mean AUROC of one encoder over the conditions of the tier.",
         "- normalise=True: embeddings are L2-normalised before the scorer; "
         "normalise=False: raw embeddings.",
         "- k: the kNN score is the distance to the k-th nearest training embedding.",
         "- covariance=ledoit_wolf: pooled within-class covariance with Ledoit-Wolf shrinkage; "
         "covariance=empirical: the same covariance without shrinkage (pseudo-inverse)."]
    for title, scorer, settings in SECTIONS:
        header = ["Encoder"] + [f"{tier}, {s}" for tier in tiers for s in settings]
        body = []
        for m in models:
            row = [config.display(m)]
            for tier in tiers:
                for s in settings:
                    vals = [r["auroc"] for r in rows
                            if (r["model"], r["tier"], r["scorer"], r["setting"])
                            == (m, tier, scorer, s)]
                    row.append(fmt(float(np.mean(vals))))
            body.append(row)
        L += ["", f"## {title}", ""] + md_table(header, body)
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

    write_csv(os.path.join(args.out_dir, "scorer_sensitivity.csv"), FIELDS, rows)
    lines = summary(models, tasks, rows)
    write_lines(os.path.join(args.out_dir, "scorer_sensitivity.md"), lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
