"""
sicap_without_benign.py
=======================
Probe metrics on SICAPv2 with the benign class removed.

The benign patches of SICAPv2 come from few patients. For every encoder and
split seed (`config.SPLIT_SEEDS`) the benign class is removed from the training
and test splits, the probe is fitted on the remaining Gleason patterns and the
metrics of src/analysis/probe_metrics.py are computed on the remaining test
patches. The number of patients that contribute benign patches, in all and to
each test split, is reported with them.

Outputs:
    <out-dir>/sicap_without_benign.csv   one row per model and seed
    <out-dir>/sicap_without_benign.md    summary tables

Run from the repository root:
    uv run python src/analysis/sicap_without_benign.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed

from src.analysis.probe_metrics import random_ranking_excess_aurc
from src.lib import config
from src.lib.conditions import hold_out
from src.lib.evaluation import probe_predict, selective_metrics
from src.lib.probe import fit_probe
from src.lib.splits import pool, resplit
from src.lib.tables import fmt, md_table, write_csv, write_lines

TASK = "sicap"
BENIGN = "NC"
METRICS = ("accuracy", "qwk", "e_aurc", "e_aurc_random", "ece")
FIELDS = ("model", "seed", "n_train", "n_test", "benign_test_patients") + METRICS


def run_split(model: str, seed: int, emb_dir: str) -> dict:
    """Metrics of one (model, seed) with the benign class removed from both sides."""
    split = resplit(config.load(model, TASK, emb_dir), seed)
    benign = [str(c) for c in split["classes"]].index(BENIGN)
    benign_test = np.unique(split["groups_test"][split["y_test"] == benign])
    reduced, _ = hold_out(split, [benign])
    n_classes = len(reduced["classes"])
    probe = fit_probe(reduced["Z_train"], reduced["y_train"], groups=reduced["groups_train"])
    y = reduced["y_test"]
    pred, conf = probe_predict(probe, reduced["Z_test"], n_classes)
    metrics = selective_metrics(y, pred, conf, n_classes)
    metrics["e_aurc_random"] = random_ranking_excess_aurc(pred == y, 1.0 - conf)
    return {"model": model, "seed": seed, "n_train": len(reduced["y_train"]), "n_test": len(y),
            "benign_test_patients": len(benign_test), **{k: metrics[k] for k in METRICS}}


def benign_patients(model: str, emb_dir: str) -> tuple[int, int, int]:
    """Patients in all, patients with benign patches, and benign patches of SICAPv2."""
    d = config.load(model, TASK, emb_dir)
    p = pool(d)
    benign = p["y"] == [str(c) for c in d["classes"]].index(BENIGN)
    return len(np.unique(p["groups"])), len(np.unique(p["groups"][benign])), int(benign.sum())


def summary(rows, models, seeds, counts) -> list[str]:
    """Lines of sicap_without_benign.md."""
    n_patients, n_benign_patients, n_benign = counts
    per_split = sorted({r["benign_test_patients"] for r in rows})
    L = ["# SICAPv2 without the benign class", "",
         f"Splits: seeds {', '.join(str(s) for s in seeds)}. The benign class (NC) is removed "
         "from the training and test splits and the probe is fitted on the three Gleason "
         "patterns. Values are means over the splits, on the remaining test patches.", "",
         f"- Benign patches: {n_benign}, from {n_benign_patients} of the {n_patients} patients; "
         f"patients with benign patches in a test split: "
         f"{', '.join(str(n) for n in per_split)}.",
         "- QWK: quadratic weighted kappa over the three Gleason patterns.",
         "- E-AURC: area under the risk-coverage curve minus that of a perfect ranking at the "
         "same error rate; confidence is the maximum softmax probability.",
         "- E-AURC, random: expected E-AURC of a ranking that is independent of correctness.",
         "- ECE: top-label expected calibration error.", ""]
    header = ["Encoder", "accuracy", "QWK", "E-AURC", "E-AURC, random", "ECE"]
    L += md_table(header, [[config.display(m)]
                           + [fmt(np.mean([r[k] for r in rows if r["model"] == m]))
                              for k in METRICS] for m in models])
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--models", nargs="+", default=None, choices=list(config.ENCODERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SPLIT_SEEDS))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers")
    args = ap.parse_args()
    models = args.models or config.list_models(args.emb_dir)

    jobs = [(m, s) for m in models for s in args.seeds]
    rows = Parallel(n_jobs=args.jobs)(delayed(run_split)(m, s, args.emb_dir) for m, s in jobs)

    write_csv(os.path.join(args.out_dir, "sicap_without_benign.csv"), FIELDS, rows)
    lines = summary(rows, models, args.seeds, benign_patients(models[0], args.emb_dir))
    write_lines(os.path.join(args.out_dir, "sicap_without_benign.md"), lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
