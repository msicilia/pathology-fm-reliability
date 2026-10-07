"""
blur_accuracy.py
================
Accuracy of the linear probe on the blurred test split, on the stored split.

For every encoder and categorical task the probe is fitted on the training split
and its accuracy is computed on the test split and on the same test patches
under Gaussian blur, once per sigma of config.BLUR_SIGMAS. The probe is the one
fitted by the detection sweep (src/pipeline/ood_sweep.py), so the accuracy of
the unblurred test split equals that of probe_stored.csv.

Outputs:
    <out-dir>/blur_accuracy.csv   model, task, sigma, n_test, accuracy (sigma 0: unblurred)
    <out-dir>/blur_accuracy.md    summary tables

Run from the repository root:
    uv run python src/analysis/blur_accuracy.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed

from src.lib import config
from src.lib.names import task_name
from src.lib.probe import fit_probe
from src.lib.splits import stored_split
from src.lib.tables import fmt, md_table, write_csv, write_lines

FIELDS = ("model", "task", "sigma", "n_test", "accuracy")
UNBLURRED = 0.0


def run_task(model: str, task: str, emb_dir: str) -> list[dict]:
    """Accuracy on the test split and on each blurred copy of it, for one (model, task)."""
    d = config.load(model, task, emb_dir)
    split = stored_split(d)
    y = split["y_test"]
    probe = fit_probe(split["Z_train"], split["y_train"], groups=split["groups_train"])
    sets = [(UNBLURRED, split["Z_test"])]
    sets += [(sigma, d[f"Z_test_blur_s{sigma}"]) for sigma in config.BLUR_SIGMAS]
    rows = []
    for sigma, Z in sets:
        if len(Z) != len(y):
            raise RuntimeError(f"{model}/{task}: blurred test set has a different size")
        rows.append(dict(model=model, task=task, sigma=float(sigma), n_test=len(y),
                         accuracy=float((probe.predict(Z) == y).mean())))
    return rows


def accuracy_change(rows) -> dict:
    """(model, task, sigma) -> accuracy under blur minus accuracy of the unblurred test split."""
    acc = {(r["model"], r["task"], r["sigma"]): r["accuracy"] for r in rows}
    return {(m, t, s): a - acc[(m, t, UNBLURRED)] for (m, t, s), a in acc.items()
            if s != UNBLURRED}


def signed(value: float) -> str:
    """Fixed-point text of a change; a change that rounds to zero is written without a sign."""
    return fmt(round(value, 3) + 0.0)


def summary(rows, models, tasks) -> list[str]:
    """Lines of blur_accuracy.md."""
    acc = {(r["model"], r["task"], r["sigma"]): r["accuracy"] for r in rows}
    change = accuracy_change(rows)
    sigmas = [float(s) for s in config.BLUR_SIGMAS]
    names = ", ".join(task_name(t) for t in tasks)
    L = ["# Accuracy of the linear probe under Gaussian blur", "",
         "Stored split. The probe is fitted on the training split and evaluated on the test "
         "split and on the same test patches under Gaussian blur. Sigma is in pixels of a "
         "224 x 224 input.", "",
         "- Accuracy: fraction of test patches classified correctly.",
         "- Change: accuracy under blur minus accuracy of the unblurred test split.", "",
         f"## Mean over the {len(tasks)} tasks ({names})", ""]
    header = (["Encoder", "accuracy, unblurred"]
              + [f"change, sigma={s:g}" for s in sigmas])
    L += md_table(header, [[config.display(m),
                            fmt(np.mean([acc[(m, t, UNBLURRED)] for t in tasks]))]
                           + [signed(np.mean([change[(m, t, s)] for t in tasks]))
                              for s in sigmas]
                           for m in models])
    for task in tasks:
        L += ["", f"## {task_name(task)}", ""]
        L += md_table(header, [[config.display(m), fmt(acc[(m, task, UNBLURRED)])]
                               + [signed(change[(m, task, s)]) for s in sigmas]
                               for m in models])
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--models", nargs="+", default=None, choices=list(config.ENCODERS))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel (model, task) jobs")
    args = ap.parse_args()
    models = args.models or config.list_models(args.emb_dir)
    tasks = list(config.CATEGORICAL)

    jobs = [(m, t) for m in models for t in tasks]
    results = Parallel(n_jobs=args.jobs)(delayed(run_task)(m, t, args.emb_dir) for m, t in jobs)
    rows = [row for task_rows in results for row in task_rows]

    write_csv(os.path.join(args.out_dir, "blur_accuracy.csv"), FIELDS, rows)
    lines = summary(rows, models, tasks)
    write_lines(os.path.join(args.out_dir, "blur_accuracy.md"), lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
