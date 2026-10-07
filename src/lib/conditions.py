"""
conditions.py
=============
Out-of-distribution conditions of a task: tiers, condition names and sets.

    tier       condition          out-of-distribution set
    far        far_<donor>        test split of the donor task config.FAR_OOD[task]
    blur       blur_s<sigma>      the test split under Gaussian blur, one condition
                                  per sigma of config.BLUR_SIGMAS (categorical tasks)
    near       holdout_<class>    test patches of one class that is held out (tasks
                                  with three or more classes)
    artefact   artefact           test patches of the classes of
                                  config.ARTEFACT_CLASSES, held out jointly (NCT-CRC-HE)

In the far and blur tiers the in-distribution set is the test split and nothing is
removed from the training split. In the near and artefact tiers the held-out
classes are removed from the training and test splits; the in-distribution set is
the remaining test patches.
"""
from __future__ import annotations

import numpy as np

from src.lib import config

SHIFT_TIERS = ("far", "blur")           # conditions built by `shifted_sets`
CLASS_TIERS = ("near", "artefact")      # conditions built by `held_out_conditions`
TIERS = SHIFT_TIERS + CLASS_TIERS
ARTEFACT = "artefact"                   # name of the artefact tier and of its condition
ARTEFACT_TASK = "nct"
NEAR_MIN_CLASSES = 3


def shifted_sets(model: str, task: str, d, emb_dir: str = config.EMB_DIR):
    """(tier, condition, embeddings) of the far and blur sets of one (model, task).

    `d` is the loaded .npz of the task.
    """
    donor = config.FAR_OOD[task]
    sets = [("far", f"far_{donor}", config.load(model, donor, emb_dir)["Z_test"])]
    if task in config.CATEGORICAL:
        n_test = len(d["y_test"])
        for sigma in config.BLUR_SIGMAS:
            z_blur = d[f"Z_test_blur_s{sigma}"]
            if len(z_blur) != n_test:
                raise RuntimeError(f"{model}/{task}: blurred test set has a different size")
            sets.append(("blur", f"blur_s{sigma}", z_blur))
    return sets


def held_out_conditions(task: str, classes):
    """(tier, condition, held-out labels) of the near and artefact conditions of one task."""
    classes = [str(c) for c in classes]
    out = []
    if len(classes) >= NEAR_MIN_CLASSES:
        out += [("near", f"holdout_{name}", [label]) for label, name in enumerate(classes)]
    if task == ARTEFACT_TASK:
        out.append((ARTEFACT, ARTEFACT, [classes.index(c) for c in config.ARTEFACT_CLASSES]))
    return out


def hold_out(split: dict, labels) -> tuple[dict, np.ndarray]:
    """Remove the classes with the given labels from a split.

    Returns the reduced split, with the remaining labels renumbered 0 .. K-1 in
    their original order, and the test embeddings of the removed classes.
    """
    n_classes = len(split["classes"])
    drop = np.unique(labels)
    if len(drop) != len(labels) or not np.isin(drop, np.arange(n_classes)).all():
        raise ValueError(f"held-out labels {list(labels)} are not distinct labels of the split")
    keep = np.setdiff1d(np.arange(n_classes), drop)
    out = {"classes": np.asarray(split["classes"])[keep]}
    for part in ("train", "test"):
        m = np.isin(split[f"y_{part}"], keep)
        out[f"Z_{part}"] = split[f"Z_{part}"][m]
        out[f"y_{part}"] = np.searchsorted(keep, split[f"y_{part}"][m])
        for key in ("paths", "groups", "sites"):
            v = split[f"{key}_{part}"]
            out[f"{key}_{part}"] = None if v is None else v[m]
    z_ood = split["Z_test"][np.isin(split["y_test"], drop)]
    n_removed = len(split["y_train"]) - len(out["y_train"])
    if not (len(out["y_train"]) and len(out["y_test"]) and len(z_ood) and n_removed):
        raise RuntimeError("empty in-distribution, out-of-distribution or held-out set")
    return out, z_ood
