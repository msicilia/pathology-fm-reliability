"""
splits.py
=========
Train/test splitting of patch sets and of the cached embeddings.

Every split is stratified by class. Tasks whose patches come from identifiable
patients or slides carry a group id per patch (BreaKHis and SICAPv2: patient;
PatchCamelyon and PANDA: slide); for these one fifth of the groups is held out
and all patches of a group stay on the same side. For the other tasks one fifth
of the patches is held out. PANDA patches also carry the data provider of
their slide, and the split is stratified by grade and provider jointly.

The exported .npz stores one such split (seed 42). Because the embeddings are
frozen, the analyses pool it and draw further splits with `resplit`.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5     # the test side is one fold


def split_indices(y, groups=None, strata=None, seed: int = 0):
    """Return (train_idx, test_idx) for one stratified, group-respecting split.

    `strata` defaults to the class labels. Without `groups` the patches are
    stratified directly. With `groups`, whole groups are assigned to folds,
    stratified by the most frequent stratum among each group's patches, so all
    rows sharing a group id fall on one side.
    """
    y = np.asarray(y)
    strata = y if strata is None else np.asarray(strata)
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    if groups is None:
        return next(folds.split(y, strata))
    ids, group_idx = np.unique(np.asarray(groups), return_inverse=True)
    values, strata_idx = np.unique(strata, return_inverse=True)
    counts = np.zeros((len(ids), len(values)), dtype=int)
    np.add.at(counts, (group_idx, strata_idx), 1)
    _, test_groups = next(folds.split(ids, counts.argmax(1)))
    test = np.isin(group_idx, test_groups)
    return np.flatnonzero(~test), np.flatnonzero(test)


def joint_strata(y, sites=None):
    """Stratification key: the class, or class and site jointly when sites are given."""
    if sites is None:
        return np.asarray(y)
    _, site_code = np.unique(np.asarray(sites), return_inverse=True)
    return np.asarray(y) * (site_code.max() + 1) + site_code


def pool(d) -> dict:
    """Concatenate the stored train and test parts of one .npz into a single pool."""
    out = {"Z": np.concatenate([d["Z_train"], d["Z_test"]]),
           "y": np.concatenate([d["y_train"], d["y_test"]]),
           "paths": np.concatenate([d["paths_train"], d["paths_test"]])}
    for key in ("groups", "sites"):
        out[key] = (np.concatenate([d[f"{key}_train"], d[f"{key}_test"]])
                    if f"{key}_train" in d else None)
    return out


def resplit(d, seed: int = 0) -> dict:
    """Draw a train/test split of the pooled embeddings for one seed.

    Returns a dict with the keys of the .npz (`Z_train`, `y_train`, `paths_train`,
    ... and `groups_*`, `sites_*`, which are None for tasks without them).
    """
    p = pool(d)
    tr, te = split_indices(p["y"], p["groups"], joint_strata(p["y"], p["sites"]), seed)
    out = {"classes": d["classes"]}
    for name, idx in (("train", tr), ("test", te)):
        out[f"Z_{name}"] = p["Z"][idx]
        out[f"y_{name}"] = p["y"][idx]
        out[f"paths_{name}"] = p["paths"][idx]
        for key in ("groups", "sites"):
            out[f"{key}_{name}"] = None if p[key] is None else p[key][idx]
    return out


def stored_split(d) -> dict:
    """The split stored in the .npz, in the same form as `resplit` returns."""
    out = {"classes": d["classes"]}
    for name in ("train", "test"):
        for key in ("Z", "y", "paths"):
            out[f"{key}_{name}"] = d[f"{key}_{name}"]
        for key in ("groups", "sites"):
            out[f"{key}_{name}"] = d[f"{key}_{name}"] if f"{key}_{name}" in d else None
    return out


def slide_ids(task: str, paths) -> np.ndarray:
    """Slide id of each patch of a whole-slide task, parsed from its file name."""
    if task == "panda":          # <image_id>_<k>.png
        return np.array([str(p).rsplit("_", 1)[0] for p in paths])
    if task == "sicap":          # <slide_id>_Block_Region_..._.jpg
        return np.array([str(p).split("_Block")[0] for p in paths])
    raise ValueError(f"{task} has no slide ids")
