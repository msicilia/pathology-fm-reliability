"""
config.py
=========
Names and constants shared by the export, analysis and figure scripts.
"""
from __future__ import annotations

import glob
import os

import numpy as np

EMB_DIR = "./embeddings"
RES_DIR = "./results"
FIG_SUBDIR = "figures"              # figures are written to <out-dir>/figures

# encoder key -> display name; pathology encoders first, general-vision baselines last
ENCODERS = {
    "lunit": "Lunit",
    "ctranspath": "CTransPath",
    "conch": "CONCH",
    "hibou_b": "Hibou-B",
    "phikon": "Phikon",
    "phikon_v2": "Phikon-v2",
    "hibou_l": "Hibou-L",
    "virchow2": "Virchow2",
    "uni": "UNI2",
    "hoptimus": "H-optimus-0",
    "midnight": "Midnight",
    "dinov2_nat": "DINOv2-nat",
    "imagenet_vit": "ImageNet-ViT",
}
BASELINES = ("dinov2_nat", "imagenet_vit")

CATEGORICAL = ("nct", "lung", "lc_colon", "breakhis", "pcam")
# categorical tasks whose probe accuracy enters the mean classification accuracy
ACCURACY_TASKS = ("nct", "lung", "breakhis", "pcam")
MEAN_TASK = "accuracy_tasks"        # task key of the mean accuracy over ACCURACY_TASKS
ORDINAL = ("sicap", "panda")
TASKS = CATEGORICAL + ORDINAL
# tasks whose patches carry a patient or slide id
GROUPED = ("breakhis", "pcam", "sicap", "panda")

# far-OOD: in-distribution task -> task whose test patches are the OOD set
FAR_OOD = {
    "nct": "breakhis",
    "lung": "nct",
    "lc_colon": "breakhis",
    "breakhis": "lung",
    "pcam": "nct",
    "panda": "nct",
    "sicap": "nct",
}

# NCT-CRC-HE classes that are not diagnostic tissue (artefact-rejection setting)
ARTEFACT_CLASSES = ("BACK", "DEB", "ADI")

BLUR_SIGMAS = (1.0, 2.0, 3.0)
EXPORT_SEED = 42                    # seed of the split stored in the .npz
SPLIT_SEEDS = (0, 1, 2, 3, 4)       # seeds of the splits drawn by the analyses
N_BOOT = 300                        # bootstrap resamples per split


def display(model: str) -> str:
    return ENCODERS.get(model, model)


def fig_dir(out_dir: str = RES_DIR) -> str:
    """Directory of the figures that belong to the results in `out_dir`."""
    return os.path.join(out_dir, FIG_SUBDIR)


def npz_path(model: str, task: str, emb_dir: str = EMB_DIR) -> str:
    return os.path.join(emb_dir, f"{model}__{task}.npz")


def load(model: str, task: str, emb_dir: str = EMB_DIR):
    """Load the cached embeddings of one (model, task); fails if they are missing."""
    return np.load(npz_path(model, task, emb_dir), allow_pickle=True)


def list_models(emb_dir: str = EMB_DIR) -> list[str]:
    """Encoders that have embeddings for every task, in the order of `ENCODERS`."""
    found = {os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(emb_dir, "*.npz"))}
    unknown = found - set(ENCODERS)
    if unknown:
        raise RuntimeError(f"embeddings of unknown encoders: {sorted(unknown)}")
    models = [m for m in ENCODERS if m in found]
    for m in models:
        missing = [t for t in TASKS if not os.path.exists(npz_path(m, t, emb_dir))]
        if missing:
            raise RuntimeError(f"{m} has no embeddings for {missing}")
    return models
