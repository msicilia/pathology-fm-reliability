"""
prepare_pcam.py
===============
Write the PatchCamelyon test split (32,768 lymph-node H&E patches, 96x96,
tumour vs normal) as class folders of PNG files:

    data/pcam/{normal,tumour}/<wsi>__<index>.png

`<wsi>` is the Camelyon16 slide the patch was cut from, taken from the official
metadata file, and `<index>` is the row in the HDF5 arrays. The slide id lets
export_embeddings.py keep all patches of one slide on the same side of a split.

Inputs, under data/pcam_raw/pcam/ (https://zenodo.org/records/2546921):
    camelyonpatch_level_2_split_test_x.h5
    camelyonpatch_level_2_split_test_y.h5
    camelyonpatch_level_2_split_test_meta.csv

Run from the repository root:
    uv run python src/data/prepare_pcam.py
"""
from __future__ import annotations

import argparse
import os
import shutil

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CLASSES = {0: "normal", 1: "tumour"}
STEM = "camelyonpatch_level_2_split_test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/pcam_raw/pcam")
    ap.add_argument("--out", default="data/pcam")
    args = ap.parse_args()

    with h5py.File(os.path.join(args.raw, f"{STEM}_x.h5"), "r") as f:
        images = f["x"][:]                                    # (N, 96, 96, 3) uint8
    with h5py.File(os.path.join(args.raw, f"{STEM}_y.h5"), "r") as f:
        labels = np.asarray(f["y"]).reshape(-1).astype(int)
    meta = pd.read_csv(os.path.join(args.raw, f"{STEM}_meta.csv"))
    if not (len(images) == len(labels) == len(meta)):
        raise SystemExit("image, label and metadata files differ in length")
    if not np.array_equal(meta["center_tumor_patch"].to_numpy().astype(int), labels):
        raise SystemExit("metadata labels do not match the label file")

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for name in CLASSES.values():
        os.makedirs(os.path.join(args.out, name))
    for i in tqdm(range(len(labels)), desc="writing"):
        name = f"{meta['wsi'].iloc[i]}__{i:05d}.png"
        Image.fromarray(images[i]).save(os.path.join(args.out, CLASSES[labels[i]], name))
    print(f"wrote {len(labels)} patches from {meta['wsi'].nunique()} slides to {args.out}")


if __name__ == "__main__":
    main()
