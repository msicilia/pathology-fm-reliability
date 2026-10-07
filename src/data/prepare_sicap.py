"""
prepare_sicap.py
================
Arrange the labelled SICAPv2 patches as class folders of symbolic links:

    data/sicap/{NC,G3,G4,G5}/<slide>_Block_Region_..._.jpg
    data/sicap/slides.csv        slide_id, patient_id

Patch labels come from the release's partition spreadsheets (Train.xlsx and
Test.xlsx, which together list every labelled patch). The primary Gleason
pattern of a patch is its class; the cribriform flag (G4C) accompanies G4 and is
not a separate class. `slides.csv` copies the slide-to-patient map of
wsi_labels.xlsx so that export_embeddings.py can split by patient.

Input: the extracted SICAPv2 release (https://data.mendeley.com/datasets/9xxm58dvs3).

Run from the repository root:
    uv run python src/data/prepare_sicap.py --src data/sicap_orig/SICAPv2
"""
from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd

CLASSES = ["NC", "G3", "G4", "G5"]     # ordinal: benign, then Gleason pattern 3 to 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="extracted SICAPv2 folder")
    ap.add_argument("--out", default="data/sicap")
    args = ap.parse_args()

    part = os.path.join(args.src, "partition", "Test")
    labels = pd.concat([pd.read_excel(os.path.join(part, f)) for f in ("Train.xlsx", "Test.xlsx")])
    if labels["image_name"].duplicated().any():
        raise SystemExit("a patch is listed in both Train.xlsx and Test.xlsx")
    if not (labels[CLASSES].sum(axis=1) == 1).all():
        raise SystemExit("a patch does not have exactly one primary label")
    labels["class"] = labels[CLASSES].idxmax(axis=1)

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for name in CLASSES:
        os.makedirs(os.path.join(args.out, name))
    images = os.path.abspath(os.path.join(args.src, "images"))
    for image_name, name in zip(labels["image_name"], labels["class"]):
        target = os.path.join(images, image_name)
        if not os.path.exists(target):
            raise SystemExit(f"missing image {target}")
        link_dir = os.path.join(args.out, name)
        os.symlink(os.path.relpath(target, os.path.abspath(link_dir)),
                   os.path.join(link_dir, image_name))

    wsi = pd.read_excel(os.path.join(args.src, "wsi_labels.xlsx"))
    wsi[["slide_id", "patient_id"]].to_csv(os.path.join(args.out, "slides.csv"), index=False)
    print(labels["class"].value_counts().reindex(CLASSES))
    print(f"wrote {len(labels)} links, {wsi['slide_id'].nunique()} slides, "
          f"{wsi['patient_id'].nunique()} patients to {args.out}")


if __name__ == "__main__":
    main()
