"""
download_panda.py
=================
Download the selected PANDA whole-slide images from the Kaggle competition
"prostate-cancer-grade-assessment", one slide at a time, with retries. Slides
already present are skipped, so the script can be re-run until it reports no
failures.

Requires Kaggle API credentials and acceptance of the competition rules.

Run from the repository root:
    uv run python src/data/download_panda.py \
        --csv data/panda_raw/slides.csv --out data/panda_raw/train_images
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

import pandas as pd

COMPETITION = "prostate-cancer-grade-assessment"


def is_tiff(path: str) -> bool:
    """True if the file exists and starts with a TIFF byte-order mark."""
    try:
        with open(path, "rb") as f:
            return f.read(2) in (b"II", b"MM")
    except OSError:
        return False


def unwrap_if_zip(path: str, out_dir: str) -> bool:
    """Kaggle may deliver the slide zipped, as `<id>.tiff.zip` or `<id>.tiff`; extract it."""
    archive = path + ".zip"
    if not os.path.exists(archive):
        try:
            with open(path, "rb") as f:
                zipped = f.read(2) == b"PK"
        except OSError:
            return False
        if zipped:
            shutil.move(path, archive)
    if os.path.exists(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        os.remove(archive)
    return is_tiff(path)


def fetch(image_id: str, out_dir: str, tries: int = 3) -> str:
    path = os.path.join(out_dir, f"{image_id}.tiff")
    if is_tiff(path):
        return "cached"
    for attempt in range(tries):
        subprocess.run(
            [sys.executable, "-m", "kaggle.cli", "competitions", "download", COMPETITION,
             "-f", f"train_images/{image_id}.tiff", "-p", out_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if unwrap_if_zip(path, out_dir):
            return "ok"
        time.sleep(1.5 * (attempt + 1))
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/panda_raw/slides.csv")
    ap.add_argument("--out", default="data/panda_raw/train_images")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids = pd.read_csv(args.csv)["image_id"].tolist()
    counts = {"ok": 0, "cached": 0, "fail": 0}
    for i, image_id in enumerate(ids, 1):
        status = fetch(image_id, args.out)
        counts[status] += 1
        if status != "cached":
            time.sleep(0.25)   # rate limit
        if i % 25 == 0 or status == "fail":
            print(f"[{i}/{len(ids)}] {counts}", flush=True)
    print(f"done: {counts}")
    if counts["fail"]:
        sys.exit(f"{counts['fail']} slides failed; re-run to retry")


if __name__ == "__main__":
    main()
