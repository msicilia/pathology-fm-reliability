"""
tile_panda.py
=============
Tile the selected PANDA prostate biopsies into 224x224 patches. Every patch
inherits the ISUP grade (0-5) of its slide. The output is a class-folder tree
read by the `panda` task of export_embeddings.py:

    <out>/<isup_grade>/<image_id>_<k>.png
    <out>/slides.csv        image_id, data_provider, isup_grade, n_tiles

Tiles are read from pyramid level 1 (4x downsample of the scan, about
1.8-2.0 um/px) on a non-overlapping grid. Tissue is located on the
lowest-resolution pyramid level: a pixel counts as tissue when it is neither
background (bright), nor scanner fill (black), nor green marker ink. A tile is
kept when at least `--tissue-frac` of its footprint is tissue; at most
`--max-per-slide` tiles are drawn at random per slide.

Run from the repository root:
    uv run python src/data/tile_panda.py \
        --csv data/panda_raw/slides.csv --images data/panda_raw/train_images \
        --out data/panda_tiles
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import tiffslide

LEVEL = 1          # pyramid level the tiles are read from
WHITE = 210        # mean intensity above which a pixel is background
BLACK = 40         # mean intensity below which a pixel is scanner fill
INK_MARGIN = 15    # green channel exceeding red and blue by this much = marker ink


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean tissue mask of an RGB uint8 array (H, W, 3)."""
    rgb = rgb.astype(np.int16)
    gray = rgb.mean(2)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ink = (g - np.maximum(r, b)) > INK_MARGIN
    return (gray < WHITE) & (gray > BLACK) & ~ink


def candidate_tiles(slide, tile: int, tissue_frac: float) -> list[tuple[int, int]]:
    """Level-0 (x, y) origins of grid tiles whose footprint is mostly tissue."""
    low = slide.level_count - 1
    thumb = np.asarray(slide.read_region((0, 0), low, slide.level_dimensions[low]).convert("RGB"))
    mask = tissue_mask(thumb)
    span = int(round(tile * slide.level_downsamples[LEVEL]))        # tile side in level-0 px
    scale = slide.level_downsamples[low]
    width, height = slide.level_dimensions[0]
    out = []
    for y0 in range(0, height - span + 1, span):
        for x0 in range(0, width - span + 1, span):
            ys, xs = int(y0 / scale), int(x0 / scale)
            block = mask[ys:int((y0 + span) / scale), xs:int((x0 + span) / scale)]
            if block.size and block.mean() >= tissue_frac:
                out.append((x0, y0))
    return out


def tile_slide(path: str, isup: int, out_dir: str, tile: int, max_per_slide: int,
               tissue_frac: float, seed: int) -> int:
    """Write the tiles of one slide; return how many were written."""
    slide = tiffslide.TiffSlide(path)
    cands = candidate_tiles(slide, tile, tissue_frac)
    if len(cands) > max_per_slide:
        rng = np.random.default_rng(seed)
        cands = [cands[i] for i in sorted(rng.choice(len(cands), max_per_slide, replace=False))]
    image_id = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(out_dir, str(isup))
    os.makedirs(dst, exist_ok=True)
    n = 0
    for x0, y0 in cands:
        img = slide.read_region((x0, y0), LEVEL, (tile, tile)).convert("RGB")
        # the thumbnail is coarse: re-check tissue content at tile resolution
        if tissue_mask(np.asarray(img)).mean() < tissue_frac:
            continue
        img.save(os.path.join(dst, f"{image_id}_{n}.png"))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/panda_raw/slides.csv")
    ap.add_argument("--images", default="data/panda_raw/train_images")
    ap.add_argument("--out", default="data/panda_tiles")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--max-per-slide", type=int, default=100)
    ap.add_argument("--tissue-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    slides = pd.read_csv(args.csv)
    missing = [i for i in slides["image_id"]
               if not os.path.exists(os.path.join(args.images, f"{i}.tiff"))]
    if missing:
        raise SystemExit(f"{len(missing)} slides are missing under {args.images}, "
                         f"e.g. {missing[0]}; run download_panda.py first")
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)        # never mix tiles from different settings
    os.makedirs(args.out)

    counts = []
    for i, row in enumerate(slides.itertuples(index=False), 1):
        n = tile_slide(os.path.join(args.images, f"{row.image_id}.tiff"), int(row.isup_grade),
                       args.out, args.tile, args.max_per_slide, args.tissue_frac, args.seed)
        counts.append(n)
        if i % 50 == 0:
            print(f"[{i}/{len(slides)}] {sum(counts)} tiles", flush=True)
    slides = slides.assign(n_tiles=counts)
    slides[["image_id", "data_provider", "isup_grade", "n_tiles"]].to_csv(
        os.path.join(args.out, "slides.csv"), index=False)
    print(slides.groupby(["data_provider", "isup_grade"])["n_tiles"].agg(["sum", "median", "min"]))
    print(f"wrote {sum(counts)} tiles from {len(slides)} slides to {args.out}")


if __name__ == "__main__":
    main()
