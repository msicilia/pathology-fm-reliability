"""
select_panda_slides.py
======================
Select the PANDA slides used in the study from the challenge's `train.csv`.

The selection is balanced over data provider and ISUP grade: the same number of
slides is drawn at random from each (provider, grade) cell, so that grade cannot
be predicted from the provider (scanner, stain, annotation protocol).

Run from the repository root:
    uv run python src/data/select_panda_slides.py \
        --train-csv data/panda_raw/train.csv --out data/panda_raw/slides.csv
"""
from __future__ import annotations

import argparse

import pandas as pd


def select(train: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    """Draw `per_cell` slides from every (data_provider, isup_grade) cell."""
    train = train.sort_values("image_id")
    picked = train.groupby(["data_provider", "isup_grade"], sort=True).sample(
        n=per_cell, random_state=seed)
    return picked.sort_values(["data_provider", "isup_grade", "image_id"])[
        ["image_id", "data_provider", "isup_grade", "gleason_score"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/panda_raw/train.csv")
    ap.add_argument("--out", default="data/panda_raw/slides.csv")
    ap.add_argument("--per-cell", type=int, default=60,
                    help="slides per (provider, ISUP grade) cell")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    slides = select(pd.read_csv(args.train_csv), args.per_cell, args.seed)
    slides.to_csv(args.out, index=False)
    print(pd.crosstab(slides["data_provider"], slides["isup_grade"]))
    print(f"wrote {args.out} ({len(slides)} slides)")


if __name__ == "__main__":
    main()
