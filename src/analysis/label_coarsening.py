"""
label_coarsening.py
===================
Within-cohort control on SICAPv2: the probe under region-level patch labels and
under labels coarsened to the slide.

For every encoder and every patient-grouped split seed, three arms are run on
the same patches:

* patch        the probe is trained and evaluated on the region-level patch
               labels;
* slide        every patch is relabelled with the most severe grade among the
               patches of its slide; the probe is trained on the relabelled
               training patches and evaluated against the relabelled test
               labels and against the region-level test labels;
* slide_abmil  the gated attention aggregator of slide_aggregation.py is
               trained on slide bags with the slide label and evaluated on the
               test slides.

The slide label is the maximum over all exported patches of the slide, training
and test together. A patient-grouped split keeps every slide on one side.

Writes label_coarsening.csv and label_coarsening.md under the output directory.

Run from the repository root:
    uv run python src/analysis/label_coarsening.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from src.analysis.slide_aggregation import (
    ATTENTION_DIM,
    EMBED_DIM,
    EPOCHS,
    LEARNING_RATE,
    METRICS,
    WEIGHT_DECAY,
    fit_predict_mil,
    make_bags,
)
from src.lib import config
from src.lib.evaluation import mean_sd, probe_predict, selective_metrics
from src.lib.probe import fit_probe, selected_c
from src.lib.scoring import ECE_BINS
from src.lib.splits import pool, resplit, slide_ids
from src.lib.tables import write_lines

TASK = "sicap"
# (arm, evaluation) in the order of the output tables
ROWS = (("patch", "patch_labels"), ("slide", "slide_labels"), ("slide", "patch_labels"),
        ("slide_abmil", "slide_labels"))


def coarsen(slides, y) -> np.ndarray:
    """Label of each patch after relabelling: the maximum label within its slide."""
    _, inverse = np.unique(np.asarray(slides), return_inverse=True)
    top = np.zeros(inverse.max() + 1, dtype=np.asarray(y).dtype)
    np.maximum.at(top, inverse, y)
    return top[inverse]


def slide_grades(d) -> dict:
    """Slide id -> most severe grade among all exported patches of the slide."""
    p = pool(d)
    slides = slide_ids(TASK, p["paths"])
    return dict(zip(slides.tolist(), coarsen(slides, p["y"]).tolist()))


def relabelling_counts(d) -> np.ndarray:
    """Counts of exported patches by region-level label (rows) and slide label (columns)."""
    p = pool(d)
    coarse = coarsen(slide_ids(TASK, p["paths"]), p["y"])
    n_classes = len(d["classes"])
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (p["y"], coarse), 1)
    return counts


def run_split(model: str, seed: int, emb_dir: str, device: str) -> list[dict]:
    """Rows of label_coarsening.csv for one encoder and one split."""
    torch.set_num_threads(1)
    d = config.load(model, TASK, emb_dir)
    grade = slide_grades(d)
    s = resplit(d, seed)
    n_classes = len(s["classes"])
    Z_train, Z_test = s["Z_train"], s["Z_test"]
    y_train, y_test = s["y_train"], s["y_test"]
    slides_train = slide_ids(TASK, s["paths_train"])
    slides_test = slide_ids(TASK, s["paths_test"])
    assert not set(slides_train) & set(slides_test), "a slide is on both sides of the split"
    coarse_train = np.array([grade[g] for g in slides_train])
    coarse_test = np.array([grade[g] for g in slides_test])
    assert (coarse_train >= y_train).all() and (coarse_test >= y_test).all()

    results = {}    # (arm, evaluation) -> (metrics, selected C, unit, n test)

    probe = fit_probe(Z_train, y_train, groups=s["groups_train"])
    pred, conf = probe_predict(probe, Z_test, n_classes)
    results["patch", "patch_labels"] = (selective_metrics(y_test, pred, conf, n_classes),
                                        selected_c(probe), "patch", len(y_test))

    probe = fit_probe(Z_train, coarse_train, groups=s["groups_train"])
    pred, conf = probe_predict(probe, Z_test, n_classes)
    for evaluation, target in (("slide_labels", coarse_test), ("patch_labels", y_test)):
        results["slide", evaluation] = (selective_metrics(target, pred, conf, n_classes),
                                        selected_c(probe), "patch", len(target))

    _, index_train, bag_train = make_bags(slides_train, coarse_train)
    _, index_test, bag_test = make_bags(slides_test, coarse_test)
    pred, conf = fit_predict_mil("abmil", Z_train, index_train, bag_train, Z_test, index_test,
                                 n_classes, seed, device)
    results["slide_abmil", "slide_labels"] = (
        selective_metrics(bag_test, pred, conf, n_classes), float("nan"), "slide", len(bag_test))

    rows = []
    for arm, evaluation in ROWS:
        metrics, c, unit, n_test = results[arm, evaluation]
        rows.append({"model": model, "seed": seed, "arm": arm, "evaluation": evaluation,
                     "unit": unit, "n_test": n_test, **{k: metrics[k] for k in METRICS},
                     "selected_c": c})
    return rows


def write_markdown(df: pd.DataFrame, counts: np.ndarray, classes, n_slides: int, models, seeds,
                   device: str, path: str) -> None:
    classes = [str(c) for c in classes]
    changed = counts.sum() - np.trace(counts)
    lines = [
        "# Label coarsening on SICAPv2",
        "",
        "Arms, on the same patient-grouped splits "
        f"(seeds {list(seeds)}):",
        "",
        "- patch: probe trained and evaluated on the region-level patch labels.",
        "- slide: every patch relabelled with the most severe grade among the patches of its "
        "slide (maximum over all exported patches of the slide); probe trained on the "
        "relabelled training patches.",
        "- slide_abmil: gated attention aggregator (Ilse et al., 2018) trained on slide bags "
        f"with the slide label and evaluated on the test slides; projection to {EMBED_DIM} "
        f"units, attention width {ATTENTION_DIM}, Adam, learning rate {LEARNING_RATE:g}, "
        f"weight decay {WEIGHT_DECAY:g}, {EPOCHS} epochs, one bag per step, embeddings "
        "standardised with the training patches, `torch.manual_seed` set to the split seed, "
        f"weights of the final epoch. Device of this run: {device}. Results obtained on GPU "
        "backends may differ in the last digits.",
        "",
        "## Relabelled patches",
        "",
        f"All exported patches ({int(counts.sum())} patches, {n_slides} slides). Rows: "
        "region-level label. Slide-label columns: number of patches of that row whose slide "
        "label is the column grade. Relabelled: patches whose slide label differs from "
        "their region-level label. Fraction: relabelled / patches.",
        "",
        "| Region-level label | Patches | " + " | ".join(f"Slide label {c}" for c in classes)
        + " | Relabelled | Fraction |",
        "|---|---|" + "---|" * len(classes) + "---|---|",
    ]
    for k, name in enumerate(classes):
        n = counts[k].sum()
        moved = n - counts[k, k]
        lines.append(f"| {name} | {n} | " + " | ".join(str(v) for v in counts[k])
                     + f" | {moved} | {moved / n:.4f} |")
    lines.append(f"| all | {counts.sum()} | " + " | ".join(str(v) for v in counts.sum(0))
                 + f" | {changed} | {changed / counts.sum():.4f} |")
    lines += [
        "",
        "## Metrics",
        "",
        "Mean +/- sample standard deviation (ddof = 1) over the splits.",
        "",
        "- Arm / evaluation: the arm and the test labels the predictions are compared with "
        "(patch_labels = region-level labels; slide_labels = labels after relabelling).",
        "- Unit: what one prediction refers to (patch or slide). Test n: mean number of test "
        "units per split.",
        "- Accuracy: fraction of correct predictions. QWK: quadratic weighted kappa.",
        "- AURC: area under the risk-coverage curve, abstaining on the least confident "
        "predictions first (confidence = maximum softmax probability). E-AURC: AURC minus "
        "the AURC of a perfect ranking at the same error rate.",
        f"- ECE: top-label expected calibration error, {ECE_BINS} equal-width bins.",
        "- Selected C: geometric mean over the splits of the inverse regularisation strength "
        "selected by cross-validation.",
        "",
        "| Encoder | Arm / evaluation | Unit | Test n | Accuracy | QWK | AURC | E-AURC | ECE "
        "| Selected C |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        for arm, evaluation in ROWS:
            sub = df[(df["model"] == m) & (df["arm"] == arm) & (df["evaluation"] == evaluation)]
            c = sub["selected_c"]
            c_text = "n/a" if c.isna().all() else f"{np.exp(np.log(c).mean()):.3g}"
            lines.append(f"| {config.display(m)} | {arm} / {evaluation} | {sub['unit'].iloc[0]} "
                         f"| {sub['n_test'].mean():.1f} | "
                         + " | ".join(mean_sd(sub[k]) for k in METRICS) + f" | {c_text} |")
    write_lines(path, lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SPLIT_SEEDS))
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--device", default="cpu", help="torch device of the slide_abmil arm")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel (model, seed) jobs")
    args = ap.parse_args()
    models = args.models if args.models is not None else config.list_models(args.emb_dir)
    models = sorted(models, key=list(config.ENCODERS).index)

    # the relabelling depends on the exported patches only, which every encoder shares
    first = config.load(models[0], TASK, args.emb_dir)
    counts = relabelling_counts(first)
    for m in models[1:]:
        assert np.array_equal(relabelling_counts(config.load(m, TASK, args.emb_dir)), counts), \
            f"{m} was exported on a different patch set"
    n_slides = len(slide_grades(first))

    jobs = [(m, seed) for m in models for seed in args.seeds]
    results = Parallel(n_jobs=args.jobs)(
        delayed(run_split)(m, seed, args.emb_dir, args.device) for m, seed in jobs)
    df = pd.DataFrame([row for rows in results for row in rows])
    order = {(m, seed, arm, evaluation): i
             for i, (m, seed, (arm, evaluation)) in enumerate(
                 (m, seed, row) for m in models for seed in sorted(args.seeds) for row in ROWS)}
    keys = zip(df["model"], df["seed"], df["arm"], df["evaluation"])
    df = df.iloc[np.argsort([order[k] for k in keys])].reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "label_coarsening.csv")
    md_path = os.path.join(args.out_dir, "label_coarsening.md")
    df.to_csv(csv_path, index=False)
    write_markdown(df, counts, first["classes"], n_slides, models, args.seeds, args.device,
                   md_path)
    print(f"wrote {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
