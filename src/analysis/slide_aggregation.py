"""
slide_aggregation.py
====================
Slide-level ISUP grading on PANDA from the frozen patch embeddings.

For every encoder and every split seed, the patches are split by slide
(stratified by grade and data provider) and grouped into one bag per slide.
Five aggregators are trained on the training slides and evaluated on the test
slides:

* patch_vote   the patch-level probe followed by a majority vote of its patch
               predictions (ties go to the lowest grade); the slide confidence
               is the fraction of patches voting for the winning grade;
* mean_pool    the per-slide mean of the patch embeddings, then the probe;
* max_pool     the per-slide maximum of the patch embeddings, then the probe;
* abmil        gated attention pooling (Ilse et al., 2018);
* transformer  a one-layer transformer encoder with a class token.

The two neural aggregators receive embeddings standardised with the per-feature
mean and standard deviation of the training patches, and are trained with one
fixed configuration for every encoder (the constants below): no validation set,
no model selection, weights of the final epoch.

The patch-level accuracy and QWK of the patch probe are recorded next to the
slide-level metrics; their unit is the patch, not the slide.

Writes slide_aggregation.csv and slide_aggregation.md to the output directory
and fig_slide_aggregation.png to its figures subdirectory.

Run from the repository root:
    uv run python src/analysis/slide_aggregation.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from src.lib import config
from src.lib.evaluation import mean_sd, probe_predict, selective_metrics
from src.lib.probe import fit_probe, selected_c
from src.lib.scoring import ECE_BINS, qwk
from src.lib.splits import resplit
from src.lib.tables import write_lines

TASK = "panda"
AGGREGATORS = ("patch_vote", "mean_pool", "max_pool", "abmil", "transformer")
AGGREGATOR_LABELS = {
    "patch_vote": "patch vote",
    "mean_pool": "mean pooling",
    "max_pool": "max pooling",
    "abmil": "ABMIL",
    "transformer": "transformer",
}
METRICS = ("accuracy", "qwk", "aurc", "e_aurc", "ece")     # metric columns of the csv

# configuration of the neural aggregators, shared by every encoder
EMBED_DIM = 256         # width of the patch projection (both aggregators)
ATTENTION_DIM = 128     # hidden width of the gated attention (abmil)
N_HEADS = 4             # attention heads (transformer)
FEEDFORWARD_DIM = 512   # feed-forward width (transformer)
DROPOUT = 0.1           # dropout inside the transformer layer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30


# --------------------------------------------------------------------------- #
# bags
# --------------------------------------------------------------------------- #
def make_bags(slides, y):
    """Group patches by slide.

    Returns (ids, index, labels): the sorted slide ids, the patch indices of each
    slide and the label of each slide. All patches of a slide must share one label.
    """
    y = np.asarray(y)
    ids, inverse = np.unique(np.asarray(slides), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    index = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    labels = np.array([y[i[0]] for i in index])
    for slide, i, label in zip(ids, index, labels):
        assert (y[i] == label).all(), f"patches of slide {slide} carry different labels"
    return ids, index, labels


def majority_vote(patch_pred, index, n_classes: int):
    """Per-slide majority vote of patch predictions (ties go to the lowest label).

    Returns the winning label of each slide and the fraction of its patches that
    voted for it.
    """
    counts = np.stack([np.bincount(patch_pred[i], minlength=n_classes) for i in index])
    return counts.argmax(1), counts.max(1) / counts.sum(1)


def pool_bags(Z, index, how: str) -> np.ndarray:
    """One vector per slide: the mean or the maximum of its patch embeddings."""
    if how == "mean":
        return np.stack([Z[i].mean(0) for i in index])
    if how == "max":
        return np.stack([Z[i].max(0) for i in index])
    raise ValueError(f"unknown pooling: {how}")


# --------------------------------------------------------------------------- #
# neural aggregators
# --------------------------------------------------------------------------- #
class GatedAttentionMIL(nn.Module):
    """Gated attention pooling over the patches of one bag (Ilse et al., 2018)."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(dim, EMBED_DIM), nn.ReLU())
        self.attention_v = nn.Linear(EMBED_DIM, ATTENTION_DIM)
        self.attention_u = nn.Linear(EMBED_DIM, ATTENTION_DIM)
        self.attention_w = nn.Linear(ATTENTION_DIM, 1)
        self.head = nn.Linear(EMBED_DIM, n_classes)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:          # bag: (n_patches, dim)
        h = self.embed(bag)
        gate = torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
        weights = torch.softmax(self.attention_w(gate), dim=0)
        return self.head((weights * h).sum(0, keepdim=True))       # (1, n_classes)


class TransformerMIL(nn.Module):
    """One transformer encoder layer over the patches of one bag and a class token."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.embed = nn.Linear(dim, EMBED_DIM)
        self.class_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.encoder = nn.TransformerEncoderLayer(EMBED_DIM, N_HEADS, FEEDFORWARD_DIM,
                                                  dropout=DROPOUT, batch_first=True)
        self.head = nn.Linear(EMBED_DIM, n_classes)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:          # bag: (n_patches, dim)
        tokens = torch.cat([self.class_token, self.embed(bag).unsqueeze(0)], dim=1)
        return self.head(self.encoder(tokens)[:, 0])               # (1, n_classes)


MIL_MODELS = {"abmil": GatedAttentionMIL, "transformer": TransformerMIL}


def standardise(Z_train, Z_test):
    """Standardise both parts with the per-feature statistics of the training patches."""
    scaler = StandardScaler().fit(Z_train)
    return (scaler.transform(Z_train).astype(np.float32),
            scaler.transform(Z_test).astype(np.float32))


def train_mil(kind: str, Z, index, labels, n_classes: int, seed: int, device: str):
    """Train one neural aggregator on bags of standardised embeddings.

    One bag per optimisation step, bags in a fresh random order every epoch,
    cross-entropy over the classes; the weights of the final epoch are returned.
    """
    torch.manual_seed(seed)
    model = MIL_MODELS[kind](Z.shape[1], n_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    Z = torch.from_numpy(Z).to(device)
    bags = [torch.from_numpy(i).to(device) for i in index]
    targets = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    model.train()
    for _ in range(EPOCHS):
        for j in torch.randperm(len(bags)).tolist():
            loss = loss_fn(model(Z[bags[j]]), targets[j:j + 1])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


@torch.no_grad()
def predict_mil(model, Z, index, device: str) -> np.ndarray:
    """Class probabilities of each bag, one row per bag."""
    model.eval()
    Z = torch.from_numpy(Z).to(device)
    probs = [torch.softmax(model(Z[torch.from_numpy(i).to(device)]), dim=1)[0] for i in index]
    return torch.stack(probs).double().cpu().numpy()


def fit_predict_mil(kind: str, Z_train, index_train, labels_train, Z_test, index_test,
                    n_classes: int, seed: int, device: str = "cpu"):
    """Standardise, train one neural aggregator and predict the test bags.

    Returns the predicted label and the maximum softmax probability of each test bag.
    """
    X_train, X_test = standardise(Z_train, Z_test)
    model = train_mil(kind, X_train, index_train, labels_train, n_classes, seed, device)
    probs = predict_mil(model, X_test, index_test, device)
    return probs.argmax(1), probs.max(1)


# --------------------------------------------------------------------------- #
# one (model, seed)
# --------------------------------------------------------------------------- #
def run_split(model: str, seed: int, emb_dir: str, device: str) -> list[dict]:
    """Rows of slide_aggregation.csv for one encoder and one split."""
    torch.set_num_threads(1)
    s = resplit(config.load(model, TASK, emb_dir), seed)
    if s["groups_train"] is None:
        raise RuntimeError(f"{config.npz_path(model, TASK, emb_dir)} has no slide ids")
    n_classes = len(s["classes"])
    Z_train, Z_test = s["Z_train"], s["Z_test"]
    ids_train, index_train, y_train = make_bags(s["groups_train"], s["y_train"])
    ids_test, index_test, y_test = make_bags(s["groups_test"], s["y_test"])
    assert not set(ids_train) & set(ids_test), "a slide is on both sides of the split"

    predictions = {}    # aggregator -> (slide prediction, slide confidence, selected C)

    patch_probe = fit_probe(Z_train, s["y_train"], groups=s["groups_train"])
    patch_pred, _ = probe_predict(patch_probe, Z_test, n_classes)
    predictions["patch_vote"] = (*majority_vote(patch_pred, index_test, n_classes),
                                 selected_c(patch_probe))
    patch_level = {"patch_level_accuracy": float((patch_pred == s["y_test"]).mean()),
                   "patch_level_qwk": qwk(s["y_test"], patch_pred, n_classes),
                   "n_test_patches": len(patch_pred)}

    for how in ("mean", "max"):
        probe = fit_probe(pool_bags(Z_train, index_train, how), y_train)
        pred, conf = probe_predict(probe, pool_bags(Z_test, index_test, how), n_classes)
        predictions[f"{how}_pool"] = (pred, conf, selected_c(probe))

    for kind in MIL_MODELS:
        pred, conf = fit_predict_mil(kind, Z_train, index_train, y_train, Z_test, index_test,
                                     n_classes, seed, device)
        predictions[kind] = (pred, conf, float("nan"))

    rows = []
    for aggregator in AGGREGATORS:
        pred, conf, c = predictions[aggregator]
        metrics = selective_metrics(y_test, pred, conf, n_classes)
        rows.append({"model": model, "seed": seed, "aggregator": aggregator,
                     "n_train_slides": len(ids_train), "n_test_slides": len(ids_test),
                     **{k: metrics[k] for k in METRICS},
                     "selected_c": c, **patch_level})
    return rows


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_markdown(df: pd.DataFrame, models, seeds, device: str, path: str) -> None:
    lines = [
        "# Slide-level aggregation on PANDA",
        "",
        f"Slide-grouped splits stratified by grade and data provider, seeds {list(seeds)}; "
        f"{int(df['n_train_slides'].iloc[0])} training and {int(df['n_test_slides'].iloc[0])} "
        "test slides in the first split. Every aggregator is trained on the training slides "
        "and evaluated on the test slides. Entries are the mean +/- the sample standard "
        "deviation (ddof = 1) over the splits.",
        "",
        "Aggregators:",
        "",
        "- patch_vote: patch-level probe, majority vote of its patch predictions per slide "
        "(ties to the lowest grade); confidence = fraction of patches voting for the "
        "winning grade.",
        "- mean_pool, max_pool: per-slide mean or maximum of the patch embeddings, then the "
        "probe; confidence = maximum softmax probability.",
        "- abmil: gated attention pooling (Ilse et al., 2018): linear projection to "
        f"{EMBED_DIM} units with ReLU, gated attention with {ATTENTION_DIM} hidden units, "
        "linear classifier; confidence = maximum softmax probability.",
        f"- transformer: linear projection to {EMBED_DIM} units, a class token, one "
        f"transformer encoder layer ({N_HEADS} heads, feed-forward width {FEEDFORWARD_DIM}, "
        f"dropout {DROPOUT}), linear classifier on the class token; confidence = maximum "
        "softmax probability.",
        "",
        "Training of abmil and transformer, identical for every encoder: embeddings "
        "standardised with the per-feature mean and standard deviation of the training "
        f"patches; Adam, learning rate {LEARNING_RATE:g}, weight decay {WEIGHT_DECAY:g}, "
        f"{EPOCHS} epochs, one bag per step, cross-entropy over the six grades, "
        "`torch.manual_seed` set to the split seed; no validation set and no model "
        f"selection, weights of the final epoch. Device of this run: {device}. Results "
        "obtained on GPU backends may differ in the last digits.",
        "",
        "## QWK",
        "",
        "- Patch-level probe: QWK of the patch-level probe over the test patches, each "
        "patch labelled with the grade of its slide. Its unit is the patch.",
        "- Other columns: QWK of each aggregator over the test slides.",
        "",
        "| Encoder | Patch-level probe | " + " | ".join(AGGREGATORS) + " |",
        "|---|---|" + "---|" * len(AGGREGATORS),
    ]
    for m in models:
        sub = df[df["model"] == m]
        patch = sub[sub["aggregator"] == AGGREGATORS[0]]["patch_level_qwk"]
        cells = [mean_sd(sub[sub["aggregator"] == a]["qwk"]) for a in AGGREGATORS]
        lines.append(f"| {config.display(m)} | {mean_sd(patch)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Slide-level accuracy, selective prediction and calibration",
        "",
        "- Accuracy: fraction of test slides graded correctly.",
        "- AURC: area under the risk-coverage curve, abstaining on the least confident "
        "slides first.",
        "- E-AURC: AURC minus the AURC of a perfect ranking at the same error rate.",
        f"- ECE: top-label expected calibration error, {ECE_BINS} equal-width bins.",
        "- Test slides: number of test slides per split (mean over the splits).",
        "",
        "| Encoder | Aggregator | Accuracy | AURC | E-AURC | ECE | Test slides |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        for a in AGGREGATORS:
            sub = df[(df["model"] == m) & (df["aggregator"] == a)]
            cells = [mean_sd(sub[k]) for k in ("accuracy", "aurc", "e_aurc", "ece")]
            lines.append(f"| {config.display(m)} | {a} | " + " | ".join(cells)
                         + f" | {sub['n_test_slides'].mean():.1f} |")
    lines += [
        "",
        "## Patch-level probe",
        "",
        "Accuracy and QWK of the patch-level probe over the test patches (unit: patch). "
        "AURC and ECE are not reported at this level.",
        "",
        "| Encoder | Patch-level accuracy | Patch-level QWK | Test patches |",
        "|---|---|---|---|",
    ]
    for m in models:
        sub = df[(df["model"] == m) & (df["aggregator"] == AGGREGATORS[0])]
        lines.append(f"| {config.display(m)} | {mean_sd(sub['patch_level_accuracy'])} | "
                     f"{mean_sd(sub['patch_level_qwk'])} | {sub['n_test_patches'].mean():.1f} |")
    write_lines(path, lines)


def plot_qwk(df: pd.DataFrame, models, path: str) -> None:
    """Patch-level QWK and the slide-level QWK of each aggregator, per encoder."""
    series = [("patch-level probe (unit: patch)", None, "#555555", "s")]
    colours = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
    markers = ("o", "^", "v", "D", "P")
    for a, colour, marker in zip(AGGREGATORS, colours, markers):
        series.append((f"{AGGREGATOR_LABELS[a]} (unit: slide)", a, colour, marker))

    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * len(models) + 2.0), 4.6))
    offsets = np.linspace(-0.35, 0.35, len(series))
    x = np.arange(len(models))
    for (label, aggregator, colour, marker), dx in zip(series, offsets):
        means, sds = [], []
        for m in models:
            sub = df[df["model"] == m]
            if aggregator is None:
                values = sub[sub["aggregator"] == AGGREGATORS[0]]["patch_level_qwk"]
            else:
                values = sub[sub["aggregator"] == aggregator]["qwk"]
            means.append(values.mean())
            sds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        ax.errorbar(x + dx, means, yerr=sds, fmt=marker, color=colour, markersize=5.5,
                    elinewidth=1.0, capsize=2.0, label=label)
    for boundary in x[:-1] + 0.5:
        ax.axvline(boundary, color="0.9", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([config.display(m) for m in models], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel("QWK (mean +/- sd over splits)")
    ax.set_title("PANDA: patch-level and slide-level QWK by encoder")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SPLIT_SEEDS))
    ap.add_argument("--out-dir", default=config.RES_DIR)
    ap.add_argument("--device", default="cpu", help="torch device of the neural aggregators")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel (model, seed) jobs")
    args = ap.parse_args()
    models = args.models if args.models is not None else config.list_models(args.emb_dir)
    models = sorted(models, key=list(config.ENCODERS).index)

    jobs = [(m, seed) for m in models for seed in args.seeds]
    results = Parallel(n_jobs=args.jobs)(
        delayed(run_split)(m, seed, args.emb_dir, args.device) for m, seed in jobs)
    df = pd.DataFrame([row for rows in results for row in rows])
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    df["aggregator"] = pd.Categorical(df["aggregator"], categories=AGGREGATORS, ordered=True)
    df = df.sort_values(["model", "seed", "aggregator"]).reset_index(drop=True)
    df = df.astype({"model": str, "aggregator": str})

    fig_dir = config.fig_dir(args.out_dir)
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "slide_aggregation.csv")
    md_path = os.path.join(args.out_dir, "slide_aggregation.md")
    fig_path = os.path.join(fig_dir, "fig_slide_aggregation.png")
    df.to_csv(csv_path, index=False)
    write_markdown(df, models, args.seeds, args.device, md_path)
    plot_qwk(df, models, fig_path)
    print(f"wrote {csv_path}, {md_path}, {fig_path}")


if __name__ == "__main__":
    main()
