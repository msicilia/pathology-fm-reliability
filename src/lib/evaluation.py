"""
evaluation.py
=============
Helpers shared by the analysis scripts: predictions of a fitted probe, the
metrics of a set of predictions, and the text of a mean with its spread.
"""
from __future__ import annotations

import numpy as np

from src.lib.probe import probe_logits
from src.lib.scoring import aurc, confidence, ece, excess_aurc, qwk


def probe_predict(probe, Z, n_classes: int):
    """Predicted label and maximum softmax probability of a fitted probe."""
    if not np.array_equal(probe[-1].classes_, np.arange(n_classes)):
        raise RuntimeError("the probe was not fitted on every class")
    logits = probe_logits(probe, Z)
    return logits.argmax(1), confidence(logits)


def selective_metrics(y, pred, conf, n_classes: int | None = None) -> dict:
    """Accuracy, AURC, E-AURC and ECE of predictions with a confidence each.

    The quadratic weighted kappa is added when `n_classes` is given (ordinal labels).
    """
    correct = np.asarray(pred) == np.asarray(y)
    uncertainty = 1.0 - np.asarray(conf, dtype=float)
    out = {"accuracy": float(correct.mean()),
           "aurc": aurc(correct, uncertainty),
           "e_aurc": excess_aurc(correct, uncertainty),
           "ece": ece(conf, correct)}
    if n_classes is not None:
        out["qwk"] = qwk(y, pred, n_classes)
    return out


def mean_sd(values) -> str:
    """Mean and sample standard deviation (ddof = 1) of the values, as text."""
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else float("nan")
    return f"{values.mean():.3f} +/- {sd:.3f}"
