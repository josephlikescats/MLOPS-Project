"""
Shared metric computation and threshold selection.

Training and monitoring both need to turn probabilities into the same set of
numbers, so the arithmetic lives here once. Keeping it shared is what makes
"precision dropped below its floor" a comparable statement between the
training report and a later monitoring run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Threshold-free and thresholded metrics for a binary churn prediction."""
    y_true = np.asarray(y_true).astype(int)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }

    # ROC-AUC and average precision are undefined when only one class is
    # present, which can legitimately happen in a small monitoring batch.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
        metrics["brier_score"] = float(brier_score_loss(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["brier_score"] = None

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    (true_negative, false_positive), (false_negative, true_positive) = matrix
    metrics["confusion_matrix"] = {
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }
    return metrics


def tune_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    min_precision: float | None = None,
) -> tuple[float, dict]:
    """Pick the decision threshold on the validation block, not the test block.

    F1 is maximised by default. ``min_precision`` additionally constrains the
    search, which is the knob to reach for before a retention campaign: a
    precision floor bounds how many contented customers get contacted
    unnecessarily, at the cost of recall.
    """
    y_true = np.asarray(y_true).astype(int)
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)

    best_threshold, best_score, best_metrics = DEFAULT_THRESHOLD, -1.0, None
    for threshold in candidates:
        metrics = classification_metrics(y_true, probabilities, float(threshold))
        if min_precision is not None and metrics["precision"] < min_precision:
            continue
        if metrics["f1"] > best_score:
            best_threshold, best_score, best_metrics = float(threshold), metrics["f1"], metrics

    if best_metrics is None:
        # No threshold satisfied the precision floor - fall back to the
        # unconstrained default rather than returning nothing.
        best_metrics = classification_metrics(y_true, probabilities, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD, best_metrics

    return best_threshold, best_metrics


def scalar_metrics(metrics: dict) -> dict[str, float]:
    """Flatten to the scalar subset MLflow can log as metrics."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
