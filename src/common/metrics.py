"""Basic classification metrics (no sklearn dependency for core calcs)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list[str] | None = None) -> dict:
    """Return a dict with acc, f1_macro, confusion matrix, and per-class report."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "acc": round(acc, 4),
        "f1_macro": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
