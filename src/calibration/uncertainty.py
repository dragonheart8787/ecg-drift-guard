"""Uncertainty quantification: entropy, margin, and confidence analysis."""

from __future__ import annotations

import numpy as np


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of predicted distribution.  shape (N,)."""
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def prediction_margin(proba: np.ndarray) -> np.ndarray:
    """Difference between top-1 and top-2 probabilities.  shape (N,)."""
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    return sorted_p[:, 0] - sorted_p[:, 1]


def confidence_analysis(
    proba: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Split confidence into correct / incorrect populations for comparison."""
    confidences = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    correct_mask = preds == y_true

    ent = predictive_entropy(proba)
    margin = prediction_margin(proba)

    return {
        "conf_correct": confidences[correct_mask].tolist(),
        "conf_incorrect": confidences[~correct_mask].tolist(),
        "entropy_correct": ent[correct_mask].tolist(),
        "entropy_incorrect": ent[~correct_mask].tolist(),
        "margin_correct": margin[correct_mask].tolist(),
        "margin_incorrect": margin[~correct_mask].tolist(),
        "mean_conf_correct": round(float(confidences[correct_mask].mean()), 4) if correct_mask.any() else 0,
        "mean_conf_incorrect": round(float(confidences[~correct_mask].mean()), 4) if (~correct_mask).any() else 0,
        "mean_entropy_correct": round(float(ent[correct_mask].mean()), 4) if correct_mask.any() else 0,
        "mean_entropy_incorrect": round(float(ent[~correct_mask].mean()), 4) if (~correct_mask).any() else 0,
    }
