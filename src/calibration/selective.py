"""Selective prediction: coverage-vs-risk and reject curves."""

from __future__ import annotations

import numpy as np


def selective_prediction_curve(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_points: int = 50,
) -> dict:
    """Compute coverage–risk curve by sweeping confidence threshold.

    Returns
    -------
    dict with lists: thresholds, coverages, risks (error rates), accuracies
    """
    confidences = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    correct = (preds == y_true)

    thresholds = np.linspace(0, 1, n_points + 1)[:-1]
    coverages: list[float] = []
    risks: list[float] = []
    accuracies: list[float] = []

    for thr in thresholds:
        mask = confidences >= thr
        cov = mask.sum() / len(y_true)
        if mask.sum() == 0:
            risk = 0.0
            acc = 0.0
        else:
            risk = float(1 - correct[mask].mean())
            acc = float(correct[mask].mean())
        coverages.append(round(cov, 4))
        risks.append(round(risk, 4))
        accuracies.append(round(acc, 4))

    return {
        "thresholds": thresholds.tolist(),
        "coverages": coverages,
        "risks": risks,
        "accuracies": accuracies,
    }


def reject_curve(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_points: int = 50,
) -> dict:
    """Reject rate vs Error rate curve (complementary view).

    As reject_rate increases, error_rate should decrease.
    """
    confidences = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    correct = (preds == y_true)

    thresholds = np.linspace(0, 1, n_points + 1)[:-1]
    reject_rates: list[float] = []
    error_rates: list[float] = []

    for thr in thresholds:
        accepted = confidences >= thr
        rejected_frac = 1 - accepted.sum() / len(y_true)
        if accepted.sum() == 0:
            err = 0.0
        else:
            err = float(1 - correct[accepted].mean())
        reject_rates.append(round(rejected_frac, 4))
        error_rates.append(round(err, 4))

    return {
        "thresholds": thresholds.tolist(),
        "reject_rates": reject_rates,
        "error_rates": error_rates,
    }


def compute_auc_risk_coverage(coverages: list[float], risks: list[float]) -> float:
    """Area Under the Risk-Coverage Curve (lower is better)."""
    c = np.array(coverages)
    r = np.array(risks)
    idx = np.argsort(c)
    return float(round(np.trapz(r[idx], c[idx]), 4))
