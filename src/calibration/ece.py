"""Expected Calibration Error (ECE), Brier score, and reliability-diagram data."""

from __future__ import annotations

import numpy as np


def compute_ece(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Compute top-label ECE.

    Parameters
    ----------
    proba : (N, C) predicted probabilities
    y_true : (N,)  true class ids
    """
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(y_true))


def compute_brier(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error of probability vector).

    Lower is better.  Range [0, 2].
    """
    N, C = proba.shape
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(N), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def reliability_diagram_data(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Return bin-level accuracy & confidence for plotting reliability diagrams."""
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
        else:
            bin_accs.append(float(correct[mask].mean()))
            bin_confs.append(float(confidences[mask].mean()))
        bin_counts.append(cnt)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
    }
