"""Temperature Scaling — post-hoc calibration for neural network logits."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood with temperature T (standard TS objective)."""
    proba = softmax(logits / T, axis=1)
    proba = np.clip(proba, 1e-12, 1.0)
    log_proba = np.log(proba)
    return float(-log_proba[np.arange(len(y)), y].mean())


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find optimal temperature T on validation set by minimising NLL."""
    result = minimize_scalar(
        _nll,
        bounds=(0.5, 5.0),
        method="bounded",
        args=(logits_val, y_val),
    )
    return round(float(result.x), 4)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Return calibrated probabilities after dividing logits by T."""
    return softmax(logits / T, axis=1).astype(np.float32)
