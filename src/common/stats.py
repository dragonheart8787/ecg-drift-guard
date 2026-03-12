"""Statistical utilities: Bootstrap CI, Spearman/Pearson correlation."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    metric_fn : callable(y_true, y_pred) -> float

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    scores = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        scores[b] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1 - ci) / 2
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return round(point, 4), round(lo, 4), round(hi, 4)


def bootstrap_metric(
    values: np.ndarray,
    metric_fn=np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for a single-array metric (e.g. mean accuracy)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    point = float(metric_fn(values))
    scores = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        scores[b] = metric_fn(values[idx])

    alpha = (1 - ci) / 2
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return round(point, 4), round(lo, 4), round(hi, 4)


def correlation_analysis(
    x: np.ndarray | list,
    y: np.ndarray | list,
) -> dict:
    """Compute Pearson and Spearman correlations with p-values."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None,
                "note": "Too few points for correlation"}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "pearson_r": round(float(pr), 4),
        "pearson_p": round(float(pp), 6),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 6),
    }
