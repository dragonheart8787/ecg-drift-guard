"""Out-of-Distribution (OOD) detection on embeddings.

Two complementary methods:
  1. Mahalanobis distance — parametric, uses class-conditional Gaussians
  2. Energy score — non-parametric, derived from logits
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import mahalanobis

from src.common.log import get_logger

log = get_logger(__name__)


# ── Mahalanobis distance ────────────────────────────────────────────────

class MahalanobisOOD:
    """Fit class-conditional Gaussians on reference embeddings,
    then score new samples by minimum Mahalanobis distance."""

    def __init__(self):
        self.class_means: dict[int, np.ndarray] = {}
        self.cov_inv: np.ndarray | None = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        classes = np.unique(labels)
        D = embeddings.shape[1]

        for c in classes:
            mask = labels == c
            self.class_means[int(c)] = embeddings[mask].mean(axis=0)

        # Shared covariance (pooled)
        centered = embeddings.copy()
        for c in classes:
            mask = labels == c
            centered[mask] -= self.class_means[int(c)]
        cov = np.cov(centered, rowvar=False) + np.eye(D) * 1e-6
        self.cov_inv = np.linalg.inv(cov)
        self._fitted = True
        log.info("MahalanobisOOD fitted: %d classes, %d dims", len(classes), D)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return per-sample minimum Mahalanobis distance (higher = more OOD)."""
        assert self._fitted, "Call fit() first"
        N = embeddings.shape[0]
        scores = np.full(N, np.inf)
        for c, mean in self.class_means.items():
            for i in range(N):
                d = mahalanobis(embeddings[i], mean, self.cov_inv)
                scores[i] = min(scores[i], d)
        return scores

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Vectorised version — much faster for large N."""
        assert self._fitted, "Call fit() first"
        N = embeddings.shape[0]
        min_dists = np.full(N, np.inf)
        for c, mean in self.class_means.items():
            diff = embeddings - mean  # (N, D)
            left = diff @ self.cov_inv  # (N, D)
            dists = np.sqrt(np.sum(left * diff, axis=1))  # (N,)
            min_dists = np.minimum(min_dists, dists)
        return min_dists


# ── Energy score ────────────────────────────────────────────────────────

def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Negative log-sum-exp of logits (higher = more OOD).

    References: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
    """
    return -T * np.log(np.sum(np.exp(logits / T), axis=1) + 1e-12)


# ── Threshold-based OOD decisions ───────────────────────────────────────

def compute_ood_stats(
    scores: np.ndarray,
    percentile_threshold: float = 95.0,
) -> dict:
    """Compute OOD statistics and a percentile-based threshold.

    Parameters
    ----------
    scores : OOD scores from reference (in-distribution) data
    percentile_threshold : percentile to use as OOD boundary

    Returns
    -------
    dict with threshold, mean, std, percentiles
    """
    thr = float(np.percentile(scores, percentile_threshold))
    return {
        "threshold": round(thr, 4),
        "percentile_used": percentile_threshold,
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "p50": round(float(np.median(scores)), 4),
        "p90": round(float(np.percentile(scores, 90)), 4),
        "p95": round(float(np.percentile(scores, 95)), 4),
        "p99": round(float(np.percentile(scores, 99)), 4),
    }


def classify_ood(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return boolean mask: True = OOD (above threshold)."""
    return scores > threshold
