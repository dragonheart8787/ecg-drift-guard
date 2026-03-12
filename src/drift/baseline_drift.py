"""Baseline B drift detection: PSI/KS on raw signal features (no embedding).

Demonstrates that embedding-based drift catches more subtle shifts.
"""

from __future__ import annotations

import numpy as np

from src.common.log import get_logger
from src.drift.ks import ks
from src.drift.psi import psi as psi_fn

log = get_logger(__name__)


def _extract_signal_features(X: np.ndarray) -> np.ndarray:
    """Hand-crafted per-beat features from raw signal.

    X : (N, L, 1)
    Returns : (N, 6) feature matrix
    """
    sig = X[:, :, 0]
    feats = np.column_stack([
        sig.mean(axis=1),
        sig.std(axis=1),
        sig.max(axis=1),
        sig.min(axis=1),
        np.median(sig, axis=1),
        (sig.max(axis=1) - sig.min(axis=1)),  # range
    ])
    return feats.astype(np.float32)


def compute_baseline_drift_score(
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    psi_bins: int = 10,
    ks_alpha: float = 0.01,
    w_psi: float = 0.6,
    w_ks: float = 0.4,
) -> dict:
    """PSI + KS on hand-crafted signal features (Baseline B).

    Same aggregation logic as embedding drift but on simple features.
    """
    ref_feat = _extract_signal_features(X_ref)
    cur_feat = _extract_signal_features(X_cur)
    D = ref_feat.shape[1]
    feature_names = ["mean", "std", "max", "min", "median", "range"]

    psi_vals = np.zeros(D)
    ks_pvals = np.zeros(D)

    for d in range(D):
        psi_vals[d] = psi_fn(ref_feat[:, d], cur_feat[:, d], bins=psi_bins)
        _, p = ks(ref_feat[:, d], cur_feat[:, d])
        ks_pvals[d] = p

    psi_mean = float(np.mean(psi_vals))
    ks_rate = float(np.mean(ks_pvals < ks_alpha))
    score = w_psi * psi_mean + w_ks * ks_rate

    top_idx = np.argsort(psi_vals)[::-1]
    top_features = [feature_names[i] for i in top_idx[:3]]

    return {
        "method": "baseline_signal_features",
        "psi_mean": round(psi_mean, 4),
        "ks_rate": round(ks_rate, 4),
        "score": round(score, 4),
        "top_features": top_features,
        "per_feature_psi": {feature_names[i]: round(float(psi_vals[i]), 4) for i in range(D)},
    }
