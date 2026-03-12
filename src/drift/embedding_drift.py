"""Embedding-level drift detection: PSI + KS aggregated across dimensions."""

from __future__ import annotations

import numpy as np

from src.drift.ks import ks
from src.drift.psi import psi as psi_fn
from src.common.log import get_logger

log = get_logger(__name__)


def compute_drift_score(
    ref_E: np.ndarray,
    cur_E: np.ndarray,
    psi_bins: int = 10,
    ks_alpha: float = 0.01,
    w_psi: float = 0.6,
    w_ks: float = 0.4,
) -> dict:
    """Aggregate embedding drift across all dimensions.

    Parameters
    ----------
    ref_E, cur_E : (N, D) embedding matrices
    psi_bins : number of bins for PSI
    ks_alpha : significance threshold for KS test
    w_psi, w_ks : weights for aggregated drift score

    Returns
    -------
    dict with keys: psi_mean, ks_rate, score, top_dims, per_dim
    """
    D = ref_E.shape[1]
    psi_vals = np.zeros(D)
    ks_pvals = np.zeros(D)

    for d in range(D):
        psi_vals[d] = psi_fn(ref_E[:, d], cur_E[:, d], bins=psi_bins)
        _, p = ks(ref_E[:, d], cur_E[:, d])
        ks_pvals[d] = p

    psi_mean = float(np.mean(psi_vals))
    ks_rate = float(np.mean(ks_pvals < ks_alpha))
    score = w_psi * psi_mean + w_ks * ks_rate

    # top-5 drifted dimensions by PSI
    top_dims = np.argsort(psi_vals)[::-1][:5].tolist()

    return {
        "psi_mean": round(psi_mean, 4),
        "ks_rate": round(ks_rate, 4),
        "score": round(score, 4),
        "top_dims": top_dims,
        "per_dim_psi": psi_vals.tolist(),
        "per_dim_ks_pval": ks_pvals.tolist(),
    }
