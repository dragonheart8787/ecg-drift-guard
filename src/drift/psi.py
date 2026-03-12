"""Population Stability Index (PSI) — binned distributional distance."""

from __future__ import annotations

import numpy as np


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two 1-D distributions.

    Uses equal-width bins defined by the *ref* quantiles to ensure
    stable binning across different drift levels.
    """
    edges = np.percentile(ref, np.linspace(0, 100, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)

    # Laplace smoothing to avoid log(0)
    ref_pct = (ref_counts + 1) / (ref_counts.sum() + bins)
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + bins)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
