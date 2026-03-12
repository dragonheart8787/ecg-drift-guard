"""Kolmogorov–Smirnov two-sample test wrapper."""

from __future__ import annotations

from scipy.stats import ks_2samp


def ks(ref, cur) -> tuple[float, float]:
    """Return (statistic, p-value) for 2-sample KS test."""
    stat, p = ks_2samp(ref, cur)
    return float(stat), float(p)
