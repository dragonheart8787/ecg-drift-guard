"""Determinism tests — same seed + config must produce identical results."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestDeterminism:
    """Verify that core computations are deterministic given fixed seed."""

    def test_normalize_deterministic(self):
        from src.common.seed import set_seed
        from src.dataset.beat_cut import normalize

        set_seed(42)
        x = np.random.randn(50, 216).astype(np.float32)
        r1 = normalize(x, mode="zscore")
        r2 = normalize(x, mode="zscore")
        np.testing.assert_array_equal(r1, r2)

    def test_noise_injection_deterministic(self):
        from src.common.seed import set_seed
        from src.drift.simulate import apply_noise

        set_seed(42)
        X = np.random.randn(20, 216, 1).astype(np.float32)

        X1 = apply_noise(X, snr_db=10.0, seed=42)
        X2 = apply_noise(X, snr_db=10.0, seed=42)
        np.testing.assert_array_almost_equal(X1, X2, decimal=6)

    def test_gain_injection_deterministic(self):
        from src.drift.simulate import apply_gain

        X = np.random.randn(20, 216, 1).astype(np.float32)
        X1 = apply_gain(X, seed=42)
        X2 = apply_gain(X, seed=42)
        np.testing.assert_array_almost_equal(X1, X2, decimal=6)

    def test_psi_deterministic(self):
        from src.drift.psi import psi

        rng = np.random.RandomState(42)
        ref = rng.randn(500)
        cur = rng.randn(500) + 0.5
        v1 = psi(ref, cur, bins=10)
        v2 = psi(ref, cur, bins=10)
        assert v1 == v2

    def test_policy_deterministic(self):
        from src.risk.policy import decide

        d1 = decide(0.20, 0.6)
        d2 = decide(0.20, 0.6)
        assert d1 == d2

    def test_temperature_scaling_deterministic(self):
        from src.calibration.temperature_scaling import apply_temperature

        logits = np.array([[2.0, 1.0, -1.0], [0.5, 0.5, 0.0]])
        p1 = apply_temperature(logits, 1.5)
        p2 = apply_temperature(logits, 1.5)
        np.testing.assert_array_almost_equal(p1, p2, decimal=8)
