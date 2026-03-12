"""Smoke tests — verify core invariants without requiring data download."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── 1. Split non-overlap ────────────────────────────────────────────────

class TestSplitLeakage:
    """Verify that DS1/DS2 splits have zero patient overlap."""

    def test_ds1_ds2_disjoint(self):
        from src.common.io import load_yaml
        cfg = load_yaml(ROOT / "config" / "splits.yaml")
        ds1 = set(cfg["ds1"])
        ds2 = set(cfg["ds2"])
        assert ds1 & ds2 == set(), f"Overlap found: {ds1 & ds2}"

    def test_saved_splits_disjoint(self):
        """If split files exist, verify they don't overlap."""
        splits_dir = ROOT / "data" / "splits"
        if not splits_dir.exists():
            pytest.skip("Split files not yet generated")
        from src.common.checks import assert_no_leakage
        assert_no_leakage(splits_dir)


# ── 2. NPZ shape ────────────────────────────────────────────────────────

class TestNpzShape:
    """Verify NPZ files have consistent shapes."""

    @pytest.fixture
    def npz_dir(self):
        d = ROOT / "data" / "processed"
        if not d.exists() or not list(d.glob("*.npz")):
            pytest.skip("NPZ files not yet generated")
        return d

    def test_train_shape(self, npz_dir):
        data = np.load(str(npz_dir / "beats_train.npz"), allow_pickle=True)
        X, y = data["X"], data["y"]
        assert X.ndim == 3, f"Expected 3D, got {X.ndim}D"
        assert X.shape[0] == y.shape[0], "X and y length mismatch"
        assert X.shape[2] == 1, "Expected single channel"

    def test_label_range(self, npz_dir):
        for name in ("beats_train.npz", "beats_val.npz", "beats_test.npz"):
            f = npz_dir / name
            if not f.exists():
                continue
            data = np.load(str(f), allow_pickle=True)
            y = data["y"]
            assert y.min() >= 0
            assert y.max() <= 4, f"Label out of AAMI range in {name}"


# ── 3. Policy boundary ──────────────────────────────────────────────────

class TestPolicy:
    """Verify risk policy produces correct decisions at boundary values."""

    def test_normal_accept(self):
        from src.risk.policy import decide
        d = decide(0.0, 0.9)
        assert d["level"] == "normal"
        assert d["action"] == "accept"
        assert d["reason_code"] == "NORMAL"

    def test_critical_reject(self):
        from src.risk.policy import decide
        d = decide(0.30, 0.3)
        assert d["level"] == "critical"
        assert d["action"] == "reject"
        assert d["reason_code"] == "DRIFT_CRIT_CONF_LOW"

    def test_critical_degrade(self):
        from src.risk.policy import decide
        d = decide(0.25, 0.8)
        assert d["level"] == "critical"
        assert d["action"] == "degrade"

    def test_warning_warn(self):
        from src.risk.policy import decide
        d = decide(0.18, 0.4)
        assert d["level"] == "warning"
        assert d["action"] == "warn"

    def test_normal_low_conf(self):
        from src.risk.policy import decide
        d = decide(0.05, 0.3)
        assert d["level"] == "normal"
        assert d["action"] == "warn"
        assert d["reason_code"] == "CONF_LOW"


# ── 4. PSI / KS basic ───────────────────────────────────────────────────

class TestDriftMetrics:
    def test_psi_identical(self):
        from src.drift.psi import psi
        x = np.random.randn(1000)
        val = psi(x, x)
        assert val < 0.01, f"PSI of identical distributions should ≈ 0, got {val}"

    def test_psi_shifted(self):
        from src.drift.psi import psi
        ref = np.random.randn(1000)
        cur = np.random.randn(1000) + 3.0
        val = psi(ref, cur)
        assert val > 0.1, f"PSI of shifted distributions should be large, got {val}"

    def test_ks_identical(self):
        from src.drift.ks import ks
        x = np.random.randn(500)
        stat, p = ks(x, x)
        assert p > 0.05, f"KS p-value for identical data should be high, got {p}"


# ── 5. Calibration ──────────────────────────────────────────────────────

class TestCalibration:
    def test_ece_perfect(self):
        from src.calibration.ece import compute_ece
        proba = np.eye(3)
        y = np.array([0, 1, 2])
        ece = compute_ece(proba, y)
        assert ece < 0.01

    def test_brier_perfect(self):
        from src.calibration.ece import compute_brier
        proba = np.eye(3).astype(np.float32)
        y = np.array([0, 1, 2])
        brier = compute_brier(proba, y)
        assert brier < 0.01

    def test_temperature_identity(self):
        from src.calibration.temperature_scaling import apply_temperature
        logits = np.array([[2.0, 1.0, 0.0]])
        p1 = apply_temperature(logits, 1.0)
        assert abs(p1.sum() - 1.0) < 1e-5


# ── 6. AAMI label mapping ───────────────────────────────────────────────

class TestLabelMapping:
    def test_known_symbols(self):
        from src.dataset.label_aami import symbol_to_aami
        assert symbol_to_aami("N") == 0
        assert symbol_to_aami("V") == 2
        assert symbol_to_aami("F") == 3

    def test_unknown_symbol(self):
        from src.dataset.label_aami import symbol_to_aami
        assert symbol_to_aami("+") is None
        assert symbol_to_aami("~") is None
