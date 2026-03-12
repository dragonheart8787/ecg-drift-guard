"""Schema validation tests — verify output files have correct structure."""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "artifacts" / "reports"


class TestSummarySchema:
    """Verify summary.json has all required top-level keys and types."""

    @pytest.fixture
    def summary(self):
        path = REPORTS_DIR / "summary.json"
        if not path.exists():
            pytest.skip("summary.json not yet generated")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    REQUIRED_KEYS = [
        "scope_statement", "dataset", "split_note", "model",
        "baseline", "calibration", "drift", "correlation",
        "hypotheses", "risk_policy", "risk_results",
        "audit_summary", "failure_modes", "model_card",
    ]

    def test_top_level_keys(self, summary):
        for key in self.REQUIRED_KEYS:
            assert key in summary, f"Missing key: {key}"

    def test_baseline_has_metrics(self, summary):
        bl = summary["baseline"]
        assert "acc" in bl
        assert "f1_macro" in bl
        assert isinstance(bl["acc"], (int, float))

    def test_calibration_has_before_after(self, summary):
        cal = summary["calibration"]
        for k in ("ece_before", "ece_after", "brier_before", "brier_after", "temperature_T"):
            assert k in cal, f"Missing calibration key: {k}"

    def test_drift_is_list(self, summary):
        assert isinstance(summary["drift"], list)
        if summary["drift"]:
            d = summary["drift"][0]
            assert "scenario" in d
            assert "drift_score" in d

    def test_hypotheses_h1_h2_h3(self, summary):
        hyp = summary["hypotheses"]
        for h in ("H1", "H2", "H3"):
            assert h in hyp, f"Missing hypothesis: {h}"
            assert "supported" in hyp[h]
            assert isinstance(hyp[h]["supported"], bool)

    def test_failure_modes_structure(self, summary):
        fm = summary["failure_modes"]
        assert isinstance(fm, list)
        assert len(fm) >= 3
        for entry in fm:
            for k in ("mode", "effect", "detection", "mitigation", "residual_risk"):
                assert k in entry, f"Failure mode missing key: {k}"

    def test_model_card_present(self, summary):
        mc = summary["model_card"]
        assert "model_name" in mc
        assert "limitations" in mc
        assert isinstance(mc["limitations"], list)


class TestDecisionsCsvSchema:
    """Verify decisions.csv has correct columns and types."""

    @pytest.fixture
    def rows(self):
        path = REPORTS_DIR / "decisions.csv"
        if not path.exists():
            pytest.skip("decisions.csv not yet generated")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    REQUIRED_COLUMNS = [
        "sample_idx", "record_id", "true_label", "pred_label",
        "drift_score", "confidence", "entropy", "margin",
        "policy_level", "action", "reason_code",
    ]

    def test_columns_present(self, rows):
        if not rows:
            pytest.skip("decisions.csv is empty")
        for col in self.REQUIRED_COLUMNS:
            assert col in rows[0], f"Missing column: {col}"

    def test_numeric_fields_parseable(self, rows):
        for row in rows[:100]:
            float(row["drift_score"])
            float(row["confidence"])
            float(row["entropy"])
            float(row["margin"])

    def test_action_values_valid(self, rows):
        valid_actions = {"accept", "warn", "reject", "degrade"}
        for row in rows:
            assert row["action"] in valid_actions, f"Invalid action: {row['action']}"

    def test_policy_level_values_valid(self, rows):
        valid_levels = {"normal", "warning", "critical"}
        for row in rows:
            assert row["policy_level"] in valid_levels

    def test_reason_codes_not_empty(self, rows):
        for row in rows:
            assert row["reason_code"], "reason_code should not be empty"


class TestBenchmarkSchema:
    """Verify benchmark.json structure if it exists."""

    @pytest.fixture
    def benchmark(self):
        path = REPORTS_DIR / "benchmark.json"
        if not path.exists():
            pytest.skip("benchmark.json not yet generated")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_model_latency_present(self, benchmark):
        assert "model_latency" in benchmark
        assert isinstance(benchmark["model_latency"], list)

    def test_latency_entry_structure(self, benchmark):
        for entry in benchmark["model_latency"]:
            assert "batch_size" in entry
            assert "per_beat_ms" in entry
            assert isinstance(entry["per_beat_ms"], (int, float))


class TestExternalSummarySchema:
    """Verify external_summary.json if it exists."""

    @pytest.fixture
    def ext(self):
        path = REPORTS_DIR / "external_summary.json"
        if not path.exists():
            pytest.skip("external_summary.json not yet generated")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_required_keys(self, ext):
        for k in ("external_db", "n_beats", "metrics", "drift_vs_internal", "ood"):
            assert k in ext, f"Missing key: {k}"

    def test_ood_has_threshold(self, ext):
        assert "threshold" in ext["ood"]
        assert "ood_rate" in ext["ood"]
