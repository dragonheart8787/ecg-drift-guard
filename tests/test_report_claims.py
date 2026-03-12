import math

from src.common.io import load_json


def test_internal_per_class_and_confusion_match_report() -> None:
    """內部 MIT-BIH 測試集的 per-class 指標與混淆矩陣，需與報告表格一致。"""
    m = load_json("artifacts/reports/test_metrics.json")

    # Overall
    assert math.isclose(m["accuracy"], 0.7832, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(m["macro_f1"], 0.2707, rel_tol=0, abs_tol=1e-4)

    # Per-class table（對應最新版 REPORT_TABLES.md 表 1）
    expected_per_class = [
        ("N", 0.9120, 0.8571, 0.8837, 44242),
        ("S", 0.0068, 0.0174, 0.0098, 1837),
        ("V", 0.9254, 0.3006, 0.4538, 3220),
        ("F", 0.0005, 0.0026, 0.0008, 388),
        ("Q", 0.0028, 0.1429, 0.0056, 7),
    ]
    per_class = {row["class"]: row for row in m["per_class"]}
    for cls, p, r, f1, sup in expected_per_class:
        row = per_class[cls]
        assert math.isclose(row["precision"], p, rel_tol=0, abs_tol=1e-4)
        assert math.isclose(row["recall"], r, rel_tol=0, abs_tol=1e-4)
        assert math.isclose(row["f1-score"], f1, rel_tol=0, abs_tol=1e-4)
        assert row["support"] == sup

    # Confusion matrix
    expected_cm = [
        [37918, 3969, 36, 1983, 336],
        [1779, 32, 3, 17, 6],
        [1554, 677, 968, 11, 10],
        [324, 25, 38, 1, 0],
        [0, 5, 1, 0, 1],
    ]
    assert m["confusion_matrix"] == expected_cm


def test_calibration_and_drift_match_report() -> None:
    """校準指標與主要 drift 表格需與報告一致（容許微小四捨五入差）。"""
    s = load_json("artifacts/reports/summary.json")

    cal = s["calibration"]
    assert math.isclose(cal["ece_before"], 0.1060, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(cal["ece_after"], 0.0692, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(cal["brier_before"], 0.3191, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(cal["brier_after"], 0.3066, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(cal["temperature_T"], 0.7464, rel_tol=0, abs_tol=1e-4)

    # Drift table: noise / resample / gain × S1~S3
    expected_rows = {
        ("noise", "S1"): (1.1221, 0.0806),
        ("noise", "S2"): (4.1650, 0.2648),
        ("noise", "S3"): (5.2553, 0.2706),
        ("resample", "S1"): (0.0220, 0.0007),
        ("resample", "S2"): (0.2423, 0.0029),
        ("resample", "S3"): (0.3115, 0.0048),
        ("gain", "S1"): (0.2161, -0.0097),
        ("gain", "S2"): (0.4621, -0.0128),
        ("gain", "S3"): (0.5904, -0.0045),
    }
    got = {(d["scenario"], d["intensity"]): d for d in s["drift"]}
    for key, (drift_score, perf_drop) in expected_rows.items():
        row = got[key]
        assert math.isclose(row["drift_score"], drift_score, rel_tol=0, abs_tol=1e-4)
        assert math.isclose(row["perf_drop_f1"], perf_drop, rel_tol=0, abs_tol=1e-4)


def test_risk_policy_and_threshold_sensitivity_match_report() -> None:
    """風控總結與三組門檻敏感度需與報告一致。"""
    s = load_json("artifacts/reports/summary.json")
    risk = s["hypotheses"]["H3"]
    assert risk["supported"] is True
    assert math.isclose(risk["error_rate_overall"], 0.2168, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(risk["error_rate_after_policy"], 0.1700, rel_tol=0, abs_tol=1e-4)

    risk_results = s["risk_results"]
    assert math.isclose(risk_results["reject_rate"], 0.0773, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(risk_results["degrade_rate"], 0.0227, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(risk_results["warn_rate"], 0.0442, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(risk_results["accept_rate"], 0.8558, rel_tol=0, abs_tol=1e-4)

    ts = load_json("artifacts/reports/threshold_sensitivity.json")
    cfgs = {c["config"]: c for c in ts["configs"]}

    def check_cfg(name: str, reject: float, coverage: float, error_after: float) -> None:
        c = cfgs[name]
        assert math.isclose(c["reject_rate"], reject, rel_tol=0, abs_tol=1e-4)
        assert math.isclose(c["coverage"], coverage, rel_tol=0, abs_tol=1e-4)
        assert math.isclose(c["error_after_policy"], error_after, rel_tol=0, abs_tol=1e-4)

    check_cfg("conservative", 0.0984, 0.9016, 0.1572)
    check_cfg("default", 0.0773, 0.9227, 0.1700)
    check_cfg("liberal", 0.0217, 0.9783, 0.2040)


def test_external_internal_vs_external_tables_match_report() -> None:
    """外部 SVDB 指標與 per-class / 混淆矩陣需與報告一致。"""
    e = load_json("artifacts/reports/external_summary.json")

    # Overall table
    assert math.isclose(e["metrics"]["acc"], 0.8532, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(e["metrics"]["f1_macro"], 0.3028, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(e["metrics"]["ece"], 0.0943, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(e["metrics"]["brier"], 0.1738, rel_tol=0, abs_tol=1e-4)

    dv = e["drift_vs_internal"]
    assert math.isclose(dv["drift_score"], 0.5827, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(dv["psi_mean"], 0.3046, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(dv["ks_rate"], 1.0, rel_tol=0, abs_tol=1e-6)

    ood = e["ood"]
    assert math.isclose(ood["ood_rate"], 0.3730, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(ood["threshold"], 18.4756, rel_tol=0, abs_tol=1e-4)

    # Per-class SVDB
    expected_per_class_ext = [
        ("N", 0.9756, 0.8816, 0.9262, 26314),
        ("S", 0.0093, 0.0563, 0.0159, 320),
        ("V", 0.9931, 0.4015, 0.5718, 1081),
        ("F", 0.0000, 0.0000, 0.0000, 5),
        ("Q", 0.0000, 0.0000, 0.0000, 0),
    ]
    per_class_ext = {row["class"]: row for row in e["per_class"]}
    for cls, p, r, f1, sup in expected_per_class_ext:
        row = per_class_ext[cls]
        assert math.isclose(row["precision"], p, rel_tol=0, abs_tol=1e-4)
        assert math.isclose(row["recall"], r, rel_tol=0, abs_tol=1e-4)
        assert math.isclose(row["f1-score"], f1, rel_tol=0, abs_tol=1e-4)
        assert row["support"] == sup

    # Confusion matrix SVDB
    expected_cm_ext = [
        [23199, 1648, 1, 2, 1464],
        [246, 18, 1, 0, 55],
        [330, 275, 434, 17, 25],
        [4, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert e["confusion_matrix"] == expected_cm_ext

