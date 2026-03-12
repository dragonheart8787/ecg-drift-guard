#!/usr/bin/env python
"""Step 5: Calibration + uncertainty + selective prediction + risk + audit trail → summary.json."""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.ece import compute_brier, compute_ece, reliability_diagram_data
from src.calibration.selective import (
    compute_auc_risk_coverage,
    reject_curve,
    selective_prediction_curve,
)
from src.calibration.temperature_scaling import apply_temperature, fit_temperature
from src.calibration.uncertainty import (
    confidence_analysis,
    prediction_margin,
    predictive_entropy,
)
from src.common.checks import class_distribution_table, environment_info
from src.common.io import load_json, load_npz, load_yaml, save_json
from src.common.log import get_logger
from src.common.metrics import compute_metrics
from src.common.seed import set_seed
from src.dataset.label_aami import AAMI_CLASSES
from src.models.infer import predict_logits, predict_proba
from src.risk.audit import audit_summary, build_decision_rows, save_decisions_csv
from src.risk.policy import evaluate_batch
from src.risk.report import build_summary, save_summary, verify_hypotheses
from src.viz.plots import (
    plot_calibration_table,
    plot_class_distribution,
    plot_confidence_distribution,
    plot_coverage_risk,
    plot_reject_curve,
    plot_reliability,
    plot_reliability_comparison,
)

log = get_logger("05_calibrate_and_risk")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration + Risk + Audit")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    val_data = load_npz(ROOT / "data" / "processed" / "beats_val.npz")
    test_data = load_npz(ROOT / "data" / "processed" / "beats_test.npz")
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    model = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "cnn1d.keras")
    fig_dir = ROOT / "artifacts" / "reports" / "figures"

    # ══════════════════════════════════════════════════════════════════════
    #  BASELINE METRICS (Baseline A — model only, no calibration/risk)
    # ══════════════════════════════════════════════════════════════════════
    proba_test = predict_proba(model, X_test)
    y_pred_test = np.argmax(proba_test, axis=1)
    baseline_metrics = compute_metrics(y_test, y_pred_test, AAMI_CLASSES)
    log.info("Baseline A — acc=%.4f  f1=%.4f", baseline_metrics["acc"], baseline_metrics["f1_macro"])

    # ══════════════════════════════════════════════════════════════════════
    #  CALIBRATION (Temperature Scaling) + ECE + Brier
    # ══════════════════════════════════════════════════════════════════════
    logits_val = predict_logits(model, X_val)
    logits_test = predict_logits(model, X_test)
    n_bins = cfg["calibration"]["n_bins"]

    # Before calibration
    proba_before = predict_proba(model, X_test)
    ece_before = compute_ece(proba_before, y_test, n_bins=n_bins)
    brier_before = compute_brier(proba_before, y_test)
    rel_before = reliability_diagram_data(proba_before, y_test, n_bins=n_bins)

    # Fit temperature on val
    T = fit_temperature(logits_val, y_val)
    log.info("Fitted Temperature T = %.4f", T)

    # After calibration
    proba_after = apply_temperature(logits_test, T)
    ece_after = compute_ece(proba_after, y_test, n_bins=n_bins)
    brier_after = compute_brier(proba_after, y_test)
    rel_after = reliability_diagram_data(proba_after, y_test, n_bins=n_bins)
    log.info("ECE  before=%.4f  after=%.4f", ece_before, ece_after)
    log.info("Brier before=%.4f  after=%.4f", brier_before, brier_after)

    save_json({"T": T}, ROOT / "artifacts" / "calibration" / "temperature.json")

    calibration_info = {
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "temperature_T": T,
    }

    # Reliability plots
    plot_reliability(rel_before, ece_before, "Before Calibration", fig_dir / "reliability_before.png")
    plot_reliability(rel_after, ece_after, "After Calibration", fig_dir / "reliability_after.png")
    plot_reliability_comparison(rel_before, ece_before, rel_after, ece_after,
                                fig_dir / "reliability_comparison.png")
    plot_calibration_table(calibration_info, fig_dir / "calibration_table.png")

    # ══════════════════════════════════════════════════════════════════════
    #  UNCERTAINTY ANALYSIS (entropy, margin, confidence dist)
    # ══════════════════════════════════════════════════════════════════════
    conf_analysis = confidence_analysis(proba_after, y_test)
    plot_confidence_distribution(conf_analysis, fig_dir / "confidence_distribution.png")
    log.info("Mean conf correct=%.4f  incorrect=%.4f",
             conf_analysis["mean_conf_correct"], conf_analysis["mean_conf_incorrect"])

    # ══════════════════════════════════════════════════════════════════════
    #  SELECTIVE PREDICTION (coverage-risk, reject curve)
    # ══════════════════════════════════════════════════════════════════════
    sel_data = selective_prediction_curve(proba_after, y_test)
    rej_data = reject_curve(proba_after, y_test)
    aurc = compute_auc_risk_coverage(sel_data["coverages"], sel_data["risks"])
    log.info("AURC (Area Under Risk-Coverage) = %.4f", aurc)

    plot_coverage_risk(sel_data, fig_dir / "coverage_risk.png")
    plot_reject_curve(rej_data, fig_dir / "reject_curve.png")

    # ══════════════════════════════════════════════════════════════════════
    #  RISK POLICY + AUDIT TRAIL
    # ══════════════════════════════════════════════════════════════════════
    drift_results_path = ROOT / "artifacts" / "reports" / "drift_results.json"
    drift_results = load_json(drift_results_path) if drift_results_path.exists() else []

    rcfg = cfg["risk"]["thresholds"]
    calibrated_confs = np.max(proba_after, axis=1)
    y_pred_cal = np.argmax(proba_after, axis=1)
    entropies = predictive_entropy(proba_after)
    margins = prediction_margin(proba_after)

    # Derive per-sample drift scores from uncertainty:
    # we use entropy-based gating as a heuristic risk indicator (not true drift).
    uncertainty_gate_q = 0.9
    if drift_results:
        # Use median drift_score across all scenarios as a moderate drift level.
        all_scores = np.array([d["drift_score"] for d in drift_results], dtype=float)
        moderate_drift = float(np.median(all_scores))
    else:
        moderate_drift = rcfg["critical"]

    high_entropy_thr = float(np.percentile(entropies, uncertainty_gate_q * 100))
    drift_scores_per_sample = np.where(
        entropies >= high_entropy_thr,
        max(moderate_drift, rcfg["critical"] + 0.05),  # ensure critical for top-uncertain
        0.0,
    ).tolist()

    risk_stats = evaluate_batch(
        drift_scores_per_sample,
        calibrated_confs.tolist(),
        thr_warning=rcfg["warning"],
        thr_critical=rcfg["critical"],
        conf_low=rcfg["conf_low"],
        conf_mid=rcfg["conf_mid"],
    )

    # Build audit trail (decisions.csv)
    record_ids = test_data.get("record_id", np.array([""] * len(y_test)))
    decision_rows = build_decision_rows(
        record_ids=record_ids,
        y_true=y_test,
        y_pred=y_pred_cal,
        drift_scores=drift_scores_per_sample,
        confidences=calibrated_confs,
        entropies=entropies,
        margins=margins,
        decisions=risk_stats["decisions"],
        class_names=AAMI_CLASSES,
    )
    save_decisions_csv(decision_rows, ROOT / "artifacts" / "reports" / "decisions.csv")
    a_stats = audit_summary(decision_rows)
    log.info("Audit: error_overall=%.4f  error_after_policy=%.4f  reject_rate=%.4f",
             a_stats["error_rate_overall"], a_stats["error_rate_after_policy"],
             a_stats["reject_rate"])

    risk_stats_summary = {k: v for k, v in risk_stats.items() if k != "decisions"}

    # ══════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS VERIFICATION
    # ══════════════════════════════════════════════════════════════════════
    corr_path = ROOT / "artifacts" / "reports" / "drift_correlation.json"
    correlation_report = load_json(corr_path) if corr_path.exists() else {}

    hyp = verify_hypotheses(
        correlation_report=correlation_report,
        calibration_info=calibration_info,
        audit_stats=a_stats,
    )
    for h_id, h_data in hyp.items():
        status = "SUPPORTED" if h_data["supported"] else "NOT SUPPORTED"
        log.info("%s [%s]: %s — %s", h_id, status, h_data["hypothesis"], h_data["evidence"])

    # ══════════════════════════════════════════════════════════════════════
    #  CLASS DISTRIBUTION + ENV INFO
    # ══════════════════════════════════════════════════════════════════════
    class_dist_path = ROOT / "artifacts" / "reports" / "class_distribution.json"
    class_dist = load_json(class_dist_path) if class_dist_path.exists() else None
    if class_dist:
        plot_class_distribution(class_dist, fig_dir / "class_distribution.png")

    env_info = environment_info()

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    mcfg = cfg["model"]
    summary = build_summary(
        baseline_metrics=baseline_metrics,
        calibration_info=calibration_info,
        drift_results=drift_results,
        correlation_report=correlation_report,
        risk_thresholds=rcfg,
        risk_stats=risk_stats_summary,
        audit_stats=a_stats,
        hypothesis_results=hyp,
        uncertainty_gate_quantile=uncertainty_gate_q,
        drift_vs_uncertainty_note=(
            "Sample-level gating uses predictive entropy as a risk heuristic; "
            "distribution drift is evaluated at window-level via embedding PSI/KS."
        ),
        class_dist=class_dist,
        env_info=env_info,
        dataset_info={
            "name": "MIT-BIH",
            "split": "inter-patient (DS1/DS2, de Chazal 2004)",
            "window_len": mcfg["input_len"],
            "aami_version": cfg["dataset"]["aami_version"],
        },
        model_info={
            "arch": "1d-cnn",
            "classes": AAMI_CLASSES,
            "n_params": int(model.count_params()),
        },
    )

    save_summary(summary, ROOT / "artifacts" / "reports" / "summary.json")
    log.info("Done — full pipeline complete.")


if __name__ == "__main__":
    main()
