#!/usr/bin/env python
"""Policy threshold sensitivity — run 3 threshold configs, record reject/coverage/error."""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.temperature_scaling import apply_temperature
from src.calibration.uncertainty import predictive_entropy, prediction_margin
from src.common.io import load_json, load_npz, load_yaml, save_json
from src.common.metrics import compute_metrics
from src.common.seed import set_seed
from src.dataset.label_aami import AAMI_CLASSES
from src.models.infer import predict_logits, predict_proba
from src.risk.audit import audit_summary, build_decision_rows
from src.risk.policy import evaluate_batch


def main() -> None:
    cfg = load_yaml(ROOT / "config" / "default.yaml")
    set_seed(cfg["seed"])

    test_data = load_npz(ROOT / "data" / "processed" / "beats_test.npz")
    X_test, y_test = test_data["X"], test_data["y"]
    record_ids = test_data.get("record_id", np.array([""] * len(y_test)))

    model = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "cnn1d.keras")
    logits = predict_logits(model, X_test)
    T = load_json(ROOT / "artifacts" / "calibration" / "temperature.json").get("T", 1.0)
    proba = apply_temperature(logits, T)
    y_pred = np.argmax(proba, axis=1)

    entropies = predictive_entropy(proba)
    rcfg = cfg["risk"]["thresholds"]
    drift_results_path = ROOT / "artifacts" / "reports" / "drift_results.json"
    drift_results = load_json(drift_results_path) if drift_results_path.exists() else []
    if drift_results:
        all_scores = np.array([d["drift_score"] for d in drift_results], dtype=float)
        moderate_drift = float(np.median(all_scores))
    else:
        moderate_drift = rcfg["critical"]

    uncertainty_gate_q = 0.9
    high_entropy_thr = float(np.percentile(entropies, uncertainty_gate_q * 100))
    drift_scores_per_sample = np.where(
        entropies >= high_entropy_thr,
        max(moderate_drift, rcfg["critical"] + 0.05),
        0.0,
    ).tolist()

    confidences = np.max(proba, axis=1)

    # 3 threshold configs: conservative, default, liberal
    configs = [
        {
            "name": "conservative",
            "warning": 0.10,
            "critical": 0.15,
            "conf_low": 0.6,
            "conf_mid": 0.8,
        },
        {
            "name": "default",
            "warning": 0.15,
            "critical": 0.22,
            "conf_low": 0.5,
            "conf_mid": 0.7,
        },
        {
            "name": "liberal",
            "warning": 0.20,
            "critical": 0.30,
            "conf_low": 0.4,
            "conf_mid": 0.6,
        },
    ]

    results = []
    for c in configs:
        risk = evaluate_batch(
            drift_scores_per_sample,
            confidences.tolist(),
            thr_warning=c["warning"],
            thr_critical=c["critical"],
            conf_low=c["conf_low"],
            conf_mid=c["conf_mid"],
        )
        rows = build_decision_rows(
            record_ids=record_ids,
            y_true=y_test,
            y_pred=y_pred,
            drift_scores=drift_scores_per_sample,
            confidences=confidences,
            entropies=entropies,
            margins=prediction_margin(proba),
            decisions=risk["decisions"],
            class_names=AAMI_CLASSES,
        )
        a = audit_summary(rows)
        coverage = 1.0 - a["reject_rate"]
        results.append({
            "config": c["name"],
            "thresholds": c,
            "reject_rate": a["reject_rate"],
            "coverage": round(coverage, 4),
            "error_overall": a["error_rate_overall"],
            "error_after_policy": a["error_rate_after_policy"],
            "error_reduction": a["error_rate_reduced"],
        })

    out = {"configs": results}
    save_json(out, ROOT / "artifacts" / "reports" / "threshold_sensitivity.json")
    print("Saved artifacts/reports/threshold_sensitivity.json")
    for r in results:
        print(f"  {r['config']}: reject={r['reject_rate']:.4f}  coverage={r['coverage']:.4f}  error_after={r['error_after_policy']:.4f}")


if __name__ == "__main__":
    main()
