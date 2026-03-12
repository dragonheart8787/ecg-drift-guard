#!/usr/bin/env python
"""Step 7: External validation — run trained model on a different ECG database.

Demonstrates cross-dataset drift detection, OOD scoring, and policy behaviour
WITHOUT retraining.  This is the strongest evidence of generalisability.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.ece import compute_brier, compute_ece
from src.calibration.temperature_scaling import apply_temperature
from src.calibration.uncertainty import confidence_analysis
from src.common.io import load_json, load_npz, load_yaml, save_json
from src.common.log import get_logger
from src.common.metrics import compute_metrics
from src.common.seed import set_seed
from src.dataset.external_loader import load_external_beats
from src.dataset.label_aami import AAMI_CLASSES
from src.drift.embedding_drift import compute_drift_score
from src.drift.ood import MahalanobisOOD, classify_ood, compute_ood_stats
from src.models.infer import predict_embeddings, predict_logits, predict_proba
from src.risk.policy import evaluate_batch
from src.viz.plots import (
    plot_confidence_distribution,
    plot_reliability,
)

log = get_logger("07_external_validation")


def main() -> None:
    parser = argparse.ArgumentParser(description="External dataset validation")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--ext-db", default="svdb", help="External PhysioNet DB name")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    ext_cfg = cfg.get("external", {})
    ext_db = args.ext_db or ext_cfg.get("db_name", "svdb")
    ext_dir = ROOT / "data" / "raw" / ext_db

    # ── Load models ──────────────────────────────────────────────────────
    model = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "cnn1d.keras")
    embedder = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "embedder.keras")

    # ── Load reference data (internal test) for drift comparison ─────────
    test_data = load_npz(ROOT / "data" / "processed" / "beats_test.npz")
    X_ref, y_ref = test_data["X"], test_data["y"]
    ref_E = predict_embeddings(embedder, X_ref)

    # ── Load external beats ──────────────────────────────────────────────
    log.info("Loading external dataset: %s", ext_db)
    X_ext, y_ext, rid_ext = load_external_beats(
        db_name=ext_db,
        db_dir=ext_dir,
        pre_sec=cfg["dataset"]["window"]["pre_sec"],
        post_sec=cfg["dataset"]["window"]["post_sec"],
        norm_mode=cfg["dataset"]["normalize"],
        max_records=args.max_records,
    )
    log.info("External: X=%s  y=%s", X_ext.shape, y_ext.shape)

    # ── Inference on external ────────────────────────────────────────────
    proba_ext = predict_proba(model, X_ext)
    y_pred_ext = np.argmax(proba_ext, axis=1)
    metrics_ext = compute_metrics(y_ext, y_pred_ext, AAMI_CLASSES)
    log.info("External raw — acc=%.4f  f1=%.4f", metrics_ext["acc"], metrics_ext["f1_macro"])

    # Per-class metrics for report
    report = metrics_ext["classification_report"]
    per_class_ext = []
    for name in AAMI_CLASSES:
        if name not in report:
            continue
        row = report[name]
        per_class_ext.append({
            "class": name,
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1-score": round(row["f1-score"], 4),
            "support": int(row["support"]),
        })

    # ── Calibration on external ──────────────────────────────────────────
    cal_path = ROOT / "artifacts" / "calibration" / "temperature.json"
    T = load_json(cal_path)["T"] if cal_path.exists() else 1.0

    logits_ext = predict_logits(model, X_ext)
    proba_cal = apply_temperature(logits_ext, T)
    ece_ext = compute_ece(proba_cal, y_ext)
    brier_ext = compute_brier(proba_cal, y_ext)
    log.info("External calibrated — ECE=%.4f  Brier=%.4f", ece_ext, brier_ext)

    # ── Embedding drift: external vs internal ────────────────────────────
    ext_E = predict_embeddings(embedder, X_ext)
    dcfg = cfg["drift"]
    drift_ext = compute_drift_score(
        ref_E, ext_E,
        psi_bins=dcfg["psi_bins"], ks_alpha=dcfg["ks_alpha"],
        w_psi=dcfg["aggregate"]["w_psi"], w_ks=dcfg["aggregate"]["w_ks"],
    )
    log.info("External drift_score = %.4f  (psi=%.4f, ks_rate=%.4f)",
             drift_ext["score"], drift_ext["psi_mean"], drift_ext["ks_rate"])

    # ── OOD detection ────────────────────────────────────────────────────
    ood_model = MahalanobisOOD()
    ood_model.fit(ref_E, y_ref)

    ref_ood_scores = ood_model.score_batch(ref_E)
    ref_ood_stats = compute_ood_stats(ref_ood_scores, percentile_threshold=95.0)

    ext_ood_scores = ood_model.score_batch(ext_E)
    ext_ood_mask = classify_ood(ext_ood_scores, ref_ood_stats["threshold"])
    ood_rate = float(ext_ood_mask.mean())
    log.info("External OOD rate (Mahalanobis): %.2f%%", ood_rate * 100)

    # ── Risk policy on external ──────────────────────────────────────────
    rcfg = cfg["risk"]["thresholds"]
    ext_confs = np.max(proba_cal, axis=1).tolist()
    ext_drift_per_sample = [drift_ext["score"]] * len(ext_confs)

    risk_ext = evaluate_batch(
        ext_drift_per_sample, ext_confs,
        thr_warning=rcfg["warning"], thr_critical=rcfg["critical"],
        conf_low=rcfg["conf_low"], conf_mid=rcfg["conf_mid"],
    )
    risk_summary = {k: v for k, v in risk_ext.items() if k != "decisions"}

    # ── Confidence analysis ──────────────────────────────────────────────
    conf_ext = confidence_analysis(proba_cal, y_ext)

    # ── Figures ──────────────────────────────────────────────────────────
    fig_dir = ROOT / "artifacts" / "reports" / "figures"
    from src.calibration.ece import reliability_diagram_data
    rel_ext = reliability_diagram_data(proba_cal, y_ext)
    plot_reliability(rel_ext, ece_ext, f"External ({ext_db}) Calibrated",
                     fig_dir / "external_reliability.png")
    plot_confidence_distribution(conf_ext, fig_dir / "external_confidence_dist.png")

    # OOD score distribution: internal vs external
    from src.viz.plots import plot_ood_distribution
    plot_ood_distribution(ref_ood_scores, ext_ood_scores, ref_ood_stats["threshold"],
                          fig_dir / "ood_score_distribution.png")

    # Drift comparison: internal (clean) vs external
    from src.viz.plots import plot_external_drift_comparison
    int_drift_path = ROOT / "artifacts" / "reports" / "drift_results.json"
    int_drift = load_json(int_drift_path) if int_drift_path.exists() else []
    plot_external_drift_comparison(int_drift, drift_ext, ext_db,
                                   fig_dir / "external_drift_vs_internal.png")

    # ── Save results ─────────────────────────────────────────────────────
    ext_summary = {
        "external_db": ext_db,
        "n_beats": int(len(y_ext)),
        "metrics": {
            "acc": metrics_ext["acc"],
            "f1_macro": metrics_ext["f1_macro"],
            "ece": round(ece_ext, 4),
            "brier": round(brier_ext, 4),
        },
        "per_class": per_class_ext,
        "confusion_matrix": metrics_ext["confusion_matrix"],
        "drift_vs_internal": {
            "drift_score": drift_ext["score"],
            "psi_mean": drift_ext["psi_mean"],
            "ks_rate": drift_ext["ks_rate"],
            "top_dims": drift_ext["top_dims"],
        },
        "ood": {
            "method": "mahalanobis",
            "threshold": ref_ood_stats["threshold"],
            "ood_rate": round(ood_rate, 4),
            "ref_stats": ref_ood_stats,
        },
        "risk_policy": risk_summary,
        "note": (
            "External validation without retraining. "
            "Drift score and OOD rate reflect distribution mismatch "
            "between MIT-BIH training data and the external dataset."
        ),
    }
    save_json(ext_summary, ROOT / "artifacts" / "reports" / "external_summary.json")
    log.info("Done — external validation saved.")


if __name__ == "__main__":
    main()
