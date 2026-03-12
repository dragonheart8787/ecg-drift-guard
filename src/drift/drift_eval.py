"""Run all drift scenarios at multiple intensities, baseline comparisons, correlation."""

from __future__ import annotations

import numpy as np

from src.common.log import get_logger
from src.common.metrics import compute_metrics
from src.common.stats import correlation_analysis
from src.dataset.label_aami import AAMI_CLASSES
from src.drift.baseline_drift import compute_baseline_drift_score
from src.drift.embedding_drift import compute_drift_score
from src.drift.simulate import INTENSITY_PRESETS, apply_scenario
from src.models.infer import predict_embeddings, predict_proba

log = get_logger(__name__)

INTENSITIES = ["S1", "S2", "S3"]


def _run_single(
    model, embedder, X_test, y_test, ref_E,
    scenario, intensity, base_metrics,
    psi_bins, ks_alpha, w_psi, w_ks, fs,
) -> dict:
    """Evaluate one scenario × one intensity."""
    X_drift = apply_scenario(X_test, scenario, intensity, fs=fs)

    proba_drift = predict_proba(model, X_drift)
    y_pred_drift = np.argmax(proba_drift, axis=1)
    drift_metrics = compute_metrics(y_test, y_pred_drift, AAMI_CLASSES)

    cur_E = predict_embeddings(embedder, X_drift)
    ds_emb = compute_drift_score(ref_E, cur_E,
                                 psi_bins=psi_bins, ks_alpha=ks_alpha,
                                 w_psi=w_psi, w_ks=w_ks)

    # Baseline B: signal-feature drift
    ds_base = compute_baseline_drift_score(X_test, X_drift,
                                           psi_bins=psi_bins, ks_alpha=ks_alpha,
                                           w_psi=w_psi, w_ks=w_ks)

    perf_drop_f1 = round(base_metrics["f1_macro"] - drift_metrics["f1_macro"], 4)
    perf_drop_acc = round(base_metrics["acc"] - drift_metrics["acc"], 4)

    return {
        "scenario": scenario,
        "intensity": intensity,
        # Embedding drift (main)
        "drift_score": ds_emb["score"],
        "psi_mean": ds_emb["psi_mean"],
        "ks_rate": ds_emb["ks_rate"],
        "top_dims": ds_emb["top_dims"],
        "per_dim_psi": ds_emb["per_dim_psi"],
        # Baseline B drift (signal-feature)
        "baseline_drift_score": ds_base["score"],
        "baseline_psi_mean": ds_base["psi_mean"],
        "baseline_top_features": ds_base["top_features"],
        # Performance
        "acc_drift": drift_metrics["acc"],
        "f1_drift": drift_metrics["f1_macro"],
        "perf_drop_f1": perf_drop_f1,
        "perf_drop_acc": perf_drop_acc,
    }


def run_drift_evaluation(
    model,
    embedder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ref_E: np.ndarray,
    *,
    scenarios: list[str] | None = None,
    intensities: list[str] | None = None,
    psi_bins: int = 10,
    ks_alpha: float = 0.01,
    w_psi: float = 0.6,
    w_ks: float = 0.4,
    fs: int = 360,
) -> dict:
    """Full drift evaluation across scenarios × intensities.

    Returns
    -------
    dict with keys:
        baseline_metrics, results (list), correlation, intensity_curve
    """
    scenarios = scenarios or ["noise", "resample", "gain"]
    intensities = intensities or INTENSITIES

    # Baseline performance (Baseline A: model only, no drift/cal/risk)
    proba_base = predict_proba(model, X_test)
    y_pred_base = np.argmax(proba_base, axis=1)
    base_metrics = compute_metrics(y_test, y_pred_base, AAMI_CLASSES)
    log.info("Baseline A — acc=%.4f  f1=%.4f", base_metrics["acc"], base_metrics["f1_macro"])

    results: list[dict] = []

    for scenario in scenarios:
        for intensity in intensities:
            if scenario not in INTENSITY_PRESETS or intensity not in INTENSITY_PRESETS[scenario]:
                continue
            log.info("=== %s @ %s ===", scenario, intensity)
            r = _run_single(model, embedder, X_test, y_test, ref_E,
                            scenario, intensity, base_metrics,
                            psi_bins, ks_alpha, w_psi, w_ks, fs)
            results.append(r)
            log.info("  emb_drift=%.4f  base_drift=%.4f  f1=%.4f  drop=%.4f",
                     r["drift_score"], r["baseline_drift_score"],
                     r["f1_drift"], r["perf_drop_f1"])

    # ── Correlation: H1 — drift_score predicts performance drop ──────────
    drift_scores = [r["drift_score"] for r in results]
    f1_drops = [r["perf_drop_f1"] for r in results]
    acc_drops = [r["perf_drop_acc"] for r in results]

    corr_f1 = correlation_analysis(drift_scores, f1_drops)
    corr_acc = correlation_analysis(drift_scores, acc_drops)

    # Compare embedding vs baseline-B drift → which correlates better?
    base_drift_scores = [r["baseline_drift_score"] for r in results]
    corr_base_f1 = correlation_analysis(base_drift_scores, f1_drops)

    correlation_report = {
        "H1_embedding_vs_f1_drop": corr_f1,
        "H1_embedding_vs_acc_drop": corr_acc,
        "baseline_B_vs_f1_drop": corr_base_f1,
    }

    # ── Intensity curve data (for plotting) ──────────────────────────────
    intensity_curve = {}
    for scenario in scenarios:
        curve = {"intensities": [], "drift_scores": [], "f1_drops": [],
                 "baseline_drift_scores": []}
        for r in results:
            if r["scenario"] == scenario:
                curve["intensities"].append(r["intensity"])
                curve["drift_scores"].append(r["drift_score"])
                curve["f1_drops"].append(r["perf_drop_f1"])
                curve["baseline_drift_scores"].append(r["baseline_drift_score"])
        intensity_curve[scenario] = curve

    return {
        "baseline_metrics": base_metrics,
        "results": results,
        "correlation": correlation_report,
        "intensity_curve": intensity_curve,
    }
