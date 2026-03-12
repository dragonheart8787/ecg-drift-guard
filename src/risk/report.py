"""Assemble the final summary.json with hypotheses, failure modes, model card."""

from __future__ import annotations

from pathlib import Path

from src.common.io import save_json
from src.common.log import get_logger

log = get_logger(__name__)

# ── Hypotheses ────────────────────────────────────────────────────────────

HYPOTHESES = {
    "H1": "When drift_score increases, model performance (F1/accuracy) drops significantly.",
    "H2": "Temperature Scaling reduces ECE and aligns confidence with accuracy.",
    "H3": "Risk policy (reject/degrade) reduces error rate at acceptable rejection cost.",
}

# ── Failure Modes ─────────────────────────────────────────────────────────

FAILURE_MODES = [
    {
        "mode": "Sampling rate mismatch",
        "effect": "Beat waveform shape distorted → increased misclassification",
        "detection": "drift_score ↑ (resample scenario)",
        "mitigation": "warning/critical → degrade to binary",
        "residual_risk": "Extreme mismatch (<100 Hz) may not be recoverable",
    },
    {
        "mode": "Powerline noise (50/60 Hz)",
        "effect": "V/S class misclassification ↑",
        "detection": "PSI/KS ↑ on embedding dims sensitive to HF noise",
        "mitigation": "reject or warn + suggest re-acquisition",
        "residual_risk": "Noise buried in QRS may evade detection",
    },
    {
        "mode": "Gain/amplitude shift",
        "effect": "Normalisation partially compensates but extreme shifts break features",
        "detection": "drift_score ↑ (gain scenario)",
        "mitigation": "warning at moderate, critical at extreme",
        "residual_risk": "Saturated signals lose information permanently",
    },
    {
        "mode": "Class prior shift",
        "effect": "Certain classes over/under-represented → biased recall",
        "detection": "Prior distribution monitor (separate from feature drift)",
        "mitigation": "Alert for retraining; flag in audit log",
        "residual_risk": "Requires labelled deployment data to fully detect",
    },
    {
        "mode": "Concept drift (pathology evolution)",
        "effect": "P(Y|X) changes → model silently degrades",
        "detection": "Performance monitoring with labelled feedback loop",
        "mitigation": "Periodic retrain triggers; external validation",
        "residual_risk": "Cannot detect without ground truth labels",
    },
]

# ── Model Card ────────────────────────────────────────────────────────────

MODEL_CARD = {
    "model_name": "ECG Drift Guard — 1D-CNN Beat Classifier with Safety Layer",
    "version": "1.0",
    "purpose": "Research demonstration of deployment-time safety monitoring for ECG classification. NOT a medical device. NOT for clinical diagnosis.",
    "intended_use": "Educational / research: illustrate drift detection, calibration, and risk-policy for medical AI.",
    "not_intended_for": "Clinical decision-making, patient diagnosis, regulatory submission.",
    "training_data": "MIT-BIH Arrhythmia Database (48 records, 109k+ beats), AAMI 5-class mapping.",
    "split_method": "DS1/DS2 inter-patient split (de Chazal 2004) to prevent patient leakage.",
    "evaluation_metrics": "Accuracy, F1 macro, ECE, Brier score, drift_score (PSI+KS), reject rate.",
    "limitations": [
        "Single-lead ECG only (MLII); multi-lead not supported.",
        "Trained on MIT-BIH only; external validation on other datasets not performed.",
        "Drift detection assumes covariate shift; concept drift requires labelled feedback.",
        "Risk thresholds are heuristic; clinical deployment requires rigorous threshold tuning.",
    ],
    "ethical_considerations": "This system is for research purposes only. Any medical application must comply with local regulations (e.g., FDA, CE marking) and undergo clinical validation.",
}

# ── Concept Drift Playbook ────────────────────────────────────────────────

CONCEPT_DRIFT_PLAYBOOK = {
    "definition": (
        "Concept drift occurs when P(Y|X) changes — the relationship between "
        "input features and labels shifts over time, even if the input distribution "
        "stays the same. Unlike covariate shift (P(X) changes), concept drift "
        "cannot be detected from features alone."
    ),
    "detection_requirements": [
        "Delayed ground-truth labels from clinical feedback loop",
        "Periodic performance audits comparing predicted vs actual labels",
        "Monitoring per-class recall / precision over time windows",
    ],
    "trigger_conditions": {
        "retrain": (
            "Trigger when: (1) test F1 drops >5% relative to baseline, sustained "
            "over 2+ evaluation windows; OR (2) per-class recall for critical "
            "classes (V, S) drops below acceptable threshold."
        ),
        "freeze": (
            "Freeze the model when: no significant performance change detected; "
            "keep current version with documented performance metrics."
        ),
        "rollback": (
            "Rollback to previous model version when: newly retrained model "
            "performs worse on held-out validation set; use model_registry.json "
            "to identify last-known-good version."
        ),
    },
    "delayed_label_strategy": (
        "In clinical settings, labels may arrive days/weeks after inference. "
        "Strategy: (1) Buffer predictions with confidence scores; "
        "(2) When labels arrive, compute delayed performance metrics; "
        "(3) Flag if delayed metrics diverge from initial calibrated confidence."
    ),
    "monitoring_cadence": {
        "covariate_drift": "Continuous (every batch)",
        "calibration_check": "Daily / weekly",
        "concept_drift_audit": "Weekly / monthly (requires labels)",
        "full_revalidation": "Quarterly or on major system changes",
    },
}


def verify_hypotheses(
    *,
    correlation_report: dict,
    calibration_info: dict,
    audit_stats: dict,
) -> dict:
    """Check H1–H3 against computed evidence."""
    h_results = {}

    # H1: drift_score ↑ → performance ↓
    h1 = correlation_report.get("H1_embedding_vs_f1_drop", {})
    sr = h1.get("spearman_r")
    sp = h1.get("spearman_p")
    h_results["H1"] = {
        "hypothesis": HYPOTHESES["H1"],
        "spearman_r": sr,
        "spearman_p": sp,
        "supported": sr is not None and sr > 0.5 and (sp is not None and sp < 0.1),
        "evidence": f"Spearman ρ={sr}, p={sp}",
    }

    # H2: Temperature Scaling reduces ECE
    ece_b = calibration_info.get("ece_before", 1)
    ece_a = calibration_info.get("ece_after", 1)
    brier_b = calibration_info.get("brier_before")
    brier_a = calibration_info.get("brier_after")
    h_results["H2"] = {
        "hypothesis": HYPOTHESES["H2"],
        "ece_before": ece_b,
        "ece_after": ece_a,
        "ece_reduction_pct": round((ece_b - ece_a) / (ece_b + 1e-12) * 100, 1),
        "brier_before": brier_b,
        "brier_after": brier_a,
        "supported": ece_a < ece_b,
        "evidence": f"ECE {ece_b:.4f} → {ece_a:.4f}",
    }

    # H3: Risk policy reduces error rate
    err_overall = audit_stats.get("error_rate_overall", 0)
    err_after = audit_stats.get("error_rate_after_policy", 0)
    reject_r = audit_stats.get("reject_rate", 0)
    h_results["H3"] = {
        "hypothesis": HYPOTHESES["H3"],
        "error_rate_overall": err_overall,
        "error_rate_after_policy": err_after,
        "error_reduction": round(err_overall - err_after, 4),
        "reject_rate": reject_r,
        "supported": err_after < err_overall,
        "evidence": f"Error {err_overall:.4f} → {err_after:.4f} (reject {reject_r:.2%})",
    }

    return h_results


def build_summary(
    *,
    baseline_metrics: dict,
    calibration_info: dict,
    drift_results: list[dict],
    correlation_report: dict,
    risk_thresholds: dict,
    risk_stats: dict,
    audit_stats: dict,
    hypothesis_results: dict,
    uncertainty_gate_quantile: float | None = None,
    drift_vs_uncertainty_note: str | None = None,
    class_dist: dict | None = None,
    env_info: dict | None = None,
    dataset_info: dict | None = None,
    model_info: dict | None = None,
) -> dict:
    summary = {
        "scope_statement": (
            "This system is a deployment-time safety layer for ECG classification. "
            "It is NOT a diagnostic tool and is intended for research/education only."
        ),
        "dataset": dataset_info or {"name": "MIT-BIH", "split": "inter-patient"},
        "split_note": (
            "We adopt the DS1/DS2 inter-patient split (de Chazal 2004) "
            "to prevent patient leakage and ensure fair evaluation."
        ),
        "model": model_info or {"arch": "1d-cnn", "classes": ["N", "S", "V", "F", "Q"]},
        "class_distribution": class_dist,
        "baseline": {
            "acc": baseline_metrics.get("acc"),
            "f1_macro": baseline_metrics.get("f1_macro"),
        },
        "calibration": calibration_info,
        "drift": [
            {
                "scenario": d["scenario"],
                "intensity": d.get("intensity", "S2"),
                "drift_score": d["drift_score"],
                "baseline_drift_score": d.get("baseline_drift_score"),
                "perf_drop_f1": d["perf_drop_f1"],
            }
            for d in drift_results
        ],
        "correlation": correlation_report,
        "hypotheses": hypothesis_results,
        "risk_policy": {"thresholds": risk_thresholds},
        "risk_results": risk_stats,
        "audit_summary": audit_stats,
        "uncertainty_gate_quantile": uncertainty_gate_quantile,
        "drift_vs_uncertainty_note": drift_vs_uncertainty_note,
        "failure_modes": FAILURE_MODES,
        "concept_drift_playbook": CONCEPT_DRIFT_PLAYBOOK,
        "model_card": MODEL_CARD,
        "environment": env_info,
    }
    return summary


def save_summary(summary: dict, path: str | Path) -> None:
    save_json(summary, path)
    log.info("Summary saved → %s", path)
