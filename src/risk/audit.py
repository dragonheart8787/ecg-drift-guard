"""Audit trail — produce decisions.csv for every inference."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.common.log import get_logger

log = get_logger(__name__)

REASON_CODES = {
    ("critical", "reject"): "DRIFT_CRIT_CONF_LOW",
    ("critical", "degrade"): "DRIFT_CRIT_DEGRADE",
    ("warning", "warn"): "DRIFT_WARN_CONF_MID",
    ("warning", "accept"): "DRIFT_WARN_CONF_OK",
    ("normal", "accept"): "NORMAL",
}

CSV_COLUMNS = [
    "sample_idx",
    "record_id",
    "true_label",
    "pred_label",
    "drift_score",
    "confidence",
    "entropy",
    "margin",
    "policy_level",
    "action",
    "reason_code",
]


def build_decision_rows(
    *,
    record_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drift_scores: list[float] | np.ndarray,
    confidences: np.ndarray,
    entropies: np.ndarray,
    margins: np.ndarray,
    decisions: list[dict],
    class_names: list[str] | None = None,
) -> list[dict]:
    """Assemble one row per sample for the audit log."""
    rows: list[dict] = []
    for i in range(len(y_true)):
        d = decisions[i]
        level = d["level"]
        action = d["action"]
        reason = REASON_CODES.get((level, action), "UNKNOWN")
        tl = class_names[int(y_true[i])] if class_names else str(y_true[i])
        pl = class_names[int(y_pred[i])] if class_names else str(y_pred[i])
        rows.append({
            "sample_idx": i,
            "record_id": str(record_ids[i]) if record_ids is not None else "",
            "true_label": tl,
            "pred_label": pl,
            "drift_score": round(float(drift_scores[i]), 4),
            "confidence": round(float(confidences[i]), 4),
            "entropy": round(float(entropies[i]), 4),
            "margin": round(float(margins[i]), 4),
            "policy_level": level,
            "action": action,
            "reason_code": reason,
        })
    return rows


def save_decisions_csv(rows: list[dict], path: str | Path) -> None:
    """Write audit trail to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Audit trail saved → %s  (%d rows)", path, len(rows))


def audit_summary(rows: list[dict]) -> dict:
    """Aggregate statistics from decision rows."""
    total = len(rows)
    if total == 0:
        return {}
    actions = [r["action"] for r in rows]
    reasons = [r["reason_code"] for r in rows]
    from collections import Counter
    action_counts = dict(Counter(actions))
    reason_counts = dict(Counter(reasons))

    # Error rate among auto-decided samples (accept + degrade + warn)
    auto = [r for r in rows if r["action"] in ("accept", "degrade", "warn")]
    if auto:
        errors_auto = sum(1 for r in auto if r["true_label"] != r["pred_label"])
        error_rate_after = round(errors_auto / len(auto), 4)
    else:
        error_rate_after = 0.0

    # Overall error rate (without risk policy)
    all_errors = sum(1 for r in rows if r["true_label"] != r["pred_label"])
    error_rate_overall = round(all_errors / total, 4)

    # Error reduction
    error_reduction = round(error_rate_overall - error_rate_after, 4) if auto else 0.0

    return {
        "total_samples": total,
        "action_counts": action_counts,
        "reason_counts": reason_counts,
        "error_rate_overall": error_rate_overall,
        "error_rate_after_policy": error_rate_after,
        "error_rate_reduced": error_reduction,
        "reject_rate": round(action_counts.get("reject", 0) / total, 4),
    }
