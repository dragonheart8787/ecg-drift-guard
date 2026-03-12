"""Risk policy engine — map (drift_score, confidence, ood_score) → action + reason code."""

from __future__ import annotations

from src.common.log import get_logger

log = get_logger(__name__)


def decide(
    drift_score: float,
    confidence: float,
    *,
    ood_flag: bool = False,
    thr_warning: float = 0.15,
    thr_critical: float = 0.22,
    conf_low: float = 0.5,
    conf_mid: float = 0.7,
) -> dict:
    """Determine risk level, action, reason code, and message.

    Parameters
    ----------
    ood_flag : if True, the sample is flagged as out-of-distribution

    Returns
    -------
    dict with keys: level, action, reason_code, message
    """
    # --- OOD override (highest priority) --------------------------------------
    if ood_flag:
        return {
            "level": "critical",
            "action": "reject",
            "reason_code": "OOD_DETECTED",
            "message": "樣本超出訓練分布（OOD）：拒絕輸出，建議人工覆核資料來源",
        }

    # --- Critical drift -------------------------------------------------------
    if drift_score >= thr_critical:
        if confidence < conf_low:
            return {
                "level": "critical",
                "action": "reject",
                "reason_code": "DRIFT_CRIT_CONF_LOW",
                "message": "高漂移 + 低可信度：拒絕輸出，建議人工覆核",
            }
        return {
            "level": "critical",
            "action": "degrade",
            "reason_code": "DRIFT_CRIT_DEGRADE",
            "message": "高漂移：降級為二分類（正常 vs 異常）並提示醫護確認",
        }

    # --- Warning drift --------------------------------------------------------
    if drift_score >= thr_warning:
        if confidence < conf_mid:
            return {
                "level": "warning",
                "action": "warn",
                "reason_code": "DRIFT_WARN_CONF_MID",
                "message": "中度漂移 + 中低可信度：接受結果但標示警示，建議人工抽查",
            }
        return {
            "level": "warning",
            "action": "accept",
            "reason_code": "DRIFT_WARN_CONF_OK",
            "message": "中度漂移但可信度尚可：接受結果，記錄漂移資訊",
        }

    # --- Normal ---------------------------------------------------------------
    if confidence < conf_low:
        return {
            "level": "normal",
            "action": "warn",
            "reason_code": "CONF_LOW",
            "message": "漂移正常但可信度低：接受結果並標示警示",
        }
    return {
        "level": "normal",
        "action": "accept",
        "reason_code": "NORMAL",
        "message": "資料品質正常，模型輸出可信",
    }


def evaluate_batch(
    drift_scores: list[float],
    confidences: list[float],
    ood_flags: list[bool] | None = None,
    **kwargs,
) -> dict:
    """Apply policy to a batch and compute aggregate risk statistics."""
    if ood_flags is None:
        ood_flags = [False] * len(drift_scores)

    decisions = [
        decide(ds, cf, ood_flag=ood, **kwargs)
        for ds, cf, ood in zip(drift_scores, confidences, ood_flags)
    ]

    total = len(decisions)
    reject_count = sum(1 for d in decisions if d["action"] == "reject")
    degrade_count = sum(1 for d in decisions if d["action"] == "degrade")
    warn_count = sum(1 for d in decisions if d["action"] == "warn")
    ood_count = sum(1 for d in decisions if d["reason_code"] == "OOD_DETECTED")

    return {
        "total": total,
        "reject_rate": round(reject_count / total, 4) if total else 0,
        "degrade_rate": round(degrade_count / total, 4) if total else 0,
        "warn_rate": round(warn_count / total, 4) if total else 0,
        "accept_rate": round(1 - (reject_count + degrade_count + warn_count) / total, 4) if total else 0,
        "ood_reject_count": ood_count,
        "decisions": decisions,
    }
