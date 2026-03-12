#!/usr/bin/env python
"""Step 6 (optional): Generate a standalone hypothesis-verification report.

Reads all artifacts and produces a concise text + JSON summary of H1–H3.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.io import load_json, save_json
from src.common.log import get_logger

log = get_logger("06_hypothesis_report")

SEPARATOR = "=" * 72


def _fmt(val, default="N/A"):
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hypothesis verification report")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    reports = ROOT / "artifacts" / "reports"
    summary = load_json(reports / "summary.json")

    hyp = summary.get("hypotheses", {})
    cal = summary.get("calibration", {})
    audit = summary.get("audit_summary", {})
    corr = summary.get("correlation", {})

    lines: list[str] = []
    lines.append(SEPARATOR)
    lines.append("  ECG DRIFT GUARD — HYPOTHESIS VERIFICATION REPORT")
    lines.append(SEPARATOR)
    lines.append("")

    # H1
    h1 = hyp.get("H1", {})
    lines.append("[H1] " + h1.get("hypothesis", ""))
    lines.append(f"  Spearman ρ = {_fmt(h1.get('spearman_r'))}")
    lines.append(f"  p-value    = {_fmt(h1.get('spearman_p'))}")
    lines.append(f"  Result     : {'✓ SUPPORTED' if h1.get('supported') else '✗ NOT SUPPORTED'}")
    # Embedding vs Baseline B comparison
    b_corr = corr.get("baseline_B_vs_f1_drop", {})
    lines.append(f"  Baseline B (signal features) Spearman ρ = {_fmt(b_corr.get('spearman_r'))}")
    lines.append(f"  → Embedding drift {'outperforms' if (h1.get('spearman_r') or 0) > (b_corr.get('spearman_r') or 0) else 'comparable to'} signal-feature drift")
    lines.append("")

    # H2
    h2 = hyp.get("H2", {})
    lines.append("[H2] " + h2.get("hypothesis", ""))
    lines.append(f"  ECE   : {_fmt(cal.get('ece_before'))} → {_fmt(cal.get('ece_after'))}")
    lines.append(f"  Brier : {_fmt(cal.get('brier_before'))} → {_fmt(cal.get('brier_after'))}")
    lines.append(f"  T     = {_fmt(cal.get('temperature_T'))}")
    lines.append(f"  ECE reduction = {_fmt(h2.get('ece_reduction_pct'))}%")
    lines.append(f"  Result: {'✓ SUPPORTED' if h2.get('supported') else '✗ NOT SUPPORTED'}")
    lines.append("")

    # H3
    h3 = hyp.get("H3", {})
    lines.append("[H3] " + h3.get("hypothesis", ""))
    lines.append(f"  Error rate (overall)       = {_fmt(audit.get('error_rate_overall'))}")
    lines.append(f"  Error rate (after policy)  = {_fmt(audit.get('error_rate_after_policy'))}")
    lines.append(f"  Error reduction            = {_fmt(h3.get('error_reduction'))}")
    lines.append(f"  Reject rate                = {_fmt(audit.get('reject_rate'))}")
    lines.append(f"  Result: {'✓ SUPPORTED' if h3.get('supported') else '✗ NOT SUPPORTED'}")
    lines.append("")

    # Failure modes
    lines.append(SEPARATOR)
    lines.append("  FAILURE MODES")
    lines.append(SEPARATOR)
    for fm in summary.get("failure_modes", []):
        lines.append(f"  Mode      : {fm['mode']}")
        lines.append(f"  Effect    : {fm['effect']}")
        lines.append(f"  Detection : {fm['detection']}")
        lines.append(f"  Mitigation: {fm['mitigation']}")
        lines.append(f"  Residual  : {fm['residual_risk']}")
        lines.append("")

    # Scope / disclaimer
    lines.append(SEPARATOR)
    lines.append(f"  SCOPE: {summary.get('scope_statement', '')}")
    lines.append(f"  SPLIT: {summary.get('split_note', '')}")
    lines.append(SEPARATOR)

    report_text = "\n".join(lines)
    print(report_text)

    # Save text version
    report_path = reports / "hypothesis_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Report saved → %s", report_path)


if __name__ == "__main__":
    main()
