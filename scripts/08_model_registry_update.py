#!/usr/bin/env python
"""Step 8: Update model registry with current model version, metrics, and calibration."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.io import load_json, load_yaml
from src.common.log import get_logger
from src.common.versioning import (
    build_registry_entry,
    generate_version_id,
    update_registry,
)

log = get_logger("08_model_registry")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update model registry")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--notes", default="", help="Optional notes for this version")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_path = Path(args.config)

    # Gather metrics
    reports = ROOT / "artifacts" / "reports"
    train_report = {}
    if (reports / "train_report.json").exists():
        train_report = load_json(reports / "train_report.json")

    baseline_metrics = {}
    if (reports / "baseline_metrics.json").exists():
        baseline_metrics = load_json(reports / "baseline_metrics.json")

    metrics = {
        "val_acc": train_report.get("val_acc"),
        "val_f1_macro": train_report.get("val_f1_macro"),
        "val_acc_ci": train_report.get("val_acc_ci"),
        "val_f1_ci": train_report.get("val_f1_ci"),
        "test_acc": baseline_metrics.get("acc"),
        "test_f1_macro": baseline_metrics.get("f1_macro"),
    }

    # Calibration T
    cal_T = None
    cal_path = ROOT / "artifacts" / "calibration" / "temperature.json"
    if cal_path.exists():
        cal_T = load_json(cal_path).get("T")

    # Version ID
    version_id = generate_version_id(cfg["seed"], config_path)

    entry = build_registry_entry(
        version_id=version_id,
        model_path=ROOT / "artifacts" / "models" / "cnn1d.keras",
        config_path=config_path,
        metrics=metrics,
        calibration_T=cal_T,
        notes=args.notes,
    )

    registry_path = ROOT / "artifacts" / "models" / "model_registry.json"
    update_registry(registry_path, entry)

    log.info("Registered version: %s", version_id)
    for k, v in entry.items():
        log.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
