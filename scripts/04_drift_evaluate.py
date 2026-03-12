#!/usr/bin/env python
"""Step 4: Run drift scenarios at multiple intensities; baseline comparison; correlation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.io import load_npz, load_yaml, save_json
from src.common.log import get_logger
from src.common.seed import set_seed
from src.drift.drift_eval import run_drift_evaluation
from src.models.infer import predict_embeddings
from src.viz.plots import (
    plot_correlation_scatter,
    plot_drift_curve,
    plot_intensity_curve,
    plot_perf_vs_drift,
    plot_top_feature_shift,
)

log = get_logger("04_drift_evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift evaluation with intensity sweep")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    test_data = load_npz(ROOT / "data" / "processed" / "beats_test.npz")
    X_test, y_test = test_data["X"], test_data["y"]
    log.info("Test: X=%s  y=%s", X_test.shape, y_test.shape)

    model = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "cnn1d.keras")
    embedder = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "embedder.keras")

    ref_E = predict_embeddings(embedder, X_test)

    dcfg = cfg["drift"]
    eval_result = run_drift_evaluation(
        model, embedder, X_test, y_test, ref_E,
        scenarios=dcfg["scenarios"],
        intensities=dcfg.get("intensities", ["S1", "S2", "S3"]),
        psi_bins=dcfg["psi_bins"],
        ks_alpha=dcfg["ks_alpha"],
        w_psi=dcfg["aggregate"]["w_psi"],
        w_ks=dcfg["aggregate"]["w_ks"],
        fs=cfg["dataset"]["fs"],
    )

    # Save full drift results
    save_json(eval_result["results"], ROOT / "artifacts" / "reports" / "drift_results.json")
    save_json(eval_result["correlation"], ROOT / "artifacts" / "reports" / "drift_correlation.json")
    save_json(eval_result["baseline_metrics"], ROOT / "artifacts" / "reports" / "baseline_metrics.json")

    fig_dir = ROOT / "artifacts" / "reports" / "figures"

    # 1) Drift curve (all scenario×intensity)
    plot_drift_curve(eval_result["results"], fig_dir / "drift_curve.png")

    # 2) Performance vs drift
    plot_perf_vs_drift(eval_result["results"], fig_dir / "perf_vs_drift.png")

    # 3) Intensity curve (S1→S2→S3)
    plot_intensity_curve(eval_result["intensity_curve"], fig_dir / "intensity_curve.png")

    # 4) Correlation scatter (H1 evidence)
    h1_corr = eval_result["correlation"].get("H1_embedding_vs_f1_drop", {})
    plot_correlation_scatter(eval_result["results"], h1_corr, fig_dir / "correlation_scatter.png")

    # 5) Top-feature-shift from worst scenario
    worst = max(eval_result["results"], key=lambda d: d["drift_score"])
    if "per_dim_psi" in worst and "top_dims" in worst:
        plot_top_feature_shift(worst["per_dim_psi"], worst["top_dims"],
                               fig_dir / "top_feature_shift.png")

    # Log correlation summary
    corr = eval_result["correlation"]
    for key, val in corr.items():
        log.info("Correlation [%s]: Spearman ρ=%s, p=%s",
                 key, val.get("spearman_r"), val.get("spearman_p"))

    log.info("Done — figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()
