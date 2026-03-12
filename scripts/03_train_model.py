#!/usr/bin/env python
"""Step 3: Train the 1D-CNN on processed NPZ data; save model + embedder."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.io import load_npz, load_yaml, save_json
from src.common.log import get_logger
from src.common.metrics import compute_metrics
from src.common.seed import set_seed
from src.common.stats import bootstrap_ci
from src.dataset.label_aami import AAMI_CLASSES
from src.models.cnn1d import build_cnn1d, build_embedder
from src.models.train import train_model

log = get_logger("03_train_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D-CNN model")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    proc = ROOT / "data" / "processed"
    train_data = load_npz(proc / "beats_train.npz")
    val_data = load_npz(proc / "beats_val.npz")

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    log.info("Train: X=%s  y=%s", X_train.shape, y_train.shape)
    log.info("Val  : X=%s  y=%s", X_val.shape, y_val.shape)

    mcfg = cfg["model"]
    model = build_cnn1d(
        input_len=mcfg["input_len"],
        n_classes=mcfg["n_classes"],
        dropout=mcfg["dropout"],
    )
    model.summary(print_fn=log.info)

    model_path = ROOT / "artifacts" / "models" / "cnn1d.keras"
    train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=mcfg["epochs"],
        batch_size=mcfg["batch_size"],
        lr=mcfg["lr"],
        model_path=model_path,
    )

    # Save embedder
    embedder = build_embedder(model, layer_name="gap")
    emb_path = ROOT / "artifacts" / "models" / "embedder.keras"
    embedder.save(str(emb_path))
    log.info("Embedder saved → %s", emb_path)

    # Evaluate on val + bootstrap CI
    proba = model.predict(X_val, batch_size=512, verbose=0)
    y_pred = np.argmax(proba, axis=1)
    m = compute_metrics(y_val, y_pred, AAMI_CLASSES)
    log.info("Val metrics — acc=%.4f  f1_macro=%.4f", m["acc"], m["f1_macro"])

    from sklearn.metrics import f1_score, accuracy_score
    bcfg = cfg.get("bootstrap", {})
    acc_pt, acc_lo, acc_hi = bootstrap_ci(
        y_val, y_pred, accuracy_score,
        n_boot=bcfg.get("n_boot", 1000), ci=bcfg.get("ci", 0.95),
    )
    f1_fn = lambda yt, yp: float(f1_score(yt, yp, average="macro", zero_division=0))
    f1_pt, f1_lo, f1_hi = bootstrap_ci(
        y_val, y_pred, f1_fn,
        n_boot=bcfg.get("n_boot", 1000), ci=bcfg.get("ci", 0.95),
    )
    log.info("Val acc  = %.4f  [%.4f, %.4f] 95%% CI", acc_pt, acc_lo, acc_hi)
    log.info("Val f1   = %.4f  [%.4f, %.4f] 95%% CI", f1_pt, f1_lo, f1_hi)

    # Save training report
    train_report = {
        "val_acc": m["acc"],
        "val_f1_macro": m["f1_macro"],
        "val_acc_ci": [acc_lo, acc_hi],
        "val_f1_ci": [f1_lo, f1_hi],
        "confusion_matrix": m["confusion_matrix"],
        "n_params": int(model.count_params()),
    }
    save_json(train_report, ROOT / "artifacts" / "reports" / "train_report.json")
    log.info("Done.")


if __name__ == "__main__":
    main()
