#!/usr/bin/env python
"""Generate test-set per-class metrics and confusion matrix for the report."""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.io import load_npz, load_yaml, save_json
from src.common.metrics import compute_metrics
from src.common.seed import set_seed
from src.dataset.label_aami import AAMI_CLASSES

def main() -> None:
    cfg = load_yaml(ROOT / "config" / "default.yaml")
    set_seed(cfg["seed"])

    test_data = load_npz(ROOT / "data" / "processed" / "beats_test.npz")
    X_test, y_test = test_data["X"], test_data["y"]

    model = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "cnn1d.keras")
    proba = model.predict(X_test, batch_size=512, verbose=0)
    y_pred = np.argmax(proba, axis=1)

    metrics = compute_metrics(y_test, y_pred, AAMI_CLASSES)

    # Per-class table: precision, recall, f1, support
    report = metrics["classification_report"]
    per_class = []
    for name in AAMI_CLASSES:
        if name not in report:
            continue
        row = report[name]
        per_class.append({
            "class": name,
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1-score": round(row["f1-score"], 4),
            "support": int(row["support"]),
        })

    out = {
        "accuracy": metrics["acc"],
        "macro_f1": metrics["f1_macro"],
        "per_class": per_class,
        "confusion_matrix": metrics["confusion_matrix"],
    }
    save_json(out, ROOT / "artifacts" / "reports" / "test_metrics.json")
    print("Saved artifacts/reports/test_metrics.json")


if __name__ == "__main__":
    main()
