"""Training loop with class-weight balancing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.common.log import get_logger

log = get_logger(__name__)


def _class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    """Inverse-frequency weighting (sklearn-style)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    model_path: str | Path | None = None,
) -> keras.callbacks.History:
    n_classes = model.output_shape[-1]
    cw = _class_weights(y_train, n_classes)
    log.info("Class weights: %s", cw)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=callbacks,
        verbose=2,
    )

    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        log.info("Model saved → %s", model_path)

    return history
