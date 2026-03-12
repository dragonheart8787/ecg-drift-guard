"""Batch inference: logits, probabilities, embeddings."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.common.log import get_logger

log = get_logger(__name__)


def predict_proba(model: keras.Model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Return softmax probabilities, shape (N, C)."""
    return model.predict(X, batch_size=batch_size, verbose=0)


def predict_logits(model: keras.Model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Build a logit-output model (strip final softmax) and predict.

    Works by creating a model that outputs the pre-softmax Dense layer.
    """
    dense_layer = model.get_layer("output")
    weights, bias = dense_layer.get_weights()

    gap_model = keras.Model(inputs=model.input,
                            outputs=model.get_layer("drop").output,
                            name="logit_extractor")
    gap_out = gap_model.predict(X, batch_size=batch_size, verbose=0)
    logits = gap_out @ weights + bias
    return logits.astype(np.float32)


def predict_embeddings(embedder: keras.Model, X: np.ndarray,
                       batch_size: int = 512) -> np.ndarray:
    """Return embedding vectors, shape (N, D)."""
    return embedder.predict(X, batch_size=batch_size, verbose=0)
