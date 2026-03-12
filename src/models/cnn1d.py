"""1D-CNN architecture for ECG beat classification + embedding extractor."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn1d(input_len: int, n_classes: int, dropout: float = 0.3) -> keras.Model:
    """Three-block 1D-CNN → GlobalAveragePooling → Dense softmax.

    Layer naming convention makes it easy to extract embeddings later.
    """
    inp = layers.Input(shape=(input_len, 1), name="ecg_input")

    x = layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(2, name="pool1")(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(2, name="pool2")(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="ecg_cnn1d")
    return model


def build_embedder(model: keras.Model, layer_name: str = "gap") -> keras.Model:
    """Create a sub-model that outputs the embedding from *layer_name*."""
    emb_layer = model.get_layer(layer_name).output
    return keras.Model(inputs=model.input, outputs=emb_layer, name="ecg_embedder")
