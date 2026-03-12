"""R-peak-aligned beat extraction and normalisation."""

from __future__ import annotations

import numpy as np

from src.dataset.label_aami import symbol_to_aami


def extract_beats(
    signal: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: list[str],
    pre_samples: int,
    post_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut fixed-length windows around each R-peak.

    Returns
    -------
    X : (N, window_len) float32
    y : (N,)            int64  (AAMI class id)
    keep_mask : (N_original,) bool — which annotations were kept
    """
    sig_len = len(signal)
    window_len = pre_samples + post_samples

    beats: list[np.ndarray] = []
    labels: list[int] = []
    keep: list[bool] = []

    for s, sym in zip(ann_samples, ann_symbols):
        aami = symbol_to_aami(sym)
        if aami is None:
            keep.append(False)
            continue
        start = s - pre_samples
        end = s + post_samples
        if start < 0 or end > sig_len:
            keep.append(False)
            continue
        beats.append(signal[start:end])
        labels.append(aami)
        keep.append(True)

    X = np.array(beats, dtype=np.float32) if beats else np.empty((0, window_len), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, np.array(keep, dtype=bool)


def normalize(x: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Per-beat normalisation.

    Parameters
    ----------
    x : (N, L) or (L,)
    mode : 'zscore' or 'robust'
    """
    axis = -1
    if mode == "zscore":
        mu = x.mean(axis=axis, keepdims=True)
        sigma = x.std(axis=axis, keepdims=True) + 1e-8
        return (x - mu) / sigma
    elif mode == "robust":
        med = np.median(x, axis=axis, keepdims=True)
        iqr = np.percentile(x, 75, axis=axis, keepdims=True) - np.percentile(x, 25, axis=axis, keepdims=True) + 1e-8
        return (x - med) / iqr
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
