"""Inference latency and resource benchmarking."""

from __future__ import annotations

import os
import time

import numpy as np

from src.common.log import get_logger

log = get_logger(__name__)


def measure_latency(
    model,
    input_shape: tuple,
    batch_sizes: list[int] | None = None,
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> list[dict]:
    """Measure inference latency (ms/beat) at various batch sizes.

    Parameters
    ----------
    model : keras.Model
    input_shape : single-sample shape, e.g. (216, 1)
    batch_sizes : list of batch sizes to test
    """
    batch_sizes = batch_sizes or [1, 32, 128, 256, 512]
    results: list[dict] = []

    for bs in batch_sizes:
        X_dummy = np.random.randn(bs, *input_shape).astype(np.float32)

        # Warmup
        for _ in range(n_warmup):
            model.predict(X_dummy, batch_size=bs, verbose=0)

        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            model.predict(X_dummy, batch_size=bs, verbose=0)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        total_ms = np.mean(times) * 1000
        per_beat_ms = total_ms / bs

        results.append({
            "batch_size": bs,
            "total_ms": round(float(total_ms), 2),
            "per_beat_ms": round(float(per_beat_ms), 4),
            "std_ms": round(float(np.std(times) * 1000), 2),
        })
        log.info("Batch %d: %.2f ms total, %.4f ms/beat", bs, total_ms, per_beat_ms)

    return results


def measure_memory() -> dict:
    """Estimate peak memory usage (cross-platform best-effort)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {
            "rss_mb": round(mem.rss / 1024 / 1024, 1),
            "vms_mb": round(mem.vms / 1024 / 1024, 1),
        }
    except ImportError:
        import resource
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return {"max_rss_kb": usage.ru_maxrss}
        except Exception:
            return {"note": "Memory measurement unavailable (install psutil)"}


def get_device_info() -> dict:
    """Report CPU / GPU availability."""
    info: dict = {}
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info["tf_device"] = "GPU" if gpus else "CPU"
        info["gpu_count"] = len(gpus)
        if gpus:
            info["gpu_names"] = [g.name for g in gpus]
    except Exception:
        info["tf_device"] = "unknown"
    return info
