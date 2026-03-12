#!/usr/bin/env python
"""Step 9: Benchmark inference latency, memory, and device info."""

import argparse
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.benchmark import get_device_info, measure_latency, measure_memory
from src.common.io import load_yaml, save_json
from src.common.log import get_logger
from src.common.seed import set_seed
from src.viz.plots import plot_latency

log = get_logger("09_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    model = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "cnn1d.keras")
    embedder = tf.keras.models.load_model(ROOT / "artifacts" / "models" / "embedder.keras")

    input_shape = (cfg["model"]["input_len"], 1)
    batch_sizes = [1, 32, 128, 256, 512]

    log.info("=== Model latency ===")
    model_lat = measure_latency(model, input_shape, batch_sizes)

    log.info("=== Embedder latency ===")
    emb_lat = measure_latency(embedder, input_shape, batch_sizes)

    mem = measure_memory()
    device = get_device_info()

    benchmark = {
        "model_latency": model_lat,
        "embedder_latency": emb_lat,
        "memory": mem,
        "device": device,
        "model_params": int(model.count_params()),
        "embedder_params": int(embedder.count_params()),
    }

    save_json(benchmark, ROOT / "artifacts" / "reports" / "benchmark.json")

    fig_dir = ROOT / "artifacts" / "reports" / "figures"
    plot_latency(model_lat, emb_lat, fig_dir / "latency_benchmark.png")

    log.info("Device: %s", device)
    log.info("Memory: %s", mem)
    log.info("Done — benchmark saved.")


if __name__ == "__main__":
    main()
