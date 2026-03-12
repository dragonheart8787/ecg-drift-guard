"""Inter-patient record split — ensures no patient leakage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.io import load_yaml, save_list
from src.common.log import get_logger

log = get_logger(__name__)


def make_splits_from_config(splits_yaml: str | Path) -> dict[str, list[str]]:
    """Load DS1/DS2 split from config and derive train/val/test.

    DS1 → train (first 80 %) + val (last 20 %)
    DS2 → test
    """
    cfg = load_yaml(splits_yaml)
    ds1 = cfg["ds1"]
    ds2 = cfg["ds2"]

    n_train = int(len(ds1) * 0.8)
    train = ds1[:n_train]
    val = ds1[n_train:]
    test = ds2

    log.info("Split → train %d, val %d, test %d records", len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}


def make_splits_random(
    records: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Randomly split record list (fallback if no pre-defined DS1/DS2)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(records))
    n = len(records)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train:n_train + n_val]]
    test = [records[i] for i in idx[n_train + n_val:]]
    log.info("Split → train %d, val %d, test %d records", len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}


def save_splits(split_dict: dict[str, list[str]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    for name, recs in split_dict.items():
        save_list(recs, out_dir / f"{name}_records.txt")
        log.info("Saved %s_records.txt (%d records)", name, len(recs))
