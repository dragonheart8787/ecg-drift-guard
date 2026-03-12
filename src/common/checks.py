"""Data integrity checks: leakage detection, class distribution, version info."""

from __future__ import annotations

import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.common.io import load_list
from src.common.log import get_logger

log = get_logger(__name__)


def assert_no_leakage(splits_dir: str | Path) -> None:
    """Verify train/val/test record lists are mutually exclusive."""
    splits_dir = Path(splits_dir)
    sets: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        f = splits_dir / f"{name}_records.txt"
        if f.exists():
            sets[name] = set(load_list(f))

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        if a in sets and b in sets:
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f"DATA LEAKAGE: {a} ∩ {b} = {overlap}")
    log.info("Leakage check PASSED — no record overlap between splits")


def class_distribution(y: np.ndarray, class_names: list[str]) -> dict:
    """Return per-class count and proportion."""
    counts = Counter(int(v) for v in y)
    total = len(y)
    dist = {}
    for i, name in enumerate(class_names):
        c = counts.get(i, 0)
        dist[name] = {"count": c, "ratio": round(c / total, 4) if total else 0}
    dist["_total"] = total
    return dist


def class_distribution_table(
    splits: dict[str, np.ndarray],
    class_names: list[str],
) -> dict:
    """Build a table of class distributions for multiple splits."""
    return {name: class_distribution(y, class_names) for name, y in splits.items()}


def environment_info() -> dict:
    """Collect runtime version info for reproducibility."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
    except ImportError:
        pass
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import wfdb
        info["wfdb"] = wfdb.__version__
    except (ImportError, AttributeError):
        info["wfdb"] = "installed (version unavailable)"
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    return info
