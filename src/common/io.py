"""Unified I/O helpers for NPZ, JSON, YAML, and text lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── YAML ────────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ── JSON ────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars / arrays that sneak into dicts."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


# ── NPZ ─────────────────────────────────────────────────────────────────

def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


# ── Text list (one item per line) ───────────────────────────────────────

def save_list(items: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items) + "\n")


def load_list(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
