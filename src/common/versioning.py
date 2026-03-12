"""Model versioning and registry management."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.common.io import load_json, save_json
from src.common.log import get_logger

log = get_logger(__name__)


def _config_hash(config_path: str | Path) -> str:
    """SHA-256 of config file content (first 8 chars)."""
    content = Path(config_path).read_bytes()
    return hashlib.sha256(content).hexdigest()[:8]


def _model_hash(model_path: str | Path) -> str:
    """SHA-256 of model file (first 8 chars)."""
    if not Path(model_path).exists():
        return "no_model"
    content = Path(model_path).read_bytes()
    return hashlib.sha256(content).hexdigest()[:8]


def generate_version_id(seed: int, config_path: str | Path) -> str:
    """Generate a deterministic version ID: timestamp + seed + config hash."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    ch = _config_hash(config_path)
    return f"v{ts}_s{seed}_{ch}"


def build_registry_entry(
    *,
    version_id: str,
    model_path: str | Path,
    config_path: str | Path,
    metrics: dict,
    calibration_T: float | None = None,
    notes: str = "",
) -> dict:
    """Create a single model registry entry."""
    return {
        "version_id": version_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_hash": _model_hash(model_path),
        "config_hash": _config_hash(config_path),
        "metrics": metrics,
        "calibration_T": calibration_T,
        "model_path": str(model_path),
        "notes": notes,
    }


def update_registry(
    registry_path: str | Path,
    entry: dict,
) -> list[dict]:
    """Append entry to registry JSON (list of entries). Creates if missing."""
    registry_path = Path(registry_path)
    if registry_path.exists():
        registry = load_json(registry_path)
        if not isinstance(registry, list):
            registry = [registry]
    else:
        registry = []

    registry.append(entry)
    save_json(registry, registry_path)
    log.info("Registry updated → %s  (total %d versions)", registry_path, len(registry))
    return registry


def get_latest_version(registry_path: str | Path) -> dict | None:
    """Return the most recent registry entry, or None."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return None
    registry = load_json(registry_path)
    if not registry:
        return None
    return registry[-1]
