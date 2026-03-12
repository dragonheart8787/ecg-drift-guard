"""Minimal loader for external ECG datasets (for cross-dataset validation).

Default target: MIT-BIH Supraventricular Arrhythmia Database (svdb)
— same annotation format as mitdb but different patients & pathologies.

Also supports: incartdb, ltdb, or any PhysioNet WFDB database with beat
annotations.  The key point is we do NOT retrain — only infer + drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from src.common.log import get_logger
from src.dataset.beat_cut import extract_beats, normalize
from src.dataset.label_aami import symbol_to_aami

log = get_logger(__name__)

# Records known to work for each database
EXTERNAL_DB_RECORDS: dict[str, list[str]] = {
    "svdb": [
        "800", "801", "802", "803", "804", "805", "806", "807",
        "808", "809", "810", "811", "812",
    ],
    "incartdb": [f"I{str(i).zfill(2)}" for i in range(1, 76)],
}


def ensure_external_downloaded(
    db_name: str,
    db_dir: str | Path,
    records: list[str] | None = None,
) -> None:
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    records = records or EXTERNAL_DB_RECORDS.get(db_name, [])
    for rec in records:
        dat_file = db_dir / f"{rec}.dat"
        if dat_file.exists():
            continue
        log.info("Downloading %s/%s ...", db_name, rec)
        wfdb.dl_database(db_name, str(db_dir), records=[rec])


def load_external_beats(
    db_name: str,
    db_dir: str | Path,
    records: list[str] | None = None,
    pre_sec: float = 0.2,
    post_sec: float = 0.4,
    norm_mode: str = "zscore",
    max_records: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, cut, and normalise beats from an external database.

    Returns (X, y, record_ids) — same schema as internal NPZ.
    X shape: (N, L, 1)
    """
    db_dir = Path(db_dir)
    records = records or EXTERNAL_DB_RECORDS.get(db_name, [])
    if max_records:
        records = records[:max_records]

    ensure_external_downloaded(db_name, db_dir, records)

    all_X, all_y, all_rid = [], [], []

    for rec in records:
        try:
            rec_obj = wfdb.rdrecord(str(db_dir / rec))
            ann = wfdb.rdann(str(db_dir / rec), "atr")
        except Exception as e:
            log.warning("Skipping %s/%s: %s", db_name, rec, e)
            continue

        signal = rec_obj.p_signal[:, 0].astype(np.float32)
        fs = rec_obj.fs

        pre_s = int(pre_sec * fs)
        post_s = int(post_sec * fs)

        X, y, _ = extract_beats(signal, np.array(ann.sample), list(ann.symbol),
                                pre_s, post_s)
        if len(X) == 0:
            continue

        X = normalize(X, mode=norm_mode)

        # Resample to 216 if window length differs (different fs)
        target_len = pre_s + post_s
        if target_len != 216:
            from scipy.signal import resample as scipy_resample
            X_resampled = np.array([scipy_resample(x, 216) for x in X], dtype=np.float32)
            X = X_resampled

        all_X.append(X)
        all_y.append(y)
        all_rid.append(np.array([rec] * len(y)))
        log.info("External %s/%s: %d beats (fs=%d)", db_name, rec, len(y), fs)

    if not all_X:
        raise RuntimeError(f"No valid beats from {db_name}")

    X_out = np.concatenate(all_X, axis=0)[..., np.newaxis]
    y_out = np.concatenate(all_y, axis=0)
    rid_out = np.concatenate(all_rid, axis=0)

    log.info("External total: %d beats from %d records", len(y_out), len(all_X))
    return X_out, y_out, rid_out
