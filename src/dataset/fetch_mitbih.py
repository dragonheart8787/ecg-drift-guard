"""Download / read MIT-BIH records + annotations via WFDB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from src.common.log import get_logger

log = get_logger(__name__)

DB_NAME = "mitdb"
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "219", "220", "221", "222",
    "223", "228", "230", "231", "232", "233", "234",
]


def ensure_downloaded(db_dir: str | Path, records: list[str] | None = None) -> None:
    """Download MIT-BIH records if not already cached locally."""
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    records = records or MITBIH_RECORDS
    for rec in records:
        dat_file = db_dir / f"{rec}.dat"
        if dat_file.exists():
            continue
        log.info("Downloading record %s ...", rec)
        wfdb.dl_database(DB_NAME, str(db_dir), records=[rec])


def load_record(record: str, db_dir: str | Path) -> tuple[np.ndarray, int]:
    """Return (signal, fs).  signal shape: (T, n_leads); we use lead 0."""
    rec = wfdb.rdrecord(str(Path(db_dir) / record))
    signal = rec.p_signal[:, 0].astype(np.float32)
    return signal, rec.fs


def load_ann(record: str, db_dir: str | Path) -> tuple[np.ndarray, list[str]]:
    """Return (sample_indices, symbols)."""
    ann = wfdb.rdann(str(Path(db_dir) / record), "atr")
    return np.array(ann.sample), list(ann.symbol)
