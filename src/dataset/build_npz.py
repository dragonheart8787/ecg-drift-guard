"""Build NPZ files from raw records — one NPZ per split."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.io import load_list, save_npz
from src.common.log import get_logger
from src.dataset.beat_cut import extract_beats, normalize
from src.dataset.fetch_mitbih import ensure_downloaded, load_ann, load_record

log = get_logger(__name__)


def build_split_npz(
    split_name: str,
    records_file: str | Path,
    db_dir: str | Path,
    out_path: str | Path,
    pre_sec: float = 0.2,
    post_sec: float = 0.4,
    norm_mode: str = "zscore",
) -> None:
    """Read all records in a split, cut beats, normalise, save NPZ."""
    records = load_list(records_file)
    ensure_downloaded(db_dir, records)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_rid: list[np.ndarray] = []

    for rec in records:
        sig, fs = load_record(rec, db_dir)
        ann_samples, ann_symbols = load_ann(rec, db_dir)

        pre_s = int(pre_sec * fs)
        post_s = int(post_sec * fs)
        X, y, _ = extract_beats(sig, ann_samples, ann_symbols, pre_s, post_s)

        if len(X) == 0:
            log.warning("Record %s: 0 valid beats, skipping", rec)
            continue

        X = normalize(X, mode=norm_mode)
        all_X.append(X)
        all_y.append(y)
        all_rid.append(np.array([rec] * len(y)))
        log.info("Record %s: %d beats", rec, len(y))

    X_all = np.concatenate(all_X, axis=0)[..., np.newaxis]  # (N, L, 1)
    y_all = np.concatenate(all_y, axis=0)
    rid_all = np.concatenate(all_rid, axis=0)

    save_npz(out_path, X=X_all, y=y_all, record_id=rid_all)
    log.info("Saved %s → %s  (N=%d)", split_name, out_path, len(y_all))
