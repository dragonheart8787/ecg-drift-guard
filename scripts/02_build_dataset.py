#!/usr/bin/env python
"""Step 2: Download MIT-BIH records, cut beats, normalise, save NPZ per split."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.checks import class_distribution_table
from src.common.io import load_npz, load_yaml, save_json
from src.common.log import get_logger
from src.common.seed import set_seed
from src.dataset.build_npz import build_split_npz
from src.dataset.label_aami import AAMI_CLASSES

log = get_logger("02_build_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NPZ dataset from MIT-BIH")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    db_dir = ROOT / cfg["dataset"]["db_dir"]
    splits_dir = ROOT / "data" / "splits"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_sec = cfg["dataset"]["window"]["pre_sec"]
    post_sec = cfg["dataset"]["window"]["post_sec"]
    norm_mode = cfg["dataset"]["normalize"]

    for split_name in ("train", "val", "test"):
        rec_file = splits_dir / f"{split_name}_records.txt"
        if not rec_file.exists():
            log.warning("Split file %s not found, skipping", rec_file)
            continue
        out_path = out_dir / f"beats_{split_name}.npz"
        build_split_npz(
            split_name=split_name,
            records_file=rec_file,
            db_dir=db_dir,
            out_path=out_path,
            pre_sec=pre_sec,
            post_sec=post_sec,
            norm_mode=norm_mode,
        )

    # Class distribution report
    splits_y = {}
    for split_name in ("train", "val", "test"):
        npz_path = out_dir / f"beats_{split_name}.npz"
        if npz_path.exists():
            data = load_npz(npz_path)
            splits_y[split_name] = data["y"]
            log.info("%s: %d beats", split_name, len(data["y"]))

    dist = class_distribution_table(splits_y, AAMI_CLASSES)
    save_json(dist, ROOT / "artifacts" / "reports" / "class_distribution.json")
    for split_name, d in dist.items():
        log.info("  %s class dist: %s", split_name, {k: v for k, v in d.items() if k != "_total"})

    log.info("Done — NPZ files saved to %s", out_dir)


if __name__ == "__main__":
    main()
