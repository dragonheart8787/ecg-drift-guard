#!/usr/bin/env python
"""Step 1: Generate inter-patient record splits and persist to text files."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.checks import assert_no_leakage
from src.common.io import load_yaml
from src.common.log import get_logger
from src.common.seed import set_seed
from src.dataset.split_records import make_splits_from_config, save_splits

log = get_logger("01_make_splits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inter-patient splits")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    splits_yaml = ROOT / "config" / "splits.yaml"
    split_dict = make_splits_from_config(splits_yaml)

    out_dir = ROOT / "data" / "splits"
    save_splits(split_dict, out_dir)

    # Leakage check
    assert_no_leakage(out_dir)

    for name, recs in split_dict.items():
        log.info("%5s: %d records → %s", name, len(recs), recs)
    log.info("Done — splits saved to %s", out_dir)


if __name__ == "__main__":
    main()
