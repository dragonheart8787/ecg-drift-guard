"""AAMI-standard 5-class mapping from MIT-BIH beat symbols."""

from __future__ import annotations

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# MIT-BIH symbol → AAMI class index
_SYMBOL_MAP: dict[str, int] = {
    # N — Normal & bundle-branch-block
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S — Supraventricular ectopic
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V — Ventricular ectopic
    "V": 2, "E": 2,
    # F — Fusion
    "F": 3,
    # Q — Unknown / paced
    "/": 4, "f": 4, "Q": 4,
}


def symbol_to_aami(symbol: str) -> int | None:
    """Map a single beat symbol to AAMI class id (0-4) or None if excluded."""
    return _SYMBOL_MAP.get(symbol, None)
