"""Same inputs, same output: a key for what a model run depends on, so a run is not repeated for it.

A run (the daily decision, every model's scores) is keyed by the data it read (the information cutoff and a hash of
that close's prices, so re-adjusted history counts as new data), the code that computes it, and the settings and
records it reads. When the newest saved run has the same key, that run is returned and nothing is fitted, scored or
saved again; asking with ``force`` runs it anyway.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]


def key(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def code_hash(*relative: str) -> str:
    """Hash of these source files (paths relative to the ``tradingagents`` package)."""
    h = hashlib.sha256()
    for rel in sorted(relative):
        path = PACKAGE / rel
        h.update(rel.encode())
        h.update(path.read_bytes() if path.is_file() else b"missing")
    return h.hexdigest()[:16]


def prices_hash(close: pd.DataFrame, cutoff) -> str:
    """The closes on the cutoff session, rounded: a later download that rewrote them is new data."""
    row = close.loc[pd.Timestamp(cutoff)].sort_index()
    values = np.round(row.to_numpy(dtype=float), 6)
    return hashlib.sha256(json.dumps([list(row.index), np.nan_to_num(values, nan=-1.0).tolist()]).encode()
                          ).hexdigest()[:16]


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None
