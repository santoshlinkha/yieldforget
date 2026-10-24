"""Index membership lists: which stocks are in the S&P 500 today.

The list comes from the open ``datasets/s-and-p-500-companies`` table (kept in
step with the index) and is cached for a week, since membership changes a few
times a quarter. Symbols are written the way Yahoo spells them (BRK.B -> BRK-B).
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
_MAX_AGE = 7 * 24 * 3600


def get_sp500_members(cache_dir: str | Path | None = None) -> dict[str, str]:
    """``{symbol: "Company (Sector)"}`` for the S&P 500; the cached copy if the download fails."""
    path = Path(cache_dir) / "sp500_members.csv" if cache_dir else None
    text = None
    if path and path.exists() and time.time() - path.stat().st_mtime < _MAX_AGE:
        text = path.read_text(encoding="utf-8")
    if text is None:
        try:
            response = requests.get(SP500_URL, timeout=30)
            response.raise_for_status()
            text = response.text
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
        except requests.RequestException:
            if not (path and path.exists()):
                return {}
            text = path.read_text(encoding="utf-8")
    table = pd.read_csv(io.StringIO(text))
    return {str(symbol).replace(".", "-"): f"{name} ({sector})"
            for symbol, name, sector in zip(table["Symbol"], table["Security"], table["GICS Sector"], strict=True)}
