"""Point-in-time S&P 500 membership: who was in the index on each date, including stocks later removed.

Source: the public dataset github.com/fja05680/sp500, "S&P 500 Historical Components &
Changes (Updated).csv": one row per change date with the full member list that day, from
1996. It is saved once and refreshed after a week.

Identity. Members are identified by **ticker within a membership interval**. Tickers get
reused (a symbol can belong to a different company years later), so for a removed stock:
prices are taken only inside its intervals, SEC fundamentals are not attached (the ticker
map is today's and could point at another company), and each such row is marked
``removed`` so it can be counted and left out. Today's members keep their SEC CIK.

Outcomes. A removed stock's prices often stop at its delisting (an acquisition or a
failure). A label whose holding period runs past the last price is **missing**, never
filled: the delisting return (an acquisition premium or a loss) is unknown here. That
leaves a residual bias, reported with every result as ``survivorship: partial``. When the
dataset cannot be loaded, the universe falls back to today's members from their date
added and every result says ``survivorship: biased``.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
       "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
MAX_AGE = 7 * 24 * 3600


def load(cache_dir: Path) -> pd.DataFrame | None:
    """``(date, tickers)`` change rows, sorted; None when unavailable."""
    path = Path(cache_dir) / "membership" / "sp500_components.csv"
    if not (path.exists() and time.time() - path.stat().st_mtime < MAX_AGE):
        try:
            r = requests.get(URL, timeout=60)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text, encoding="utf-8")
        except requests.RequestException as exc:
            logger.warning("historical S&P 500 membership unavailable: %s", exc)
            if not path.exists():
                return None
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def intervals(changes: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]]:
    """``{ticker: [(start, end), ...]}``, end exclusive (None while still a member)."""
    out: dict[str, list] = {}
    current: set[str] = set()
    for _, row in changes.iterrows():
        day = row["date"]
        members = {t.strip().replace(".", "-") for t in str(row["tickers"]).split(",") if t.strip()}
        for t in members - current:
            out.setdefault(t, []).append([day, None])
        for t in current - members:
            out[t][-1][1] = day
        current = members
    return {t: [(a, b) for a, b in spans] for t, spans in out.items()}


def mask(spans: dict, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Boolean ``dates x tickers``: a member on that date."""
    out = pd.DataFrame(False, index=dates, columns=tickers)
    for t in tickers:
        for start, end in spans.get(t, []):
            hit = (dates >= start) & ((dates < end) if end is not None else True)
            out.loc[hit, t] = True
    return out


def members_since(spans: dict, start: str) -> list[str]:
    """Every ticker that was a member at some point on or after ``start``."""
    s = pd.Timestamp(start)
    return sorted(t for t, ivs in spans.items() if any(end is None or end > s for _, end in ivs))
