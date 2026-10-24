"""Closing prices as they were fetched, so a past recommendation's inputs can be rebuilt exactly.

Adjusted closes are rewritten whenever a fund pays a dividend: every earlier
close is scaled down. Storing a full copy each day would cost megabytes per
recommendation, so each ticker keeps one *base* chunk (its whole series as
fetched), plus small *append* chunks with the new days. A new base is written
only when an already-stored close changes. The series as known at time T is
the last base created by T and the appends after it.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from .db import now_utc, transaction

REL_TOL = 1e-9


def _pack(series: pd.Series) -> bytes:
    rows = [[d.strftime("%Y-%m-%d"), float(v)] for d, v in series.dropna().items()]
    return zlib.compress(json.dumps(rows, separators=(",", ":")).encode(), 6)


def _unpack(blob: bytes) -> pd.Series:
    rows = json.loads(zlib.decompress(blob))
    return pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]), dtype=float)


def series_as_of(con, ticker: str, at: str | None = None) -> pd.Series:
    """The series of ``ticker`` as stored by time ``at`` (an ISO UTC timestamp); now if None."""
    at = at or "9999"
    rows = con.execute("SELECT kind, payload FROM price_chunks WHERE ticker = ? AND created_at <= ? ORDER BY id",
                       (ticker, at)).fetchall()
    parts: list[pd.Series] = []
    for r in rows:
        if r["kind"] == "base":
            parts = []
        parts.append(_unpack(r["payload"]))
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def save(con, closes: pd.DataFrame, tickers=None) -> str:
    """Store what is new in ``closes`` for ``tickers`` (all columns by default); returns the
    timestamp a later ``frame_as_of`` reads back exactly this data at."""
    at = now_utc()
    with transaction(con):
        for t in (tickers if tickers is not None else closes.columns):
            if t not in closes:
                continue
            new = closes[t].dropna()
            if new.empty:
                continue
            old = series_as_of(con, t)
            shared = old.index.intersection(new.index)
            same = (len(shared) == len(old) and
                    np.allclose(old.loc[shared].to_numpy(), new.loc[shared].to_numpy(), rtol=REL_TOL, atol=0.0))
            if old.empty or not same:
                kind, chunk = "base", new
            else:
                kind, chunk = "append", new[new.index > old.index[-1]]
                if chunk.empty:
                    continue
            con.execute("INSERT INTO price_chunks (ticker, kind, created_at, first_date, last_date, rows, payload)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (t, kind, at, chunk.index[0].strftime("%Y-%m-%d"), chunk.index[-1].strftime("%Y-%m-%d"),
                         len(chunk), _pack(chunk)))
    return at


def frame_as_of(con, tickers, at: str) -> pd.DataFrame:
    """The closes of ``tickers`` exactly as stored at ``at``."""
    return pd.DataFrame({t: series_as_of(con, t, at) for t in tickers}).sort_index()
