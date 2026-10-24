"""Economic features as they were known on each date, from FRED and its vintage archive (ALFRED).

Revised series (CPI, unemployment, payrolls, financial conditions) are read with
every vintage: on a date, each observation takes the value of the latest vintage
published **before** that date. Release times are not in the data, so a vintage
counts from the day after its publication date. Today's revised numbers never
leak into a backtest.

Market series that are not revised (the 10-year minus 2-year Treasury spread, the
Baa corporate minus 10-year spread, the effective fed funds rate) use their
observations from the day after their date.

Features (the same for every stock on a date): CPI inflation over 12 months,
the 3-month change in unemployment, payroll growth over 12 months, the Chicago
Fed's financial conditions index (NFCI), the yield-curve spread, the credit
spread and its 3-month change, and the fed funds rate.

Each series is downloaded once and saved (``macro/<SERIES>.json`` under the cache
folder), refreshed after a week. Needs ``FRED_API_KEY``; without it the stock
module runs without economic features and says so.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

URL = "https://api.stlouisfed.org/fred/series/observations"
REVISED = {"CPIAUCSL": "monthly", "UNRATE": "monthly", "PAYEMS": "monthly", "NFCI": "weekly"}
UNREVISED = ("T10Y2Y", "BAA10Y", "DFF")
from tradingagents.research.features import MACRO_FEATURES as FEATURES  # noqa: E402  (one list, used by the models too)
MAX_AGE = 7 * 24 * 3600
# Publication timing, audited per series (release times from the publishers' schedules):
# - CPIAUCSL, UNRATE, PAYEMS: BLS releases at 8:30 ET; ALFRED's realtime_start is the release
#   date, used from the next day (as_known: strictly before the decision date).
# - NFCI: Chicago Fed, Wednesdays 8:30 ET, data through the prior Friday; vintages, same rule.
# - DFF: the New York Fed publishes the effective rate for day d at about 9:00 ET on d+1; usable
#   at the close of d+1: one business day.
# - T10Y2Y (from H.15 constant-maturity yields) and BAA10Y (Moody's via H.15): FRED posts day d's
#   value on the next business day, not always before that day's close: two business days.
LAG_BDAYS = {"DFF": 1, "T10Y2Y": 2, "BAA10Y": 2}


def _fetch(series: str, vintages: bool, cache_dir: Path) -> list[dict]:
    path = Path(cache_dir) / "macro" / f"{series}{'_vintages' if vintages else ''}.json"
    if path.exists() and time.time() - path.stat().st_mtime < MAX_AGE:
        return json.loads(path.read_text(encoding="utf-8"))
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set")
    params = {"series_id": series, "api_key": key, "file_type": "json", "observation_start": "2002-01-01",
              "limit": 100000}
    if vintages:
        params |= {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}
    rows = []
    for attempt in range(4):
        try:
            response = requests.get(URL, params=params, timeout=60)
            response.raise_for_status()
            rows = [{"date": o["date"], "value": float(o["value"]), "from": o.get("realtime_start")}
                    for o in response.json().get("observations", []) if o["value"] not in (".", "")]
            break
        except (requests.RequestException, ValueError) as exc:
            if attempt == 3:
                raise
            time.sleep(1.5 * 2 ** attempt)
            logger.info("FRED retry %s: %s", series, exc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def as_known(rows: list[dict], dates: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Series]:
    """For each date, the series as published strictly before it: each observation's latest vintage
    whose publication date is earlier than the date."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {d: pd.Series(dtype=float) for d in dates}
    frame["date"], frame["from"] = pd.to_datetime(frame["date"]), pd.to_datetime(frame["from"])
    frame = frame.sort_values("from")
    out = {}
    for d in dates:
        seen = frame[frame["from"] < d]
        out[d] = seen.groupby("date")["value"].last().sort_index()
    return out


def _change(series: pd.Series, months: int, ratio: bool) -> float:
    if series.empty:
        return np.nan
    last_date, last = series.index[-1], series.iloc[-1]
    prior = series[series.index <= last_date - pd.DateOffset(months=months)]
    if prior.empty:
        return np.nan
    return float(last / prior.iloc[-1] - 1) if ratio else float(last - prior.iloc[-1])


def features(dates: pd.DatetimeIndex, cache_dir: Path) -> pd.DataFrame:
    """The economic features at each date (NaN where a series is not available)."""
    out = pd.DataFrame(index=dates, columns=FEATURES, dtype=float)
    known = {}
    for series in REVISED:
        try:
            known[series] = as_known(_fetch(series, True, cache_dir), dates)
        except Exception as exc:  # no key or no network: the stock module runs without it, and says so
            logger.warning("FRED %s unavailable: %s", series, exc)
    for d in dates:
        if "CPIAUCSL" in known:
            out.at[d, "cpi_yoy"] = _change(known["CPIAUCSL"][d], 12, True)
        if "UNRATE" in known:
            out.at[d, "unrate_chg_3m"] = _change(known["UNRATE"][d], 3, False)
        if "PAYEMS" in known:
            out.at[d, "payrolls_yoy"] = _change(known["PAYEMS"][d], 12, True)
        if "NFCI" in known and not known["NFCI"][d].empty:
            out.at[d, "nfci"] = float(known["NFCI"][d].iloc[-1])
    for series, name in (("T10Y2Y", "curve_10y2y"), ("BAA10Y", "credit_spread"), ("DFF", "fed_funds")):
        try:
            rows = _fetch(series, False, cache_dir)
        except Exception as exc:
            logger.warning("FRED %s unavailable: %s", series, exc)
            continue
        s = pd.Series({pd.Timestamp(r["date"]): r["value"] for r in rows}).sort_index()
        s.index = s.index + pd.offsets.BDay(LAG_BDAYS[series])    # usable at the close after publication
        daily = s.reindex(s.index.union(dates)).ffill()
        out[name] = daily.reindex(dates).to_numpy()
        if name == "credit_spread":
            out["credit_chg_3m"] = (daily - daily.shift(63)).reindex(dates).to_numpy() if len(daily) else np.nan
    return out
