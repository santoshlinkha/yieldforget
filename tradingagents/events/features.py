"""Market-wide calendar and index features for a decision cutoff and holding window.

The decision is made after the close of ``cutoff``; the position is entered at the next session's close and held
``horizon`` sessions (the existing next-close execution rule). The calendar features count releases inside that
holding window and the sessions until the next of each kind; the index features take the latest value that was
available at the cutoff.

Two modes:

- ``strict``: only rows this app had received by the cutoff (``first_seen_at``). Right for live decisions and for
  replaying the app's own record; empty for dates before the store existed.
- ``research``: archived history with documented availability assumptions, for walk-forward studies. Release
  schedules are assumed known ``SCHEDULE_LEAD_DAYS`` before the release (BLS and BEA publish their schedules for the
  year ahead; the FOMC its meeting dates a year or more ahead). Index values are assumed available ``INDEX_LAG`` after
  their period ends (GPR is updated weekly; EPU early in the following month). Values come from the latest stored
  vintage, so later revisions can leak in: every research feature row carries ``exploratory = True``.

A future event's *date* may be known at the cutoff; its *outcome* never is, and nothing here reads one.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from tradingagents.ledger.sessions import next_session, sessions_between

from . import store as ST

FEATURE_VERSION = "event-features-v1"
CALENDAR_KINDS = ("cpi", "employment", "pce", "gdp", "fomc_statement", "ppi")
SCHEDULE_LEAD_DAYS = 30
INDEX_LAG = {"gpr_daily": timedelta(days=7), "epu_monthly": timedelta(days=38)}   # from the obs date (EPU: month start)
CAP = 60   # sessions: "none within this many" reads the same as "far away"


def _as_dt(cutoff) -> datetime:
    """The decision moment: the cutoff session's 16:00 New York close, in UTC (naive)."""
    from zoneinfo import ZoneInfo

    d = pd.Timestamp(cutoff).date()
    return datetime(d.year, d.month, d.day, 16, tzinfo=ZoneInfo("America/New_York")).astimezone(
        ZoneInfo("UTC")).replace(tzinfo=None)


def calendar_view(con, cutoff, mode: str = "strict", decided_at: datetime | None = None) -> pd.DataFrame:
    """The latest known version of each scheduled release as of the cutoff (one row per ``source_event_id``).
    ``decided_at`` (UTC, naive): when the decision was actually made, for strict mode; default the cutoff's close."""
    at = decided_at or _as_dt(cutoff)
    rows = pd.read_sql_query("SELECT source, source_event_id, event_type, scheduled_at, status, first_seen_at, row_id "
                             "FROM events", con)
    if rows.empty:
        return rows
    rows["scheduled_at"] = pd.to_datetime(rows["scheduled_at"], utc=True).dt.tz_localize(None)
    rows["first_seen_at"] = pd.to_datetime(rows["first_seen_at"], utc=True).dt.tz_localize(None)
    if mode == "strict":
        rows = rows[rows["first_seen_at"] <= at]
    elif mode == "research":
        rows = rows[rows["scheduled_at"] - timedelta(days=SCHEDULE_LEAD_DAYS) <= at]
    else:
        raise ValueError("mode is strict or research")
    rows = rows.sort_values("row_id").groupby(["source", "source_event_id"], as_index=False).last()
    return rows[~rows["status"].isin(["cancelled", "withdrawn"])]


def calendar_features(con, cutoff, horizon: int = 20, mode: str = "strict", decided_at: datetime | None = None) -> dict:
    cut = pd.Timestamp(cutoff).date()
    entry = next_session(cut)
    exit_ = next_session(entry, horizon)
    view = calendar_view(con, cutoff, mode, decided_at)
    out = {"feature_version": FEATURE_VERSION, "mode": mode, "exploratory": mode == "research"}
    for kind in CALENDAR_KINDS:
        days = sorted(d.date() for d in view.loc[view["event_type"] == kind, "scheduled_at"]) if len(view) else []
        upcoming = [d for d in days if d > cut]
        nxt = upcoming[0] if upcoming else None
        out[f"sessions_to_{kind}"] = min(sessions_between(cut, nxt), CAP) if nxt else np.nan
        out[f"{kind}_in_window"] = sum(1 for d in upcoming if entry <= d <= exit_)
        past = [d for d in days if d <= cut]
        out[f"sessions_since_{kind}"] = min(sessions_between(past[-1], cut), CAP) if past else np.nan
    out["releases_in_window"] = sum(out[f"{k}_in_window"] for k in CALENDAR_KINDS)
    return out


def series_as_of(con, series: str, cutoff, mode: str = "strict", decided_at: datetime | None = None) -> pd.Series:
    """The series' values that were available at the cutoff, indexed by observation date."""
    at = decided_at or _as_dt(cutoff)
    rows = pd.read_sql_query("SELECT obs_date, value, first_seen_at, row_id FROM observations WHERE series = ?", con,
                             params=(series,))
    if rows.empty:
        return pd.Series(dtype=float)
    rows["obs_date"] = pd.to_datetime(rows["obs_date"])
    rows["first_seen_at"] = pd.to_datetime(rows["first_seen_at"], utc=True).dt.tz_localize(None)
    if mode == "strict":
        rows = rows[rows["first_seen_at"] <= at]
    else:
        lag = INDEX_LAG.get(series, timedelta(days=7))
        rows = rows[rows["obs_date"] + lag <= at]
    if rows.empty:
        return pd.Series(dtype=float)
    rows = rows.sort_values("row_id").groupby("obs_date").last()        # the latest vintage known for each date
    return rows["value"].sort_index()


def index_features(con, cutoff, mode: str = "strict", decided_at: datetime | None = None) -> dict:
    cut = pd.Timestamp(cutoff)
    out = {}
    for series, short, window in (("gpr_daily", "gpr", 252), ("epu_monthly", "epu", 60)):
        s = series_as_of(con, series, cutoff, mode, decided_at)
        if s.empty:
            out.update({f"{short}_level": np.nan, f"{short}_z": np.nan, f"{short}_chg": np.nan,
                        f"{short}_stale_days": np.nan, f"{short}_missing": 1})
            continue
        if series == "gpr_daily":
            s = s.rolling(7, min_periods=3).mean().dropna()                # a week's average: daily values are noisy
        last = float(s.iloc[-1])
        hist = s.iloc[-window:]
        sd = float(hist.std())
        out[f"{short}_level"] = last
        out[f"{short}_z"] = (last - float(hist.mean())) / sd if sd > 0 else np.nan   # trailing history only
        back = s.iloc[-21] if series == "gpr_daily" and len(s) > 21 else (s.iloc[-4] if len(s) > 4 else np.nan)
        out[f"{short}_chg"] = float(np.log(last / back)) if back and back > 0 and last > 0 else np.nan
        out[f"{short}_stale_days"] = int((cut - s.index[-1]).days)
        out[f"{short}_missing"] = 0
    return out


def market_event_features(cache_dir, cutoff, horizon: int = 20, mode: str = "strict",
                          decided_at: datetime | None = None) -> dict:
    """``decided_at``: for a live decision, when it is made (now); strict mode then sees what had arrived by then."""
    con = ST.connect(cache_dir)
    try:
        return {**calendar_features(con, cutoff, horizon, mode, decided_at),
                **index_features(con, cutoff, mode, decided_at), "decided_at": str(decided_at or _as_dt(cutoff))}
    finally:
        con.close()


def _unused(_: date) -> None:
    return None
