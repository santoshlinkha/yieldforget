"""Wall Street analyst consensus, rebuilt for each past date from dated records (not the AI agents' ratings).

No free source sells historical consensus, so it is reconstructed from two dated Yahoo lists already in the local
store (``dataflows/vendors/yahoo/history``):

- **analyst actions** (``upgrades``): every rating change or reiteration with the firm, the new grade and the price
  target, from about 2012. At a date, each firm's latest action within the last 365 days is its current view.
- **earnings reports** (``earnings``): each report's consensus EPS estimate, reported EPS and surprise, from about
  2002.

Features at the close of a date use only records known by then:

- an action is used from the session *after* its calendar date (its hour is not reliable enough to use the same
  day);
- an earnings report from the first session whose close follows it (after 16:00 New York time: the next session).

| feature | meaning |
|---|---|
| ``an_firms`` | firms with a rating or target action in the last 365 days |
| ``an_score`` | the average of each firm's latest grade: +1 buy-type, 0 hold-type, -1 sell-type |
| ``an_net_90`` | upgrades minus downgrades in the last 90 days |
| ``an_pt_change_90`` | the average change of price targets set in the last 90 days (new over prior, scale-free) |
| ``an_pt_upside`` | the median of firms' latest targets set in the last 120 days over the close, minus 1 |
| ``eps_surprise_last`` | the last reported EPS surprise, % |
| ``eps_beats_4`` | how many of the last 4 reports beat the estimate |
| ``eps_growth_yoy`` | the last reported EPS over the one 4 reports earlier, minus 1 (both positive) |

Caveat: both lists are Yahoo's **as of today**. A firm that stopped covering a stock, or an estimate Yahoo later
corrected, may be missing or changed, so these features are not strictly point in time. Stocks Yahoo has no list
for get no value (missing, never zero).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BULL = ("buy", "strong buy", "long-term buy", "outperform", "market outperform", "sector outperform", "overweight",
        "accumulate", "add", "positive", "top pick", "conviction buy")
BEAR = ("sell", "strong sell", "underperform", "market underperform", "sector underperform", "underweight",
        "reduce", "negative")
PT_DAYS = 120          # a firm's target counts toward the upside only if it was set within this many days
COLS = ["an_firms", "an_score", "an_net_90", "an_pt_change_90", "an_pt_upside", "eps_surprise_last", "eps_beats_4",
        "eps_growth_yoy"]


def grade_score(grade: str) -> float:
    g = str(grade or "").strip().lower()
    if not g:
        return np.nan
    return 1.0 if g in BULL else -1.0 if g in BEAR else 0.0


def _known_after_day(times: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Session position from which a dated action is used: the first session after its calendar date."""
    days = pd.DatetimeIndex(times).normalize()
    return sessions.searchsorted(days, side="right")


def _known_after_close(times: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Session position from which an earnings report is known: its own day if before 16:00 New York time on a
    session day, else the next session."""
    t = pd.DatetimeIndex(times)
    t = t.tz_convert("America/New_York") if t.tz is not None else t
    days = t.tz_localize(None).normalize() if t.tz is not None else t.normalize()
    pos = sessions.searchsorted(days, side="left")
    same = (pos < len(sessions)) & (sessions[np.minimum(pos, len(sessions) - 1)] == days)
    return pos + (same & (t.hour >= 16)).astype(int)


def ticker_features(actions: pd.DataFrame, reports: pd.DataFrame, positions: np.ndarray, sessions: pd.DatetimeIndex,
                    close: pd.Series | None, splits: list | None = None) -> pd.DataFrame:
    """One ticker's features at each session position in ``positions``. ``close``: split-adjusted (today's shares)
    as-traded closes; targets are put in today's shares with ``splits`` before they are compared."""
    out = np.full((len(positions), len(COLS)), np.nan)
    if actions is not None and len(actions):
        a = actions.sort_index()
        k = _known_after_day(a.index, sessions)
        order = np.argsort(k, kind="stable")                       # by the session each action becomes known
        k = k[order]
        firm = a["Firm"].astype(str).to_numpy()[order]
        score = a["ToGrade"].map(grade_score).to_numpy()[order]
        direction = a["Action"].map({"up": 1.0, "down": -1.0}).fillna(0.0).to_numpy()[order]
        cur = pd.to_numeric(a.get("currentPriceTarget"), errors="coerce").to_numpy()[order]
        prior = pd.to_numeric(a.get("priorPriceTarget"), errors="coerce").to_numpy()[order]
        # Yahoo mixes bases: around a split some firms' targets are in the old shares and some already in the new
        # (AAPL, 2020-07-31: $450 and $95 on the same day). Each target is put in today's shares in whichever
        # reading is closer to the price when it was set; one far from both (5x either way) is dropped.
        pt_raw = np.where(cur > 0, cur, np.nan)
        as_old = pt_raw / split_factor_after(a.index, splits)[order]
        if close is not None:
            px_then = close.to_numpy()[np.minimum(k, len(sessions) - 1)]
            gap = lambda x: np.abs(np.log(x / px_then))             # noqa: E731
            pt = np.where(gap(as_old) <= gap(pt_raw), as_old, pt_raw)
            pt = np.where(np.minimum(gap(as_old), gap(pt_raw)) <= np.log(5), pt, np.nan)
        else:
            pt = as_old
        chg = np.where((cur > 0) & (prior > 0), cur / np.where(prior > 0, prior, 1) - 1, np.nan)
        days = sessions.values.astype("datetime64[D]").astype(np.int64)      # whatever the index's time unit
        act_day = days[np.minimum(k, len(sessions) - 1)]           # the day each action is first used
        # Running sums for the 90-day windows; a sweep keeps each firm's latest view for the 365-day one.
        c_dir = np.r_[0.0, np.cumsum(direction)]
        c_chg = np.r_[0.0, np.cumsum(np.nan_to_num(chg))]
        c_n = np.r_[0, np.cumsum(np.isfinite(chg))]
        latest: dict[str, tuple[int, float, float]] = {}
        j = 0
        for i, p in enumerate(positions):                          # positions ascend with the dates
            if p < 0 or p >= len(sessions):
                continue
            while j < len(k) and k[j] <= p:
                latest[firm[j]] = (act_day[j], score[j], pt[j])
                j += 1
            if j == 0:
                continue
            views = [v for v in latest.values() if days[p] - v[0] <= 365]
            if not views:
                continue
            out[i, 0] = len(views)
            sc = [v[1] for v in views if np.isfinite(v[1])]
            out[i, 1] = float(np.mean(sc)) if sc else np.nan
            lo = int(np.searchsorted(act_day[:j], days[p] - 90, side="left"))
            out[i, 2] = c_dir[j] - c_dir[lo]
            n = c_n[j] - c_n[lo]
            out[i, 3] = (c_chg[j] - c_chg[lo]) / n if n else np.nan
            pts = [v[2] for v in views if np.isfinite(v[2]) and days[p] - v[0] <= PT_DAYS]
            if close is not None and pts and np.isfinite(close.iloc[p]) and close.iloc[p] > 0:
                out[i, 4] = float(np.median(pts)) / close.iloc[p] - 1
    if reports is not None and len(reports):
        r = reports.dropna(subset=["Reported EPS"]).copy()
        if len(r):
            k = _known_after_close(r.index, sessions)
            sur = pd.to_numeric(r["Surprise(%)"], errors="coerce").to_numpy()
            rep = pd.to_numeric(r["Reported EPS"], errors="coerce").to_numpy()
            est = pd.to_numeric(r["EPS Estimate"], errors="coerce").to_numpy()
            for i, p in enumerate(positions):
                n = int((k <= p).sum())                            # reports known by this close (time order)
                if n == 0:
                    continue
                out[i, 5] = sur[n - 1]
                last4 = slice(max(0, n - 4), n)
                beats = (rep[last4] > est[last4])[np.isfinite(rep[last4]) & np.isfinite(est[last4])]
                out[i, 6] = beats.sum() if len(beats) else np.nan
                if n >= 5 and rep[n - 1] > 0 and rep[n - 5] > 0:
                    out[i, 7] = rep[n - 1] / rep[n - 5] - 1
    return pd.DataFrame(out, columns=COLS)


def split_factor_after(when: pd.DatetimeIndex, splits: list) -> np.ndarray:
    """For each time, the product of the share splits after it: a dollar figure then (an analyst's target) divided by
    this is in today's shares, comparable with split-adjusted prices (an NVDA target of $900 in March 2024 is $90)."""
    f = np.ones(len(when))
    w = pd.DatetimeIndex(when)
    w = w.tz_localize(None) if w.tz is not None else w
    for d, r in splits or []:
        if float(r) > 0:
            f[np.asarray(w < pd.Timestamp(d))] *= float(r)
    return f


def features(cache_dir, dates: pd.DatetimeIndex, tickers: list[str], sessions: pd.DatetimeIndex,
             close: pd.DataFrame | None = None, actions: dict | None = None) -> pd.DataFrame:
    """Per (date, ticker) for every ticker with a stored Yahoo analyst or earnings list."""
    from tradingagents.dataflows.vendors.yahoo import history as yh

    positions = sessions.get_indexer(dates)
    frames = []
    for t in tickers:
        try:
            acts, reps = yh.upgrades(t, Path(cache_dir)), yh.earnings(t, Path(cache_dir))
        except Exception:   # no list for this ticker: its features stay missing
            continue
        c = close[t].reindex(sessions) if close is not None and t in close else None
        f = ticker_features(acts, reps, positions, sessions, c, ((actions or {}).get(t) or {}).get("splits"))
        f.index = pd.MultiIndex.from_arrays([dates, [t] * len(dates)], names=["date", "ticker"])
        frames.append(f)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.concat(frames).reindex(idx) if frames else pd.DataFrame(index=idx, columns=COLS, dtype=float)
