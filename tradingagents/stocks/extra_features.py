"""Extra point-in-time features for the stock panel: residual momentum, earnings timing, and the
geopolitical / policy-uncertainty regime with each stock's sensitivity to it.

Everything is known at the decision close t:

**Residual momentum.** Each day, the stock's excess return over cash is regressed over the last 126
sessions on SPY's excess return and its sector fund's return over SPY (GICS sector -> SPDR fund):
``r_i - r_f = a + b1 (r_SPY - r_f) + b2 (r_sec - r_SPY) + e``. ``res_mom_20`` / ``res_mom_60`` are the sum of
the last 20 / 60 residuals divided by the residual volatility times sqrt(window): momentum with the market
and sector parts removed.

**Earnings timing** (SEC EDGAR submissions, 8-K with Item 2.02 "Results of Operations", by acceptance
time; a filing counts from the session after it was accepted): ``days_since_earn`` (calendar days),
``days_to_earn_est`` (91 days after the last one, minus today: an estimate), ``earn_abret_2d`` (the stock's
return over SPY on the two sessions from the filing).

**Regime (the same for every stock on a date).**
- ``gpr_threat_z``: Caldara-Iacoviello daily Geopolitical Threats index, log(1+x), lagged one session (the
  index for a day is compiled after it), as a 63-session z-score; ``gpr_act_diff``: the Acts z-score
  minus the Threats z-score.
- ``epu_z``: the daily US Economic Policy Uncertainty index (FRED USEPUINDXD), log, lagged one session,
  63-session z-score.
- ``geo_mkt_z5`` / ``geo_mkt_z20``: a market-implied geopolitical hedge factor, the average daily log return
  of gold, US aerospace & defense and oil (GLD, ITA, USO) minus SPY's, summed over 5 / 20 sessions and
  divided by its 252-session volatility times sqrt(window).
- ``vix_term``: VIX / VIX3M (above 1: near-term fear above longer-term).

**Each stock's sensitivity.** ``geo_beta_126`` (126-session slope of the stock's return over SPY on the
geopolitical factor), ``epu_beta_126`` (on the daily change in log policy uncertainty, lagged); the
interactions ``geo_shock_int = geo_beta_126 x geo_mkt_z5`` and ``epu_shock_int = epu_beta_126 x epu_z`` give
a date-grouped ranker something to split on when the regime moves.

Known limits: sectors are today's GICS labels (applied to the past); 8-K history comes from each filer's
"recent" list (about the last 1,000 filings), so early years can be missing (left empty, never guessed).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GEO_BASKET = ("GLD", "ITA", "USO")
SECTOR_FUND = {"Information Technology": "XLK", "Communication Services": "XLC", "Consumer Discretionary": "XLY",
               "Financials": "XLF", "Health Care": "XLV", "Industrials": "XLI", "Energy": "XLE", "Materials": "XLB",
               "Consumer Staples": "XLP", "Utilities": "XLU", "Real Estate": "XLRE"}
STOCK_COLS = ["res_mom_20", "res_mom_60", "days_since_earn", "days_to_earn_est", "earn_abret_2d",
              "geo_beta_126", "epu_beta_126", "geo_shock_int", "epu_shock_int"]
REGIME_COLS = ["gpr_threat_z", "gpr_act_diff", "epu_z", "geo_mkt_z5", "geo_mkt_z20", "vix_term"]
MAX_AGE = 6 * 24 * 3600


# --- downloads (cached) ---

def gpr_daily(cache_dir) -> pd.DataFrame:
    import requests

    path = Path(cache_dir) / "macro" / "gpr_daily.xls"
    if not path.exists() or time.time() - path.stat().st_mtime > MAX_AGE:
        r = requests.get(GPR_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    df = pd.read_excel(path)
    df = df[pd.to_numeric(df["DAY"], errors="coerce").notna()]
    df.index = pd.to_datetime(df["DAY"].astype(int).astype(str), format="%Y%m%d")
    return df[["GPRD", "GPRD_ACT", "GPRD_THREAT"]].astype(float)


def epu_daily(cache_dir) -> pd.Series:
    from .macro import _fetch

    rows = _fetch("USEPUINDXD", False, Path(cache_dir))
    return pd.Series({pd.Timestamp(r["date"]): r["value"] for r in rows}, dtype=float).sort_index()


def earnings_dates(tickers, cache_dir) -> dict[str, list[pd.Timestamp]]:
    """Acceptance times of each company's 8-K filings with Item 2.02, from SEC EDGAR submissions (cached)."""
    from tradingagents.dataflows.vendors import sec_edgar as E

    out = {}
    for t in tickers:
        try:
            cik = E.cik_for(t)
            if cik is None:
                continue
            recent = E._cached_json(E._SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik}.json") \
                .get("filings", {}).get("recent", {})
        except Exception as exc:   # one company missing never stops the rest
            logger.info("no SEC submissions for %s: %s", t, exc)
            continue
        acc = [pd.Timestamp(recent["acceptanceDateTime"][i][:19]) for i, form in enumerate(recent.get("form", []))
               if form.startswith("8-K") and "2.02" in str(recent.get("items", [""] * (i + 1))[i])]
        out[t] = sorted(acc)
    return out


# --- regime ---

def _z(x: pd.Series, n: int) -> pd.Series:
    return (x - x.ewm(span=n, min_periods=n // 2).mean()) / x.rolling(n, min_periods=n // 2).std()


def regime(sessions: pd.DatetimeIndex, close: pd.DataFrame, cache_dir) -> pd.DataFrame:
    out = pd.DataFrame(index=sessions)
    try:
        g = gpr_daily(cache_dir)
        # Lag one session: the index for day d is compiled after it; on session t use the newest day before t.
        lagged = g.reindex(g.index.union(sessions)).ffill().shift(1).reindex(sessions)
        thr, act = np.log1p(lagged["GPRD_THREAT"]), np.log1p(lagged["GPRD_ACT"])
        out["gpr_threat_z"] = _z(thr, 63)
        out["gpr_act_diff"] = _z(act, 63) - out["gpr_threat_z"]
    except Exception as exc:
        logger.warning("geopolitical risk index unavailable: %s", exc)
        out["gpr_threat_z"] = out["gpr_act_diff"] = np.nan
    try:
        e = epu_daily(cache_dir)
        el = np.log(e.reindex(e.index.union(sessions)).ffill().shift(1).reindex(sessions))
        out["epu_z"] = _z(el, 63)
        out["_epu_chg"] = el.diff()
    except Exception as exc:
        logger.warning("policy uncertainty unavailable: %s", exc)
        out["epu_z"] = out["_epu_chg"] = np.nan
    lr = np.log(close).diff()
    basket = [t for t in GEO_BASKET if t in lr]
    geo = lr[basket].mean(axis=1, skipna=False) - lr["SPY"] if basket else pd.Series(np.nan, index=sessions)
    vol = geo.rolling(252, min_periods=126).std()
    out["_geo_ret"] = geo
    out["geo_mkt_z5"] = geo.rolling(5).sum() / (vol * np.sqrt(5))
    out["geo_mkt_z20"] = geo.rolling(20).sum() / (vol * np.sqrt(20))
    out["vix_term"] = close["^VIX"] / close["^VIX3M"] if {"^VIX", "^VIX3M"} <= set(close) else np.nan
    return out


# --- per stock ---

def _rolling_slope(y: pd.DataFrame, x: pd.Series, n: int = 126) -> pd.DataFrame:
    xm = x.rolling(n, min_periods=n // 2).mean()
    vx = x.rolling(n, min_periods=n // 2).var()
    cxy = y.mul(x, axis=0).rolling(n, min_periods=n // 2).mean().sub(y.rolling(n, min_periods=n // 2).mean().mul(xm, axis=0))
    return cxy.div(vx * (n - 1) / n, axis=0)


def residual_momentum(stock_ret: pd.DataFrame, spy_ret: pd.Series, cash_ret: pd.Series, sector_ret: pd.DataFrame,
                      n: int = 126) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(res_mom_20, res_mom_60): standardized sums of the residuals of a rolling two-factor regression."""
    y = stock_ret.sub(cash_ret, axis=0)
    x1 = (spy_ret - cash_ret)
    x2 = sector_ret.sub(spy_ret, axis=0)                        # per stock: its sector fund over SPY
    mp = n // 2
    roll = lambda f: f.rolling(n, min_periods=mp).mean()       # noqa: E731
    m1, my = roll(x1), roll(y)
    m2 = roll(x2)
    s11 = roll(x1 * x1) - m1 ** 2
    s22 = roll(x2 * x2) - m2 ** 2
    s12 = roll(x2.mul(x1, axis=0)) - m2.mul(m1, axis=0)
    s1y = roll(y.mul(x1, axis=0)) - my.mul(m1, axis=0)
    s2y = roll(y * x2) - my * m2
    det = s22.mul(s11, axis=0) - s12 ** 2
    b1 = (s1y.mul(s22) - s2y * s12) / det
    b2 = (s2y.mul(s11, axis=0) - s1y.mul(s12)) / det
    a = my - b1.mul(m1, axis=0) - b2 * m2
    resid = y - a - b1.mul(x1, axis=0) - b2 * x2
    sd = resid.rolling(n, min_periods=mp).std()
    return (resid.rolling(20).sum() / (sd * np.sqrt(20)), resid.rolling(60).sum() / (sd * np.sqrt(60)))


def stock_features(dates: pd.DatetimeIndex, close: pd.DataFrame, stocks: list[str], sectors: dict[str, str],
                   cash: str, reg: pd.DataFrame, earn: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    """Rows for (date, ticker) on ``dates``; ``close`` holds the stocks, SPY, the cash fund and the sector funds."""
    lr = np.log(close).diff()
    sret = lr[stocks]
    spy, csh = lr["SPY"], lr[cash].fillna(0.0)
    sec = pd.DataFrame({t: lr[SECTOR_FUND[sectors[t]]] if SECTOR_FUND.get(sectors.get(t, "")) in lr else spy
                        for t in stocks}, index=lr.index)
    r20, r60 = residual_momentum(sret, spy, csh, sec)
    excess = sret.sub(spy, axis=0)
    geo_beta = _rolling_slope(excess, reg["_geo_ret"].reindex(lr.index))
    epu_beta = _rolling_slope(excess, reg["_epu_chg"].reindex(lr.index))
    frames = {"res_mom_20": r20, "res_mom_60": r60, "geo_beta_126": geo_beta, "epu_beta_126": epu_beta,
              "geo_shock_int": geo_beta.mul(reg["geo_mkt_z5"].reindex(lr.index), axis=0),
              "epu_shock_int": epu_beta.mul(reg["epu_z"].reindex(lr.index), axis=0)}
    out = pd.concat({k: v.reindex(dates) for k, v in frames.items()}, axis=1).stack(future_stack=True)
    out.index.names = ["date", "ticker"]
    # Earnings timing, as known at each decision date (an 8-K counts from the session after its acceptance).
    sessions = close.index
    rows = []
    for t in stocks:
        acc = earn.get(t) or []
        known = pd.DatetimeIndex([sessions[min(sessions.searchsorted(a.normalize(), side="right"), len(sessions) - 1)]
                                  for a in acc]) if acc else pd.DatetimeIndex([])
        for d in dates:
            prior = known[known <= d]
            if not len(prior):
                rows.append((d, t, np.nan, np.nan, np.nan))
                continue
            last = prior[-1]
            i = sessions.get_loc(last)
            ab = float(excess[t].iloc[i - 1:i + 1].sum()) if i >= 1 and t in excess else np.nan
            rows.append((d, t, (d - last).days, (last + timedelta(days=91) - d).days, ab))
    e = pd.DataFrame(rows, columns=["date", "ticker", "days_since_earn", "days_to_earn_est", "earn_abret_2d"]) \
        .set_index(["date", "ticker"])
    return out.join(e, how="left")


def build(panel: pd.DataFrame, data, cache_dir, progress=None) -> pd.DataFrame:
    """The panel with the extra columns joined (rows and existing columns unchanged)."""
    from tradingagents.dataflows.vendors.index_members import get_sp500_members
    from tradingagents.research.prices import bars

    say = progress or (lambda *_: None)
    from .data import CASH

    stocks = sorted(set(panel.index.get_level_values("ticker")))
    dates = pd.DatetimeIndex(sorted(set(panel.index.get_level_values("date"))))
    members = get_sp500_members(cache_dir)
    sectors = {t: lab[lab.rfind("(") + 1:-1] for t, lab in members.items() if lab.endswith(")")}
    extra = sorted({*SECTOR_FUND.values(), *GEO_BASKET, "SPY", CASH, "^VIX", "^VIX3M"})
    say("extra features: prices for sectors, gold, defense, oil")
    side = bars(extra, cache_dir)["Close"]
    close = pd.concat([data.close[[t for t in stocks if t in data.close]], side.drop(
        columns=[c for c in side if c in data.close and c not in extra], errors="ignore")], axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    say("extra features: geopolitical and policy regime")
    reg = regime(close.index, close, cache_dir)
    say("extra features: earnings dates from SEC")
    earn = earnings_dates(stocks, cache_dir)
    say("extra features: residual momentum and sensitivities")
    st = stock_features(dates, close, [t for t in stocks if t in close], sectors, CASH, reg, earn)
    out = panel.join(st, how="left")
    out = out.join(reg[REGIME_COLS].reindex(dates).rename_axis("date"), on="date")
    return out


def coverage(panel: pd.DataFrame) -> dict:
    return {c: round(float(panel[c].notna().mean()), 3) for c in [*STOCK_COLS, *REGIME_COLS] if c in panel}


def save_note(cache_dir, info: dict) -> None:
    (Path(cache_dir) / "macro" / "extra_features.json").write_text(json.dumps(info, default=str), encoding="utf-8")
