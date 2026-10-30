"""Yahoo analyst, earnings and options data, through the local store so each is fetched rarely.

Two kinds carry their own dated history and are used to build features for the
past:

- ``upgrades``: every analyst action Yahoo lists (firm, from/to grade, price
  target and its change), stamped with its time; for large US stocks back to about 2012.
- ``earnings``: report dates with the consensus EPS estimate, reported EPS and
  the surprise; back to about 2002 (``limit=100`` quarters).

They are refetched at most weekly. Caveat: these are Yahoo's lists **as of
today**. A firm that stopped covering a stock, or an estimate Yahoo later
corrected, may be missing or changed; that cannot be checked without a
point-in-time vendor. Results using them are labelled accordingly.

The rest exist only as today's values (consensus price target, EPS trend and
revisions, the recommendation counts, an option-chain summary). ``collect_today``
saves them once per session, so a history of them builds up from now on;
nothing in the past can use them yet.
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from tradingagents.dataflows import local_store

logger = logging.getLogger(__name__)
HISTORY_TTL = timedelta(days=7)


def _records(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or len(frame) == 0:
        return []
    if isinstance(frame.index, pd.RangeIndex):   # a plain row number carries nothing to keep
        return json_safe(frame.to_dict("records"))
    out = frame.reset_index()
    first = out.columns[0]
    out[first] = out[first].astype(str)   # dates and labels as text (pandas 3 refuses an in-place cast)
    return json_safe(out.to_dict("records"))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _ticker(symbol: str):
    import yfinance as yf

    from tradingagents.dataflows.symbols import normalize_symbol
    return yf.Ticker(normalize_symbol(symbol))


def intraday_close(symbol: str, period: str, interval: str) -> pd.Series:
    """Intraday closes from Yahoo (display only; not stored)."""
    return _ticker(symbol).history(period=period, interval=interval)["Close"].dropna()


def live_quotes(symbols) -> dict[str, tuple[pd.Timestamp, float]]:
    """Each symbol's latest one-minute regular-session price from Yahoo, with its time, in one request
    (display only; not stored). A symbol Yahoo has no bar for today is left out."""
    import yfinance as yf

    from tradingagents.dataflows.symbols import normalize_symbol
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    names = {normalize_symbol(s): s for s in symbols}
    frame = yf_retry(lambda: yf.download(list(names), period="1d", interval="1m", prepost=False, auto_adjust=False,
                                         progress=False, threads=False, group_by="column"))
    if frame is None or frame.empty:
        return {}
    close = frame["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(next(iter(names)))
    out = {}
    for col in close.columns:
        s = close[col].dropna()
        if not s.empty and col in names:
            out[names[col]] = (s.index[-1], float(s.iloc[-1]))
    return out


def upgrades(symbol: str, cache_dir=None) -> pd.DataFrame:
    """Dated analyst actions, oldest first, indexed by their time (no time zone)."""
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    def fetch():
        return _records(yf_retry(lambda: _ticker(symbol).upgrades_downgrades))

    store = local_store.store(cache_dir)
    rows = store.snapshot(symbol, "upgrades", fetch, ttl=HISTORY_TTL) if store else fetch()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame.pop(frame.columns[0]), utc=True).dt.tz_localize(None)
    return frame.sort_index()


def earnings(symbol: str, cache_dir=None) -> pd.DataFrame:
    """Earnings dates with estimate, reported EPS and surprise, oldest first, in New York time."""
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    def fetch():
        return _records(yf_retry(lambda: _ticker(symbol).get_earnings_dates(limit=100)))

    store = local_store.store(cache_dir)
    rows = store.snapshot(symbol, "earnings", fetch, ttl=HISTORY_TTL) if store else fetch()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame.pop(frame.columns[0]), utc=True).dt.tz_convert("America/New_York")
    return frame.sort_index()


def _option_summary(t) -> dict:
    """At-the-money implied volatility and put/call figures for the first expiry 7 or more days out."""
    from datetime import date

    expiries = [e for e in (t.options or []) if (date.fromisoformat(e) - date.today()).days >= 7]
    if not expiries:
        return {}
    chain = t.option_chain(expiries[0])
    spot = t.fast_info.get("lastPrice")
    out = {"expiry": expiries[0], "spot": spot}
    if spot:
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            if len(frame):
                atm = frame.iloc[(frame["strike"] - spot).abs().argsort()[:2]]
                out[f"atm_iv_{side}"] = float(atm["impliedVolatility"].mean())
        out["put_call_volume"] = float(chain.puts["volume"].fillna(0).sum() / max(chain.calls["volume"].fillna(0).sum(), 1))
        out["put_call_oi"] = float(chain.puts["openInterest"].fillna(0).sum() / max(chain.calls["openInterest"].fillna(0).sum(), 1))
    return out


def collect_today(symbols, cache_dir=None, options: bool = True, progress=None) -> dict[str, int]:
    """Save today's current-only Yahoo data for each symbol, once per session. Returns counts."""
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    store = local_store.store(cache_dir)
    if store is None:
        raise ValueError("the local store is switched off (TRADINGAGENTS_LOCAL_STORE=off)")
    counts = {"saved": 0, "failed": 0}
    kinds = {
        "price_targets": lambda t: t.analyst_price_targets or {},
        "eps_trend": lambda t: _records(t.eps_trend),
        "eps_revisions": lambda t: _records(t.eps_revisions),
        "recommendations": lambda t: _records(t.recommendations_summary),
        **({"options": _option_summary} if options else {}),
    }
    for n, symbol in enumerate(symbols, 1):
        if progress:
            progress(n, len(symbols), symbol)
        t = None
        for kind, get in kinds.items():
            try:
                def fetch(get=get):
                    nonlocal t
                    t = t or _ticker(symbol)
                    return json_safe(yf_retry(lambda: get(t)))
                store.snapshot(symbol, kind, fetch)
                counts["saved"] += 1
            except Exception as exc:  # one symbol's missing data must not stop the rest
                logger.info("snapshot %s %s failed: %s", symbol, kind, exc)
                counts["failed"] += 1
    return counts


PROFILE_KEYS = ("longName", "shortName", "quoteType", "sector", "industry", "category", "fundFamily", "exchange",
                "currency", "marketCap", "totalAssets", "netExpenseRatio", "annualReportExpenseRatio", "yield",
                "dividendYield", "trailingPE", "forwardPE", "priceToBook", "beta", "beta3Year",
                "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiftyDayAverage", "twoHundredDayAverage",
                "averageVolume", "sharesOutstanding", "shortPercentOfFloat", "trailingEps", "forwardEps",
                "targetMeanPrice", "targetMedianPrice", "targetHighPrice", "targetLowPrice", "recommendationKey",
                "recommendationMean", "numberOfAnalystOpinions", "longBusinessSummary", "website",
                "fundInceptionDate", "firstTradeDateMilliseconds")


def profile(symbol: str, cache_dir=None) -> dict:
    """What Yahoo says about the instrument today (company or fund), saved once per session."""
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    def fetch():
        info = yf_retry(lambda: _ticker(symbol).info) or {}
        return json_safe({k: info.get(k) for k in PROFILE_KEYS if info.get(k) is not None})

    store = local_store.store(cache_dir)
    return store.snapshot(symbol, "profile", fetch) if store else fetch()


STATEMENT_ROWS = {
    "income": ("Total Revenue", "Gross Profit", "Operating Income", "Interest Expense", "Pretax Income", "Tax Provision",
               "Net Income", "Diluted EPS", "EBITDA"),
    "balance": ("Total Assets", "Current Assets", "Total Liabilities Net Minority Interest", "Current Liabilities",
                "Stockholders Equity", "Cash And Cash Equivalents", "Total Debt", "Ordinary Shares Number"),
    "cashflow": ("Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", "Repurchase Of Capital Stock",
                 "Cash Dividends Paid"),
}


def financials(symbol: str, cache_dir=None) -> dict:
    """Annual and quarterly income statement, balance sheet and cash flow (key lines), refetched weekly.

    These are Yahoo's figures as of today: a period's numbers may have been restated since
    they were first reported. For what was on file at a past date, the SEC EDGAR vendor serves
    each figure at its filing date.
    """
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    def table(frame, rows):
        if frame is None or len(frame) == 0:
            return None
        keep = [r for r in rows if r in frame.index]
        sub = frame.loc[keep]
        return {"periods": [str(c)[:10] for c in sub.columns],
                "rows": {r: [None if v != v else float(v) for v in sub.loc[r].tolist()] for r in keep}}

    def fetch():
        t = _ticker(symbol)
        out = {}
        for kind, rows in STATEMENT_ROWS.items():
            annual = {"income": "income_stmt", "balance": "balance_sheet", "cashflow": "cashflow"}[kind]
            out[kind] = {"annual": table(yf_retry(lambda a=annual: getattr(t, a)), rows),
                         "quarterly": table(yf_retry(lambda a=annual: getattr(t, "quarterly_" + a)), rows)}
        return json_safe(out)

    store = local_store.store(cache_dir)
    # Saved under a new kind whenever STATEMENT_ROWS grows, so the added lines are fetched once.
    return store.snapshot(symbol, "financials-v3", fetch, ttl=HISTORY_TTL) if store else fetch()


def ohlcv(tickers: list[str], cache_dir) -> dict:
    """Daily open, high, low, close and volume (split- and dividend-adjusted) per ticker, for the few
    models that need opens; each ticker downloaded once and saved under ``ohlcv/``."""
    from pathlib import Path

    import yfinance as yf

    folder = Path(cache_dir) / "ohlcv"
    folder.mkdir(parents=True, exist_ok=True)
    out, missing = {}, []
    for t in tickers:
        p = folder / f"{t}.pkl"
        if p.exists():
            out[t] = pd.read_pickle(p)
        else:
            missing.append(t)
    if missing:
        raw = yf.download(missing, start="2000-01-01", auto_adjust=True, progress=False, group_by="ticker", threads=True)
        for t in missing:
            try:
                f = raw[t].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            except KeyError:
                continue
            f.index = pd.to_datetime(f.index).tz_localize(None)
            f.to_pickle(folder / f"{t}.pkl")
            out[t] = f
    return out


def actions(symbol: str, cache_dir=None) -> dict:
    """Dividends (split-adjusted, per share) and splits only, as ``[date, value]`` pairs oldest
    first: one light request per stock, refetched weekly. ``corporate_actions`` adds share counts."""
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    def fetch():
        frame = yf_retry(lambda: _ticker(symbol).actions)
        if frame is None or not len(frame):
            return {"dividends": [], "splits": []}
        pts = lambda col: [[str(i)[:10], float(v)] for i, v in frame[col].items() if v and v == v]  # noqa: E731
        return {"dividends": pts("Dividends") if "Dividends" in frame else [],
                "splits": pts("Stock Splits") if "Stock Splits" in frame else []}

    store = local_store.store(cache_dir)
    return store.snapshot(symbol, "actions", fetch, ttl=HISTORY_TTL) if store else fetch()


def corporate_actions(symbol: str, cache_dir=None) -> dict:
    """Every dividend (split-adjusted, per share), every split, and the shares-outstanding history
    (as reported: not adjusted for splits), as ``[date, value]`` pairs oldest first; refetched weekly."""
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    def points(series):
        if series is None or not len(series):
            return []
        return [[str(i)[:10], float(v)] for i, v in series.items() if v == v]

    def fetch():
        t = _ticker(symbol)
        try:
            shares = yf_retry(lambda: t.get_shares_full(start="2000-01-01"))
        except Exception as exc:  # a fund, or Yahoo has no share history for it
            logger.info("%s share history unavailable: %s", symbol, exc)
            shares = None
        return {"dividends": points(yf_retry(lambda: t.dividends)), "splits": points(yf_retry(lambda: t.splits)),
                "shares": points(shares)}

    store = local_store.store(cache_dir)
    return store.snapshot(symbol, "corporate_actions", fetch, ttl=HISTORY_TTL) if store else fetch()


def current_analysts(symbol: str, cache_dir=None) -> dict:
    """Today's Wall Street consensus, saved once per session: price targets, the buy/hold/sell
    counts over the last four months, and the EPS and revenue estimates (average, low, high,
    analyst count, growth) for this and next quarter and year, with their trend and revisions."""
    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    store = local_store.store(cache_dir)

    def get(kind, fn):
        def fetch():
            return json_safe(yf_retry(lambda: fn(_ticker(symbol))))
        try:
            return store.snapshot(symbol, kind, fetch) if store else fetch()
        except Exception as exc:  # an ETF has none of this
            logger.info("%s %s unavailable: %s", symbol, kind, exc)
            return None

    return {"price_targets": get("price_targets", lambda t: t.analyst_price_targets or {}),
            "recommendations": get("recommendations", lambda t: _records(t.recommendations_summary)),
            "eps_trend": get("eps_trend", lambda t: _records(t.eps_trend)),
            "earnings_estimate": get("earnings_estimate", lambda t: _records(t.earnings_estimate)),
            "revenue_estimate": get("revenue_estimate", lambda t: _records(t.revenue_estimate)),
            "eps_revisions": get("eps_revisions", lambda t: _records(t.eps_revisions)),
            "growth_estimates": get("growth_estimates", lambda t: _records(t.growth_estimates))}
