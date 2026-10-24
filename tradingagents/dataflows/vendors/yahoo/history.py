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
    out = frame.reset_index()
    out.iloc[:, 0] = out.iloc[:, 0].astype(str)
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
