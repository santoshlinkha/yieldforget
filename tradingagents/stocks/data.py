"""The stock module's inputs: which tickers are stocks, and every shared dataset fetched once.

Tickers come from a pasted list, a CSV, the saved watchlist or the stock holdings.
Each is checked: S&P 500 members are stocks by definition; anything else is a stock
only when Yahoo calls it an equity. Funds, crypto, futures, options and indexes are
turned away with the reason, and an unknown symbol is reported, never guessed.

The training universe is today's S&P 500 plus the requested stocks. A member counts
from the date it was added to the index (``Date added`` in the member list), which
removes the run-up that got it added; members that were later removed are still
missing (survivorship bias), and every result says so.

Shared data is fetched once per batch: daily bars for all stocks, SPY, the cash
fund, the eleven sector funds and the market series (the local store downloads
only new sessions); SEC company facts and Yahoo splits and dividends per stock,
four at a time with SEC's rate limit and retries, each saved for a week. A stock
whose data fails gets that error recorded; the batch goes on without it.
"""

from __future__ import annotations

import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CASH = "BIL"
SECTOR_ETF = {"Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
              "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE", "Industrials": "XLI",
              "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication Services": "XLC"}
YAHOO_TO_GICS = {"Technology": "Information Technology", "Financial Services": "Financials",
                 "Healthcare": "Health Care", "Consumer Cyclical": "Consumer Discretionary",
                 "Consumer Defensive": "Consumer Staples", "Energy": "Energy", "Industrials": "Industrials",
                 "Basic Materials": "Materials", "Utilities": "Utilities", "Real Estate": "Real Estate",
                 "Communication Services": "Communication Services"}
NOT_STOCKS = {"ETF": "a fund (ETF)", "MUTUALFUND": "a mutual fund", "CRYPTOCURRENCY": "crypto", "FUTURE": "a future",
              "OPTION": "an option", "INDEX": "an index", "CURRENCY": "a currency", "MONEYMARKET": "a money-market fund"}


def parse_tickers(text: str = "", csv_text: str = "") -> list[str]:
    """Symbols from pasted text (commas, spaces or lines) and from a CSV: the column named like
    ticker/symbol, else the first column. Upper case, class shares with a dash (BRK.B -> BRK-B)."""
    found = re.split(r"[\s,;]+", text or "")
    if csv_text and csv_text.strip():
        frame = pd.read_csv(io.StringIO(csv_text))
        col = next((c for c in frame.columns if str(c).strip().lower() in ("ticker", "symbol", "tickers", "symbols")),
                   frame.columns[0])
        found += [str(v) for v in frame[col].dropna()]
    out = []
    for raw in found:
        t = raw.strip().strip("$\"'").upper().replace(".", "-")
        if t and re.fullmatch(r"[A-Z][A-Z0-9\-]{0,9}", t) and t not in out:
            out.append(t)
    return out


def members(cache_dir: Path) -> pd.DataFrame:
    """Today's S&P 500, indexed by ticker: name, GICS sector, CIK and date added."""
    from tradingagents.dataflows.vendors.index_members import get_sp500_members

    get_sp500_members(cache_dir)   # refreshes the saved list when it is old
    frame = pd.read_csv(Path(cache_dir) / "sp500_members.csv")
    frame["Symbol"] = frame["Symbol"].astype(str).str.replace(".", "-", regex=False)
    frame["added"] = pd.to_datetime(frame["Date added"], errors="coerce")
    return frame.set_index("Symbol")


def classify(tickers: list[str], cache_dir: Path, table: pd.DataFrame | None = None,
             workers: int = 4) -> tuple[dict[str, dict], dict[str, str]]:
    """``(stocks, rejected)``: each stock's name, sector, sector fund, CIK and index entry date;
    each rejected symbol's reason."""
    from tradingagents.dataflows.vendors.yahoo import history as yh

    table = members(cache_dir) if table is None else table
    stocks, rejected, unknown = {}, {}, []
    for t in tickers:
        if t in table.index:
            row = table.loc[t]
            stocks[t] = {"name": row["Security"], "sector": row["GICS Sector"],
                         "sector_etf": SECTOR_ETF.get(row["GICS Sector"]), "cik": f"{int(row['CIK']):010d}",
                         "added": None if pd.isna(row["added"]) else str(row["added"].date()), "sp500": True}
        else:
            unknown.append(t)

    def look(t):
        try:
            return t, yh.profile(t, cache_dir), None
        except Exception as exc:  # one bad symbol never stops the rest
            return t, None, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(look, t) for t in unknown]):
            t, p, error = fut.result()
            if error is not None:
                rejected[t] = f"lookup failed: {str(error)[:120]}"
                continue
            kind = (p or {}).get("quoteType")
            if kind == "EQUITY":
                sector = YAHOO_TO_GICS.get(p.get("sector"), p.get("sector"))
                stocks[t] = {"name": p.get("longName") or p.get("shortName") or t, "sector": sector,
                             "sector_etf": SECTOR_ETF.get(sector), "cik": None, "added": None, "sp500": False}
            elif kind in NOT_STOCKS:
                rejected[t] = f"{NOT_STOCKS[kind]}: this module forecasts stocks only"
            else:
                rejected[t] = "not found on Yahoo Finance" if not p else f"not a stock ({kind or 'unknown type'})"
    return stocks, rejected


@dataclass
class StockData:
    cache_dir: Path
    info: dict[str, dict]                     # every stock in the universe
    requested: list[str]                      # the stocks asked for, in order
    rejected: dict[str, str]
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    facts: dict[str, dict | None] = field(default_factory=dict)
    actions: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, list[str]] = field(default_factory=dict)
    spans: dict | None = None                 # point-in-time membership intervals (``membership``)
    survivorship: str = "biased"              # "partial" with historical membership, else "biased"

    @property
    def stocks(self) -> list[str]:
        return [t for t in self.info if t in self.close.columns]


def load(requested: list[str], cache_dir: Path, universe: bool = True, progress=None, workers: int = 4,
         fundamentals: bool = True) -> StockData:
    """Everything a batch needs, fetched once. ``universe`` adds today's S&P 500 for training."""
    from tradingagents.dataflows.vendors.yahoo import history as yh
    from tradingagents.research.prices import MARKET, bars

    from .fundamentals import facts_for

    say = progress or (lambda stage, **_: None)
    cache_dir = Path(cache_dir)
    table = members(cache_dir)
    from . import membership

    say("checking which tickers are stocks")
    asked, rejected = classify(requested, cache_dir, table, workers)
    info = dict(asked)
    spans, survivorship = None, "biased"
    if universe:
        extra, _ = classify([t for t in table.index if t not in info], cache_dir, table, workers)
        info.update(extra)
        changes = membership.load(cache_dir)
        if changes is not None:
            spans, survivorship = membership.intervals(changes), "partial"
            # Former members since 2005: prices only inside their intervals, no SEC facts (see membership).
            for t in membership.members_since(spans, "2005-01-01"):
                if t not in info:
                    info[t] = {"name": t, "sector": None, "sector_etf": None, "cik": None, "added": None,
                               "sp500": True, "removed": True}
    tickers = list(info)
    support = [CASH, *MARKET, *sorted({v for v in SECTOR_ETF.values()})]
    say("loading daily prices", count=len(tickers))
    b = bars([*tickers, *support], cache_dir)
    data = StockData(cache_dir, info, [t for t in requested if t in asked], rejected,
                     b["Close"], b["High"], b["Low"], b["Volume"], spans=spans, survivorship=survivorship)
    for t in tickers:
        if t not in data.close.columns or data.close[t].dropna().empty:
            data.errors.setdefault(t, []).append("no prices from Yahoo")

    def one(t):
        got, problems = {}, []
        if fundamentals and not info[t].get("removed"):
            try:
                got["facts"] = facts_for(t, cache_dir, info[t].get("cik"))
            except Exception as exc:  # recorded per stock; the batch carries on
                problems.append(f"SEC facts failed: {str(exc)[:120]}")
        try:
            got["actions"] = yh.actions(t, cache_dir)
        except Exception as exc:
            problems.append(f"splits and dividends failed: {str(exc)[:120]}")
        return t, got, problems

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tickers]):
            t, got, problems = fut.result()
            data.facts[t] = got.get("facts")
            data.actions[t] = got.get("actions") or {"dividends": [], "splits": []}
            if problems:
                data.errors.setdefault(t, []).extend(problems)
            done += 1
            if done % 25 == 0 or done == len(tickers):
                say("company filings, splits and dividends", done=done, count=len(tickers))
    return data


def raw_close(adjusted: pd.Series, actions: dict) -> pd.Series:
    """The price as it traded each day: Yahoo's closes are adjusted for later dividends and splits;
    both are undone, so the price matches the share count companies reported at the time."""
    from tradingagents.research.quality import raw_closes

    price = raw_closes(adjusted, actions.get("dividends") or [])
    factor = pd.Series(1.0, index=price.index)
    for d, ratio in actions.get("splits") or []:
        if ratio and ratio > 0:
            factor[factor.index < pd.Timestamp(d)] *= ratio
    return price * factor


def quality(close: pd.Series, volume: pd.Series, cutoff: pd.Timestamp) -> list[str]:
    """Reasons a stock's data cannot be trusted at ``cutoff``: no price, stale, too short, halted."""
    seen = close.loc[:cutoff].dropna()
    if seen.empty:
        return ["no price on record"]
    out = []
    if seen.index[-1] < cutoff:
        out.append(f"stale: last price {seen.index[-1].date()}")
    if len(seen) < 252:
        out.append(f"insufficient history: {len(seen)} sessions, a year is needed")
    v = volume.loc[:cutoff].dropna().iloc[-5:]
    if len(v) and (v == 0).all():
        out.append("no trading volume in the last 5 sessions (suspended?)")
    if len(seen) >= 2 and not np.isfinite(np.log(seen.iloc[-1] / seen.iloc[-2])):
        out.append("unusable latest price")
    return out


def gather(store=None, text: str = "", csv_text: str = "", watchlist: bool = False, holdings: bool = False,
           as_of=None) -> tuple[list[str], dict[str, str]]:
    """The tickers to analyze and where each came from: pasted, CSV, the saved watchlist, the account's
    holdings (funds among them are turned away later by ``classify``)."""
    from datetime import date

    sources: dict[str, str] = {}
    for t in parse_tickers(text):
        sources.setdefault(t, "pasted")
    for t in parse_tickers(csv_text=csv_text):
        sources.setdefault(t, "csv")
    if watchlist and store is not None:
        for t in parse_tickers(" ".join(store.get("global", "stock_watchlist") or [])):
            sources.setdefault(t, "watchlist")
    if holdings and store is not None:
        state = store.ledger.state("combined", as_of or date.today())
        for t in parse_tickers(" ".join(t for t, q in state.positions.items() if q)):
            sources.setdefault(t, "holding")
    return list(sources), sources
