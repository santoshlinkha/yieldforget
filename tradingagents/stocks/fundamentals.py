"""Company fundamentals as they were known on each date, from SEC EDGAR's company facts.

Point in time. Every fact carries the date it was filed. A value counts from the
day **after** its filing date (EDGAR gives the date, not the acceptance time, so a
filing accepted after the close is not used that day). A later restatement is a
new fact with a later filing date: until that date the original value stands, so
history is never rewritten. A reporting period's end date is never treated as
the day its numbers became known.

Per stock, the facts are compacted to the concepts below and saved under the
cache folder (``sec_pit2/<TICKER>.json``), refetched after a week. Requests go
through one rate limiter (SEC asks for no more than 10 a second) with retries.

Features, from the latest fiscal year and quarter on file at each date:
revenue growth (year on year, annual and the latest quarter against the same
quarter a year before), gross, operating, net and free-cash-flow margins, return
on assets, liabilities to assets, dilution (share count change over a year), and
how many days old the latest filing is. Valuation (earnings, sales, free-cash-flow
and book yields) needs the market value on the date, from the price as it then
traded (``stocks.data.raw_close``) times the share count then reported.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
MAX_AGE = 7 * 24 * 3600
CACHE_DIR = "sec_pit2"   # renamed when CONCEPTS change, so the added concepts are fetched once

CONCEPTS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"),
    "op_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "ocf": ("NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "shares": ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
}
INSTANT = {"assets", "liabilities", "equity", "shares"}
FEATURES = ["rev_growth_y", "rev_growth_q", "gross_margin", "op_margin", "net_margin", "fcf_margin", "roa",
            "liabilities_to_assets", "dilution_1y"]
VALUATION = ["earnings_yield", "sales_yield", "fcf_yield", "book_yield"]
RAW = ["revenue_y", "net_income_y", "fcf_y", "equity_i", "shares_i", "shares_end_ord", "filing_age_days"]


class RateLimiter:
    """At most ``per_second`` calls, across threads."""

    def __init__(self, per_second: float = 8.0):
        self.gap, self.lock, self.last = 1.0 / per_second, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            delay = self.last + self.gap - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


SEC_LIMIT = RateLimiter()


def _get(url: str, tries: int = 4) -> dict:
    from tradingagents.dataflows.vendors.sec_edgar import _user_agent

    for attempt in range(tries):
        SEC_LIMIT.wait()
        try:
            response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=60)
            if response.status_code == 404:
                return {}
            if response.status_code in (429, 503):
                raise requests.HTTPError(f"{response.status_code} from SEC")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == tries - 1:
                raise
            logger.info("SEC retry %d for %s: %s", attempt + 1, url, exc)
            time.sleep(1.5 * 2 ** attempt)
    return {}


def compact(facts: dict) -> dict[str, list[list]]:
    """``{concept: [[start, end, filed, form, value], ...]}`` from a company-facts document; for
    each period the first tag in ``CONCEPTS`` order that reports it wins (tags change over the
    years, e.g. revenue under ASC 606), and every filing of that period is kept."""
    taxonomies = facts.get("facts", {})
    out = {}
    for concept, tags in CONCEPTS.items():
        seen_periods, rows = set(), []
        for tag in tags:
            found = None
            for tax in ("us-gaap", "dei"):
                if tag in taxonomies.get(tax, {}):
                    found = taxonomies[tax][tag]
                    break
            if not found:
                continue
            units = found.get("units", {})
            values = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
            mine = {}
            for f in values:
                if "filed" not in f or "end" not in f or f.get("val") is None:
                    continue
                key = (f.get("start"), f["end"])
                if key in seen_periods:
                    continue
                mine.setdefault(key, []).append([f.get("start"), f["end"], f["filed"], f.get("form", ""), float(f["val"])])
            for key, fs in mine.items():
                rows.extend(fs)
            seen_periods |= set(mine)
        out[concept] = sorted(rows, key=lambda r: (r[2], r[1]))
    return out


def facts_for(ticker: str, cache_dir: Path, cik: str | None = None) -> dict | None:
    """The compact facts for one stock: from the cache when under a week old, else from SEC.
    None for a company EDGAR does not know (a foreign filer, a fund)."""
    from tradingagents.dataflows.vendors.sec_edgar import cik_for

    path = Path(cache_dir) / CACHE_DIR / f"{ticker.upper()}.json"
    if path.exists() and time.time() - path.stat().st_mtime < MAX_AGE:
        return json.loads(path.read_text(encoding="utf-8")) or None
    cik = cik or cik_for(ticker)
    data = compact(_get(FACTS_URL.format(cik=cik))) if cik else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
    return data or None


def _day(s: str) -> date:
    return date.fromisoformat(s)


def _kind(start, end) -> str | None:
    """'instant' (a balance on a day), 'annual' (330-400 days), 'quarter' (80-100), else None."""
    if not start:
        return "instant"
    span = (_day(end) - _day(start)).days
    return "annual" if 330 <= span <= 400 else "quarter" if 80 <= span <= 100 else None


def _latest(known: dict):
    if not known:
        return None, None
    end = max(known)
    return end, known[end]


def _year_before(known: dict, end: str, slack: int = 25):
    target = _day(end) - timedelta(days=365)
    for e, v in known.items():
        if abs((_day(e) - target).days) <= slack:
            return v
    return None


def _ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else np.nan


def _split_factor(splits: list, after: str, until: str) -> float:
    """The product of the split ratios between two dates: how many of today's shares one share was."""
    factor = 1.0
    for d, ratio in splits or []:
        if after < d <= until and ratio > 0:
            factor *= ratio
    return factor


def _features(known: dict, splits: list | None) -> dict:
    """The features from what is on file: ``known[(concept, kind)] = {period end: value}``, each
    period holding its latest filing so far."""
    get = lambda c, k: known.get((c, k), {})  # noqa: E731
    out = {k: np.nan for k in FEATURES + RAW}
    rev_y = get("revenue", "annual")
    end_y, revenue = _latest(rev_y)
    if end_y:
        out["revenue_y"] = revenue
        out["rev_growth_y"] = _ratio(revenue, _year_before(rev_y, end_y)) - 1
        gross = get("gross_profit", "annual").get(end_y)
        if gross is None and get("cost_of_revenue", "annual").get(end_y) is not None:
            gross = revenue - get("cost_of_revenue", "annual")[end_y]   # revenue less cost of revenue
        out["gross_margin"] = _ratio(gross, revenue)
        out["op_margin"] = _ratio(get("op_income", "annual").get(end_y), revenue)
        out["net_margin"] = _ratio(get("net_income", "annual").get(end_y), revenue)
        ocf, capex = get("ocf", "annual").get(end_y), get("capex", "annual").get(end_y)
        if ocf is not None and capex is not None:     # a missing capex is unknown, not zero
            fcf = ocf - capex
            out["fcf_y"], out["fcf_margin"] = fcf, _ratio(fcf, revenue)
        out["net_income_y"] = get("net_income", "annual").get(end_y, np.nan)
    end_q, rq = _latest(get("revenue", "quarter"))
    if end_q:
        out["rev_growth_q"] = _ratio(rq, _year_before(get("revenue", "quarter"), end_q)) - 1
    end_a, a = _latest(get("assets", "instant"))
    equity = get("equity", "instant")
    if end_a:
        liabilities = get("liabilities", "instant").get(end_a)
        if liabilities is None and equity.get(end_a) is not None:
            liabilities = a - equity[end_a]                      # assets less equity
        out["liabilities_to_assets"] = _ratio(liabilities, a)
        if end_y:
            out["roa"] = _ratio(out["net_income_y"], a)
    out["equity_i"] = _latest(equity)[1]
    shares = get("shares", "instant")
    end_s, sh = _latest(shares)
    if end_s:
        out["shares_i"] = sh
        out["shares_end_ord"] = float(_day(end_s).toordinal())   # the date the count is as of
        before = _year_before(shares, end_s, slack=60)
        if before is not None:
            prior_end = (_day(end_s) - timedelta(days=365)).isoformat()
            out["dilution_1y"] = _ratio(sh, before * _split_factor(splits, prior_end, end_s)) - 1
    return {k: (np.nan if v is None else float(v)) for k, v in out.items()}


def _events(facts: dict) -> list[tuple]:
    """Every fact as ``(filed, concept, kind, end, value)``, in filing order."""
    rows = []
    for concept, facts_of in facts.items():
        for start, end, filed, _form, value in facts_of:
            kind = _kind(start, end)
            if kind:
                rows.append((filed, concept, kind, end, value))
    return sorted(rows)


def snapshot(facts: dict, cutoff: str, splits: list | None = None) -> dict:
    """Every feature from the facts filed on or before ``cutoff``. ``splits`` (``[[date, ratio]]``)
    keep a split from reading as dilution: a year-earlier share count is restated first."""
    known = {}
    for filed, concept, kind, end, value in _events(facts):
        if filed > cutoff:
            break
        known.setdefault((concept, kind), {})[end] = value
    return _features(known, splits)


def history(facts: dict, dates: pd.DatetimeIndex, splits: list | None = None) -> pd.DataFrame:
    """Features at each date, from the filings made strictly before it.

    The known set changes only on filing days, so it is updated one filing day at a
    time and the features recomputed then; dates in between carry them forward.
    """
    events = _events(facts)
    if not events:
        return pd.DataFrame(index=dates, columns=FEATURES + RAW, dtype=float)
    known, states, days = {}, [], []
    i = 0
    while i < len(events):
        day = events[i][0]
        while i < len(events) and events[i][0] == day:
            _, concept, kind, end, value = events[i]
            known.setdefault((concept, kind), {})[end] = value
            i += 1
        states.append(_features(known, splits))
        days.append(day)
    # Usable from the day after filing: a decision on the filing date itself does not see it.
    usable = pd.to_datetime(days) + pd.Timedelta(days=1)
    frame = pd.DataFrame(states, index=usable)
    out = frame.reindex(frame.index.union(dates)).ffill().reindex(dates)
    seen = pd.Series(usable, index=usable)
    seen = seen.reindex(seen.index.union(dates)).ffill().reindex(dates)
    out["filing_age_days"] = (dates - pd.DatetimeIndex(seen)).days + 1
    return out


def valuation(fund: pd.DataFrame, raw_price: pd.Series, splits: list | None = None) -> pd.DataFrame:
    """Yields on the market value at each date: the price as it then traded times the share count
    reported, carried to that date's split basis (a split between the count's date and the
    valuation date multiplies the count), so price and shares always match."""
    shares = fund["shares_i"].copy()
    for i, (d, ordv) in enumerate(zip(fund.index, fund.get("shares_end_ord", pd.Series(np.nan, index=fund.index)),
                                      strict=True)):
        if np.isfinite(ordv):
            shares.iloc[i] *= _split_factor(splits, date.fromordinal(int(ordv)).isoformat(), d.date().isoformat())
    cap = raw_price.reindex(fund.index) * shares
    cap = cap.where(cap > 0)
    return pd.DataFrame({"earnings_yield": fund["net_income_y"] / cap, "sales_yield": fund["revenue_y"] / cap,
                         "fcf_yield": fund["fcf_y"] / cap, "book_yield": fund["equity_i"] / cap}, index=fund.index)
