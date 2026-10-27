"""The trading budget: SGOV's monthly interest and the broker's match, and what is left of them to trade.

The account works like this: every market session ``daily_sgov_usd`` (Settings) is deposited and
bought into SGOV, a fund of 0-3 month Treasury bills, which pays its interest as a monthly
dividend. SGOV is the base and is never traded. The interest it pays, and the broker's match on
the deposits, are what get traded in stocks:

    budget = SGOV interest earned + match earned - money put into stocks + money back from sales

all counted from the day the strategy started (``global.income_start``). Each part is
**recorded** when the ledger has the events (``dividend`` or ``interest`` events on SGOV,
``contribution`` events), else **estimated**: interest as SGOV's paid dividend per share on each
ex-date times the shares held the day before, the match as the rate times the planned deposit for
every session. An estimate is labelled as one, and recorded events replace it.

``health`` says whether the interest is drying up: SGOV's latest monthly dividend annualized, its
trailing twelve months, and the 13-week bill rate (^IRX, what SGOV earns next) now and six months
ago. When the yield falls under ``dry_yield`` or the bill rate is falling fast, ``alternatives``
lists where the daily deposit could go instead, with each fund's yield and worst fall.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

RESERVE = "SGOV"
# Where the daily deposit could go if SGOV's interest dries up: (ticker, what it is, when it fits).
ALTERNATIVES = [
    ("USFR", "Floating-rate Treasury notes", "keeps a cash-like price; its rate resets weekly, so it falls with bills too"),
    ("SHY", "1-3 year Treasuries", "locks today's yields for 1-3 years and gains a little when rates fall"),
    ("IEI", "3-7 year Treasuries", "locks yields longer; gains more when rates fall, loses when they rise"),
    ("IEF", "7-10 year Treasuries", "the most gain from falling rates among these, and real price swings"),
    ("TLT", "20+ year Treasuries", "large gains if rates fall a lot, large losses if they rise (fell ~45% in 2020-23)"),
    ("JEPI", "Covered-call US stocks", "high monthly income from options, stock-market risk"),
    ("SCHD", "US dividend stocks", "dividends that grow over time, stock-market risk"),
    ("VTI", "The whole US stock market", "growth rather than income: best over decades, can fall 50%"),
]


def start(store) -> date | None:
    value = store.get("global", "income_start")
    return date.fromisoformat(value) if value else None


def set_start(store, day) -> None:
    store.set("global", "income_start", str(day))


def _dividends(cache_dir, ticker: str) -> list[tuple[date, float]]:
    from tradingagents.dataflows.vendors.yahoo import history as yh

    try:
        rows = (yh.actions(ticker, cache_dir) or {}).get("dividends") or []
    except Exception:   # no data: no estimate, said so by the caller
        return []
    return [(date.fromisoformat(str(d)[:10]), float(a)) for d, a in rows]


def _events(store, kinds: tuple[str, ...], since: date, as_of: date, ticker: str | None = None) -> list[dict]:
    return [e for e in store.ledger.events(None, as_of) if e["kind"] in kinds and str(e["trade_date"]) >= str(since)
            and (ticker is None or e["ticker"] == ticker)]


def interest(store, cache_dir, since: date, as_of: date) -> dict:
    """SGOV interest earned from ``since`` to ``as_of``: recorded events, else estimated from its dividends."""
    recorded = _events(store, ("dividend", "interest"), since, as_of, RESERVE)
    if recorded:
        items = [{"date": str(e["trade_date"]), "amount": round(float(e["cash"]), 2), "basis": "recorded"} for e in recorded]
        return {"earned": round(sum(i["amount"] for i in items), 2), "basis": "recorded", "items": items}
    items = []
    for ex, per_share in _dividends(cache_dir, RESERVE):
        if since < ex <= as_of:
            held = store.ledger.state(None, ex - timedelta(days=1)).positions.get(RESERVE, 0.0)
            if held > 0:
                items.append({"date": str(ex), "amount": round(held * per_share, 2), "per_share": per_share,
                              "shares": round(held, 4), "basis": "estimated"})
    return {"earned": round(sum(i["amount"] for i in items), 2), "basis": "estimated", "items": items}


def match(store, since: date, as_of: date, daily: float, rate: float) -> dict:
    """The broker's match from ``since`` to ``as_of``: on recorded deposits, else on the planned daily deposit."""
    from tradingagents.ledger.sessions import sessions_between

    recorded = _events(store, ("contribution",), since, as_of)
    if recorded:
        deposits = sum(float(e["cash"]) for e in recorded)
        return {"earned": round(deposits * rate, 2), "deposits": round(deposits, 2), "basis": "recorded"}
    sessions = sessions_between(since, as_of) if as_of > since else 0
    deposits = daily * sessions
    return {"earned": round(deposits * rate, 2), "deposits": round(deposits, 2), "sessions": sessions,
            "basis": "estimated"}


def traded(store, since: date, as_of: date) -> dict:
    """Money that went into stocks (buys) and came back (sales and their dividends), outside SGOV."""
    fills = [e for e in _events(store, ("fill", "fee", "dividend"), since, as_of) if e["ticker"] != RESERVE]
    spent = -sum(float(e["cash"]) for e in fills if float(e["cash"]) < 0)
    back = sum(float(e["cash"]) for e in fills if float(e["cash"]) > 0)
    return {"spent": round(spent, 2), "returned": round(back, 2)}


def budget(store, cache_dir, as_of: date | None = None) -> dict:
    """What the interest and the match have made available for trading, and what is left of it."""
    from . import capital

    as_of = as_of or date.today()
    cap = capital.load(store)
    since = start(store)
    if since is None:
        return {"started": None, "available": 0.0, "note": "the strategy has no start date yet"}
    i = interest(store, cache_dir, since, as_of)
    m = match(store, since, as_of, cap.daily_sgov_usd, cap.match_rate)
    t = traded(store, since, as_of)
    available = i["earned"] + m["earned"] - t["spent"] + t["returned"]
    notes = []
    if "estimated" in (i["basis"], m["basis"]):
        notes.append("Estimated parts are replaced by what you record: SGOV dividends as dividend events, deposits as "
                     "contributions. Robinhood pays the match on deposits, not on purchases.")
    notes.append("SGOV is never sold for stocks beyond this budget. To trade it, sell that much SGOV (or use the "
                 "dividend cash if it was paid out); the sale settles the next session (T+1).")
    return {"started": str(since), "as_of": str(as_of), "interest": i, "match": m, "traded": t,
            "available": round(max(available, 0.0), 2), "net": round(available, 2), "notes": notes,
            "settings": {"daily_sgov_usd": cap.daily_sgov_usd, "match_rate": cap.match_rate}}


def _closes(cache_dir, tickers) -> pd.DataFrame:
    from tradingagents.research.prices import bars

    return bars(list(tickers), cache_dir)["Close"]


def trailing_yield(dividends: list[tuple[date, float]], price: float, as_of: date) -> float | None:
    paid = [a for d, a in dividends if as_of - timedelta(days=365) < d <= as_of]
    return sum(paid) / price if paid and price > 0 else None


def health(store, cache_dir, as_of: date | None = None, closes: pd.DataFrame | None = None) -> dict:
    """Is SGOV's interest drying up? Its latest and trailing yield, and where bill rates are heading."""
    from . import capital

    as_of = as_of or date.today()
    dry = capital.load(store).dry_yield
    if closes is None:
        try:
            closes = _closes(cache_dir, [RESERVE, "^IRX"])
        except Exception as exc:   # no prices: the page says so instead of failing
            return {"status": "unknown", "message": f"SGOV prices could not be read: {exc}"}
    divs = [(d, a) for d, a in _dividends(cache_dir, RESERVE) if d <= as_of]
    px = closes[RESERVE].dropna() if RESERVE in closes else pd.Series(dtype=float)
    if px.empty or not divs:
        return {"status": "unknown", "message": "no SGOV prices or dividends in the local store yet"}
    price = float(px.iloc[-1])
    latest = divs[-1][1] * 12 / price
    trailing = trailing_yield(divs, price, as_of)
    bill = closes["^IRX"].dropna() / 100 if "^IRX" in closes else pd.Series(dtype=float)
    bill_now = float(bill.iloc[-1]) if len(bill) else None
    bill_6m = float(bill.iloc[-127]) if len(bill) > 126 else None
    change = bill_now - bill_6m if bill_now is not None and bill_6m is not None else None
    reasons = []
    if latest < dry:
        status = "dried"
        reasons.append(f"SGOV's latest monthly dividend is {latest:.2%} a year, under your {dry:.2%} line")
    elif change is not None and change <= -0.005:
        status = "falling"
        reasons.append(f"the 13-week bill rate fell {-change:.2%} in six months, to {bill_now:.2%}: SGOV's interest "
                       "follows it down within a month or two")
    elif trailing is not None and latest < trailing - 0.003:
        status = "falling"
        reasons.append(f"the latest dividend ({latest:.2%} a year) is below the last twelve months ({trailing:.2%})")
    else:
        status = "ok"
    monthly = store.ledger.state(None, as_of).positions.get(RESERVE, 0.0) * divs[-1][1]
    return {"status": status, "reasons": reasons, "price": round(price, 4), "latest_yield": round(latest, 5),
            "trailing_yield": round(trailing, 5) if trailing is not None else None,
            "bill_rate": round(bill_now, 5) if bill_now is not None else None,
            "bill_rate_6m_ago": round(bill_6m, 5) if bill_6m is not None else None,
            "last_dividend": {"date": str(divs[-1][0]), "per_share": divs[-1][1]},
            "monthly_interest_now": round(monthly, 2), "dry_yield": dry,
            "alternatives": alternatives(cache_dir, as_of, status) if status != "ok" else None}


def alternatives(cache_dir, as_of: date, status: str) -> dict:
    """Where the daily deposit could go instead: each fund's yield, one-year return and worst fall, and a pick."""
    tickers = [t for t, _, _ in ALTERNATIVES]
    closes = _closes(cache_dir, tickers)
    rows = []
    for t, what, fits in ALTERNATIVES:
        px = closes[t].dropna() if t in closes else pd.Series(dtype=float)
        if px.empty:
            continue
        price = float(px.iloc[-1])
        y = trailing_yield([(d, a) for d, a in _dividends(cache_dir, t) if d <= as_of], price, as_of)
        year = px[px.index <= px.index[-1] - pd.Timedelta(days=365)]
        ret = float(price / year.iloc[-1] - 1) if len(year) else None
        five = px[px.index >= px.index[-1] - pd.Timedelta(days=5 * 365)]
        worst = float((five / five.cummax() - 1).min()) if len(five) else None
        rows.append({"ticker": t, "what": what, "fits": fits, "yield": None if y is None else round(y, 4),
                     "one_year": None if ret is None else round(ret, 4), "worst_fall_5y": None if worst is None else round(worst, 4)})
    # The pick: while rates are still falling, lock them for a few years with little price risk (the safest bond
    # fund with a positive yield and a 5-year worst fall under 10%); once interest has dried up, the highest
    # yield whose 5-year worst fall stayed under 20%. A stock fund is never the automatic pick.
    bonds = [r for r in rows if r["ticker"] in ("USFR", "SHY", "IEI", "IEF") and r["yield"]]
    pick = None
    if status == "falling":
        safe = [r for r in bonds if r["worst_fall_5y"] is not None and r["worst_fall_5y"] > -0.10]
        pick = max(safe, key=lambda r: r["yield"], default=None)
    elif status == "dried":
        ok = [r for r in rows if r["yield"] and r["worst_fall_5y"] is not None and r["worst_fall_5y"] > -0.20]
        pick = max(ok, key=lambda r: r["yield"], default=None)
    return {"rows": rows, "pick": pick["ticker"] if pick else None,
            "why": (f"{pick['ticker']} ({pick['what']}): {pick['fits']}. Yield {pick['yield']:.2%}, worst fall in five "
                    f"years {pick['worst_fall_5y']:.1%}." if pick else "no fund meets the rule; keep SGOV for now"),
            "note": "A suggestion from yields and past falls, not a forecast. Changing the daily buy is your call; "
                    "nothing changes until you change the recurring buy at Robinhood and in Settings."}


def projection(cap, health_row: dict, value: float) -> dict:
    """A month of interest and match at today's rates, and the deposits' yearly total."""
    y = health_row.get("latest_yield") or 0.0
    month_sessions = 21
    yearly = cap.daily_sgov_usd * 252
    return {"interest_per_month": round(value * y / 12, 2),
            "match_per_month": round(cap.daily_sgov_usd * month_sessions * cap.match_rate, 2),
            "deposits_per_year": round(yearly, 2),
            "limit_note": (f"${cap.daily_sgov_usd:,.2f} each session is about ${yearly:,.0f} a year: check it stays "
                           "within the IRA contribution limit for the year (the match does not count toward it).")}


def finite(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


__all__ = ["ALTERNATIVES", "RESERVE", "alternatives", "budget", "health", "interest", "match", "projection", "start",
           "set_start", "traded"]
