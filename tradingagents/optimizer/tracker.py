"""Paper tracking: what the portfolio would be worth had you followed the plan from a start day.

Two paper portfolios start from your real holdings:

- **Followed the plan**: each trading day it settles yesterday's sales, runs the
  optimizer on that day's closing prices, and carries out the day's orders
  exactly as the app suggests them (buys from settled cash only, sales settling
  the next trading day).
- **Kept as is**: your holdings untouched, plus your recurring buys, funded from
  the cash fund under the same T+1 rule.

Orders fill at the day's close. A day is only processed once its close is final
(after 4:30 pm New York time), so an intraday price never becomes part of the
record. The ledger is a JSON file next to your holdings; reading it never
changes it, and a missed day is caught up on the next update.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

from tradingagents.portfolio import PortfolioContext, Position

from .plan import OptimizerSettings, build_plan, daily_orders

NEW_YORK = pytz.timezone("America/New_York")
PORTFOLIOS = ("plan", "hold")


@dataclass
class Book:
    shares: dict[str, float]
    settled: float
    unsettled: float

    @classmethod
    def from_json(cls, d: dict) -> Book:
        return cls(dict(d["shares"]), float(d["settled"]), float(d["unsettled"]))

    def to_json(self) -> dict:
        return {"shares": {t: q for t, q in self.shares.items() if q > 1e-9},
                "settled": round(self.settled, 6), "unsettled": round(self.unsettled, 6)}

    def value(self, prices: dict[str, float]) -> float:
        return sum(q * prices.get(t, 0.0) for t, q in self.shares.items()) + self.settled + self.unsettled

    def settle(self):
        self.settled += self.unsettled
        self.unsettled = 0.0

    def sell(self, ticker: str, amount: float, price: float) -> float:
        qty = min(amount / price, self.shares.get(ticker, 0.0))
        self.shares[ticker] = self.shares.get(ticker, 0.0) - qty
        self.unsettled += qty * price
        return qty * price

    def buy(self, ticker: str, amount: float, price: float) -> float:
        spend = min(amount, self.settled)
        if spend <= 0:
            return 0.0
        self.shares[ticker] = self.shares.get(ticker, 0.0) + spend / price
        self.settled -= spend
        return spend


def _final_days(closes: pd.DataFrame, ref: str, now: datetime | None = None) -> list[pd.Timestamp]:
    """Trading days whose close is final: every day before today, and today after 4:30 pm New York."""
    now = now or datetime.now(NEW_YORK)
    days = list(closes[ref].dropna().index)
    today = pd.Timestamp(now.date())
    after_close = (now.hour, now.minute) >= (16, 30)
    return [d for d in days if d < today or (d == today and after_close)]


def _prices(closes: pd.DataFrame, day) -> dict[str, float]:
    row = closes.loc[:day].ffill().iloc[-1]
    return {t: float(p) for t, p in row.items() if pd.notna(p)}


def _trade_plan(book: Book, day, closes, schedule, settings, universe, prices) -> list[list]:
    positions = [Position(ticker=t, quantity=q) for t, q in book.shares.items() if q > 1e-9]
    if not any(p.ticker == settings.cash_ticker for p in positions):
        positions.append(Position(ticker=settings.cash_ticker, quantity=0.0))
    context = PortfolioContext(positions=positions, cash=book.settled + book.unsettled)
    plan = build_plan(context, schedule, settings, closes=closes.loc[:day], universe=universe, summary=False)
    done = []
    for order in sorted(daily_orders(plan, book.settled).orders, key=lambda o: o.side != "sell"):
        price = prices.get(order.ticker)
        if not price:
            continue
        filled = (book.sell if order.side == "sell" else book.buy)(order.ticker, order.amount, price)
        if filled >= 0.01:
            done.append([order.side, order.ticker, round(filled, 2)])
    return done


def _trade_hold(book: Book, schedule: dict[str, float], cash_ticker: str, prices) -> list[list]:
    """Your recurring buys as they are: settled cash buys the schedule, the cash fund funds tomorrow."""
    done = []
    budget = sum(schedule.values())
    if budget <= 0:
        return done
    share = min(1.0, book.settled / budget)
    for ticker, amount in schedule.items():
        if ticker in prices and (filled := book.buy(ticker, amount * share, prices[ticker])) >= 0.01:
            done.append(["buy", ticker, round(filled, 2)])
    if cash_ticker in prices and (sold := book.sell(cash_ticker, budget, prices[cash_ticker])) >= 0.01:
        done.append(["sell", cash_ticker, round(sold, 2)])
    return done


def start_ledger(portfolio: PortfolioContext, settings: OptimizerSettings, closes: pd.DataFrame,
                 start: str | None = None) -> dict:
    """A new ledger from the current holdings, starting at ``start`` or the latest close."""
    ref = settings.cash_proxy if settings.cash_proxy in closes else settings.cash_ticker
    days = list(closes[ref].dropna().index)
    first = days[-1] if not start else next((d for d in days if d >= pd.Timestamp(start)), None)
    if first is None:
        raise ValueError(f"no trading day on or after {start}")
    shares = {p.ticker.upper(): p.quantity for p in portfolio.positions}
    book = Book(shares, float(portfolio.cash or 0.0), 0.0).to_json()
    return {"start": first.strftime("%Y-%m-%d"), "loss_limit": settings.loss_limit, "last_day": None,
            "backfilled": first < days[-1], "books": {name: dict(book) for name in PORTFOLIOS}, "history": []}


def update_ledger(ledger: dict, schedule: dict[str, float], settings: OptimizerSettings,
                  closes: pd.DataFrame, universe: list[str], now: datetime | None = None,
                  progress=None) -> dict:
    """Process every final trading day since the last one recorded.

    Days are traded at ``settings.loss_limit``, the risk level in force today, so a
    change of level applies from the next day on and the record is never restarted.
    """
    ledger["loss_limit"] = settings.loss_limit
    ref = settings.cash_proxy if settings.cash_proxy in closes else settings.cash_ticker
    start = pd.Timestamp(ledger["start"])
    last = pd.Timestamp(ledger["last_day"]) if ledger["last_day"] else None
    days = [d for d in _final_days(closes, ref, now) if d >= start and (last is None or d > last)]
    if ledger["last_day"] is None and not days:
        # Started before today's close: open at the latest prices so the record has a day one.
        days = [d for d in closes[ref].dropna().index if d >= start][:1]
    books = {name: Book.from_json(ledger["books"][name]) for name in PORTFOLIOS}
    for n, day in enumerate(days, 1):
        if progress:
            progress(n, len(days))
        prices = _prices(closes, day)
        for book in books.values():
            book.settle()
        orders = {
            "plan": _trade_plan(books["plan"], day, closes, schedule, settings, universe, prices),
            "hold": _trade_hold(books["hold"], schedule, settings.cash_ticker, prices),
        }
        values = {name: round(books[name].value(prices), 2) for name in PORTFOLIOS}
        plan = books["plan"]
        cash_value = plan.shares.get(settings.cash_ticker, 0.0) * prices.get(settings.cash_ticker, 0.0)
        cash_value += plan.settled + plan.unsettled
        ledger["history"].append({
            "date": day.strftime("%Y-%m-%d"),
            **values,
            "loss_limit": settings.loss_limit,
            "risk_model": settings.risk_model,
            "target_growth": settings.target_growth,
            "invested": round(1 - cash_value / values["plan"], 4) if values["plan"] else 0.0,
            "orders": orders,
        })
        ledger["last_day"] = day.strftime("%Y-%m-%d")
        ledger["books"] = {name: books[name].to_json() for name in PORTFOLIOS}
    return ledger


def read_ledger(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def archive_ledger(path: Path) -> Path | None:
    """Move the ledger aside instead of deleting it, so a reset never loses the record."""
    if not path.exists():
        return None
    ledger = read_ledger(path)
    target = path.with_name(f"{path.stem}_{ledger['start']}_to_{ledger['last_day'] or ledger['start']}.json")
    n = 1
    while target.exists():
        n += 1
        target = target.with_name(f"{path.stem}_{ledger['start']}_to_{ledger['last_day'] or ledger['start']}_{n}.json")
    path.replace(target)
    return target


def write_ledger(path: Path, ledger: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    tmp.replace(path)
