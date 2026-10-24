"""Recurring investments ("auto-invest") that keep new money going into the target mix.

Robinhood's IRA recurring investments (checked 2026-10-24 on its "Retirement
investing" help page) buy a fixed dollar amount of one stock or ETF **every market
day, every week, every 2 weeks or every month**, possibly in fractional shares, paid
from a chosen payment method with an optional backup bank account. Money from an
external bank account counts toward the year's IRA contribution limit. The page
states no minimum order. **A $1 minimum per order is assumed here** (Robinhood's
fractional-share minimum elsewhere); check the order screen.

The recommendation splits your daily budget across the plan's target mix, cash fund
included, in proportion to the target weights. A fund whose share is too small for
a $1 order every day is bought weekly, every 2 weeks or monthly, at the same pace.
Recurring buys only buy: they cannot sell or rebalance, and they follow the target
of the plan they were set from. Set them again when the plan's mix changes (a new
risk level, say).

Nothing here places or changes anything at Robinhood: you set the recurring
investments yourself in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Robinhood's choices, and how many trading sessions each one spans.
FREQUENCIES = (("every market day", 1), ("every week", 5), ("every 2 weeks", 10), ("every month", 21))
SESSIONS_PER_YEAR = 252
IRA_LIMIT_2026 = {"under 50": 7500.0, "50 or older": 8600.0}


@dataclass
class Recurring:
    ticker: str
    name: str
    amount: float          # dollars per order
    frequency: str
    per_day: float         # the same pace, per trading session
    weight: float          # share of the budget


@dataclass
class AutoInvest:
    budget_per_day: float
    schedule: list[Recurring]
    changes: list[dict]                 # against your current recurring buys
    per_year: float
    notes: list[str] = field(default_factory=list)


def _frequency(per_day: float, min_order: float) -> tuple[str, float]:
    for name, sessions in FREQUENCIES:
        if per_day * sessions >= min_order - 1e-9:
            return name, round(per_day * sessions, 2)
    name, sessions = FREQUENCIES[-1]
    return name, round(max(per_day * sessions, min_order), 2)


def recommend(plan, budget_per_day: float, current: dict[str, float], min_order: float = 1.0) -> AutoInvest:
    """Recurring buys for ``budget_per_day`` dollars a session, split by the plan's target mix.

    ``current`` is your present schedule in dollars per session, by ticker.
    """
    weights = {r.ticker: r.target for r in plan.rows if r.target > 1e-4}
    if plan.cash.target > 1e-4:
        weights[plan.cash.ticker] = plan.cash.target
    names = {r.ticker: r.name for r in plan.rows} | {plan.cash.ticker: plan.cash.name or "cash"}
    total = sum(weights.values()) or 1.0
    schedule = []
    for ticker, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        per_day = budget_per_day * w / total
        if per_day * FREQUENCIES[-1][1] < min_order:
            continue   # too small even monthly; its share goes to the others below
        frequency, amount = _frequency(per_day, min_order)
        schedule.append(Recurring(ticker, names.get(ticker, ""), amount, frequency, round(per_day, 4), w / total))
    # Rounding and dropped crumbs: rescale so the pace adds up to the budget.
    kept = sum(r.per_day for r in schedule)
    if schedule and kept > 0:
        for r in schedule:
            r.per_day = round(r.per_day * budget_per_day / kept, 4)
            sessions = dict(FREQUENCIES)[r.frequency]
            r.amount = round(r.per_day * sessions, 2)
    changes = []
    for ticker in sorted(set(current) | {r.ticker for r in schedule}):
        now = current.get(ticker, 0.0)
        new = next((r for r in schedule if r.ticker == ticker), None)
        if new is None and now > 0:
            changes.append({"ticker": ticker, "action": "cancel", "now_per_day": now})
        elif new is not None and now <= 0:
            changes.append({"ticker": ticker, "action": "add", "amount": new.amount, "frequency": new.frequency})
        elif new is not None and abs(new.per_day - now) > 0.005:
            changes.append({"ticker": ticker, "action": "change", "now_per_day": now, "amount": new.amount,
                            "frequency": new.frequency})
    per_year = budget_per_day * SESSIONS_PER_YEAR
    notes = [f"About ${per_year:,.0f} a year. Money from your bank counts toward the IRA limit: "
             + ", ".join(f"${v:,.0f} ({k})" for k, v in IRA_LIMIT_2026.items()) + " for 2026."]
    if per_year > IRA_LIMIT_2026["under 50"]:
        notes.append("This pace exceeds the under-50 limit; Robinhood may refuse deposits once the year's limit is reached.")
    if plan.days_to_target:
        notes.append(f"Until the mix reaches its target (about {plan.days_to_target:.0f} trading days at this pace, "
                     f"or sooner with today's trades), the daily orders on this page put new money where the gaps are; "
                     f"the schedule above is for after that.")
    notes.append("Recurring buys only buy. Rebalancing and sales stay with the daily orders. Update the schedule "
                 "when the plan's mix changes, for example after changing the risk level.")
    notes.append(f"A ${min_order:.0f} minimum per order is assumed; Robinhood's help page states none for IRA recurring buys.")
    return AutoInvest(round(budget_per_day, 2), schedule, changes, round(per_year, 2), notes)
