"""The prospective shadow account: the frozen experimental decisions, followed day by day with the same
position engine, against putting the same capital into SPY on the same day (cash-flow matched).

It starts with the Trading capital in force on its first day (frozen in its first snapshot; later
setting changes do not touch it). Each update replays the sessions since the last snapshot:
settlement, dividends, close-confirmed exits decided at the previous close, and the PAPER ENTER
buys of the latest decision at their execution close, sized by the same rules on the shadow's own
cash. Every update is a new snapshot; nothing is rewritten, and no decision is ever regenerated.
This record, not the historical simulation, is what can confirm a policy.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tradingagents.research.walkforward import block_bootstrap

from .positions import Book, Lot, executable_prices

# Version 2 (2026-10-25): an audit found that re-running a decision for the same cutoff appended its orders
# again (39 pending orders, $9,350 against $1,000, 25 duplicates, before any had executed) and that a lot
# bought on the latest session got that same session as its deadline. Version-1 snapshots stay as they are;
# the first update under version 2 starts a new account at that update's cutoff, with no trades carried over.
SHADOW_VERSION = 2
RESTART_REASON = ("version 2: orders are frozen once per cutoff and pending orders count against the name "
                  "limit; deadlines come from the exchange calendar. The version-1 account had executed no trade.")


def _load(con):
    row = con.execute("SELECT state_json FROM shadow_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(row["state_json"]) if row else None


def _book(state, sessions) -> Book:
    book = Book(sessions, cash=state["cash"])
    book.settling = {pd.Timestamp(k): v for k, v in state["settling"].items()}
    book.lots = [Lot(**{**lot, "entry_date": pd.Timestamp(lot["entry_date"]), "deadline": pd.Timestamp(lot["deadline"])})
                 for lot in state["lots"]]
    book.costs = state.get("costs", 0.0)
    return book


def _dump(book: Book) -> dict:
    return {"cash": book.cash, "settling": {str(k.date()): v for k, v in book.settling.items()},
            "lots": [{**lot.__dict__, "entry_date": str(lot.entry_date.date()), "deadline": str(lot.deadline.date())}
                     for lot in book.lots], "costs": book.costs}


def advance(store, data, cutoff: pd.Timestamp, decision_id: str, entry_lines: list[dict], policy, capital_snapshot: dict,
            rules: dict | None = None, prices=None) -> dict:
    from tradingagents.ledger.db import now_utc, transaction

    from .daily import rules as account_rules

    r = rules or account_rules(store)
    stocks = list(data.stocks)
    # ``prices``: (executable prices, dividends) from the full history, cut at the cutoff by the caller.
    px, div = prices if prices is not None else executable_prices(data.close[stocks], data.actions)
    sessions = px.index
    spy = data.close["SPY"]
    state = _load(store.con)
    if state is None or state.get("version", 1) < SHADOW_VERSION:
        previous = state
        capital = float(capital_snapshot["trading_capital_usd"])
        state = {"version": SHADOW_VERSION, "inception": str(cutoff.date()), "capital": capital,
                 "capital_version": capital_snapshot["version"], "cash": capital, "settling": {}, "lots": [],
                 "costs": 0.0, "pending": [], "last": str(cutoff.date()), "spy_units": None, "closed": [],
                 "equity": [], "policy": policy.version(), "orders_frozen_for": [],
                 **({"restarted": {"reason": RESTART_REASON, "previous_inception": previous.get("inception"),
                                   "previous_trades": len(previous.get("closed", [])),
                                   "previous_lots": len(previous.get("lots", []))}} if previous else {})}
    book = _book(state, sessions)
    pending = state.get("pending", [])
    due = state.get("due", [])
    days = sessions[(sessions > pd.Timestamp(state["last"])) & (sessions <= cutoff)]
    closed = state.get("closed", [])
    for day in days:
        book.settle(day)
        book.dividends(day, div.loc[day])
        prices = px.loc[day]
        if state.get("spy_units") is None and pending:
            state["spy_units"] = state["capital"] / float(spy.loc[day])   # SPY bought with the same capital, same day
        for order in [o for o in pending if pd.Timestamp(o["execute"]) <= day]:
            p = float(prices.get(order["ticker"], np.nan))
            book.buy(day, order["ticker"], min(order["dollars"], book.cash), p, policy, order["sigma"], order["cost_rt"],
                     source=order["decision"])
        pending = [o for o in pending if pd.Timestamp(o["execute"]) > day]
        for item in due:
            lot = next((x for x in book.lots if x.ticker == item["ticker"] and str(x.entry_date.date()) == item["entry"]), None)
            p = float(prices.get(item["ticker"], np.nan))
            if lot is not None and np.isfinite(p) and day > lot.entry_date:
                gross = p / lot.entry_px - 1
                spy_ret = float(spy.loc[day] / spy.loc[lot.entry_date] - 1)
                book.sell(day, lot, p, item["reason"])
                closed.append({"ticker": lot.ticker, "entry": str(lot.entry_date.date()), "exit": str(day.date()),
                               "reason": item["reason"], "net": gross - lot.cost_rt, "spy": spy_ret})
        due = [{"ticker": lot.ticker, "entry": str(lot.entry_date.date()), "reason": why}
               for lot, why in book.exits_due(day, prices)]
        value = book.value(px.ffill().loc[day])     # last known price for a lot without one today
        bench = state["spy_units"] * float(spy.loc[day]) if state.get("spy_units") else state["capital"]
        state.setdefault("equity", []).append([str(day.date()), value, bench])
    value_now = book.value(px.loc[cutoff]) if cutoff in px.index else book.cash
    # Orders are frozen once per cutoff: a decision re-run for a cutoff already taken adds nothing, so the
    # record follows the first decision made for each day. Pending orders count like held names.
    frozen = state.setdefault("orders_frozen_for", [])
    take_orders = str(cutoff.date()) not in frozen
    held = {lot.ticker for lot in book.lots} | {o["ticker"] for o in pending}
    slots = max(0, r["max_names"] - len(held))
    reserved = sum(o["dollars"] for o in pending)
    for line in entry_lines if take_orders else []:
        if line.get("paper_action") != "PAPER ENTER" or slots <= 0 or line["ticker"] in held:
            continue
        dollars = min(r["max_position_share"] * value_now, max(book.cash - reserved, 0.0) / max(slots, 1))
        if dollars >= r["min_order"]:
            reserved += dollars
            held.add(line["ticker"])
            pending.append({"ticker": line["ticker"], "dollars": dollars, "execute": line["execute"],
                            "cutoff": str(cutoff.date()),
                            "sigma": float(np.nan_to_num(line.get("sigma", 0.02), nan=0.02)) if "sigma" in line else 0.02,
                            "cost_rt": float(line.get("cost_rt") or 0.004), "decision": decision_id})
            slots -= 1
    if take_orders:
        frozen.append(str(cutoff.date()))
    excess = [c["net"] - c["spy"] for c in closed]
    eq = state.get("equity", [])
    # The account against SPY, day by day: the mean daily excess and its block-bootstrap interval (20-session
    # blocks, the holding horizon), so a few lucky trades cannot validate the model on their own.
    daily_excess = np.diff(np.log([e[1] for e in eq])) - np.diff(np.log([e[2] for e in eq])) if len(eq) > 1 else []
    ci = block_bootstrap(np.asarray(daily_excess), block=20) if len(daily_excess) >= 40 else (None, None)
    record = {"round_trips": len(closed), "excess_vs_spy": float(np.mean(excess)) if excess else None,
              "win_rate": float(np.mean([c["net"] > 0 for c in closed])) if closed else None,
              "value": eq[-1][1] if eq else state["capital"], "spy_value": eq[-1][2] if eq else state["capital"],
              "since": state["inception"], "capital": state["capital"], "version": state.get("version", 1),
              "policy": state.get("policy"),
              "account_excess": (eq[-1][1] - eq[-1][2]) / state["capital"] if eq else None,
              "account_excess_ci": [None if v is None or not np.isfinite(v) else float(v) for v in ci],
              "account_days": len(daily_excess)}
    new_state = {**state, **_dump(book), "pending": pending, "due": due, "closed": closed, "last": str(cutoff.date()),
                 "record": record}
    with transaction(store.con):
        store.con.execute("INSERT INTO shadow_snapshots (created_at, as_of, decision_id, state_json) VALUES (?, ?, ?, ?)",
                          (now_utc(), str(cutoff.date()), decision_id, json.dumps(new_state, default=str)))
    return record
