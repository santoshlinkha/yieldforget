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

from .positions import Book, Lot, executable_prices


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
            rules: dict | None = None) -> dict:
    from tradingagents.ledger.db import now_utc, transaction

    from .daily import rules as account_rules

    r = rules or account_rules(store)
    stocks = list(data.stocks)
    px, div = executable_prices(data.close[stocks], data.actions)
    sessions = px.index
    spy = data.close["SPY"]
    state = _load(store.con)
    if state is None:
        capital = float(capital_snapshot["trading_capital_usd"])
        state = {"inception": str(cutoff.date()), "capital": capital, "capital_version": capital_snapshot["version"],
                 "cash": capital, "settling": {}, "lots": [], "costs": 0.0, "pending": [], "last": str(cutoff.date()),
                 "spy_units": None, "closed": [], "equity": []}
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
        value = book.value(prices)
        bench = state["spy_units"] * float(spy.loc[day]) if state.get("spy_units") else state["capital"]
        state.setdefault("equity", []).append([str(day.date()), value, bench])
    value_now = book.value(px.loc[cutoff]) if cutoff in px.index else book.cash
    held = {lot.ticker for lot in book.lots}
    slots = max(0, r["max_names"] - len(held))
    for line in entry_lines:
        if line.get("paper_action") != "PAPER ENTER" or slots <= 0 or line["ticker"] in held:
            continue
        dollars = min(r["max_position_share"] * value_now, book.cash / max(slots, 1))
        if dollars >= r["min_order"]:
            pending.append({"ticker": line["ticker"], "dollars": dollars, "execute": line["execute"],
                            "sigma": float(np.nan_to_num(line.get("sigma", 0.02), nan=0.02)) if "sigma" in line else 0.02,
                            "cost_rt": float(line.get("cost_rt") or 0.004), "decision": decision_id})
            slots -= 1
    excess = [c["net"] - c["spy"] for c in closed]
    eq = state.get("equity", [])
    record = {"round_trips": len(closed), "excess_vs_spy": float(np.mean(excess)) if excess else None,
              "win_rate": float(np.mean([c["net"] > 0 for c in closed])) if closed else None,
              "value": eq[-1][1] if eq else state["capital"], "spy_value": eq[-1][2] if eq else state["capital"],
              "since": state["inception"], "capital": state["capital"]}
    new_state = {**state, **_dump(book), "pending": pending, "due": due, "closed": closed, "last": str(cutoff.date()),
                 "record": record}
    with transaction(store.con):
        store.con.execute("INSERT INTO shadow_snapshots (created_at, as_of, decision_id, state_json) VALUES (?, ?, ?, ?)",
                          (now_utc(), str(cutoff.date()), decision_id, json.dumps(new_state, default=str)))
    return record
