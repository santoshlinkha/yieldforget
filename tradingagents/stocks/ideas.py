"""Reviewable stock recommendations from a saved run, within the account's rules, and their record.

From a run's forecasts at one horizon (20 sessions by default), for one sleeve
(Trading by default; ``stock_rules`` in settings):

- **Buy candidate**: expected return net of the estimated round-trip cost above zero,
  return evidence ``backed`` (the study's final test confirmed skill), probability of a
  large loss within the cap, and no data-quality flag. At most ``max_names``, best
  net expected return first.
- **Watch**: positive net expectation without backed evidence, or a flag: listed, not sized.
- **Avoid**: a negative net expectation. **No short orders**, ever: a held stock with a
  negative expectation is marked "consider reducing", sized at most to what is held.

Sizing: equal dollars across buy candidates, each at most ``max_position_share`` of the
sleeve's value and together within its **settled** cash (a Roth IRA is a cash
account: unsettled proceeds cannot buy). Fractional shares are fine; orders under
``min_order`` are dropped. A position already held is topped up to its target only.

No day trading: a buy executes at the earliest on the session after the cutoff, and
the plan holds it through the forecast horizon; selling it on the day it was bought
would be a day trade and is never suggested.

Exposure: before anything is suggested, the whole IRA (stocks and funds) is priced
and its daily returns over the last year compared with each candidate: correlation,
beta to the IRA and to SPY, and how much the IRA's beta to SPY moves if the candidate
is added at its size. Fund holdings are not looked through to their sectors (no
holdings data), and the page says so.

Recommendations and decisions are separate records (``stock_ideas``,
``stock_idea_decisions``); orders and fills stay in the ledger. ``prospective`` compares
each saved recommendation's buys, as if executed at the next close, with putting the
same dollars on the same dates into SPY (the frozen static policy), once the horizon
has passed: a Day-1 record that is never regenerated from today's models.
"""

from __future__ import annotations

import json
import uuid
from datetime import date

import numpy as np
import pandas as pd


def _prices(cache_dir, tickers):
    from tradingagents.research.prices import bars

    return bars(list(dict.fromkeys(tickers)), cache_dir)["Close"]


def exposure(store, cache_dir, candidates: list[str], sizes: dict[str, float], as_of: date) -> dict:
    """How each candidate would sit with what the whole IRA already holds."""
    state = store.ledger.state("combined", as_of)
    held = {t: q for t, q in state.positions.items() if q}
    close = _prices(cache_dir, [*held, *candidates, "SPY"]).iloc[-253:]
    last = close.ffill().iloc[-1]
    values = {t: q * float(last.get(t, np.nan)) for t, q in held.items()}
    values = {t: v for t, v in values.items() if np.isfinite(v)}
    total = sum(values.values()) + state.cash.total
    r = np.log(close).diff().iloc[1:]
    out = {"ira_value": round(total, 2), "holdings": len(values), "cash_share": state.cash.total / total if total else None}
    if values:
        w = pd.Series(values) / total
        port = (r[w.index].fillna(0.0) * w).sum(axis=1)
        spy = r["SPY"]
        beta_ira = float(port.cov(spy) / spy.var())
        out["ira_beta_spy"] = beta_ira
        out["candidates"] = {}
        for t in candidates:
            if t not in r:
                continue
            c = r[t]
            share = sizes.get(t, 0.0) / total if total else 0.0
            b_spy = float(c.cov(spy) / spy.var())
            out["candidates"][t] = {"corr_with_ira": float(c.corr(port)), "beta_to_ira": float(c.cov(port) / port.var()),
                                    "beta_spy": b_spy, "ira_beta_after": beta_ira + share * (b_spy - beta_ira),
                                    "share_of_ira": share}
    out["note"] = "Funds are not looked through to their holdings or sectors; correlations use the last year of daily returns."
    return out


def make(con, store, cache_dir, run_id: str | None = None, horizon: int = 20, as_of: date | None = None) -> dict:
    """Build and save the recommendation for a run; see the module docstring."""
    from tradingagents.ledger.db import now_utc, transaction

    from .pipeline import load_run

    rules = store.get("global", "stock_rules")
    as_of = as_of or date.today()
    run = load_run(con, run_id)
    if run is None:
        raise ValueError("no stock run yet: run a batch first")
    sleeve = rules["sleeve"]
    state = store.ledger.state(sleeve, as_of)
    rows = [f for f in run["forecasts"] if f["horizon"] == horizon]
    tickers = [f["ticker"] for f in rows]
    # Prices as of the decision date, never later ones; a price over a week old is flagged below.
    known = _prices(cache_dir, [*tickers, *state.positions]).ffill().loc[:pd.Timestamp(as_of)]
    last = known.iloc[-1] if len(known) else pd.Series(dtype=float)
    price_day = known.index[-1].date() if len(known) else None
    held_value = sum(q * float(last.get(t, 0) or 0) for t, q in state.positions.items())
    sleeve_value = held_value + state.cash.total
    available = state.cash.available
    lines, buys = [], []
    for f in rows:
        fc = f.get("forecast") or {}
        t = f["ticker"]
        held = state.positions.get(t, 0.0)
        line = {"ticker": t, "name": f.get("name"), "held": held, "price": float(last.get(t, np.nan)),
                "net": fc.get("net"), "expected": fc.get("mean"), "cost_rt": fc.get("cost_rt"),
                "p_large_loss": fc.get("p_large_loss"), "evidence": (f.get("evidence_status") or {}).get("return"),
                "reasons": list(f.get("reasons") or [])}
        flagged = f["status"] != "ok" or any(r.startswith(("stale", "insufficient", "no trading")) for r in line["reasons"])
        if price_day is None or (as_of - price_day).days > 7:
            flagged = True
            line["reasons"].append(f"price as of {price_day}: too old for {as_of}")
        risk = fc.get("p_large_loss")
        risk_ok = isinstance(risk, (int, float)) and np.isfinite(risk) and 0 <= risk <= 1
        if f["status"] != "ok" or fc.get("net") is None:
            line["action"] = "abstain"
        elif not risk_ok and fc["net"] > 0:
            # A missing or invalid risk estimate blocks approval; it is never read as zero risk.
            line["action"] = "watch (risk estimate missing or invalid: approval blocked)"
        elif fc["net"] <= 0:
            line["action"] = "consider reducing (no short orders)" if held > 0 else "avoid (no short orders)"
        elif line["evidence"] == "backed" and risk <= rules["max_large_loss_prob"] and not flagged:
            line["action"] = "buy candidate"
            buys.append(line)
        else:
            why = "return evidence not backed by the final test" if line["evidence"] != "backed" else \
                "large-loss probability above the cap" if risk > rules["max_large_loss_prob"] \
                else "data-quality flag"
            line["action"] = f"watch ({why})"
        lines.append(line)
    # The name limit counts what Trading already holds, not only new buys.
    held_names = {t for t, q in state.positions.items() if q > 1e-9}
    new_slots = max(0, rules["max_names"] - len(held_names))
    ranked = sorted(buys, key=lambda x: -x["net"])
    buys = [b for b in ranked if b["ticker"] in held_names] + [b for b in ranked if b["ticker"] not in held_names][:new_slots]
    for b in ranked:
        if b not in buys:
            b["action"] = f"watch (name limit: {rules['max_names']} names including holdings)"
    sizes = {}
    if buys:
        per = min(rules["max_position_share"] * sleeve_value, available / len(buys))
        for line in buys:
            have = line["held"] * (line["price"] if np.isfinite(line["price"]) else 0)
            dollars = max(0.0, per - have)
            if dollars < rules["min_order"] or not np.isfinite(line["price"]):
                line["order"] = None
                line["action"] += f" (no order: under ${rules['min_order']:.0f} or already at target)"
                continue
            line["order"] = {"side": "buy", "dollars": round(dollars, 2), "shares": round(dollars / line["price"], 6)}
            sizes[line["ticker"]] = dollars
    from tradingagents.ledger.sessions import next_session

    execute = next_session(date.fromisoformat(run["info_cutoff"]))
    payload = {"run_id": run["id"], "info_cutoff": run["info_cutoff"], "earliest_execution": str(execute),
               "hold_until": str(next_session(execute, horizon)), "sleeve": sleeve, "sleeve_value": round(sleeve_value, 2),
               "settled_cash": round(available, 2), "rules": rules, "lines": lines,
               "exposure": exposure(store, cache_dir, list(sizes), sizes, as_of),
               "note": ("Research, not orders: nothing is sent to the broker. Buys need settled cash; hold through the "
                        "horizon (selling the day you buy would be a day trade). Negative forecasts never become short orders.")}
    idea_id = uuid.uuid4().hex
    with transaction(con):
        con.execute("INSERT INTO stock_ideas VALUES (?, ?, ?, ?, ?, ?)",
                    (idea_id, run["id"], now_utc(), sleeve, horizon, json.dumps(payload, default=str)))
    return {"id": idea_id, **payload}


def decide(con, idea_id: str, ticker: str, decision: str, note: str | None = None) -> None:
    from tradingagents.ledger.db import now_utc, transaction

    if decision not in ("accepted", "rejected", "deferred"):
        raise ValueError("a decision is accepted, rejected or deferred")
    if not con.execute("SELECT 1 FROM stock_ideas WHERE id = ?", (idea_id,)).fetchone():
        raise KeyError(idea_id)
    with transaction(con):
        con.execute("INSERT INTO stock_idea_decisions (idea_id, ticker, decided_at, decision, note) VALUES (?, ?, ?, ?, ?)",
                    (idea_id, ticker, now_utc(), decision, note))


def prospective(con, cache_dir) -> list[dict]:
    """Each saved recommendation's buys against SPY with the same dollars and dates, once matured."""
    from tradingagents.ledger.sessions import next_session

    from .data import CASH

    out = []
    for row in con.execute("SELECT id, created_at, horizon, payload_json FROM stock_ideas ORDER BY created_at"):
        p = json.loads(row["payload_json"])
        orders = {line["ticker"]: line["order"]["dollars"] for line in p["lines"] if line.get("order")}
        if not orders:
            continue
        entry = date.fromisoformat(p["earliest_execution"])
        exit_ = next_session(entry, row["horizon"])
        close = _prices(cache_dir, [*orders, "SPY", CASH])
        if pd.Timestamp(exit_) > close.index[-1]:
            out.append({"idea": row["id"], "status": "open", "matures": str(exit_)})
            continue
        a, b = close.loc[pd.Timestamp(entry)], close.loc[pd.Timestamp(exit_)]
        strategy = sum(d * (b[t] / a[t] - 1) for t, d in orders.items())
        spy = sum(orders.values()) * (b["SPY"] / a["SPY"] - 1)
        out.append({"idea": row["id"], "status": "matured", "entry": str(entry), "exit": str(exit_),
                    "invested": round(sum(orders.values()), 2), "strategy_pnl": round(strategy, 2),
                    "spy_pnl": round(spy, 2), "difference": round(strategy - spy, 2),
                    "note": "simulated at the closes of the entry and exit sessions; no costs beyond the forecast's estimate"})
    return out
