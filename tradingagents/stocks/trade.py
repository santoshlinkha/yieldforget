"""Today's trades in plain terms: the best stock to buy with the trading budget, at what price, and what to sell.

Everything comes from the latest saved daily decision (``daily.run``), the internal model's own
order, never from an AI rating:

- **Best buy**: the highest-ranked stock that passes the model's gates (P(profit after costs),
  expected net P&L, P(large loss)) and is not already held. Its **entry** is the policy the model was
  trained and tested on (``positions.execution_spec``): a buy at the next session's close, whatever
  that close is. There is no buy limit: a limit at the last close would be a different policy (it
  skips the stocks that rise, and fills intraday at other prices), never tested. The last close is
  shown as a dated reference. The **exits** are the barrier policy's, set from the actual fill: a
  stop one horizon-volatility below it, a target one above, and a 20-session deadline, all
  close-confirmed; dollar levels from the last close illustrate the percentages. When the stock
  has a saved signal forecast, a patient limit at its median expected dip is shown as an
  **untested policy** (its chance is of the price trading there, not of a fill).
- **Runners-up**: the next few that pass, in case the first is not wanted.
- **Sell**: each stock held, with its action (EXIT at the next close when it has closed through
  its stop or target or reached its deadline, REDUCE above the size cap, else HOLD) and the
  prices at which it would be sold.

In experimental mode (the model has not yet passed its decision study and prospective record),
the buy is labelled a paper suggestion: the model's best idea, not a validated signal.
"""

from __future__ import annotations

import math

from tradingagents.ledger.store import GLOBAL_DEFAULTS

from .sizing import position_size

RUNNERS_UP = 4
STOCK_RULES = GLOBAL_DEFAULTS["stock_rules"]


def _exits(entry: float, sigma_daily: float, horizon: int) -> dict:
    s = (sigma_daily if sigma_daily and math.isfinite(sigma_daily) else 0.02) * math.sqrt(horizon)
    return {"stop": round(entry * (1 + math.expm1(-s)), 2), "target": round(entry * (1 + math.expm1(s)), 2),
            "stop_pct": math.expm1(-s), "target_pct": math.expm1(s), "sigma_h": s}


def _candidate(e: dict, size: dict, horizon: int, patient: dict | None, data_to: str | None) -> dict:
    # The policy buys at the next session's close, whatever it is; the last close is a reference, not an order
    # price, and the stop and target are set from the actual fill. Dollar levels from the last close are shown
    # as an illustration of the percentages. ``size`` is the allowed size (``_size``), never the whole budget.
    price = float(e["price"])
    dollars = size["allowed_usd"]
    shares = round(dollars / price, 6) if dollars > 0 and price > 0 else 0.0
    out = {"ticker": e["ticker"], "last_close": round(price, 2), "buy_limit": round(price, 2), "reference_date": data_to,
           "entry": {"basis": "next_close", "session": e.get("execute"),
                     "text": f"at the close of {e.get('execute') or 'the next session'}, whatever it is"},
           "shares": shares, "shares_at_reference": shares, "dollars": round(dollars, 2), "size": size,
           "expected_net": e.get("expected_net"), "p_win": e.get("p_win"), "p_risk": e.get("p_risk"),
           "score_pct": e.get("score_pct"), "expected_vs_spy": e.get("expected_excess_vs_spy"),
           "tied_with": e.get("tied_with", 0), "score_tied_with": e.get("score_tied_with", 0),
           "horizon": horizon, **_exits(price, e.get("sigma"), horizon)}
    if patient:
        # A limit below the price is a different policy from the one the model was tested on: its chance is of the
        # price trading there (the signal model, itself untested), not of a fill, and returns after such fills
        # were never evaluated.
        out["patient"] = {**patient, "status": "untested policy"}
    return out


def _size(e: dict, rules: dict, sleeve_value: float | None, left: float, slots: int, budget: float) -> dict:
    """The allowed size of one pick: the decision's sizing rule (``sizing.position_size``) on what is left of
    today's budget, and never more than the saved decision allowed. ``budget_usd`` is the whole budget, shown
    only as context: it is not an order size."""
    allowed = position_size(rules, sleeve_value, left, slots)
    basis = "the decision's sizing rule on today's budget"
    saved = e.get("dollars")
    if saved is not None and math.isfinite(float(saved)):
        if allowed >= float(saved):
            allowed, basis = float(saved), "the saved decision's size"
        else:
            basis = "the saved decision's size, reduced to fit today's budget"
    min_order = float(rules.get("min_order") or 0.0)
    if 0 < allowed < min_order:
        allowed, basis = 0.0, f"below the ${min_order:,.2f} minimum order"
    if slots <= 0:
        basis = f"no free name slot ({rules['max_names']} names including holdings)"
    return {"allowed_usd": round(allowed, 2), "basis": basis, "budget_usd": round(float(budget or 0.0), 2),
            "cap_usd": round(rules["max_position_share"] * sleeve_value, 2) if sleeve_value is not None else None,
            "slots_left": slots}


def plan(decision: dict | None, budget: float, patient_limits: dict[str, dict] | None = None) -> dict:
    """The best buy, runners-up and sells from a saved decision. Each pick's size is what the decision's sizing
    rule allows within ``budget`` (today's trading budget), never more than the decision itself allowed."""
    if not decision:
        return {"ready": False, "note": "No daily decision yet: run it (Decisions, or the button here) after the close."}
    horizon = int(decision.get("horizon") or 20)
    patient_limits = patient_limits or {}
    held = {lot["ticker"] for lot in decision.get("lots", [])}
    # Passing the model's own gates = it has a paper action or an action ENTER (account limits aside).
    passing = [e for e in decision.get("entries", [])
               if (e.get("paper_action") or e.get("action") == "ENTER") and e["ticker"] not in held
               and e.get("price") and math.isfinite(float(e["price"]))]
    rules = {**STOCK_RULES, **((decision.get("settings") or {}).get("rules") or {})}
    sleeve_value = (decision.get("account") or {}).get("sleeve_value")
    staying = {lot["ticker"] for lot in decision.get("lots", []) if lot.get("action") != "EXIT"}
    slots, left, picks = max(0, int(rules["max_names"]) - len(staying)), max(float(budget or 0.0), 0.0), []
    for e in passing[:1 + RUNNERS_UP]:
        size = _size(e, rules, sleeve_value, left, slots, budget)
        picks.append(_candidate(e, size, horizon, patient_limits.get(e["ticker"]), decision.get("info_cutoff")))
        if size["allowed_usd"] > 0:                  # each pick is sized on what the ones before it left
            left, slots = left - size["allowed_usd"], slots - 1
    best = picks[0] if picks else None
    # Nothing passes: the top of the ranking with what blocks each, rather than an empty page or a placeholder.
    blocked = [] if picks else [{"ticker": e["ticker"], "score_pct": e.get("score_pct"),
                                 "score_tied_with": e.get("score_tied_with", 0), "blocked": e.get("blocked", [])}
                                for e in decision.get("entries", [])[:1 + RUNNERS_UP] if e["ticker"] not in held]
    sells = []
    for lot in decision.get("lots", []):
        entry = lot.get("entry_px")
        row = {"ticker": lot["ticker"], "shares": lot.get("shares"), "action": lot.get("action"),
               "reasons": lot.get("reasons", []), "last_close": lot.get("price"), "entry": entry,
               "unrealized": lot.get("unrealized"), "deadline": lot.get("deadline"), "execute": lot.get("execute")}
        if entry and lot.get("profit_level") is not None:
            row["sell_target"] = round(entry * (1 + lot["profit_level"]), 2)
            row["sell_stop"] = round(entry * (1 + lot["risk_level"]), 2)
        sells.append(row)
    sells.sort(key=lambda r: {"EXIT": 0, "REDUCE": 1, "REVIEW": 2}.get(r["action"], 3))
    mode = decision.get("mode")
    notes = []
    if mode != "validated":
        notes.append("Experimental: the model has not yet beaten SPY in its decision study and the live record, so "
                     "the buy is its best idea on paper, not a validated signal. Size it as an experiment.")
    if not best:
        notes.append("No stock passes the model's gates today: waiting is the answer.")
    for n in dict.fromkeys(n for e in passing[:1 + RUNNERS_UP] for n in e.get("notes") or []):
        notes.append(n[0].upper() + n[1:] + ".")
    from .positions import POLICIES, execution_spec

    pol = POLICIES.get(((decision.get("model") or {}).get("policy") or {}).get("name") or "barrier_20", POLICIES["barrier_20"])
    return {"ready": True, "decision_id": decision.get("id"), "data_to": decision.get("info_cutoff"),
            "universe": decision.get("universe"),
            "executes": decision.get("execution_date"), "mode": mode, "budget": round(budget, 2),
            "best": best, "runners_up": picks[1:], "blocked_top": blocked, "sells": sells, "notes": notes,
            "ranker": (decision.get("model") or {}).get("ranker"),
            "execution": execution_spec(pol)}
