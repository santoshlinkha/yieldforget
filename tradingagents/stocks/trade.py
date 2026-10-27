"""Today's trades in plain terms: the best stock to buy with the trading budget, at what price, and what to sell.

Everything comes from the latest saved daily decision (``daily.run``), the internal model's own
order, never from an AI rating:

- **Best buy**: the highest-ranked stock that passes the model's gates (P(profit after costs),
  expected net P&L, P(large loss)) and is not already held. Its **buy price** is a limit at the
  last close: the model's entry is the next session's close, and a limit there keeps a gap up from
  filling it. When the stock has a saved signal forecast, a patient limit at its median expected
  dip (about an even chance the price trades there within 20 sessions; a touch is not a fill) is offered beside it. The **exits** are
  the decision study's barrier policy: a stop one horizon-volatility below the buy price, a
  target one above, and a 20-session deadline, all close-confirmed. **Shares** are the budget
  over the buy price (fractional).
- **Runners-up**: the next few that pass, in case the first is not wanted.
- **Sell**: each stock held, with its action (EXIT at the next close when it has closed through
  its stop or target or reached its deadline, REDUCE above the size cap, else HOLD) and the
  prices at which it would be sold.

In experimental mode (the model has not yet passed its decision study and prospective record),
the buy is labelled a paper suggestion: the model's best idea, not a validated signal.
"""

from __future__ import annotations

import math

RUNNERS_UP = 4


def _exits(entry: float, sigma_daily: float, horizon: int) -> dict:
    s = (sigma_daily if sigma_daily and math.isfinite(sigma_daily) else 0.02) * math.sqrt(horizon)
    return {"stop": round(entry * (1 + math.expm1(-s)), 2), "target": round(entry * (1 + math.expm1(s)), 2),
            "stop_pct": math.expm1(-s), "target_pct": math.expm1(s), "sigma_h": s}


def _candidate(e: dict, budget: float, horizon: int, patient: dict | None) -> dict:
    price = float(e["price"])
    out = {"ticker": e["ticker"], "last_close": round(price, 2), "buy_limit": round(price, 2),
           "shares": round(budget / price, 6) if budget > 0 and price > 0 else 0.0, "dollars": round(budget, 2),
           "expected_net": e.get("expected_net"), "p_win": e.get("p_win"), "p_risk": e.get("p_risk"),
           "score_pct": e.get("score_pct"), "expected_vs_spy": e.get("expected_excess_vs_spy"),
           "tied_with": e.get("tied_with", 0),
           "horizon": horizon, **_exits(price, e.get("sigma"), horizon)}
    if patient:
        out["patient"] = patient
    return out


def plan(decision: dict | None, budget: float, patient_limits: dict[str, dict] | None = None) -> dict:
    """The best buy, runners-up and sells from a saved decision, sized on ``budget`` dollars."""
    if not decision:
        return {"ready": False, "note": "No daily decision yet: run it (Decisions, or the button here) after the close."}
    horizon = int(decision.get("horizon") or 20)
    patient_limits = patient_limits or {}
    held = {lot["ticker"] for lot in decision.get("lots", [])}
    # Passing the model's own gates = it has a paper action or an action ENTER (account limits aside).
    passing = [e for e in decision.get("entries", [])
               if (e.get("paper_action") or e.get("action") == "ENTER") and e["ticker"] not in held
               and e.get("price") and math.isfinite(float(e["price"]))]
    picks = [_candidate(e, budget, horizon, patient_limits.get(e["ticker"])) for e in passing[:1 + RUNNERS_UP]]
    best = picks[0] if picks else None
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
    if best and budget < 1:
        notes.append("The trading budget is empty: the buy waits for SGOV's next monthly interest and the match.")
    if not best:
        notes.append("No stock passes the model's gates today: waiting is the answer.")
    return {"ready": True, "decision_id": decision.get("id"), "data_to": decision.get("info_cutoff"),
            "executes": decision.get("execution_date"), "mode": mode, "budget": round(budget, 2),
            "best": best, "runners_up": picks[1:], "sells": sells, "notes": notes}
