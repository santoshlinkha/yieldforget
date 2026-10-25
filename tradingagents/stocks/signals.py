"""Per-stock signals for the ticker page and the screener: the saved trend, breakout and crash
forecasts, turned into plain warnings, next to the daily decision's action for that stock.

Reads only what is saved (stock runs and the latest daily decision): nothing is fitted or fetched.
Every number comes from the fitted models; the warnings are fixed arithmetic on them:

- **Lift**: the model's probability divided by the base rate (how often the event happened across
  all stocks in training). 1.0 = an ordinary stock; the warnings use the lift so a common event
  (most stocks break out of a 20-day range sometime in 20 sessions) is not mistaken for a signal.
- **Peer rank**: where the probability stands among today's stocks (0 = lowest, 1 = highest). When
  the whole market's probability moves (a jittery week raises every stock's breakdown chance), the
  lift rises for everyone; requiring the top fifth of today's stocks keeps a warning about the stock.
- **Trend**: the class (up / down / sideways) whose probability is highest relative to its usual
  share, if at least 1.15x usual and in today's top fifth for that class; otherwise "no clear trend"
  ("unclear" is the most common outcome). Classes the model does not tell stocks apart on are skipped.
- **Breakout likely**: P(close above the recent range) at least 1.2x usual and in today's top fifth.
- **Breakdown risk**: P(close below the recent range) at least 1.2x usual and in today's top fifth.
- **Crash warning**: P(a large loss at the end of the horizon; terminal, not the worst point while held) at least 1.5x usual, at least 10%, and in
  today's top fifth.
- **Deep drawdown warning**: the same with at least 20%.
- **Can it tell stocks apart?** When a model gives nearly every stock the same probability (the
  middle 80% of today's stocks within 2 points), it has no stock-specific view and says so instead
  of warning: it is then only repeating the base rate.

A warning is only as good as the model behind it; each signal carries the evidence status recorded
for it (untested, failed, passed) and none of it was validated for trading.
"""

from __future__ import annotations

import json

import numpy as np

LIFT_BREAK = 1.2
LIFT_CRASH = 1.5
LIFT_TREND = 1.15
MIN_CRASH = 0.10
MIN_DEEP = 0.20
TOP_SHARE = 0.8          # peer rank a warning needs: the top fifth of today's stocks
FLAT_SPREAD = 0.02       # 10th-90th percentile spread below which a model does not tell stocks apart
MIN_PEERS = 50           # fewer stocks than this: no peer test, no flatness test
EVENTS = ("up_break", "down_break", "large_loss", "deep_drawdown")
TRENDS = (("trend_up", "up", 0), ("trend_down", "down", 1), ("trend_sideways", "sideways", 2))


def _lift(p, base):
    return p / base if p is not None and base else None


def peers(forecasts: list[dict]) -> dict:
    """Today's spread of each probability across stocks: sorted values, and whether the model tells stocks apart."""
    out = {}
    for k, col in [*((k, f"p_{k}") for k in EVENTS), *((n, k) for k, n, _ in TRENDS)]:
        v = np.sort([f[col] for f in forecasts if f.get(col) is not None])
        enough = len(v) >= MIN_PEERS
        spread = float(np.percentile(v, 90) - np.percentile(v, 10)) if enough else None
        out[k] = {"values": v if enough else None, "spread": spread,
                  "flat": bool(enough and spread < FLAT_SPREAD), "n": int(len(v))}
    return out


def _rank(p, peer):
    v = (peer or {}).get("values")
    return None if v is None or p is None else float(np.searchsorted(v, p, side="right") / len(v))


def describe(forecast: dict, base_rates: dict, peer: dict | None = None) -> dict:
    """Trend, lifts, peer ranks and warnings from one saved forecast's numbers (pure arithmetic)."""
    f, b, peer = forecast or {}, base_rates or {}, peer or {}
    mix = b.get("trend_mix") or []
    shares = [m / sum(mix) for m in mix] if mix and sum(mix) else []
    trend_lifts = {name: _lift(f.get(k), shares[i] if i < len(shares) else None) for k, name, i in TRENDS}
    # Only a class the model tells stocks apart on, high for this stock among today's, counts as its trend.
    usable = [n for _, n, _ in TRENDS if not (peer.get(n) or {}).get("flat")]
    best = max(((trend_lifts[n], n) for n in usable if trend_lifts[n] is not None
                and (_rank(f.get(f"trend_{n}"), peer.get(n)) or 1.0) >= TOP_SHARE), default=(None, None))
    if not usable:
        trend = "no stock-specific view"
    else:
        trend = best[1] if best[0] is not None and best[0] >= LIFT_TREND else "no clear trend"
    trend_p = f.get(f"trend_{best[1]}") if best[1] and trend == best[1] else None
    lifts = {k: _lift(f.get(f"p_{k}"), b.get(k)) for k in EVENTS}
    ranks = {k: _rank(f.get(f"p_{k}"), peer.get(k)) for k in EVENTS}
    flat = [k for k in EVENTS if (peer.get(k) or {}).get("flat")]

    def top(k):
        return ranks[k] is None or ranks[k] >= TOP_SHARE

    warnings = []
    if (lifts["up_break"] or 0) >= LIFT_BREAK and top("up_break") and "up_break" not in flat:
        warnings.append({"kind": "breakout", "level": "info",
                         "text": f"breakout likely: {f['p_up_break']:.0%} vs {b['up_break']:.0%} usual"})
    if (lifts["down_break"] or 0) >= LIFT_BREAK and top("down_break") and "down_break" not in flat:
        warnings.append({"kind": "breakdown", "level": "warn",
                         "text": f"breakdown risk: {f['p_down_break']:.0%} vs {b['down_break']:.0%} usual"})
    if (lifts["large_loss"] or 0) >= LIFT_CRASH and f.get("p_large_loss", 0) >= MIN_CRASH and top("large_loss") \
            and "large_loss" not in flat:
        warnings.append({"kind": "crash", "level": "high",
                         "text": f"crash warning: {f['p_large_loss']:.0%} chance of a large loss vs {b['large_loss']:.0%} usual"})
    if (lifts["deep_drawdown"] or 0) >= LIFT_CRASH and f.get("p_deep_drawdown", 0) >= MIN_DEEP and top("deep_drawdown") \
            and "deep_drawdown" not in flat:
        warnings.append({"kind": "drawdown", "level": "high",
                         "text": f"deep drawdown warning: {f['p_deep_drawdown']:.0%} vs {b['deep_drawdown']:.0%} usual"})
    return {"trend": trend, "trend_p": trend_p, "trend_lifts": trend_lifts, "current_trend": f.get("current_trend"),
            "lifts": lifts, "ranks": ranks, "flat": flat, "warnings": warnings}


def _row(r) -> dict:
    payload = json.loads(r["payload_json"])
    return {"ticker": r["ticker"], "horizon": r["horizon"], "status": r["status"], "run_id": r["run_id"],
            "data_to": r["info_cutoff"], "made_at": r["created_at"], "forecast": payload.get("forecast") or {},
            "base_rates": payload.get("base_rates") or {}, "evidence": payload.get("evidence_status") or {},
            "reasons": json.loads(r["reasons_json"]), "warnings": []}


SELECT = ("SELECT f.run_id, f.ticker, f.horizon, f.status, f.reasons_json, f.payload_json, r.created_at, r.info_cutoff"
          " FROM stock_forecasts f JOIN stock_runs r ON r.id = f.run_id")


def latest(con, horizon: int = 20, ticker: str | None = None) -> dict[str, dict]:
    """The newest saved forecast per stock at this horizon, described against today's other stocks
    (the newest forecast of every stock with the same data date); one stock when ``ticker`` is given."""
    out = {}
    for r in con.execute(SELECT + " WHERE f.horizon = ? ORDER BY r.created_at DESC", (horizon,)):
        out.setdefault(r["ticker"], _row(r))
    ok = [x for x in out.values() if x["status"] == "ok"]
    by_date: dict[str, list] = {}
    for x in ok:
        by_date.setdefault(x["data_to"], []).append(x["forecast"])
    spreads = {d: peers(fs) for d, fs in by_date.items()}
    for x in ok:
        x.update(describe(x["forecast"], x["base_rates"], spreads[x["data_to"]]))
        x["peers"] = {k: {"spread": v["spread"], "n": v["n"], "flat": v["flat"]} for k, v in spreads[x["data_to"]].items()}
    if ticker:
        return {ticker.upper(): out[ticker.upper()]} if ticker.upper() in out else {}
    return out


def actions(decision: dict | None) -> dict[str, dict]:
    """The daily decision's action per stock: candidates (ENTER / WAIT) and held lots (HOLD / REDUCE / EXIT)."""
    if not decision:
        return {}
    out = {}
    for e in decision.get("entries", []):
        out[e["ticker"]] = {"action": e["action"], "paper_action": e.get("paper_action"), "blocked": e.get("blocked", []),
                            "expected_net": e.get("expected_net"), "p_win": e.get("p_win"), "dollars": e.get("dollars"),
                            "tied_with": e.get("tied_with"), "kind": "candidate"}
    for lot in decision.get("lots", []):
        out[lot["ticker"]] = {"action": lot.get("action"), "reasons": lot.get("reasons", []), "kind": "held"}
    return out


def board(con, decision: dict | None, horizon: int = 20, ticker: str | None = None) -> dict:
    """Signals and actions together, for one stock or all stocks with a saved forecast."""
    sig = latest(con, horizon, ticker)
    act = actions(decision)
    names = sorted({*sig, *([t for t in act if not ticker or t == ticker.upper()])})
    rows = [{"ticker": t, "signal": sig.get(t), "decision": act.get(t)} for t in names]
    head = {"decision_id": decision.get("id"), "mode": decision.get("mode"), "data_to": decision.get("info_cutoff"),
            "executes": decision.get("execution_date")} if decision else None
    return {"horizon": horizon, "decision": head, "rows": rows,
            "rules": {"lift_break": LIFT_BREAK, "lift_crash": LIFT_CRASH, "lift_trend": LIFT_TREND, "min_crash": MIN_CRASH,
                      "min_deep": MIN_DEEP, "top_share": TOP_SHARE, "flat_spread": FLAT_SPREAD}}


def entry_plan(forecast: dict, close: float, horizon: int = 20) -> dict | None:
    """Where to buy, and the exits that go with it, from the stock models' own numbers (pure arithmetic).

    Limit prices, each with the model's chance it is reached within ``horizon`` sessions:
    - the 20-day low (reached when the stock breaks down: P(breakdown)),
    - the median dip (the model's median drawdown from today's close: about an even chance),
    - a deep dip (its 5% drawdown quantile: about a 1-in-20 chance).
    Exits for an entry E follow the barrier policy the decision study uses: stop-loss at
    E * expm1(-sigma_h) below, take-profit at E * expm1(+sigma_h) above (sigma_h = the stock's
    volatility over the horizon), with the deadline of ``horizon`` sessions.
    """
    f = forecast or {}
    if not close or not (close > 0) or f.get("sigma_h") is None:
        return None
    import math

    sigma = float(f["sigma_h"])
    levels = []
    if f.get("lower") and f["lower"] < close:
        levels.append({"name": "At the 20-day low", "price": float(f["lower"]), "chance": f.get("p_down_break")})
    if f.get("dd_q50") is not None and f["dd_q50"] < 0:
        levels.append({"name": "Median dip", "price": close * (1 + float(f["dd_q50"])), "chance": 0.5})
    if f.get("dd_q05") is not None and f["dd_q05"] < 0:
        levels.append({"name": "Deep dip", "price": close * (1 + float(f["dd_q05"])), "chance": 0.05})
    levels.sort(key=lambda x: -x["price"])

    def exits(entry):
        return {"stop": entry * (1 + math.expm1(-sigma)), "profit": entry * (1 + math.expm1(sigma))}

    rows = [{"name": "Buy at the next close", "price": close, "chance": 1.0, **exits(close)},
            *({**lv, **exits(lv["price"])} for lv in levels)]
    for r in rows:
        r["vs_close"] = r["price"] / close - 1
    return {"close": close, "horizon": horizon, "sigma_h": sigma, "rows": rows}
