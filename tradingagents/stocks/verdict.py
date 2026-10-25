"""One plain verdict per stock (BUY, WATCH, HOLD, SELL or AVOID) from what the app already recorded.

It combines, by fixed rules (no model is fitted and nothing is fetched here):

- whether the stock is **held** (any sleeve),
- the **daily decision**'s action for it (ENTER / EXIT / REDUCE / HOLD), when it is a candidate or a lot,
- the stock models' **signals** (breakdown, crash and deep-drawdown warnings; breakout),
- the **quality and Wall Street screen** (quality score, analyst consensus, price-target upside),
- the agents' latest **rating** (Buy / Overweight / Hold / Underweight / Sell) if it is recent.

Rules, for a stock **not held**:

- **BUY**: the daily decision says ENTER (validated mode), or every test below passes: quality score at
  least 75, consensus at least "buy" (2.2 or lower), target upside at least 10%, no breakdown, crash or
  deep-drawdown warning, and no recent agent rating of Underweight or Sell.
- **AVOID**: a recent agent rating of Underweight or Sell, a crash or deep-drawdown warning, a quality
  score under 50, or a consensus of "hold" or worse (3.0 or higher).
- **WATCH**: everything else; the failed tests say what to wait for.

For a stock **held**:

- **SELL**: the daily decision says EXIT or REDUCE, or a recent agent rating of Underweight or Sell
  together with a breakdown, crash or deep-drawdown warning.
- **HOLD**: everything else, with a caution when one of those two appears alone.

None of these inputs was validated as a trading signal (the decision study did not pass), so the verdict
is a summary of the evidence on the page, not a tested forecast. The app recommends; you decide.
"""

from __future__ import annotations

from datetime import date

MIN_QUALITY_BUY = 75
MAX_CONSENSUS_BUY = 2.2
MIN_UPSIDE_BUY = 0.10
MIN_QUALITY = 50
MAX_CONSENSUS = 3.0
RATING_DAYS = 30            # an agent rating older than this no longer counts
BEARISH = ("Underweight", "Sell")
BAD_SIGNALS = ("breakdown", "crash", "drawdown")
WAITING_FOR = {"Quality score": f"a quality score of {MIN_QUALITY_BUY}+", "Analyst consensus": "a buy consensus from Wall Street",
               "Price-target upside": f"at least +{MIN_UPSIDE_BUY:.0%} to the price target",
               "No breakdown or crash warning": "the breakdown or crash warning to clear",
               "Daily decision": "the daily decision to stop saying exit"}


def _fresh(rating: dict | None, today: date) -> dict | None:
    if not rating or not rating.get("rating") or not rating.get("date"):
        return None
    try:
        age = (today - date.fromisoformat(str(rating["date"])[:10])).days
    except ValueError:
        return None
    return {**rating, "age_days": age} if 0 <= age <= RATING_DAYS else None


def verdict(held: bool, decision: dict | None, signal: dict | None, screen: dict | None, rating: dict | None,
            today: date | None = None) -> dict:
    """``{"verdict", "summary", "checks": [{name, ok, detail}], "inputs"}``; ``ok`` None means no data."""
    today = today or date.today()
    screen, rating = screen or {}, _fresh(rating, today)
    warnings = {w["kind"]: w["text"] for w in (signal or {}).get("warnings", [])} if signal and \
        signal.get("status") == "ok" else {}
    bad = {k: v for k, v in warnings.items() if k in BAD_SIGNALS}
    action = (decision or {}).get("action")
    q, c, u = screen.get("quality_score"), screen.get("analyst_mean"), screen.get("target_upside")

    def check(name, ok, detail):
        return {"name": name, "ok": ok, "detail": detail}

    checks = [
        check("Quality score", None if q is None else q >= MIN_QUALITY_BUY,
              "not screened" if q is None else f"{q:.0f} (BUY needs {MIN_QUALITY_BUY}+)"),
        check("Analyst consensus", None if c is None else c <= MAX_CONSENSUS_BUY,
              "no analysts" if c is None else f"{c:.2f} on 1 strong buy to 5 strong sell (BUY needs {MAX_CONSENSUS_BUY} or lower)"),
        check("Price-target upside", None if u is None else u >= MIN_UPSIDE_BUY,
              "no target" if u is None else f"{u:+.0%} (BUY needs +{MIN_UPSIDE_BUY:.0%})"),
        check("No breakdown or crash warning", None if not signal or signal.get("status") != "ok" else not bad,
              "no signals saved" if not signal or signal.get("status") != "ok" else ("; ".join(bad.values()) or "none")),
        check("Agents' rating", None if not rating else rating["rating"] not in BEARISH,
              "no recent agent analysis" if not rating else f"{rating['rating']} on {rating['date']} ({rating['age_days']} days ago)"),
    ]
    if action:
        checks.append(check("Daily decision", action not in ("EXIT", "REDUCE"),
                            f"{action}{' (' + decision['paper_action'] + ')' if decision.get('paper_action') else ''}"))
    bearish = bool(rating and rating["rating"] in BEARISH)

    if held:
        if action in ("EXIT", "REDUCE"):
            v, why = "SELL", f"the daily decision says {action}"
        elif bearish and bad:
            v, why = "SELL", f"the agents rate it {rating['rating']} and the models warn: {'; '.join(bad.values())}"
        else:
            v = "HOLD"
            why = ("held; caution: " + (f"the agents rate it {rating['rating']}" if bearish else "; ".join(bad.values()))) \
                if bearish or bad else "held, and nothing on record says to sell"
    else:
        failed = [x["name"] for x in checks if x["ok"] is False]
        unknown = [x["name"] for x in checks if x["ok"] is None and x["name"] != "Agents' rating"]
        if action == "ENTER":
            v, why = "BUY", "the daily decision says ENTER"
        elif bearish:
            v, why = "AVOID", f"the agents rate it {rating['rating']}"
        elif "crash" in bad or "drawdown" in bad:
            v, why = "AVOID", "; ".join(bad[k] for k in ("crash", "drawdown") if k in bad)
        elif q is not None and q < MIN_QUALITY:
            v, why = "AVOID", f"quality score {q:.0f} is under {MIN_QUALITY}"
        elif c is not None and c >= MAX_CONSENSUS:
            v, why = "AVOID", f"Wall Street's consensus is {c:.2f}: hold or worse"
        elif not failed and not unknown:
            v, why = "BUY", "every test passes"
        else:
            v = "WATCH"
            why = ("waiting for " + " and ".join(WAITING_FOR.get(n, n.lower()) for n in failed)) if failed else \
                ("not enough data: " + ", ".join(unknown).lower())
    return {"verdict": v, "summary": why, "checks": checks, "held": held,
            "rules": {"min_quality_buy": MIN_QUALITY_BUY, "max_consensus_buy": MAX_CONSENSUS_BUY,
                      "min_upside_buy": MIN_UPSIDE_BUY, "min_quality": MIN_QUALITY, "max_consensus": MAX_CONSENSUS,
                      "rating_days": RATING_DAYS}}
