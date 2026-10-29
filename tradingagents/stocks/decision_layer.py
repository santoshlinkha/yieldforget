"""The conservative decision layer: buy now, buy on a limit, or stay in cash; hold or sell what is held.

Inputs are the outcome models' predictions for each policy (``outcome_models``), the validation status
of those models (``policy_study``), data freshness and the account. Every action is compared by its
**expected excess return over cash** from now, after costs, under the policy's exit rule:

- not held: ``cash`` (0 by definition), ``next_close`` and ``limit_dip`` (no-fill counted as cash);
- held: ``hold`` (the position from today's close, no entry cost: what it was bought at plays no part)
  against ``sell`` (the exit half of the cost, then cash).

An action is taken only if it clears the **no-trade region**: its expected excess is at least
``margin`` (chosen on validation data only, from a small fixed grid) and its 10th-percentile excess is
no worse than ``-downside_floor`` (a risk budget, configured, not tuned); a buy with no finite
10th-percentile estimate is not eligible unless ``allow_missing_downside`` says so, visibly. A held position is sold only
when holding is expected to do worse than selling by more than ``hold_margin``, so a small edge does not
churn. The layer **abstains** ("insufficient evidence") when the models are not validated, the data is
stale or the features are missing; that is a valid answer, shown with its reason.

Probabilities are reported separately and never merged: P(stop first) of a trade is not P(losing money),
not a crash probability, and not P(breakdown below the 20-day low).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

ACTIONS_NEW = ("next_close", "limit_dip")


@dataclass(frozen=True)
class LayerConfig:
    margin: float = 0.005            # minimum expected excess over cash for a buy (overwritten by validation)
    downside_floor: float = 0.15     # 10th-percentile excess must be above -15%
    hold_margin: float = 0.005       # sell only when holding is expected to trail selling by more than this
    max_names: int = 5
    max_stale_days: int = 4
    # A buy needs a finite downside estimate. Turning this on lets one through without it; each such
    # option then carries ``risk_bypass`` so the page can say so.
    allow_missing_downside: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def decide(preds: dict[str, dict], held: bool, cost_rt: float, status: dict, cfg: LayerConfig,
           stale_days: int = 0) -> dict:
    """One stock's action. ``preds``: policy -> {p_*, mean_excess, q10_excess}; ``status``: validation."""
    reasons, abstain = [], False
    if not status.get("validated"):
        abstain = True
        reasons.append(f"insufficient evidence: the outcome models are {status.get('state', 'not validated')}"
                       + (f" ({status['why']})" if status.get("why") else ""))
    if stale_days > cfg.max_stale_days:
        abstain = True
        reasons.append(f"data is {stale_days} days old")
    if not preds or any(_num((preds.get(p) or {}).get("mean_excess")) is None
                        for p in (("next_close",) if held else ACTIONS_NEW)):
        abstain = True
        reasons.append("no model estimate for this stock (missing features or prices)")
    options = []
    if held:
        h = preds.get("next_close") or {}
        hold = _num(h.get("mean_excess"))
        if hold is not None:
            hold += cost_rt / 2                      # held already: no entry cost
            sell = -cost_rt / 2
            options = [{"action": "hold", "expected_excess": hold, "q10_excess": _num(h.get("q10_excess")),
                        "p_target": _num(h.get("p_target")), "p_stop": _num(h.get("p_stop"))},
                       {"action": "sell", "expected_excess": sell, "q10_excess": sell}]
            choice = "sell" if hold < sell - cfg.hold_margin else "hold"
            why = (f"holding is expected to trail selling by {sell - hold:.2%}, more than the {cfg.hold_margin:.1%} "
                   "no-trade margin" if choice == "sell" else
                   "holding is not expected to trail selling by more than the no-trade margin")
        else:
            choice, why = "hold", "no estimate: nothing says to sell"
    else:
        options = [{"action": "cash", "expected_excess": 0.0, "q10_excess": 0.0}]
        for p in ACTIONS_NEW:
            v = preds.get(p) or {}
            m, q = _num(v.get("mean_excess")), _num(v.get("q10_excess"))
            if m is None:
                continue
            row = {"action": p, "expected_excess": m, "q10_excess": q,
                   **{k: _num(v.get(k)) for k in ("p_target", "p_stop", "p_expiry", "p_nofill")}}
            fails = []
            if m < cfg.margin:
                fails.append(f"expected excess {m:+.2%} is under the {cfg.margin:.2%} margin")
            if q is None and not cfg.allow_missing_downside:
                fails.append("no downside estimate (10th-percentile outcome), so the risk budget cannot be checked")
            elif q is None:
                row["risk_bypass"] = "no downside estimate; eligible anyway because allow_missing_downside is on"
            elif q < -cfg.downside_floor:
                fails.append(f"10th-percentile outcome {q:+.1%} is below the -{cfg.downside_floor:.0%} risk budget")
            row["eligible"], row["blocked"] = not fails, fails
            options.append(row)
        eligible = [o for o in options if o.get("eligible")]
        best = max(eligible, key=lambda o: o["expected_excess"], default=None)
        choice = best["action"] if best else "cash"
        why = (f"{choice.replace('_', ' ')} has the highest expected excess over cash that clears the margin and the "
               "risk budget" if best else "no action is expected to beat cash by the margin within the risk budget")
    action = "insufficient evidence" if abstain else choice
    return {"action": action, "model_choice": choice, "why": why, "reasons": reasons, "options": options,
            "validated": bool(status.get("validated")), "config": cfg.as_dict()}


def portfolio(decisions: dict[str, dict], held: set[str], cfg: LayerConfig) -> list[str]:
    """The new buys that fit the name limit, best expected excess first."""
    slots = max(0, cfg.max_names - len(held))
    buys = [(t, d) for t, d in decisions.items() if t not in held and d["model_choice"] in ACTIONS_NEW]
    buys.sort(key=lambda x: -max(o["expected_excess"] for o in x[1]["options"] if o["action"] == x[1]["model_choice"]))
    return [t for t, _ in buys[:slots]]


# --- live use: the bundle written by ``policy_study`` ---

def latest_bundle(cache_dir, cutoff=None):
    """The newest policy bundle fitted on data through ``cutoff`` (or any), or None."""
    import pickle
    from pathlib import Path

    folder = Path(cache_dir) / "decision_models"
    found = sorted(folder.glob("policy_bundle_*.pkl")) if folder.exists() else []
    if cutoff is not None:
        found = [f for f in found if f.stem.rsplit("_", 1)[-1] <= str(cutoff)[:10]]
    if not found:
        return None
    return pickle.loads(found[-1].read_bytes())  # noqa: S301  (the app's own model cache)


def live_view(bundle, rows, held: set[str], cost: dict, stale_days: int) -> dict:
    """Every stock's options and action from the bundle; the bundle's validation status decides abstention."""
    if bundle is None:
        return {"available": False, "note": "no policy model fitted yet: run experiments/policy_study.py"}
    cfg = LayerConfig(**{**bundle["config"], "margin": bundle["margin"]})
    preds = {p: models["lgbm"].predict(rows) for p, models in bundle["models"].items()}
    tickers = rows.index.get_level_values("ticker")
    out = {}
    for i, t in enumerate(tickers):
        per = {p: {k: _num(v) for k, v in preds[p].iloc[i].items()} for p in preds}
        out[t] = decide(per, t in held, float(cost.get(t, 0.004)), bundle["status"], cfg, stale_days)
    return {"available": True, "model_version": bundle["version"], "trained_to": bundle["cutoff"],
            "status": bundle["status"], "margin": bundle["margin"], "config": cfg.as_dict(),
            "policies": bundle["policies"], "stocks": out}


def agent_evidence(decision: dict | None, ticker: str) -> str:
    """The policy model's view for the agents, as evidence to weigh, only when it is validated; else empty.

    It states what the model estimates and how it was validated, never what the agents should conclude."""
    pv = (decision or {}).get("policy") or {}
    st = pv.get("status") or {}
    row = (pv.get("stocks") or {}).get(ticker.upper())
    if not (pv.get("available") and st.get("validated") and row):
        return ""
    lines = [f"Quantitative evidence (trade outcome model {pv.get('model_version')}, validated on a holdout on "
             f"{st.get('evaluated')}; estimates, not certainties; weigh them against your own analysis):"]
    for o in row["options"]:
        probs = ", ".join(f"{k[2:]} {v:.0%}" for k, v in o.items() if k.startswith("p_") and v is not None)
        lines.append(f"- {o['action']}: expected excess over cash {o['expected_excess']:+.2%}"
                     + (f", 10th percentile {o['q10_excess']:+.1%}" if o.get("q10_excess") is not None else "")
                     + (f"; outcome probabilities: {probs}" if probs else ""))
    return "\n".join(lines)
