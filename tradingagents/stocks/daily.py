"""The daily stock decision: ENTER, WAIT, HOLD, REDUCE or EXIT, for the Trading sleeve, after the close.

Inputs: the reference universe at the cutoff, the Trading sleeve's lots (rebuilt first-in first-out
from recorded fills), the whole IRA's holdings, the capital settings, and a fitted model bundle
(``fit_bundle``: a primary ranker, a cross-fitted map from score to expected net P&L under the exit
policy, a trade-quality probability and a position-risk probability). Numbers come from those
models and arithmetic only.

Evidence and modes. A policy is **validated** only when (1) its decision study passed on the
development period (beats cash-flow-matched SPY after costs with a positive interval, beats cash,
drawdown within limit, calibration no worse than the base rate) and (2) the prospective shadow
record has confirmed it (at least ``prospective_min_trades`` completed round trips, ahead of SPY).
Until then the mode is **experimental**: buys appear as ``PAPER ENTER`` and the validated action
is ``WAIT``. A model never inherits another model's evidence: the study must match the policy
version, thresholds version and feature list.

Exits (any mode, deterministic and explicit): a close-confirmed stop or target, or the holding
deadline, schedules ``EXIT`` at the next close; a lot above the per-name cap is ``REDUCE``-d to it;
a lot without a current price is ``REVIEW``. A lot is never sold on its entry session, and its
deadline is never extended. The ML continuation exit is shown for information until validated.

Sizing: equal dollars across new names (volatility-aware sizing is studied, not yet deployed),
each name at most ``max_position_share`` of the sleeve including what is already held, at most
``max_names`` names including holdings, within settled cash less reservations, orders under
``min_order`` dropped, and no new position that would make one stock more than
``combined_name_cap`` of the whole IRA (Investing and Trading together). WAIT is a fine answer.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import panel as P
from .adapters import Ranker, TradeQuality, apply_map, member_pct, score_to_net
from .positions import POLICIES, Book, Lot, executable_prices
from .sizing import position_size
from .targets import Thresholds

BUNDLE_VERSION = "decision-v1"
CASH_FUNDS = ("SGOV", "BIL")
TIE_ORDER = "alphabetical"        # the listing order inside a tie of model scores: not a preference      # cash equivalents: valued as cash, never reviewed as stock lots
DEFAULT_RULES = {"min_expected_net": 0.005, "min_p_win": 0.5, "combined_name_cap": 0.10, "prospective_min_trades": 20,
                 "max_drawdown_limit": -0.35,
                 # When the risk model gives every member the same probability it cannot rank risk: "skip" (pass the
                 # gate, said on every line and on the Trade page; the default) or "block" (nothing passes it).
                 "risk_gate_when_uninformative": "skip",
                 # A buy needs a finite downside estimate (the 10th percentile of net P&L). False lets one through
                 # without it, and every such line says so.
                 "require_downside_estimate": True}
RISK_MIN_SPREAD = 0.005    # P(large loss) spread (max - min over today's members) below which it is uninformative


def risk_informative(p_risk: np.ndarray) -> bool:
    """Whether the risk model tells today's members apart at all. Its early stopping can keep only the forced
    minimum of trees with no split, which returns the training base rate for every stock."""
    p = np.asarray(p_risk, dtype=float)
    p = p[np.isfinite(p)]
    return len(p) > 1 and float(p.max() - p.min()) >= RISK_MIN_SPREAD


def rules(store) -> dict:
    return {**DEFAULT_RULES, **store.get("global", "stock_rules")}


# --- the account ---

def lots_from_ledger(store, sleeve: str, as_of) -> list[dict]:
    """Open lots, first in first out, from the sleeve's fills and share transfers. A split, recorded as a
    corporate action that adds (or, for a reverse split, removes) shares with no cash, scales every open lot
    of the stock by the same ratio and divides its entry price by it, so cost and position stay whole."""
    lots: dict[str, list[dict]] = {}
    for e in store.ledger.events(sleeve, as_of):
        if e["kind"] == "corporate_action" and e["ticker"] and e["quantity"] and not e["cash"]:
            group = lots.get(e["ticker"], [])
            held = sum(lot["shares"] for lot in group)
            if held > 1e-12 and held + e["quantity"] > 1e-12:
                ratio = (held + e["quantity"]) / held
                for lot in group:
                    lot["shares"] *= ratio
                    lot["entry_px"] = lot["entry_px"] / ratio if lot["entry_px"] else lot["entry_px"]
            continue
        if not e["ticker"] or e["kind"] not in ("fill", "transfer", "opening_position") or not e["quantity"]:
            continue
        t, q = e["ticker"], e["quantity"]
        if q > 0:
            lots.setdefault(t, []).append({"ticker": t, "shares": q, "entry_date": e["trade_date"],
                                           "entry_px": e["price"]})
        else:
            left = -q
            for lot in lots.get(t, []):
                take = min(left, lot["shares"])
                lot["shares"] -= take
                left -= take
                if left <= 1e-12:
                    break
            lots[t] = [lot for lot in lots.get(t, []) if lot["shares"] > 1e-12]
    return [lot for group in lots.values() for lot in group]


# --- evidence ---

def evidence(con, policy_name: str, features: list[str], thresholds: Thresholds, r: dict) -> dict:
    """Whether the policy's study passed on development, and whether the prospective record confirms it.

    The study must be for this exact model: the policy's horizon and version, the features, the thresholds
    and the model code (a study recorded before the code hash was stored counts as untested). The
    prospective record must be the shadow account's current version, run under this same policy, and beat
    SPY at the account level with a block-bootstrap interval above zero."""
    from .shadow import SHADOW_VERSION

    policy = POLICIES[policy_name]
    code = model_code_hash()
    found = None
    for row in con.execute("SELECT id, config_json, scores_json FROM research_trials WHERE name = ? ORDER BY id DESC",
                           (f"decision_study:h{policy.horizon}",)):
        cfg = json.loads(row["config_json"] or "{}")
        if cfg.get("smoke") is False and cfg.get("thresholds_version") == thresholds.version() \
                and cfg.get("features") == features and policy_name in (cfg.get("policies") or {}) \
                and cfg.get("model_code") == code \
                and (cfg.get("policy_versions") or {}).get(policy_name) == policy.version():
            found = row
            break
    if found is None:
        return {"trial": None, "development": "untested", "prospective": "none", "validated": False,
                "note": "no full decision study for this policy version, thresholds, features and model code"}
    s = json.loads(found["scores_json"])
    sim = (s.get("simulation") or {}).get(policy_name) or {}
    dev = sim.get("development:equal") or {}
    q = ((s.get("quality") or {}).get(policy_name) or {}).get("development") or {}
    checks = {
        "beats_spy_after_costs": bool((dev.get("vs_spy_ci") or [None])[0] is not None and dev["vs_spy_ci"][0] > 0),
        "beats_cash": bool(dev.get("total", -1) > dev.get("cash_total", 0)),
        "drawdown_ok": bool(dev.get("max_drawdown", -1) >= r["max_drawdown_limit"]),
        "calibration_ok": bool(q and (q.get("quality_logit_sig") or {}).get("log_loss", 9) <= q.get("log_loss_base", 0)),
    }
    passed = all(checks.values())
    shadow = con.execute("SELECT state_json FROM shadow_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    pros = json.loads(shadow["state_json"]).get("record", {}) if shadow else {}
    same_record = pros.get("version") == SHADOW_VERSION and pros.get("policy") == policy.version()
    low = (pros.get("account_excess_ci") or [None])[0]
    confirmed = passed and same_record and pros.get("round_trips", 0) >= r["prospective_min_trades"] \
        and low is not None and low > 0
    return {"trial": found["id"], "development": "passed" if passed else "failed", "checks": checks,
            "prospective": pros, "prospective_matches": same_record, "validated": bool(confirmed),
            "chosen_rule": sim.get("chosen")}


# --- the model bundle: fitted once per week of data, reused for every decision and every sizing ---

def model_code_hash() -> str:
    """A hash of the source that shapes the fitted models (not of decision or sizing code), so a change to
    sizing or display reuses this week's models while a change to features, labels or models refits them."""
    import inspect

    from tradingagents.research import forecast, models

    from . import adapters, fundamentals, macro, panel, positions, study, targets

    text = "".join(inspect.getsource(m) for m in (targets, panel, fundamentals, macro, adapters, positions, study,
                                                     models, forecast))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def bundle_key(cutoff: pd.Timestamp, policy, features, thresholds) -> str:
    week = f"{cutoff.isocalendar().year}-W{cutoff.isocalendar().week:02d}"
    text = json.dumps({"week": week, "policy": policy.version(), "features": features, "thresholds": thresholds.version(),
                       "version": BUNDLE_VERSION, "model_code": model_code_hash()}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def as_of_data(data, cutoff: pd.Timestamp):
    """The prices as they stood at the cutoff's close: every row after it dropped, so nothing later can
    reach features, labels or fits. Corporate actions stay whole: executable prices need them to undo the
    adjustment, and are computed on the full history before this cut (``executable_prices``)."""
    from dataclasses import replace

    return replace(data, close=data.close.loc[:cutoff], high=data.high.loc[:cutoff], low=data.low.loc[:cutoff],
                   volume=data.volume.loc[:cutoff])


def fit_bundle(data, macro, cutoff, policy, cache_dir: Path, device="cpu", progress=None, train=True,
               prices=None) -> tuple[dict, dict]:
    """The fitted models for this week's data, policy, features and thresholds (training cache).

    A cached bundle is reused only when it was fitted with data through the cutoff or earlier; a bundle from
    later in the same week would carry what happened after the cutoff, so that cutoff gets its own fit.
    ``prices`` is ``(executable prices, dividends)`` already cut at the cutoff."""
    from tradingagents.research.features import decision_dates

    from . import study as S

    say = progress or (lambda *_: None)
    feats, th = P.features_for(), Thresholds()
    key = bundle_key(cutoff, policy, feats, th)
    folder = Path(cache_dir) / "decision_models"
    for name in (key, f"{key}_{cutoff.date()}"):
        path = folder / f"{name}.pkl"
        if path.exists():
            blob = path.read_bytes()
            bundle = pickle.loads(blob)  # noqa: S301  (the app's own training cache)
            if str(bundle.get("meta", {}).get("cutoff", "9999")) <= str(cutoff.date()):
                return bundle, {"key": name, "sha256": hashlib.sha256(blob).hexdigest(), "reused": True}
    path = folder / f"{key}.pkl"
    if path.exists():      # the week's bundle is from a later cutoff: this cutoff is fitted on its own
        key = f"{key}_{cutoff.date()}"
        path = folder / f"{key}.pkl"
    if not train:
        raise ValueError("the decision models for this week, data and code are not fitted yet: run `stocks today --train` "
                         "(or POST /api/decisions/run with {\"train\": true})")
    t0 = time.time()
    dates = decision_dates(data.close.index[data.close.index <= cutoff], start="2006-01-01", every=S.EVERY)
    say("building the training panel")
    panel = P.build(data, policy.horizon, dates=dates, macro=macro)
    stocks = list(data.stocks)
    px, div = prices if prices is not None else executable_prices(data.close[stocks], data.actions)
    px, div = px.loc[:cutoff, stocks], div.loc[:cutoff, stocks]
    say("simulating the policy's trades")
    tr = S.attach_trades(panel, px, data.low[stocks], data.close[stocks], div, policy)
    # The training set, fingerprinted: the same data fitted before reuses that fit (stocks.training_store).
    from . import training_store as TS

    member = panel[panel["in_index"]]
    table = member[[*feats, "label", "label_end", "pos_large_loss"]].join(tr[["net", "complete", "exit_date"]])
    table = table[(pd.to_datetime(table["label_end"]) <= cutoff) | (pd.to_datetime(table["exit_date"]) <= cutoff)]
    fp = TS.fingerprint(table, {"policy": policy.version(), "features": feats, "thresholds": th.version(),
                                "version": BUNDLE_VERSION, "model_code": model_code_hash()})
    seen = TS.find(cache_dir, fp)
    if seen:
        say("same training data as an earlier fit: reusing it")
        blob = Path(seen["bundle_path"]).read_bytes()
        bundle = pickle.loads(blob)  # noqa: S301  (the app's own training cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)       # this week's key points at it too
        return bundle, {"key": key, "sha256": hashlib.sha256(blob).hexdigest(), "reused": True,
                        "same_data_as": seen["bundle_key"], "fingerprint": fp}
    edges = S.fold_edges(panel.index.get_level_values("date"))
    say("cross-fitting the ranker")
    oof = S.crossfit_primary(panel, feats, edges, [lambda: Ranker(device)])
    rows = oof.join(tr[["net", "complete", "exit_date"]]).join(member[[*feats, "label_end", "pos_large_loss"]])
    # Only trades closed by the cutoff, and only outcomes known by then: nothing after it is learned from.
    rows = rows[rows["complete"] & (pd.to_datetime(rows["exit_date"]) <= cutoff)]
    rows["win"] = (rows["net"] > 0).astype(float)
    rows["score_pct"] = rows["xgb_rank_pct"]
    spy = data.close["SPY"]
    excess = rows["net"].to_numpy() - (spy.reindex(pd.to_datetime(tr.loc[rows.index, "exit_date"])).to_numpy()
                                       / spy.reindex(pd.to_datetime(tr.loc[rows.index, "entry_date"])).to_numpy() - 1)
    say("fitting the final models")
    train_rows = member[(pd.to_datetime(member["label_end"]) <= cutoff) & member["label"].notna()]
    from .models import XGBEvent

    bundle = {
        "ranker": Ranker(device).fit(train_rows, feats),
        "map_net": score_to_net(rows["score_pct"].to_numpy(), rows["net"].to_numpy()),
        "map_excess": score_to_net(rows["score_pct"].to_numpy(), excess),
        "quality": TradeQuality("logit", "sigmoid", device).fit(rows, [*feats, "score_pct"]),
        "risk": XGBEvent(device).fit(member[member["pos_large_loss"].notna()
                                            & (pd.to_datetime(member["label_end"]) <= cutoff)], feats, "pos_large_loss"),
        "features": feats, "policy": policy, "thresholds": th.version(),
        "meta": {"cutoff": str(cutoff.date()), "train_rows": len(train_rows), "oof_rows": len(rows),
                 "seconds": round(time.time() - t0, 1), "device": device, "survivorship": data.survivorship,
                 # The last decision date any model in the bundle learned from: scoring it or earlier is refused.
                 "trained_through": str(max(train_rows.index.get_level_values("date").max(),
                                            rows.index.get_level_values("date").max()).date()),
                 "fingerprint": fp},
    }
    blob = pickle.dumps(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    try:
        TS.save(cache_dir, fp, table, bundle["meta"], key, path)
    except Exception as exc:   # saving the set must never lose the fit
        say(f"could not save the training set: {exc}")
    return bundle, {"key": key, "sha256": hashlib.sha256(blob).hexdigest(), "reused": False, "fingerprint": fp}


def score_today(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Inference at the cutoff (the inference step: no fitting). Rows on or before the last date the bundle
    trained on are refused: the model is never scored on its own training data."""
    from .training_store import guard

    guard(bundle.get("meta", {}).get("trained_through"), rows.index.get_level_values("date"))
    s = bundle["ranker"].predict(rows)
    # Percentiles among the date's index members only, like the out-of-fold rows the maps were learned on; former
    # members and your extra companies are placed among them without moving them.
    pct = member_pct(s, rows["in_index"].to_numpy(dtype=bool))
    out = pd.DataFrame(index=rows.index.get_level_values("ticker"))
    out["score_pct"] = pct
    out["expected_net"] = apply_map(bundle["map_net"], pct)
    out["net_sd"] = apply_map(bundle["map_net"], pct, "sd")
    out["net_q10"] = apply_map(bundle["map_net"], pct, "q10")
    out["expected_excess"] = apply_map(bundle["map_excess"], pct)
    out["p_win"] = bundle["quality"].predict(rows.assign(score_pct=pct))
    out["p_risk"] = bundle["risk"].predict_proba(rows)
    out["sigma"] = rows["ewma_vol_daily"].to_numpy()
    out["cost_rt"] = rows["cost_rt"].to_numpy()
    out["in_index"] = rows["in_index"].to_numpy()
    out["risk_informative"] = risk_informative(out.loc[out["in_index"], "p_risk"].to_numpy())
    return out


def next_earnings(ticker: str, cutoff: pd.Timestamp, cache_dir) -> pd.Timestamp | None:
    """The next scheduled earnings date after the cutoff, from Yahoo's cached earnings calendar (held lots only)."""
    try:
        from tradingagents.dataflows.vendors.yahoo import history as yh

        e = yh.earnings(ticker, cache_dir)
    except Exception:  # no calendar: the review says nothing rather than guess
        return None
    if e is None or not len(e):
        return None
    idx = pd.to_datetime(e.index)
    idx = idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx
    later = sorted(d.normalize() for d in idx if d.normalize() > cutoff)
    return later[0] if later else None


# --- decisions ---

def review_lot(lot: dict, cutoff: pd.Timestamp, sessions: pd.DatetimeIndex, px: pd.Series, sigma_at_entry: float,
               policy, sleeve_value: float, r: dict, today: pd.DataFrame) -> dict:
    """HOLD, REDUCE, EXIT or REVIEW for one lot, close-confirmed at the cutoff, executed the next close."""
    from tradingagents.ledger.sessions import next_session

    entry = pd.Timestamp(lot["entry_date"])
    t = lot["ticker"]
    price = float(px.get(t, np.nan))
    nxt = pd.Timestamp(next_session(cutoff.date()))
    entry_i = sessions.searchsorted(entry)
    deadline = sessions[min(entry_i + policy.horizon, len(sessions) - 1)] if entry_i + policy.horizon < len(sessions) \
        else pd.Timestamp(next_session(entry.date(), policy.horizon))
    profit, risk = policy.levels(np.array([sigma_at_entry if np.isfinite(sigma_at_entry) else 0.02]))
    out = {"ticker": t, "shares": lot["shares"], "entry_date": str(entry.date()), "entry_px": lot["entry_px"],
           "deadline": str(deadline.date()), "price": price, "profit_level": float(profit[0]), "risk_level": float(risk[0]),
           "age_sessions": int(((sessions > entry) & (sessions <= cutoff)).sum()), "reasons": []}
    if not np.isfinite(price) or not lot.get("entry_px"):
        return {**out, "action": "REVIEW", "reasons": ["no current price or no entry price: a person decides"]}
    ret = price / lot["entry_px"] - 1
    value = lot["shares"] * price
    out.update({"unrealized": ret, "value": value, "share_of_sleeve": value / sleeve_value if sleeve_value else None})
    if entry >= cutoff:
        return {**out, "action": "HOLD", "reasons": ["bought this session: no same-day exit"]}
    if nxt >= deadline:
        return {**out, "action": "EXIT", "execute": str(nxt.date()), "reasons": ["holding deadline reached (no extension)"]}
    if ret <= risk[0]:
        return {**out, "action": "EXIT", "execute": str(nxt.date()),
                "reasons": [f"close-confirmed risk level: {ret:+.1%} at or below {risk[0]:+.1%}; the next close may gap further"]}
    if ret >= profit[0]:
        return {**out, "action": "EXIT", "execute": str(nxt.date()),
                "reasons": [f"close-confirmed profit level: {ret:+.1%} at or above {profit[0]:+.1%}"]}
    cap = r["max_position_share"] * sleeve_value
    if value > cap * 1.05:
        fraction = 1 - cap / value
        return {**out, "action": "REDUCE", "fraction": fraction, "execute": str(nxt.date()),
                "reasons": [f"{value / sleeve_value:.0%} of the sleeve, above the {r['max_position_share']:.0%} cap"]}
    if t in today.index:
        out["continuation_info"] = {"expected_net_new_entry": float(today.loc[t, "expected_net"]),
                                    "note": "the ML continuation exit is information only until validated"}
    return {**out, "action": "HOLD", "reasons": ["within its levels and deadline"]}


def entries(today: pd.DataFrame, held: set, exiting: set, r: dict, sleeve_value: float, spendable: float,
            ira: dict, prices: pd.Series, validated: bool, cutoff, execute, extra: set | None = None) -> list[dict]:
    """ENTER (validated) or PAPER ENTER (experimental) and WAIT, with sizes and blocked constraints."""
    out = []
    # The model's own order: its score, then the ticker (TIE_ORDER), the same order as the saved ranks. Stocks with
    # the same score are tied: the ranker's few small trees put many stocks in identical leaves, and P(profit) was
    # measured as no better than its base rate (docs/QUANT_UPGRADES.md), so it no longer breaks ties; the order
    # inside a tie is alphabetical and says nothing about preference. Not the expected net either: that comes from
    # a binned map that need not rise with the score. Both kinds of tie are marked on each line.
    # Candidates: today's S&P 500 members and the extra companies chosen in Settings (the watchlist).
    cand = today[today["in_index"] | today.index.isin(list(extra or ()))].reset_index().rename(columns={"index": "ticker"})
    cand = cand.sort_values(["score_pct", "ticker"], ascending=[False, True])
    ties = cand.groupby("expected_net")["ticker"].transform("count")
    score_ties = cand.groupby("score_pct")["ticker"].transform("count")
    cand = cand.assign(tied_with=ties - 1, score_tied_with=score_ties - 1).set_index("ticker")
    active = (held - exiting)
    slots = max(0, r["max_names"] - len(active))
    budget = spendable
    for t, row in cand.head(25).iterrows():
        # Model gates decide whether the signal is good enough at all (they also decide PAPER ENTER, which
        # the shadow account sizes on its own cash); account gates decide whether this account can act.
        model_blocks, account_blocks, model_notes = [], [], []
        risk = row["p_risk"]
        if not (isinstance(risk, float) and np.isfinite(risk)):
            model_blocks.append("risk estimate missing or invalid")
        elif not row.get("risk_informative", True):
            # The same value for every stock is the training base rate, not an assessment of this stock.
            why = (f"risk gate not assessed: the risk model gives every stock the same {risk:.1%} "
                   "(its training base rate), so it cannot tell risky stocks apart")
            if r.get("risk_gate_when_uninformative", "block") == "skip":
                model_notes.append(f"{why}; passed anyway (setting risk_gate_when_uninformative = skip)")
            else:
                model_blocks.append(why)
        elif risk > r["max_large_loss_prob"]:
            model_blocks.append(f"P(large loss at the exit close) {risk:.0%} above {r['max_large_loss_prob']:.0%}")
        q10 = row.get("net_q10")
        if q10 is None or not np.isfinite(q10):
            if r.get("require_downside_estimate", True):
                model_blocks.append("no downside estimate (10th percentile of net P&L) for this stock")
            else:
                model_notes.append("no downside estimate; passed anyway (setting require_downside_estimate = false)")
        if not np.isfinite(row["expected_net"]) or row["expected_net"] < r["min_expected_net"]:
            model_blocks.append(f"expected net P&L {row['expected_net']:+.2%} below {r['min_expected_net']:+.2%}")
        if not np.isfinite(row["p_win"]) or row["p_win"] < r["min_p_win"]:
            model_blocks.append(f"P(profitable after costs) {row['p_win']:.0%} below {r['min_p_win']:.0%}")
        price = float(prices.get(t, np.nan))
        if not np.isfinite(price):
            model_blocks.append("no current price")
        if t in held:
            account_blocks.append("already held")
        dollars = position_size(r, sleeve_value, budget, slots)
        if not slots:
            account_blocks.append(f"name limit: {r['max_names']} names including holdings")
        if dollars < r["min_order"]:
            account_blocks.append(f"trading budget: ${budget:,.2f} available after reservations")
        ira_total = ira["total"] or 1.0
        if (ira["by_ticker"].get(t, 0.0) + dollars) / ira_total > r["combined_name_cap"]:
            account_blocks.append(f"IRA concentration: would exceed {r['combined_name_cap']:.0%} of the whole IRA in {t}")
        line = {"ticker": t, "expected_net": row["expected_net"], "net_sd": row["net_sd"], "net_q10": row["net_q10"],
                "expected_excess_vs_spy": row["expected_excess"], "p_win": row["p_win"], "p_risk": risk,
                "cost_rt": row["cost_rt"], "score_pct": row["score_pct"], "sigma": row["sigma"], "price": price,
                "tied_with": int(row["tied_with"]), "score_tied_with": int(row["score_tied_with"]),
                "blocked": model_blocks + account_blocks, "notes": model_notes, "execute": str(execute)}
        if not model_blocks:
            line["paper_action"] = "PAPER ENTER"           # the shadow account sizes it on its own cash
        if model_blocks or account_blocks:
            line["action"] = "WAIT"
        else:
            line.update({"dollars": round(dollars, 2), "shares": round(dollars / price, 6),
                         "expected_benefit_usd": round(dollars * row["expected_net"], 2),
                         "action": "ENTER" if validated else "WAIT"})
            if not validated:
                line["blocked"] = ["evidence: not validated (experimental mode shows PAPER ENTER)"]
            budget -= dollars
            slots -= 1
        out.append(line)
    return out


def apply_ensemble(today: pd.DataFrame, zoo: dict | None, bundle: dict, rows: pd.DataFrame, cutoff) -> tuple:
    """Rank by the ensemble's percentile, with its own expected net and excess over SPY; P(profit) is re-read for the
    new percentile. Unchanged (and said so) when the run for this close has no ensemble."""
    if not zoo or zoo.get("info_cutoff") != str(pd.Timestamp(cutoff).date()) or not zoo.get("ensemble"):
        return today, {"name": "xgb_rank", "note": "the internal XGBoost ranker (no ensemble run for this close)"}
    ens = {t: m["ensemble"] for t, m in zoo["stocks"].items() if "ensemble" in m}
    have = today.index.isin(list(ens))
    if not have.any():
        return today, {"name": "xgb_rank", "note": "the internal XGBoost ranker (the ensemble scored no stock)"}
    out = today.copy()
    out.loc[have, "score_pct"] = [ens[t]["pct"] for t in out.index[have]]
    out.loc[have, "expected_net"] = [ens[t]["forecast"] for t in out.index[have]]
    out.loc[have, "expected_excess"] = [ens[t].get("excess_vs_spy", np.nan) for t in out.index[have]]
    out.loc[have, "net_sd"] = [ens[t].get("net_sd", np.nan) for t in out.index[have]]
    out.loc[have, "net_q10"] = [np.nan if ens[t].get("net_q10") is None else ens[t]["net_q10"] for t in out.index[have]]
    r = rows.droplevel("date") if "date" in rows.index.names else rows
    r = r.reindex(out.index)
    out["p_win"] = bundle["quality"].predict(r.assign(score_pct=out["score_pct"].to_numpy()))
    return out, {"name": "ensemble", "weights": zoo["ensemble"]["weights"], "version": zoo["ensemble"].get("version"),
                 "note": "learned-weight ensemble of the stock models (Models tab)"}


def run(store, cache_dir: Path, as_of: date | None = None, policy_name: str = "barrier_20", train: bool = False,
        progress=None, use_ensemble: bool = True) -> dict:
    """Make, save and return today's decision. ``train`` allows fitting this week's models when missing."""
    from tradingagents.ledger import capital
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.sessions import next_session
    from tradingagents.research.compute import choose_device

    from . import data as D, macro as M
    from .targets import ewma_vol

    import inspect

    sink = progress or (lambda *_: None)
    try:
        two = len(inspect.signature(sink).parameters) >= 2
    except (TypeError, ValueError):
        two = False

    def say(text, frac=None):
        """A short line on what is happening, and how far along the run is (0 to 1) when known."""
        sink(text, frac) if two else sink(text)

    # The training stages' share of the bar (only when this week's models must be fitted).
    fit_frac = {"building the training panel": 0.40, "simulating the policy's trades": 0.50,
                "cross-fitting the ranker": 0.58, "fitting the final models": 0.70}
    r = rules(store)
    as_of = as_of or date.today()
    policy = POLICIES[policy_name]
    all_lots = lots_from_ledger(store, r["sleeve"], as_of)
    lots = [lot for lot in all_lots if lot["ticker"] not in CASH_FUNDS]
    cash_fund = {}
    for lot in all_lots:
        if lot["ticker"] in CASH_FUNDS:
            cash_fund[lot["ticker"]] = cash_fund.get(lot["ticker"], 0.0) + lot["shares"]
    say("Loading prices, dividends and filings for the S&P 500 and your companies", 0.02)
    # Besides the Trading lots: every stock held in any sleeve and the watchlist, so each gets a score
    # (outside the S&P 500 a stock is ranked against that day's members; funds are turned away by classify).
    wanted = {lot["ticker"] for lot in lots}
    wanted |= {t for t, q in store.ledger.state("combined", as_of).positions.items() if q and t not in CASH_FUNDS}
    watch = {str(t).upper() for t in store.get("global", "stock_watchlist") or []}
    wanted |= watch
    data = D.load(sorted(wanted), cache_dir, progress=lambda stage, done=None, count=None, **_: say(
        f"Loading data: {stage}" + (f" ({done} of {count})" if done is not None else f" ({count})" if count else ""),
        0.05 + 0.25 * (done / count) if done is not None and count else None))
    say("Turning adjusted prices back into the prices stocks traded at", 0.31)
    spy = data.close["SPY"].dropna()
    cutoff = spy.index[spy.index <= pd.Timestamp(as_of)][-1]
    freshness = (pd.Timestamp(as_of) - cutoff).days
    stocks = list(data.stocks)
    # Prices as traded, from the full history (undoing the adjustment needs every dividend), then everything
    # is cut at the cutoff: a past ``as_of`` sees only what was known at that close.
    px, div = executable_prices(data.close[stocks], data.actions)
    px, div = px.loc[:cutoff], div.loc[:cutoff]
    data = as_of_data(data, cutoff)
    device, _ = choose_device(store.get("global", "compute_backend"), 300_000)
    sessions = data.close.index
    say("Reading the economic data known at the close", 0.34)
    macro = M.features(pd.DatetimeIndex([cutoff]), cache_dir)
    say("Loading this week's fitted models" + (" (fitting them if missing: several minutes)" if train else ""), 0.37)
    bundle, art = fit_bundle(data, M.features(P.decision_dates_for(sessions, cutoff), cache_dir) if train else None,
                             cutoff, policy, cache_dir, device,
                             lambda text: say(f"Fitting the models: {text}", fit_frac.get(text)), train, prices=(px, div))
    say("Scoring every stock on today's close", 0.78)
    at = P.build(data, policy.horizon, dates=pd.DatetimeIndex([cutoff]), macro=macro)
    today = score_today(bundle, at.xs(cutoff, level="date", drop_level=False))
    # The ranking: the learned-weight ensemble of every stock model when its run for this close exists (Models tab,
    # ``stocks.model_zoo``); else the internal ranker alone. Run it here when it is missing or older than the cutoff.
    ranker_used = {"name": "xgb_rank", "note": "the internal XGBoost ranker"}
    if use_ensemble:
        from . import model_zoo

        zoo = model_zoo.latest(store.con)
        if model_zoo.ensemble_bundle(cache_dir) and (not zoo or zoo.get("info_cutoff") != str(cutoff.date())):
            say("Running every stock model for the ensemble (several minutes)", 0.80)
            model_zoo.run(store, cache_dir, lambda text, frac=None: say(f"Models: {text}", 0.80))
            zoo = model_zoo.latest(store.con)
        today, ranker_used = apply_ensemble(today, zoo, bundle, at.xs(cutoff, level="date", drop_level=False), cutoff)
    # The live record, once enough has matured, recalibrates P(profit) and expected net (stocks.feedback).
    from . import feedback

    live = feedback.calibration(store.con)
    today = feedback.apply(today, live)
    prices = px.loc[cutoff]
    from tradingagents.optimizer.data import latest_prices

    outside = {lot["ticker"] for lot in all_lots if lot["ticker"] not in prices.index or not np.isfinite(prices.get(lot["ticker"], np.nan))}
    fallback = latest_prices(cache_dir, outside) if outside else {}

    def price_of(t):
        p = float(prices.get(t, np.nan)) if t in prices.index else np.nan
        return p if np.isfinite(p) else float(fallback.get(t, np.nan))

    state = store.ledger.state(r["sleeve"], as_of)
    cash_fund_value = sum(q * price_of(t) for t, q in cash_fund.items())
    held_value = sum(lot["shares"] * price_of(lot["ticker"]) for lot in lots)
    # SGOV is the base and is not traded: stocks are bought only with the trading budget (SGOV's interest and the
    # match, less what is already in stocks), and positions are sized against that budget plus the stocks held.
    from tradingagents.ledger import income

    pot = income.budget(store, cache_dir, as_of)
    sleeve_value = pot["available"] + held_value
    ev = evidence(store.con, policy_name, bundle["features"], Thresholds(), r)
    ew = ewma_vol(data.close[stocks])
    reviews = []
    for lot in lots:
        e = pd.Timestamp(lot["entry_date"])
        prior = ew.index[ew.index < e]
        sigma = float(ew.loc[prior[-1], lot["ticker"]]) if len(prior) and lot["ticker"] in ew else np.nan
        line = review_lot(lot, cutoff, sessions, prices, sigma, policy, sleeve_value, r, today)
        upcoming = next_earnings(lot["ticker"], cutoff, cache_dir)
        if upcoming is not None:
            gap = int(((sessions > cutoff) & (sessions <= upcoming)).sum()) if upcoming <= sessions[-1] else None
            line["next_earnings"] = str(upcoming.date())
            if upcoming - cutoff <= pd.Timedelta(days=8):
                line["reasons"].append(f"earnings on {upcoming.date()}: a known gap risk (no automatic action)")
        reviews.append(line)
    combined = store.ledger.state("combined", as_of)
    ira_prices = latest_prices(cache_dir, set(combined.positions))
    by_ticker = {t: q * float(ira_prices.get(t, np.nan) or 0) for t, q in combined.positions.items()}
    ira = {"total": sum(by_ticker.values()) + combined.cash.total, "by_ticker": by_ticker}
    execute = next_session(cutoff.date())
    spendable = max(0.0, pot["available"] - state.cash.reserved)
    entry_lines = entries(today, {lot["ticker"] for lot in lots},
                          {x["ticker"] for x in reviews if x["action"] == "EXIT"}, r, sleeve_value, spendable, ira,
                          prices, ev["validated"], cutoff, execute, extra=watch)
    if freshness > 4:
        for line in entry_lines:
            line["action"], line["blocked"] = "WAIT", [*line["blocked"], f"data is {freshness} days old"]
    mode = "validated" if ev["validated"] else "experimental"
    snap = capital.snapshot(store)
    # Every ranked stock's score, not only the shortlist, so any stock's page can show the model's view.
    # Tied scores share a rank (competition ranking: three stocks tied first are all #1, the next is #4) and a
    # percentile (score_pct, ties averaged), so the fifths never split identical scores; ``tied`` counts the
    # other members with the same score. The listing order inside a tie is alphabetical (TIE_ORDER).
    ranked = today[today["in_index"]].reset_index().rename(columns={"index": "ticker"})
    ranked = ranked.sort_values(["score_pct", "ticker"], ascending=[False, True]).set_index("ticker")
    n = len(ranked)
    member_pct = ranked["score_pct"].to_numpy()

    def score_of(row, rank, rank_pct, member):
        same = int((member_pct == row["score_pct"]).sum()) - (1 if member else 0)
        return {"rank": rank, "of": n, "rank_pct": round(rank_pct, 4), "member": member, "kind": "stock",
                "tied": same, "tie_order": TIE_ORDER,
                "score_pct": round(float(row["score_pct"]), 4),
                "expected_net": round(float(row["expected_net"]), 5), "net_sd": round(float(row["net_sd"]), 5),
                "p_win": round(float(row["p_win"]), 4), "p_risk": round(float(row["p_risk"]), 4),
                "p_risk_informative": bool(row["risk_informative"]),
                "cost_rt": round(float(row["cost_rt"]), 5)}

    scores = {t: score_of(row, int((member_pct > row["score_pct"]).sum()) + 1, float(row["score_pct"]), True)
              for t, row in ranked.iterrows()}
    # A requested stock outside the index: where it would stand among the members (they keep their ranks).
    for t, row in today[~today["in_index"]].iterrows():
        if t in scores or not np.isfinite(row["score_pct"]):
            continue
        ahead = int((member_pct > row["score_pct"]).sum())
        scores[t] = score_of(row, ahead + 1, float(row["score_pct"]), False)
    # The policy models (buy now / buy on a limit / cash; hold / sell), from the bundle the policy study wrote;
    # kept apart from the rules above, with their own validation status.
    from .decision_layer import latest_bundle, live_view

    say("Checking holdings and candidates, sizing buys to the budget", 0.86)

    try:
        rows_today = at.xs(cutoff, level="date", drop_level=False)
        rows_today = rows_today[rows_today["in_index"] | rows_today.index.get_level_values("ticker").isin(
            [lot["ticker"] for lot in lots] + sorted(watch))]
        policy_view = live_view(latest_bundle(cache_dir, cutoff.date()), rows_today, {lot["ticker"] for lot in lots},
                                dict(zip(today.index, today["cost_rt"], strict=False)), freshness)
    except Exception as exc:   # the rules' decision stands without it; the page says why
        policy_view = {"available": False, "note": f"the policy model could not score today: {exc}"}
    payload = {"lots": reviews, "entries": entry_lines, "freshness_days": freshness, "scores": scores,
               "universe": {"sp500": int(today["in_index"].sum()),
                            "extra": sorted(t for t in watch if t in today.index and not today.loc[t, "in_index"]),
                            "extra_missing": sorted(t for t in watch if t not in today.index)},
               "policy": policy_view,
               "horizon": policy.horizon,
               "summary": {a: sum(1 for x in reviews + entry_lines if x["action"] == a)
                           for a in ("ENTER", "WAIT", "HOLD", "REDUCE", "EXIT", "REVIEW")},
               "paper_enters": sum(1 for x in entry_lines if x.get("paper_action") == "PAPER ENTER"),
               "cash_fund": {"holdings": cash_fund, "value": round(cash_fund_value, 2),
                             "note": ("the base: never traded beyond the budget. To fund a buy, sell that much SGOV; the "
                                      "sale settles the next session (T+1), so the buy comes a session later")
                             if cash_fund else None},
               "budget": {k: pot.get(k) for k in ("started", "available", "net", "interest", "match", "traded")},
               "limitations": ["Levels are close-confirmed: exits execute at the next close and gaps can lose more "
                               "than the stated risk level.", "Recommendations only: nothing is sent to the broker.",
                               f"Universe survivorship: {data.survivorship}."]}
    decision_id = uuid.uuid4().hex
    record = {"id": decision_id, "info_cutoff": str(cutoff.date()), "execution_date": str(execute), "mode": mode,
              "sleeve": r["sleeve"], "settings": {"capital": snap, "rules": r},
              "model": {"policy": policy.__dict__, "policy_version": policy.version(), "bundle": art, "ranker": ranker_used,
                        "bundle_meta": bundle["meta"], "thresholds": bundle["thresholds"], "evidence": ev,
                        "live_calibration": {k: v for k, v in live.items() if k != "p_win_map"},
                        "code": code_version()},
              "account": {"sleeve_value": round(sleeve_value, 2), "cash_fund_value": round(cash_fund_value, 2),
                          "budget": pot["available"],
                          "settled_cash": round(state.cash.settled, 2),
                          "reserved": round(state.cash.reserved, 2), "unsettled": round(state.cash.unsettled, 2),
                          "lots": len(lots), "ira_total": round(ira["total"], 2)},
              **payload}
    say("Saving the decision and every prediction", 0.92)
    with transaction(store.con):
        store.con.execute("INSERT INTO stock_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (decision_id, now_utc(), str(cutoff.date()), str(execute), r["sleeve"], mode,
                           json.dumps(record["settings"], default=str), json.dumps(record["model"], default=str),
                           json.dumps(record["account"], default=str), json.dumps(payload, default=str)))
    # Every prediction is saved before its outcome is known, and earlier ones that have matured are scored.
    feedback.record(store.con, record, model_key=str(art.get("key") if isinstance(art, dict) else art))
    try:
        feedback.resolve(store.con, cache_dir)
        feedback.export(store.con, store.folder / "results" / "model_history")
    except Exception as exc:   # scoring past predictions never blocks today's decision
        say(f"could not score past predictions: {exc}")
    from .shadow import advance

    say("Updating the paper (shadow) account", 0.97)
    record["shadow"] = advance(store, data, cutoff, decision_id, entry_lines, policy, snap, prices=(px, div))
    return record


def history(con, limit: int = 30) -> list[dict]:
    out = []
    for row in con.execute("SELECT * FROM stock_decisions ORDER BY created_at DESC LIMIT ?", (limit,)):
        p = json.loads(row["payload_json"])
        out.append({"id": row["id"], "created_at": row["created_at"], "info_cutoff": row["info_cutoff"],
                    "execution_date": row["execution_date"], "mode": row["mode"], "summary": p.get("summary"),
                    "paper_enters": p.get("paper_enters"),
                    "settings_version": json.loads(row["settings_json"])["capital"]["version"]})
    return out


def load(con, decision_id: str | None = None) -> dict | None:
    row = con.execute("SELECT * FROM stock_decisions WHERE id = ?" if decision_id else
                      "SELECT * FROM stock_decisions ORDER BY created_at DESC LIMIT 1",
                      (decision_id,) if decision_id else ()).fetchone()
    if not row:
        return None
    out = {k: row[k] for k in ("id", "created_at", "info_cutoff", "execution_date", "sleeve", "mode")}
    for k in ("settings", "model", "account"):
        out[k] = json.loads(row[f"{k}_json"])
    return {**out, **json.loads(row["payload_json"])}
