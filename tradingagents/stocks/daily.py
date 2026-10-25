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
from .adapters import Ranker, TradeQuality, apply_map, score_to_net, within_date_pct
from .positions import POLICIES, Book, Lot, executable_prices
from .targets import Thresholds

BUNDLE_VERSION = "decision-v1"
CASH_FUNDS = ("SGOV", "BIL")      # cash equivalents: valued as cash, never reviewed as stock lots
DEFAULT_RULES = {"min_expected_net": 0.005, "min_p_win": 0.5, "combined_name_cap": 0.10, "prospective_min_trades": 20,
                 "max_drawdown_limit": -0.35}


def rules(store) -> dict:
    return {**DEFAULT_RULES, **store.get("global", "stock_rules")}


# --- the account ---

def lots_from_ledger(store, sleeve: str, as_of) -> list[dict]:
    """Open lots, first in first out, from the sleeve's fills and share transfers."""
    lots: dict[str, list[dict]] = {}
    for e in store.ledger.events(sleeve, as_of):
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
    """Whether the policy's study passed on development, and whether the prospective record confirms it."""
    found = None
    for row in con.execute("SELECT id, config_json, scores_json FROM research_trials WHERE name = 'decision_study:h20'"
                           " ORDER BY id DESC"):
        cfg = json.loads(row["config_json"] or "{}")
        if cfg.get("smoke") is False and cfg.get("thresholds_version") == thresholds.version() \
                and cfg.get("features") == features and policy_name in (cfg.get("policies") or {}):
            found = row
            break
    if found is None:
        return {"trial": None, "development": "untested", "prospective": "none", "validated": False,
                "note": "no full decision study for this policy, thresholds and features"}
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
    confirmed = passed and pros.get("round_trips", 0) >= r["prospective_min_trades"] and pros.get("excess_vs_spy", -1) > 0
    return {"trial": found["id"], "development": "passed" if passed else "failed", "checks": checks,
            "prospective": pros, "validated": bool(confirmed),
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
    edges = S.fold_edges(panel.index.get_level_values("date"))
    say("cross-fitting the ranker")
    oof = S.crossfit_primary(panel, feats, edges, [lambda: Ranker(device)])
    member = panel[panel["in_index"]]
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
                 "seconds": round(time.time() - t0, 1), "device": device, "survivorship": data.survivorship},
    }
    blob = pickle.dumps(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return bundle, {"key": key, "sha256": hashlib.sha256(blob).hexdigest(), "reused": False}


def score_today(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Inference at the cutoff (the inference step: no fitting)."""
    s = bundle["ranker"].predict(rows)
    pct = within_date_pct(rows, s)
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
            ira: dict, prices: pd.Series, validated: bool, cutoff, execute) -> list[dict]:
    """ENTER (validated) or PAPER ENTER (experimental) and WAIT, with sizes and blocked constraints."""
    out = []
    # The model's own order: its score, then P(profit), then the ticker (the same order as the saved ranks),
    # so the input order never decides which tied stock comes first. Not the expected net: that comes from
    # a binned map that need not rise with the score, and would put lower-ranked stocks first. Ties of the
    # expected net are still marked, since tied stocks share one estimate.
    cand = today[today["in_index"]].reset_index().rename(columns={"index": "ticker"})
    cand = cand.sort_values(["score_pct", "p_win", "ticker"], ascending=[False, False, True])
    ties = cand.groupby("expected_net")["ticker"].transform("count")
    cand = cand.assign(tied_with=ties - 1).set_index("ticker")
    active = (held - exiting)
    slots = max(0, r["max_names"] - len(active))
    budget = spendable
    for t, row in cand.head(25).iterrows():
        # Model gates decide whether the signal is good enough at all (they also decide PAPER ENTER, which
        # the shadow account sizes on its own cash); account gates decide whether this account can act.
        model_blocks, account_blocks = [], []
        risk = row["p_risk"]
        if not (isinstance(risk, float) and np.isfinite(risk)):
            model_blocks.append("risk estimate missing or invalid")
        elif risk > r["max_large_loss_prob"]:
            model_blocks.append(f"P(large loss at the exit close) {risk:.0%} above {r['max_large_loss_prob']:.0%}")
        if not np.isfinite(row["expected_net"]) or row["expected_net"] < r["min_expected_net"]:
            model_blocks.append(f"expected net P&L {row['expected_net']:+.2%} below {r['min_expected_net']:+.2%}")
        if not np.isfinite(row["p_win"]) or row["p_win"] < r["min_p_win"]:
            model_blocks.append(f"P(profitable after costs) {row['p_win']:.0%} below {r['min_p_win']:.0%}")
        price = float(prices.get(t, np.nan))
        if not np.isfinite(price):
            model_blocks.append("no current price")
        if t in held:
            account_blocks.append("already held")
        dollars = min(r["max_position_share"] * sleeve_value, budget / max(slots, 1)) if slots else 0.0
        if not slots:
            account_blocks.append(f"name limit: {r['max_names']} names including holdings")
        if dollars < r["min_order"]:
            account_blocks.append(f"settled cash: ${budget:,.2f} available after reservations")
        ira_total = ira["total"] or 1.0
        if (ira["by_ticker"].get(t, 0.0) + dollars) / ira_total > r["combined_name_cap"]:
            account_blocks.append(f"IRA concentration: would exceed {r['combined_name_cap']:.0%} of the whole IRA in {t}")
        line = {"ticker": t, "expected_net": row["expected_net"], "net_sd": row["net_sd"], "net_q10": row["net_q10"],
                "expected_excess_vs_spy": row["expected_excess"], "p_win": row["p_win"], "p_risk": risk,
                "cost_rt": row["cost_rt"], "score_pct": row["score_pct"], "sigma": row["sigma"], "price": price,
                "tied_with": int(row["tied_with"]), "blocked": model_blocks + account_blocks, "execute": str(execute)}
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


def run(store, cache_dir: Path, as_of: date | None = None, policy_name: str = "barrier_20", train: bool = False,
        progress=None) -> dict:
    """Make, save and return today's decision. ``train`` allows fitting this week's models when missing."""
    from tradingagents.ledger import capital
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.sessions import next_session
    from tradingagents.research.compute import choose_device

    from . import data as D
    from . import macro as M
    from .targets import ewma_vol

    say = progress or (lambda *_: None)
    r = rules(store)
    as_of = as_of or date.today()
    policy = POLICIES[policy_name]
    all_lots = lots_from_ledger(store, r["sleeve"], as_of)
    lots = [lot for lot in all_lots if lot["ticker"] not in CASH_FUNDS]
    cash_fund = {}
    for lot in all_lots:
        if lot["ticker"] in CASH_FUNDS:
            cash_fund[lot["ticker"]] = cash_fund.get(lot["ticker"], 0.0) + lot["shares"]
    say("loading data")
    # Besides the Trading lots: every stock held in any sleeve and the watchlist, so each gets a score
    # (outside the S&P 500 a stock is ranked against that day's members; funds are turned away by classify).
    wanted = {lot["ticker"] for lot in lots}
    wanted |= {t for t, q in store.ledger.state("combined", as_of).positions.items() if q and t not in CASH_FUNDS}
    wanted |= {str(t).upper() for t in store.get("global", "stock_watchlist") or []}
    data = D.load(sorted(wanted), cache_dir)
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
    macro = M.features(pd.DatetimeIndex([cutoff]), cache_dir)
    bundle, art = fit_bundle(data, M.features(P.decision_dates_for(sessions, cutoff), cache_dir) if train else None,
                             cutoff, policy, cache_dir, device, say, train, prices=(px, div))
    say("scoring today")
    at = P.build(data, policy.horizon, dates=pd.DatetimeIndex([cutoff]), macro=macro)
    today = score_today(bundle, at.xs(cutoff, level="date", drop_level=False))
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
    sleeve_value = state.cash.total + held_value + cash_fund_value
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
    spendable = max(0.0, state.cash.available)      # available = settled minus reservations
    entry_lines = entries(today, {lot["ticker"] for lot in lots},
                          {x["ticker"] for x in reviews if x["action"] == "EXIT"}, r, sleeve_value, spendable, ira,
                          prices, ev["validated"], cutoff, execute)
    if freshness > 4:
        for line in entry_lines:
            line["action"], line["blocked"] = "WAIT", [*line["blocked"], f"data is {freshness} days old"]
    mode = "validated" if ev["validated"] else "experimental"
    snap = capital.snapshot(store)
    # Every ranked stock's score, not only the shortlist, so any stock's page can show the model's view.
    # The ranker gives many stocks the same score, so the order breaks ties by P(profit), then ticker, and
    # rank_pct (1 = first, 0 = last) splits today's stocks into even fifths where score_pct, tied, cannot.
    ranked = today[today["in_index"]].reset_index().rename(columns={"index": "ticker"})
    ranked = ranked.sort_values(["score_pct", "p_win", "ticker"], ascending=[False, False, True]).set_index("ticker")
    n = len(ranked)

    def score_of(row, rank, rank_pct, member):
        return {"rank": rank, "of": n, "rank_pct": round(rank_pct, 4), "member": member, "kind": "stock",
                "score_pct": round(float(row["score_pct"]), 4),
                "expected_net": round(float(row["expected_net"]), 5), "net_sd": round(float(row["net_sd"]), 5),
                "p_win": round(float(row["p_win"]), 4), "p_risk": round(float(row["p_risk"]), 4),
                "cost_rt": round(float(row["cost_rt"]), 5)}

    scores = {t: score_of(row, i + 1, 1 - i / max(n - 1, 1), True) for i, (t, row) in enumerate(ranked.iterrows())}
    # A requested stock outside the index: where it would stand among the members (they keep their ranks).
    member_pct = ranked["score_pct"].to_numpy()
    for t, row in today[~today["in_index"]].iterrows():
        if t in scores or not np.isfinite(row["score_pct"]):
            continue
        ahead = int((member_pct > row["score_pct"]).sum())
        scores[t] = score_of(row, ahead + 1, 1 - ahead / max(n - 1, 1), False)
    payload = {"lots": reviews, "entries": entry_lines, "freshness_days": freshness, "scores": scores,
               "horizon": policy.horizon,
               "summary": {a: sum(1 for x in reviews + entry_lines if x["action"] == a)
                           for a in ("ENTER", "WAIT", "HOLD", "REDUCE", "EXIT", "REVIEW")},
               "paper_enters": sum(1 for x in entry_lines if x.get("paper_action") == "PAPER ENTER"),
               "cash_fund": {"holdings": cash_fund, "value": round(cash_fund_value, 2),
                             "note": ("held as a cash equivalent; selling it frees settled cash the next session (T+1), "
                                      "so buys funded by it come a session later") if cash_fund else None},
               "limitations": ["Levels are close-confirmed: exits execute at the next close and gaps can lose more "
                               "than the stated risk level.", "Recommendations only: nothing is sent to the broker.",
                               f"Universe survivorship: {data.survivorship}."]}
    decision_id = uuid.uuid4().hex
    record = {"id": decision_id, "info_cutoff": str(cutoff.date()), "execution_date": str(execute), "mode": mode,
              "sleeve": r["sleeve"], "settings": {"capital": snap, "rules": r},
              "model": {"policy": policy.__dict__, "policy_version": policy.version(), "bundle": art,
                        "bundle_meta": bundle["meta"], "thresholds": bundle["thresholds"], "evidence": ev,
                        "code": code_version()},
              "account": {"sleeve_value": round(sleeve_value, 2), "cash_fund_value": round(cash_fund_value, 2),
                          "settled_cash": round(state.cash.settled, 2),
                          "reserved": round(state.cash.reserved, 2), "unsettled": round(state.cash.unsettled, 2),
                          "lots": len(lots), "ira_total": round(ira["total"], 2)},
              **payload}
    with transaction(store.con):
        store.con.execute("INSERT INTO stock_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (decision_id, now_utc(), str(cutoff.date()), str(execute), r["sleeve"], mode,
                           json.dumps(record["settings"], default=str), json.dumps(record["model"], default=str),
                           json.dumps(record["account"], default=str), json.dumps(payload, default=str)))
    from .shadow import advance

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
