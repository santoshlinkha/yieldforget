"""Decision study: entry models, exit policies and the capital simulation, recorded whatever they show.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/decision_study.py
Design: tradingagents/stocks/study.py. Capital comes from Settings (its snapshot is recorded).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
HERE = Path(__file__).parent / "output"


def log(*m):
    print(time.strftime("%H:%M:%S"), *m, flush=True)


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ledger import capital
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.store import Store
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates
    from tradingagents.research.walkforward import _ic_by_date, block_bootstrap
    from tradingagents.stocks import data as D
    from tradingagents.stocks import macro as M
    from tradingagents.stocks import panel as P
    from tradingagents.stocks import study as S
    from tradingagents.stocks.adapters import Momentum, Ranker, WithinDateLinear
    from tradingagents.stocks.models import log_loss
    from tradingagents.stocks.positions import POLICIES, benchmark, executable_prices
    from tradingagents.stocks.targets import Thresholds, ewma_vol
    from tradingagents.stocks.daily import model_code_hash

    t0 = time.time()
    store = Store(Path("my_portfolio.json").resolve().parent)
    snap = capital.snapshot(store)
    cash_amount = snap["trading_capital_usd"]
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    data = D.load([], cache)
    if os.getenv("DECISION_SMOKE") == "1":      # a quick check that everything runs: 80 stocks
        data.info = {t: data.info[t] for t in list(data.info)[:80]}
    sessions = data.close.index
    dates = decision_dates(sessions, start="2006-01-01", every=S.EVERY)
    log("universe", len(data.stocks), "dates", len(dates), data.survivorship)
    panel = P.build(data, 20, dates=dates, macro=M.features(dates, cache))
    log("panel", panel.shape)
    feats = P.features_for()
    stocks = [t for t in data.stocks]
    px, div = executable_prices(data.close[stocks], data.actions)
    low, close = data.low[stocks], data.close[stocks]
    device = "cuda" if cuda_usable() else "cpu"
    tr = {name: S.attach_trades(panel, px, low, close, div, POLICIES[name]) for name in ("fixed_20", "barrier_20")}
    log("trades", {k: int(v["complete"].sum()) for k, v in tr.items()})
    edges = S.fold_edges(panel.index.get_level_values("date"))
    oof = S.crossfit_primary(panel, feats, edges, [Momentum, WithinDateLinear, lambda: Ranker(device)])
    log("primary OOF", oof.shape)
    out = {"capital": snap, "thresholds": Thresholds().version(), "survivorship": data.survivorship,
           "policies": {k: v.__dict__ for k, v in POLICIES.items() if k in tr}, "dev_end": str(S.DEV_END.date())}
    member = panel[panel["in_index"]]
    od = oof.index.get_level_values("date")
    parts = {"development": od <= S.DEV_END, "exploratory": od > S.DEV_END}

    # 1. Ranking: IC of each primary score against the tradable label and the policy's net P&L.
    out["ranking"] = {}
    for part, sel in parts.items():
        o = oof[sel].join(member[["label"]]).join(tr["barrier_20"][["net"]])
        res = {}
        for m in ("momentum", "linear_xs", "xgb_rank"):
            ic = _ic_by_date(o[["label"]].assign(mean=o[m]), "mean")
            icn = _ic_by_date(o[["net"]].rename(columns={"net": "label"}).assign(mean=o[m]), "mean")
            res[m] = {"ic_label": float(ic.mean()), "ic_label_ci": list(block_bootstrap(ic.to_numpy(), block=4)),
                      "ic_net": float(icn.mean()), "dates": int(o.index.get_level_values("date").nunique()),
                      "ranked_dates": int(ic.size)}
        out["ranking"][part] = res
    primary = max(("linear_xs", "xgb_rank"), key=lambda m: out["ranking"]["development"][m]["ic_net"])
    out["primary"] = primary
    log("primary chosen on development:", primary)

    # 2. Cross-fitted trade quality, expected net P&L and position risk.
    second = {}
    for name in ("fixed_20", "barrier_20"):
        second[name] = S.crossfit_second(panel, oof, tr[name], edges, primary, feats, device)
        log("second stage", name, second[name].shape)
    out["quality"] = {}
    for name, sec in second.items():
        j = sec.join(tr[name][["net", "complete"]])
        j = j[j["complete"]]
        y = (j["net"] > 0).astype(float).to_numpy()
        res = {}
        for part in ("development", "exploratory"):
            sel = (j.index.get_level_values("date") <= S.DEV_END) if part == "development" else \
                (j.index.get_level_values("date") > S.DEV_END)
            base = np.full(sel.sum(), y[sel].mean())
            r = {"base_rate": float(y[sel].mean()), "n": int(sel.sum()),
                 "log_loss_base": log_loss(y[sel], base), "brier_base": float(np.mean((base - y[sel]) ** 2))}
            for col in ("quality_logit", "quality_logit_sig", "quality_xgb", "quality_xgb_sig"):
                p = j[col].to_numpy()[sel]
                r[col] = {"log_loss": log_loss(y[sel], p), "brier": float(np.mean((p - y[sel]) ** 2)),
                          "reliability": S.calibration_table(p, y[sel])}
            e = j["expected_net"].to_numpy()[sel]
            r["expected_net_vs_realized"] = {"top_decile_expected": float(np.quantile(e, 0.9)),
                                             "realized_top_decile": float(j["net"].to_numpy()[sel][e >= np.quantile(e, 0.9)].mean()),
                                             "realized_all": float(j["net"].to_numpy()[sel].mean())}
            res[part] = r
        out["quality"][name] = res

    # 3. Rule selection on development, then the capital simulation in both periods.
    def signals_for(name):
        sec = second[name].join(oof[[primary, f"{primary}_pct"]]).join(member[["ewma_vol_daily", "cost_rt"]])
        sec["sigma"] = sec["ewma_vol_daily"].fillna(0.02)
        sec["cost_rt"] = sec["cost_rt"].fillna(0.004)
        sec["rank_key"] = sec["expected_net"]
        return sec

    spy, bil = data.close["SPY"], data.close["BIL"]
    grid = [(net, p) for net in (0.0, 0.0025, 0.005, 0.01) for p in (0.5, 0.55)]
    out["simulation"] = {}
    for name in ("fixed_20", "barrier_20"):
        sig = signals_for(name)
        dev_sig = sig[sig.index.get_level_values("date") <= S.DEV_END]
        best, scores = None, []
        for net, p in grid:
            rule = (lambda r, net=net, p=p: r["expected_net"] >= net and r["quality_logit_sig"] >= p and r["p_risk"] <= 0.15)
            sim = S.simulate(dev_sig, px, div, POLICIES[name], cash_amount, "2013-01-01", str(S.DEV_END.date()),
                             rule=rule, label=f"{name}:{net}:{p}")
            scores.append({"net": net, "p": p, "total": sim["total"], "max_drawdown": sim["max_drawdown"],
                           "trades": sim["trades"]})
            if sim["trades"] >= 50 and (best is None or sim["total"] > best[2]):
                best = (net, p, sim["total"])
        out["simulation"][name] = {"grid_development": scores, "chosen": best}
        if best is None:
            continue
        rule = (lambda r, net=best[0], p=best[1]: r["expected_net"] >= net and r["quality_logit_sig"] >= p
                and r["p_risk"] <= 0.15)
        for part, (a, b) in {"development": ("2013-01-01", str(S.DEV_END.date())),
                             "exploratory": (str((S.DEV_END + pd.Timedelta(days=1)).date()), str(sessions[-1].date()))}.items():
            for sizing in ("equal", "vol"):
                sim = S.simulate(sig, px, div, POLICIES[name], cash_amount, a, b, rule=rule, sizing=sizing,
                                 label=f"{name}:{sizing}")
                bench = S.buy_and_hold(spy, cash_amount, a, b)
                cash_b = S.buy_and_hold(bil, cash_amount, a, b)
                out["simulation"][name][f"{part}:{sizing}"] = S.summary(sim, bench, cash_b)
            mom = sig.assign(rank_key=sig.join(member[["mom_12_1"]])["mom_12_1"].fillna(-1))
            sim = S.simulate(mom, px, div, POLICIES["fixed_20"], cash_amount, a, b, label="momentum_fixed_20")
            out["simulation"][name][f"{part}:momentum_baseline"] = S.summary(sim, S.buy_and_hold(spy, cash_amount, a, b),
                                                                            S.buy_and_hold(bil, cash_amount, a, b))
        log("simulated", name)

    # 4. Exits on identical entries: fixed vs ML continuation (chronologically cross-fitted).
    from tradingagents.stocks.models import Relative  # noqa: F401  (kept for symmetry; not used)
    from tradingagents.research.models import XGB

    ew = ewma_vol(data.close[stocks])
    entries = tr["fixed_20"][tr["fixed_20"]["complete"]]
    top = oof[f"{primary}_pct"].reindex(entries.index)
    entries = entries[top >= 0.8]                                   # the same candidate entries for every exit rule
    states = S.continuation_states(entries, px, div, data.close[stocks], ew, bil)
    log("continuation states", states.shape)
    sfeat = ["age", "unrealized", "unrealized_sigma", "vol_ratio", "mom_5", "check"]
    # One row per (checkpoint date, ticker, entry): overlapping entries of a stock share checkpoint dates.
    states = states.set_index(["check_date", "ticker", "entry_decision"]).sort_index()
    states.index.names = ["date", "ticker", "entry_decision"]
    states["label"] = states["benefit"]
    states["label_end"] = pd.to_datetime(states["exit_date"])
    preds = []
    sd = states.index.get_level_values("date")
    for start, end in edges:
        test = states[(sd >= start) & (sd < end)]
        train = states[states["label_end"] < start]
        if test.empty or train.index.get_level_values("date").nunique() < 50:
            continue
        m = XGB("regression", device=device).fit(train, sfeat, 1)
        preds.append(pd.Series(m.predict(test), index=test.index))
    states["pred"] = pd.concat(preds).reindex(states.index)
    # Rule (persistence): exit at the next close when the predicted benefit is below tau at this checkpoint
    # and the one before; tau chosen on development from a small grid.
    res = {}
    states = states.reset_index()
    for tau in (-0.01, -0.005, 0.0):
        realized = []
        for (entry_date, ticker), g in states.groupby(["entry_decision", "ticker"]):
            g = g.sort_values("check")
            low = (g["pred"] < tau).to_numpy()
            fire = next((i for i in range(1, len(g)) if low[i] and low[i - 1]), None)
            # Selling at the next close forgoes the remaining benefit: the change in P&L is minus it.
            realized.append({"date": g["entry_decision"].iloc[0], "delta": -float(g["benefit"].iloc[fire]) if fire is not None else 0.0,
                             "fired": fire is not None})
        r = pd.DataFrame(realized)
        dev = r[pd.to_datetime(r["date"]) <= S.DEV_END]
        exp_ = r[pd.to_datetime(r["date"]) > S.DEV_END]
        res[str(tau)] = {"development_mean_delta": float(dev["delta"].mean()), "development_fired": float(dev["fired"].mean()),
                         "exploratory_mean_delta": float(exp_["delta"].mean()) if len(exp_) else None,
                         "trades": int(len(r))}
    out["ml_exit_vs_fixed"] = res
    out["seconds"] = time.time() - t0
    out["block_note"] = "bootstrap blocks: 4 decision dates for IC (20-session overlap at a 5-session cadence), 20 sessions for daily excess"
    with transaction(store.con):
        store.con.execute(
            "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
            " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now_utc(), "decision_study:h20", f"sp500_pit_{data.survivorship}", 20, "2012-01-01", str(sessions[-1].date()),
             json.dumps({"capital": snap, "features": feats, "smoke": os.getenv("DECISION_SMOKE") == "1", "thresholds_version": Thresholds().version(),
                         "policies": out["policies"], "every": S.EVERY, "dev_end": out["dev_end"],
                         # What evidence() binds a validation to: the exact model code and policy versions.
                         "model_code": model_code_hash(),
                         "policy_versions": {k: POLICIES[k].version() for k in out["policies"]}}, default=str),
             code_version(), json.dumps(out, default=str), "{}", out["seconds"]))
    (HERE / "decision_study.json").write_text(json.dumps(out, indent=1, default=str))
    log("wrote", HERE / "decision_study.json", f"{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
