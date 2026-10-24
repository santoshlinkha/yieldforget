"""Walk-forward study of the stock models: returns, events, trend, severity, portfolio, ablation, volatility.

Run from the TradingAgents folder:
    PYTHONPATH=. .venv/Scripts/python.exe experiments/stock_study.py 20        (one horizon)
    PYTHONPATH=. .venv/Scripts/python.exe experiments/stock_study.py 5 20 60   (each in its own process)

Settings are fixed here before any result is seen, thresholds included (their version
is recorded), and every run goes to the trial registry whatever it finds.

- Universe: today's S&P 500, each stock from the date it joined the index (still
  survivorship-biased: members later removed are missing).
- Folds: test blocks of two years from 2012, the same dates for every stock; each
  block's models are fitted on rows whose tradable labels ended before it began.
- Development period 2012 to June 2024; the **final test** is July 2024 on. Models
  are compared, the best chosen and the ensemble weights learned on development
  only; the final test is scored once, with everything frozen.
- Returns: momentum (benchmark), Elastic Net and quantile boosted trees, each wrapped
  for quantiles and probabilities; boosted trees for the return above SPY.
- Events (up and down breakouts, large loss, deep drawdown): base rate, logistic,
  boosted trees, and their average weighted by development skill.
- Trend states: multinomial logistic and boosted trees. Severity: quantile trees.
- Portfolio: top fifth by expected return net of cost, long only, against all stocks,
  momentum and trend rules, on non-overlapping holding periods.
- Ablation (20 sessions): technical; + market; + fundamental; + macro (all).
- Volatility (20 sessions): EWMA against GJR-GARCH(1,1) with Student-t errors on 40
  stocks, QLIKE loss on the realized variance of the next 20 sessions.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
HERE = Path(__file__).parent / "output"
HERE.mkdir(exist_ok=True)
EDGES = ["2012-01-01", "2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-07-01"]
HOLDOUT = "2024-07-01"
SMOKE = os.getenv("STOCK_STUDY_SMOKE") == "1"   # 60 stocks, recent blocks only: a check that it runs
ABLATION = [("technical",), ("technical", "market"), ("technical", "market", "fundamental"),
            ("technical", "market", "fundamental", "macro")]


def log(h, *msg):
    print(f"[h={h} {time.strftime('%H:%M:%S')}]", *msg, flush=True)


def prepare(h):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.stocks import data as D
    from tradingagents.stocks import macro as M
    from tradingagents.stocks import panel as P

    from tradingagents.research.features import decision_dates

    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    data = D.load([], cache)
    if SMOKE:
        keep = list(data.info)[:60]
        data.info = {t: data.info[t] for t in keep}
    sessions = data.close.index
    dates = decision_dates(sessions, start="2006-01-01", every=P.every_for(h)).union(pd.DatetimeIndex([sessions[-1]]))
    panel = P.build(data, h, dates=dates, macro=M.features(dates, cache))
    return data, panel


def blocks(panel):
    dates = panel.index.get_level_values("date")
    last = dates.max() + pd.Timedelta(days=1)
    edges = [pd.Timestamp(e) for e in (EDGES[-2:] if SMOKE else EDGES)] + [last]
    return [(a, b) for a, b in zip(edges, edges[1:], strict=False)]


def run(h):
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.forecast import Probabilistic
    from tradingagents.research.models import XGB, ElasticNet, Momentum
    from tradingagents.stocks import panel as P
    from tradingagents.stocks.models import BaseRate, Logit, Relative, Severity, Trend, XGBEvent
    from tradingagents.stocks.targets import EVENTS, TREND_STATES

    t0 = time.time()
    data, panel = prepare(h)
    log(h, "panel", panel.shape, f"{time.time() - t0:.0f}s")
    device = "cuda" if (h == 5 and cuda_usable()) else "cpu"
    feats = P.features_for()
    dates = panel.index.get_level_values("date")
    usable = panel[panel["in_index"]]
    chunks, fit_log = [], []
    returns = {"momentum": Momentum, "elastic_net": ElasticNet, "xgb_quantile": lambda: XGB("quantile", device=device)}
    ablate = h == 20
    for n, (start, end) in enumerate(blocks(usable), 1):
        udates = usable.index.get_level_values("date")
        train = usable[(pd.to_datetime(usable["label_end"]) < start) & usable["label"].notna()]
        test = usable[(udates >= start) & (udates < end)]
        if test.empty:
            continue
        log(h, f"block {n} {start.date()} train {len(train)} test {len(test)}")
        out = test[["label", "label_rel", "cash_h", "cost_rt", "mom_12_1", "current_trend", "drawdown",
                    "dd_breach_day", "failed_up", "failed_down", "trend_state", *EVENTS]].copy()
        for name, make in returns.items():
            t = time.time()
            m = Probabilistic(make, name=name).fit(train, feats, h)
            got = m.predict(test)
            for c in ("mean", "median", "q05", "q95", "p_gain"):
                out[f"{name}:{c}"] = got[c].to_numpy()
            fit_log.append({"block": str(start.date()), "model": name, "seconds": round(time.time() - t, 1)})
        out["relative:mean"] = Relative(device).fit(train, feats).predict(test)
        for target in EVENTS:
            for kind, model in (("base_rate", BaseRate()), ("logit", Logit()), ("xgb", XGBEvent(device))):
                t = time.time()
                out[f"p_{target}_{kind}"] = model.fit(train, feats, target).predict_proba(test)
                fit_log.append({"block": str(start.date()), "model": f"{target}:{kind}", "seconds": round(time.time() - t, 1)})
        for kind in ("logit", "xgb"):
            tm = Trend(kind, device).fit(train, feats)
            p = tm.predict_proba(test)
            for k, state in enumerate(TREND_STATES):
                out[f"p_trend_{kind}_{state}"] = p[:, k]
        out["trend_prior"] = json.dumps(tm.prior.tolist())
        sev = Severity(device).fit(train, feats).predict(test)
        out["dd_q05"], out["dd_q50"] = sev[:, 0], sev[:, 1]
        if ablate:
            for groups in ABLATION:
                tag = "+".join(groups)
                fs = P.features_for(groups)
                for name, make in (("elastic_net", ElasticNet), ("xgb", lambda: XGB("regression", device=device))):
                    out[f"abl:{tag}:{name}"] = make().fit(train, fs, h).predict(test)
        chunks.append(out)
    pred = pd.concat(chunks)
    pred.to_pickle(HERE / f"stock_pred_{h}.pkl")
    summary = summarize(h, pred, data, fit_log)
    summary["volatility"] = vol_study(data, h) if h == 20 else None
    summary["seconds"] = time.time() - t0
    return h, summary


def summarize(h, pred, data, fit_log):
    from tradingagents.research.walkforward import _ic_by_date, block_bootstrap, holm, score
    from tradingagents.stocks import evaluate as E
    from tradingagents.stocks.targets import EVENTS, TREND_STATES, Thresholds

    pdates = pred.index.get_level_values("date")
    parts = {"development": pred[pdates < pd.Timestamp(HOLDOUT)].copy(),
             "final_test": pred[pdates >= pd.Timestamp(HOLDOUT)].copy()}
    returns = ["momentum", "elastic_net", "xgb_quantile"]
    out = {"horizon": h, "thresholds": Thresholds().version(), "holdout_from": HOLDOUT,
           "dev_dates": int(parts["development"].index.get_level_values("date").nunique()),
           "test_dates": int(parts["final_test"].index.get_level_values("date").nunique()), "parts": {}}
    # Choices made on development only: the best return model, ensemble weights.
    dev = parts["development"]
    dev_scores = {m: score(dev, m) for m in returns}
    adjusted = holm({m: s.p_value for m, s in dev_scores.items()})
    weights = {m: max(s.mean_ic, 0.0) if adjusted[m] < 0.05 else 0.0 for m, s in dev_scores.items()}
    total = sum(weights.values())
    weights = {m: w / total for m, w in weights.items()} if total else {m: 1 / len(returns) for m in returns}
    best = max(dev_scores, key=lambda m: dev_scores[m].mean_ic)
    out["choices"] = {"best_return_model": best, "ensemble_weights": weights, "dev_holm": adjusted}
    for part in parts.values():
        for c in ("mean", "median", "q05", "q95", "p_gain"):
            part[f"ensemble:{c}"] = sum(weights[m] * part[f"{m}:{c}"] for m in returns)
    event_w = {}
    for target in EVENTS:
        sk = {k: E.event_scores(dev, target, f"p_{target}_{k}", f"p_{target}_base_rate").get("skill", 0) or 0
              for k in ("logit", "xgb")}
        pos = {k: max(v, 0.0) for k, v in sk.items()}
        tot = sum(pos.values())
        event_w[target] = {k: v / tot for k, v in pos.items()} if tot else {"logit": 0.5, "xgb": 0.5}
    out["choices"]["event_ensemble_weights"] = event_w
    for name, part in parts.items():
        if part.empty:
            continue
        for target in EVENTS:
            part[f"p_{target}_ensemble"] = sum(w * part[f"p_{target}_{k}"] for k, w in event_w[target].items())
        r = {"returns": {}, "events": {}, "trend": {}}
        sc = {m: score(part, m) for m in [*returns, "ensemble"]}
        adj = holm({m: s.p_value for m, s in sc.items()})
        for m, s in sc.items():
            d = dict(s.__dict__)
            d["p_value_holm"] = adj[m]
            # Dates on which the model told stocks apart at all (a constant forecast has no rank).
            d["ranked_dates"] = int(_ic_by_date(part[["label"]].assign(mean=part[f"{m}:mean"]), "mean").size)
            r["returns"][m] = d
        rel = _ic_by_date(part[["label_rel"]].rename(columns={"label_rel": "label"}).assign(mean=part["relative:mean"]),
                          "mean")
        r["relative_ic"] = {"mean": float(rel.mean()), "ci": list(block_bootstrap(rel.to_numpy()))}
        for target in EVENTS:
            extra = {"lead_time_sessions": "dd_breach_day"} if target == "deep_drawdown" else \
                {"failed_share": "failed_up"} if target == "up_break" else \
                {"failed_share": "failed_down"} if target == "down_break" else None
            r["events"][target] = {k: E.event_scores(part, target, f"p_{target}_{k}", f"p_{target}_base_rate", extra)
                                   for k in ("logit", "xgb", "ensemble")}
        prior = np.array(json.loads(part["trend_prior"].iloc[-1]))
        for kind in ("logit", "xgb"):
            r["trend"][kind] = E.trend_scores(part, [f"p_trend_{kind}_{s}" for s in TREND_STATES], prior)
        r["severity"] = E.severity_scores(part, "dd_q05")
        part = part.assign(net_score=part[f"{best}:mean"] - part["cost_rt"])
        r["portfolio"] = E.portfolio(part, "net_score", data.close.index, h)
        abl = [c for c in part.columns if c.startswith("abl:")]
        if abl:
            r["ablation"] = {}
            for c in abl:
                ic = _ic_by_date(part[["label"]].assign(mean=part[c]), "mean")
                r["ablation"][c[4:]] = {"ic": float(ic.mean()), "ci": list(block_bootstrap(ic.to_numpy()))}
        out["parts"][name] = r
    out["fits"] = len(fit_log)
    out["fit_seconds"] = sum(f["seconds"] for f in fit_log)
    return out


def vol_study(data, h, n=40, seed=3):
    """EWMA against GJR-GARCH(1,1)-t: QLIKE on the next ``h`` sessions' realized variance."""
    from arch import arch_model

    rng = np.random.default_rng(seed)
    names = [t for t in data.stocks if data.close[t].loc[:"2010"].notna().sum() > 500]
    names = list(rng.choice(names, size=min(n, len(names)), replace=False))
    losses = {"ewma": [], "gjr_garch_t": []}
    for t in names:
        r = np.log(data.close[t]).diff().dropna() * 100
        ewma = r.pow(2).ewm(alpha=0.06, min_periods=60).mean()
        for year in range(2012, 2026):
            fit_end = pd.Timestamp(f"{year}-01-01")
            hist = r[r.index < fit_end]
            if len(hist) < 1000:
                continue
            try:
                res = arch_model(hist.iloc[-2500:], vol="GARCH", p=1, o=1, q=1, dist="t", mean="Constant").fit(disp="off")
            except Exception:  # a fit that does not converge is skipped, not guessed
                continue
            year_r = r[(r.index >= fit_end) & (r.index < pd.Timestamp(f"{year + 1}-01-01"))]
            if len(year_r) < h + 25:
                continue
            fixed = arch_model(r[r.index < pd.Timestamp(f"{year + 1}-01-01")].iloc[-(2500 + len(year_r)):], vol="GARCH",
                               p=1, o=1, q=1, dist="t", mean="Constant").fix(res.params)
            fc = fixed.forecast(horizon=h, start=fit_end, reindex=False).variance.sum(axis=1)
            for d in year_r.index[: -h : 20]:
                future = r[r.index > d].iloc[:h]
                if len(future) < h or d not in fc.index:
                    continue
                realized = float((future ** 2).sum())
                for name, var in (("ewma", float(ewma.loc[d]) * h), ("gjr_garch_t", float(fc.loc[d]))):
                    if var > 0 and realized > 0:
                        losses[name].append(realized / var - np.log(realized / var) - 1)
    return {k: {"qlike": float(np.mean(v)), "n": len(v)} for k, v in losses.items()} | {"stocks": len(names)}


def main():
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.store import Store
    from tradingagents.stocks import panel as P
    from tradingagents.stocks.targets import Thresholds

    horizons = [int(x) for x in sys.argv[1:]] or [5, 20, 60]
    if len(horizons) > 1:
        with Pool(len(horizons)) as pool:
            results = pool.map(run, horizons)
    else:
        results = [run(horizons[0])]
    if SMOKE:
        for h, summary in results:
            print(json.dumps(summary, default=str)[:3000])
        return
    store = Store(Path("my_portfolio.json").resolve().parent)
    for h, summary in results:
        config = {"universe": "S&P 500 today, each from its index entry date", "edges": EDGES, "holdout_from": HOLDOUT,
                  "features": P.features_for(), "thresholds": Thresholds().__dict__,
                  "thresholds_version": Thresholds().version(), "every": P.every_for(h)}
        dev = summary["parts"].get("development", {})
        test = summary["parts"].get("final_test", {})
        with transaction(store.con):
            store.con.execute(
                "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
                " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now_utc(), f"stock_study:h{h}", "stock_pit", h, EDGES[0], str(pd.Timestamp.now().date()),
                 json.dumps(config, default=str), code_version(), json.dumps(summary, default=str),
                 json.dumps({"development": dev.get("relative_ic"), "final_test": test.get("relative_ic")}, default=str),
                 summary["seconds"]))
            for tag, v in (dev.get("ablation") or {}).items():   # each ablation variant is a trial of its own
                store.con.execute(
                    "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
                    " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (now_utc(), f"stock_study:h{h}:ablation:{tag}", "stock_pit", h, EDGES[0], HOLDOUT,
                     json.dumps({"groups": tag}), code_version(), json.dumps(v), "{}", 0.0))
        (HERE / f"stock_study_{h}.json").write_text(json.dumps(summary, indent=1, default=str))
        print("wrote", HERE / f"stock_study_{h}.json", flush=True)


if __name__ == "__main__":
    main()
