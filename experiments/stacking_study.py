"""Learned weights for the stock models (stacking), tested walk-forward, and the bundle the live app uses.

    QUICK=1 .venv/Scripts/python.exe -m experiments.stacking_study     # 60 stocks, a smoke run
    .venv/Scripts/python.exe -m experiments.stacking_study             # S&P 500 members, about an hour

1. **Out-of-fold scores.** Each member model is fitted per year on rows whose labels ended before the year and scores
   that year (``stocks.model_zoo`` functions; Mamba from ``stocks.seq_model``): one model per family, no
   near-duplicates (XGBoost on 54 features, LightGBM LambdaRank, LightGBM regression, pooled ridge, ridge per stock,
   Mamba). Chronos-2 (prices only, and with market covariates) is scored from 2023 every 10 sessions and reported
   apart: too short a history to weigh.
2. **Weights.** Each model's score becomes its within-date rank (-0.5 to 0.5). For each year from 2014, non-negative
   least squares fits the within-date rank of the trade's net P&L on those ranks, using only rows whose trades had
   exited before the year: weights never see the year they are judged on. The weights sum to one.
3. **Evaluation.** The stacked score against the equal-weight average and each model alone, on the same rows:
   rank IC with net P&L and the top fifth less the average, 2014-2024 and 2025 on (looked at before: development
   evidence; the forward record is the test).
4. **The live bundle** (``<cache>/optimizer/decision_models/ensemble.json``): weights fitted on every out-of-fold
   year, and the map from the stacked percentile to net P&L and to the excess over SPY, learned on the walk-forward
   stacked scores (out of sample), as the daily decision's own map is.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.model_zoo_study import evaluate, log

MEMBERS = ["xgb_rank", "lgbm_lambdarank", "lgbm_regression", "ridge_pooled", "ridge_per_stock", "mamba"]
OUT = Path("experiments/output")
FIRST_WEIGHT_YEAR = 2014


def within_rank(s: pd.Series) -> pd.Series:
    return s.groupby(level="date").rank(pct=True) - 0.5


def nnls_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls

    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    w, _ = nnls(X[ok], y[ok])
    return w / w.sum() if w.sum() > 0 else np.full(X.shape[1], 1 / X.shape[1])


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates
    from tradingagents.research.foundation import Chronos2, Chronos2Covariates
    from tradingagents.stocks import data as D, macro as M, model_zoo as Z, panel as P, study as S
    from tradingagents.stocks.adapters import score_to_net
    from tradingagents.stocks.positions import POLICIES, executable_prices
    from tradingagents.stocks.seq_model import MambaRanker, SequenceBank

    t0 = time.time()
    quick = os.getenv("QUICK") == "1"
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    device = "cuda" if cuda_usable() else "cpu"
    data = D.load([], cache)
    if quick:
        data.info = {t: data.info[t] for t in list(data.info)[:60]}
    sessions = data.close.index
    dates = decision_dates(sessions, start="2006-01-01", every=S.EVERY)
    panel = P.build(data, 20, dates=dates, macro=M.features(dates, cache))
    panel = panel[panel["in_index"]]
    stocks = list(data.stocks)
    tickers = sorted(set(panel.index.get_level_values("ticker")))
    px, div = executable_prices(data.close[stocks], data.actions)
    tr = S.attach_trades(panel.assign(in_index=True), px, data.low[stocks], data.close[stocks], div, POLICIES["barrier_20"])
    net = tr["net"].where(tr["complete"])
    spy = data.close["SPY"]
    entry, exit_ = pd.to_datetime(tr["entry_date"]), pd.to_datetime(tr["exit_date"])
    excess = net - (spy.reindex(exit_).to_numpy() / spy.reindex(entry).to_numpy() - 1)
    extra = Z.extra_features(cache, data, pd.DatetimeIndex(sorted(set(panel.index.get_level_values("date")))), tickers, px)
    panel = panel.join(extra)
    sets = Z.feature_sets(P.features_for(), list(extra.columns))
    edges = S.fold_edges(panel.index.get_level_values("date"))
    label_end = pd.to_datetime(panel["label_end"])
    d = panel.index.get_level_values("date")
    log("panel", panel.shape, "net rows", int(net.notna().sum()))

    def crossfit(fit_predict, cached=None):
        """Out-of-fold scores, one fit per year. A year whose rows are all in ``cached`` keeps them: its fit used only
        rows that ended before the year, which never change. Only a year with new rows (the current one, as data
        arrives) is fitted again, so a refresh every 20 sessions costs one year's fit per model."""
        outs = []
        for start, stop in edges:
            test = (d >= start) & (d < stop)
            train = (label_end < start) & panel["label"].notna()
            if not test.any() or panel.index[train].get_level_values("date").nunique() < 100:
                continue
            idx = panel.index[test]
            if cached is not None and idx.isin(cached.index).all():
                outs.append(cached.reindex(idx))
                continue
            log("fitting fold", start.date())
            outs.append(pd.Series(fit_predict(panel[train], panel[test]), index=idx))
        return pd.concat(outs)

    # 1. Out-of-fold scores, saved so the weights can be refitted without refitting the models.
    bank = SequenceBank(data)
    oof = {}
    for name in MEMBERS:
        path = OUT / f"oof_{name}{'_quick' if quick else ''}.pkl"
        cached = pd.read_pickle(path) if path.is_file() else None
        if cached is not None:
            log("loaded", name, "through", cached.index.get_level_values("date").max().date())
        if name == "mamba":
            oof[name] = crossfit(lambda a, b: MambaRanker(bank, device).fit(a, sets["base"]).predict(b), cached)
        else:
            fn, which = Z.FITTED[name]
            oof[name] = crossfit(lambda a, b, fn=fn, which=which: fn(a, b, sets[which], device), cached)
        if cached is not None and oof[name].index.equals(cached.index):
            continue
        oof[name].to_pickle(path)
        log("oof", name, json.dumps(evaluate(oof[name], net)))

    # Chronos-2, prices only and with covariates: from 2023, every 10 sessions; reported, not weighted.
    res = {"members": MEMBERS, "chronos": {}}
    cd = sorted({x for x in d if x >= pd.Timestamp("2023-01-01")})[::2]
    rows = panel[d.isin(cd)]
    etf = {t: (data.info.get(t) or {}).get("sector_etf") for t in tickers}
    close_t = data.close
    # A scheduled refresh (STACKING_SKIP_CHRONOS=1) skips this report: it is not used in the weights and takes most
    # of the run; the last one stays in model_zoo.json.
    chronos = [] if os.getenv("STACKING_SKIP_CHRONOS") == "1" else [
        ("prices only", lambda: Chronos2(close_t, device)),
        ("with covariates", lambda: Chronos2Covariates(close_t, data.volume, etf, device))]
    for label, make in chronos:
        model = make()
        try:
            s = pd.Series(model.fit(None, None, 20).predict(rows), index=rows.index)
            res["chronos"][label] = evaluate(s, net)
            log("chronos", label, json.dumps(res["chronos"][label]))
        except Exception as exc:  # noqa: BLE001
            res["chronos"][label] = {"error": str(exc)[:300]}
            log("chronos failed", label, exc)

    # 2. Walk-forward weights.
    R = pd.DataFrame({k: within_rank(v) for k, v in oof.items()}).dropna()
    y = within_rank(net.reindex(R.index))
    ex = exit_.reindex(R.index)
    rd = R.index.get_level_values("date")
    stacked, weights_by_year = [], {}
    for year in sorted({x.year for x in rd if x.year >= FIRST_WEIGHT_YEAR}):
        start = pd.Timestamp(f"{year}-01-01")
        fit = (rd < start) & (ex < start).to_numpy() & y.notna().to_numpy()
        w = nnls_weights(R[fit].to_numpy(), y[fit].to_numpy())
        weights_by_year[year] = dict(zip(R.columns, np.round(w, 4).tolist(), strict=True))
        test = rd.year == year
        stacked.append(pd.Series(R[test].to_numpy() @ w, index=R.index[test]))
    stacked = pd.concat(stacked)
    on = stacked.index
    res["evaluation"] = {"stacked (learned weights)": evaluate(stacked, net),
                         "equal weights": evaluate(R.loc[on].mean(axis=1), net),
                         **{f"{k} alone": evaluate(R.loc[on, k], net) for k in R.columns}}
    res["weights_by_year"] = weights_by_year
    for k, v in res["evaluation"].items():
        log("eval", k, json.dumps(v))

    # 3. The live bundle: weights on every year; maps learned on the walk-forward stacked scores.
    known = y.notna().to_numpy()
    final = nnls_weights(R[known].to_numpy(), y[known].to_numpy())
    pct = stacked.groupby(level="date").rank(pct=True)
    ok = net.reindex(pct.index).notna()
    bundle = {"version": "ensemble-v1", "members": list(R.columns), "weights": dict(zip(R.columns, final.tolist(), strict=True)),
              "map_net": score_to_net(pct[ok].to_numpy(), net.reindex(pct.index)[ok].to_numpy()),
              "map_excess": score_to_net(pct[ok].to_numpy(), excess.reindex(pct.index)[ok].to_numpy()),
              "trained_through": str(ex.max().date()), "evaluation": res["evaluation"]["stacked (learned weights)"],
              "created": time.strftime("%Y-%m-%d %H:%M"), "rows": int(len(R))}
    res["final_weights"] = bundle["weights"]
    log("final weights", json.dumps({k: round(v, 3) for k, v in bundle["weights"].items()}))
    if not quick:
        (cache / "decision_models").mkdir(parents=True, exist_ok=True)
        (cache / "decision_models" / "ensemble.json").write_text(json.dumps(bundle, indent=1, default=float))
        (OUT / "stacking.json").write_text(json.dumps(res, indent=1, default=float))
        zoo = OUT / "model_zoo.json"                  # beside the other models, where the Models tab reads records
        study = json.loads(zoo.read_text(encoding="utf-8")) if zoo.is_file() else {}
        study.setdefault("models", {})["ensemble (learned weights)"] = res["evaluation"]["stacked (learned weights)"]
        if res["chronos"]:
            study["chronos"] = {f"chronos2 {k}": v for k, v in res["chronos"].items()}
        zoo.write_text(json.dumps(study, indent=1, default=float))
    log("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
