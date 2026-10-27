"""Stock forecasting models and data, compared on identical purged yearly walk-forward folds.

    QUICK=1 .venv/Scripts/python.exe -m experiments.model_zoo_study      # 60 stocks, a smoke run
    .venv/Scripts/python.exe -m experiments.model_zoo_study              # S&P 500 members, about an hour

Rows: S&P 500 members on decision dates every 5 sessions from 2006 (point-in-time membership). Target for every
model: the stock's 20-session return over cash from the next close (``label``); each model's score is ranked within
the date. Evaluation: rank correlation (IC) with the barrier trade's net P&L after costs, and the top fifth's net
minus the date's average, with date-block bootstrap intervals; development 2012-2024, then 2025 on, which earlier
studies have looked at (so: development evidence too). The forward record is the only untouched test.

Data sets added to the current 54 features (each known by the decision close):
- ``news``: SEC 8-K events (``stocks.news_archive``): filings in the last 5 / 20 sessions, sessions since the last
  results release, executive changes, deals and announcements in the last 60.
- ``consensus``: Wall Street analysts (``stocks.consensus``): firms covering, average grade, net upgrades, target
  changes and upside, last EPS surprise, beats, EPS growth. Not the AI agents' ratings.
- ``peers``: competitors: the 10 stocks of the same sector most correlated with it over the last 252 sessions;
  their average return over 20 and 60 sessions and the stock's return less theirs.

Models (``label`` target, same folds):
- ``xgb_rank``: XGBoost pairwise ranker (the current model), on the 54 features and on all features.
- ``lgbm_lambdarank``: LightGBM LambdaRank on all features.
- ``lgbm_regression``: LightGBM regression of the return on all features.
- ``ridge_pooled``: one ridge regression for all stocks on all features.
- ``ridge_per_stock``: a ridge regression **for each stock** on its own history: the market, its sector, its
  competitors, the economy, analysts, news and its own momentum and volatility.
- baselines with no fitting: 12-1 momentum, low volatility, analysts' grade, analysts' target upside.
- ``chronos2``: Amazon's pretrained time-series model, zero-shot on each stock's daily prices, on dates from 2023
  (every 20 sessions); its training data has no published cutoff, so results after its release are shown apart.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("experiments/output/model_zoo.json")
HOLDOUT = pd.Timestamp("2025-01-01")
CHRONOS_FROM = pd.Timestamp("2023-01-01")
PEERS = 10


def log(*m):
    print(time.strftime("%H:%M:%S"), *m, flush=True)


def block_ci(x: pd.Series, block: int = 4, reps: int = 1000) -> list[float]:
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 2 * block:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(0)
    k = int(np.ceil(n / block))
    m = [np.concatenate([v[s:s + block] for s in rng.integers(0, n - block + 1, k)])[:n].mean() for _ in range(reps)]
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def evaluate(score: pd.Series, net: pd.Series) -> dict:
    df = pd.DataFrame({"s": score, "y": net}).dropna()
    out = {}
    d = df.index.get_level_values("date")
    for part, sel in (("development", d < HOLDOUT), ("2025_on", d >= HOLDOUT)):
        g = df[sel].groupby(level="date")
        ic = g.apply(lambda x: x["s"].rank().corr(x["y"].rank()) if len(x) > 30 else np.nan).dropna()
        tma = g.apply(lambda x: x.nlargest(max(1, len(x) // 5), "s")["y"].mean() - x["y"].mean() if len(x) > 30 else np.nan).dropna()
        out[part] = {"dates": int(ic.size), "ic": float(ic.mean()) if ic.size else None, "ic_ci": block_ci(ic),
                     "top_fifth_minus_all": float(tma.mean()) if tma.size else None, "tma_ci": block_ci(tma)}
    return out


def peer_features(close: pd.DataFrame, sectors: dict, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """The 10 most correlated same-sector stocks (252 sessions to the date) and their recent returns."""
    r = np.log(close[tickers]).diff()
    r20, r60 = np.log(close[tickers]).diff(20), np.log(close[tickers]).diff(60)
    rows = []
    for d in dates:
        hist = r.loc[:d].iloc[-252:]
        for sec in {sectors.get(t) for t in tickers} - {None}:
            names = [t for t in tickers if sectors.get(t) == sec and hist[t].notna().sum() > 150]
            if len(names) < 3:
                continue
            corr = np.array(hist[names].corr().to_numpy(), copy=True)
            np.fill_diagonal(corr, -np.inf)
            a20, a60 = r20.loc[d, names].to_numpy(), r60.loc[d, names].to_numpy()
            for i, t in enumerate(names):
                top = np.argsort(-np.nan_to_num(corr[i], nan=-np.inf))[:PEERS]
                p20, p60 = np.nanmean(a20[top]), np.nanmean(a60[top])
                rows.append((d, t, p20, p60, a20[i] - p20))
    f = pd.DataFrame(rows, columns=["date", "ticker", "peer_ret_20", "peer_ret_60", "own_minus_peer_20"])
    return f.set_index(["date", "ticker"])


def crossfit(panel, edges, fit_predict, label_end):
    d = panel.index.get_level_values("date")
    outs = []
    for start, stop in edges:
        test = (d >= start) & (d < stop)
        train = (label_end < start) & panel["label"].notna()
        if not test.any() or panel.index[train].get_level_values("date").nunique() < 100:
            continue
        outs.append(pd.Series(fit_predict(panel[train], panel[test]), index=panel.index[test]))
    return pd.concat(outs)


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates
    from tradingagents.stocks import consensus as C, data as D, macro as M, news_archive as N, panel as P
    from tradingagents.stocks import study as S
    from tradingagents.stocks.adapters import Ranker
    from tradingagents.stocks.positions import POLICIES, executable_prices

    t0 = time.time()
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    device = "cuda" if cuda_usable() else "cpu"
    data = D.load([], cache)
    if os.getenv("QUICK") == "1":
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
    log("panel", panel.shape, "net rows", int(net.notna().sum()))

    # --- the new data, joined by (date, ticker) ---
    news = N.features(cache, dates, tickers, sessions)
    import sqlite3
    with sqlite3.connect(cache / "market_data.sqlite3") as con:          # only stocks whose Yahoo lists are stored
        listed = {r[0] for r in con.execute("SELECT DISTINCT ticker FROM snapshots WHERE kind = 'upgrades'")}
    cons = C.features(cache, dates, [t for t in tickers if t in listed], sessions, px, data.actions)
    sectors = {t: (data.info.get(t) or {}).get("sector") for t in tickers}
    peers = peer_features(data.close, sectors, dates, [t for t in tickers if t in data.close])
    panel = panel.join(news).join(cons).join(peers)
    news_cols, cons_cols, peer_cols = list(news.columns), list(cons.columns), list(peers.columns)
    cov = {c: float(panel[c].notna().mean()) for c in news_cols + cons_cols + peer_cols}
    log("coverage", json.dumps({k: round(v, 3) for k, v in cov.items()}))
    base = P.features_for()
    groups = {"current_54": base, "+news": base + news_cols, "+consensus": base + cons_cols, "+peers": base + peer_cols,
              "+all": base + news_cols + cons_cols + peer_cols}
    allf = groups["+all"]
    edges = S.fold_edges(panel.index.get_level_values("date"))
    label_end = pd.to_datetime(panel["label_end"])
    res = {"rows": len(panel), "coverage": cov, "device": device, "survivorship": data.survivorship,
           "evidence": "development: every period was inspected in earlier studies; the forward record is the test",
           "data_sets": {}, "models": {}, "baselines": {}}

    def rank_pct(train, test, fs):
        s = Ranker(device).fit(train, fs).predict(test)
        return s

    # 1. Which data helps: the current model on each feature set.
    for name, fs in groups.items():
        res["data_sets"][name] = evaluate(crossfit(panel, edges, lambda a, b, fs=fs: rank_pct(a, b, fs), label_end), net)
        log("data", name, json.dumps(res["data_sets"][name]))

    # 2. Which model, on all the data.
    import lightgbm as lgb

    lgb_params = {"num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 500, "feature_fraction": 0.7,
                  "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7}

    def lambdarank(train, test):
        tr_ = train.sort_index(level="date")
        y = tr_["label"].groupby(level="date").rank(pct=True).mul(5).clip(upper=4.999).astype(int)
        grp = tr_.groupby(level="date").size().to_numpy()
        m = lgb.train({**lgb_params, "objective": "lambdarank", "metric": "ndcg"},
                      lgb.Dataset(tr_[allf], y, group=grp), 300)
        return m.predict(test[allf])

    def regression(train, test):
        y = train["label"].clip(*train["label"].quantile([0.01, 0.99]))
        return lgb.train({**lgb_params, "objective": "regression"}, lgb.Dataset(train[allf], y), 300).predict(test[allf])

    def ridge_pooled(train, test):
        from sklearn.linear_model import Ridge
        med = train[allf].median()
        mu, sd = train[allf].fillna(med).mean(), train[allf].fillna(med).std().replace(0, 1)
        z = lambda f: ((f[allf].fillna(med) - mu) / sd).fillna(0).to_numpy()   # noqa: E731
        y = train["label"].clip(*train["label"].quantile([0.01, 0.99]))
        return Ridge(alpha=1000.0).fit(z(train), y).predict(z(test))

    per_stock_cols = ["spy_mom_3m", "vix", "vix_slope", "tnx_chg_3m", "credit_spread", "curve_10y2y", "cpi_yoy",
                      "nfci", "rel_sector_1m", "rel_sector_3m", "rel_spy_3m", "mom_1m", "mom_3m", "mom_12_1",
                      "vol_3m", "beta_1y", *peer_cols, *cons_cols, *news_cols]

    def ridge_per_stock(train, test):
        from sklearn.linear_model import Ridge
        cols = [c for c in per_stock_cols if c in train]
        out = pd.Series(np.nan, index=test.index)
        for t, te in test.groupby(level="ticker"):
            own = train.xs(t, level="ticker", drop_level=False) if t in train.index.get_level_values("ticker") else None
            if own is None or len(own) < 80:
                continue
            med = own[cols].median()
            mu, sd = own[cols].fillna(med).mean(), own[cols].fillna(med).std().replace(0, 1).fillna(1)
            z = lambda f: ((f[cols].fillna(med) - mu) / sd).fillna(0).to_numpy()   # noqa: E731
            m = Ridge(alpha=50.0).fit(z(own), own["label"].clip(*own["label"].quantile([0.02, 0.98])))
            out[te.index] = m.predict(z(te))
        return out.to_numpy()

    for name, fn in (("xgb_rank (current, 54 features)", None), ("xgb_rank (all features)", None),
                     ("lgbm_lambdarank", lambdarank), ("lgbm_regression", regression), ("ridge_pooled", ridge_pooled),
                     ("ridge_per_stock", ridge_per_stock)):
        if fn is None:
            res["models"][name] = res["data_sets"]["current_54" if "54" in name else "+all"]
            continue
        res["models"][name] = evaluate(crossfit(panel, edges, fn, label_end), net)
        log("model", name, json.dumps(res["models"][name]))

    # 3. Baselines, no fitting.
    sel = panel.index.get_level_values("date") >= edges[0][0]
    for name, col, sign in (("momentum_12_1", "mom_12_1", 1), ("low_volatility", "vol_3m", -1),
                            ("analysts_grade", "an_score", 1), ("analysts_target_upside", "an_pt_upside", 1),
                            ("analysts_net_upgrades", "an_net_90", 1), ("eps_surprise", "eps_surprise_last", 1)):
        res["baselines"][name] = evaluate(sign * panel.loc[sel, col], net)
        log("baseline", name, json.dumps(res["baselines"][name]))

    # 4. Chronos-2, zero-shot, on dates from 2023 (every 20 sessions to bound the compute).
    try:
        from tradingagents.research.foundation import RELEASE, Chronos2
        cd = sorted({d for d in panel.index.get_level_values("date") if d >= CHRONOS_FROM})[::4]
        rows = panel[panel.index.get_level_values("date").isin(cd)]
        m = Chronos2(data.close[[t for t in tickers if t in data.close]], device).fit(None, None, 20)
        score = pd.Series(m.predict(rows), index=rows.index)
        res["models"]["chronos2 (zero-shot, from 2023)"] = evaluate(score, net)
        after = score[score.index.get_level_values("date") >= pd.Timestamp(RELEASE)]
        res["models"]["chronos2 (after its release)"] = evaluate(after, net)
        log("model chronos2", json.dumps(res["models"]["chronos2 (zero-shot, from 2023)"]))
    except Exception as exc:  # noqa: BLE001
        res["models"]["chronos2"] = {"error": str(exc)[:300]}
        log("chronos2 failed", exc)

    res["seconds"] = round(time.time() - t0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log("done", res["seconds"], "s")


if __name__ == "__main__":
    main()
