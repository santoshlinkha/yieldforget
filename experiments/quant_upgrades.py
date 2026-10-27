"""Quant upgrades, each measured against the current model on identical rows (purged yearly walk-forward).

    PYTHONPATH=. TRADINGAGENTS_CACHE_DIR=<a copy of the cache> python experiments/quant_upgrades.py

Development folds 2012-2024; holdout 2025-01-01 on (scored with the same frozen recipe). Results go to
``experiments/output/quant_upgrades.json`` and ``docs/QUANT_UPGRADES.md``.

1. Position risk: label in each stock's own volatility (current, ``pos_large_loss``) vs a fixed 8% fall
   (``pos_dd8``), and the fixed label with the regime features added.
2. Trade quality: logistic + sigmoid (current) vs LightGBM with a monotone constraint on the ranker
   percentile and a few other features, isotonic-calibrated on a later slice of the training years.
3. Ranker label: 20-session return over cash (current) vs the barrier trade's net return over its
   volatility (graded into fifths per date like the current one).
4. Features: the current 54 vs + residual momentum, + earnings timing, + geopolitical/policy regime and
   sensitivities, + all of them.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / "TradingAgents" / ".env")   # the keys (FRED) of the main checkout
OUT = Path(__file__).parent / "output"
HOLDOUT = pd.Timestamp("2025-01-01")


def log(*m):
    print(time.strftime("%H:%M:%S"), *m, flush=True)


def block_ci(x, block=4, reps=2000):
    v = np.asarray(pd.Series(x).dropna())
    if len(v) < 2 * block:
        return [float("nan")] * 2
    rng = np.random.default_rng(0)
    n, k = len(v), int(np.ceil(len(v) / block))
    m = [np.concatenate([v[s:s + block] for s in rng.integers(0, n - block + 1, k)])[:n].mean() for _ in range(reps)]
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def per_date_ic(score, target):
    df = pd.DataFrame({"s": score, "y": target}).dropna()
    return df.groupby(level="date").apply(lambda g: g["s"].rank().corr(g["y"].rank()) if len(g) > 30 else np.nan).dropna()


def top_minus_all(score, net, share=0.2):
    df = pd.DataFrame({"s": score, "y": net}).dropna()

    def one(g):
        k = max(1, int(len(g) * share))
        return g.nlargest(k, "s")["y"].mean() - g["y"].mean()
    return df.groupby(level="date").apply(one)


def summarize_rank(score, label, net):
    out = {}
    d = score.index.get_level_values("date")
    for part, sel in (("development", d < HOLDOUT), ("holdout", d >= HOLDOUT)):
        s = score[sel]
        ic = per_date_ic(s, net.reindex(s.index))
        icl = per_date_ic(s, label.reindex(s.index))
        tma = top_minus_all(s, net.reindex(s.index))
        out[part] = {"dates": int(ic.size), "ic_net": float(ic.mean()), "ic_net_ci": block_ci(ic),
                     "ic_label": float(icl.mean()), "top_fifth_minus_all_net": float(tma.mean()), "tma_ci": block_ci(tma)}
    return out


def brier_summary(p, y, base):
    out = {}
    d = p.index.get_level_values("date")
    for part, sel in (("development", d < HOLDOUT), ("holdout", d >= HOLDOUT)):
        pp, yy, bb = p[sel], y.reindex(p.index)[sel], base[sel]
        ok = yy.notna()
        pp, yy, bb = pp[ok].to_numpy(), yy[ok].to_numpy(), bb[ok].to_numpy()
        bs, bsb = float(np.mean((pp - yy) ** 2)), float(np.mean((bb - yy) ** 2))
        from sklearn.metrics import roc_auc_score
        xs = pd.Series(pp, index=p[sel][ok].index).groupby(level="date").std()
        q = pd.Series(pp).rank(pct=True).to_numpy()
        out[part] = {"n": int(len(yy)), "base_rate": float(yy.mean()), "brier": bs, "brier_base": bsb,
                     "skill": 1 - bs / bsb if bsb else None, "auc": float(roc_auc_score(yy, pp)) if 0 < yy.mean() < 1 else None,
                     "cross_sectional_sd": float(xs.median()), "min": float(pp.min()), "max": float(pp.max()),
                     "observed_top_decile": float(yy[q >= .9].mean()), "observed_bottom_decile": float(yy[q <= .1].mean())}
    return out


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates
    from tradingagents.stocks import data as D
    from tradingagents.stocks import extra_features as X
    from tradingagents.stocks import macro as M
    from tradingagents.stocks import panel as P
    from tradingagents.stocks import study as S
    from tradingagents.stocks.adapters import Ranker, TradeQuality, within_date_pct
    from tradingagents.stocks.models import XGBEvent
    from tradingagents.stocks.positions import POLICIES, executable_prices

    t0 = time.time()
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    log("cache", cache)
    device = "cuda" if cuda_usable() else "cpu"
    data = D.load([], cache)
    if os.getenv("QUICK") == "1":   # a smoke run on 60 stocks
        data.info = {t: data.info[t] for t in list(data.info)[:60]}
    sessions = data.close.index
    dates = decision_dates(sessions, start="2006-01-01", every=S.EVERY)
    panel = P.build(data, 20, dates=dates, macro=M.features(dates, cache))
    panel = panel[panel["in_index"]]
    log("panel", panel.shape, data.survivorship)
    panel = X.build(panel, data, cache, lambda m: log(m))
    cov = X.coverage(panel)
    log("extra coverage", cov)
    stocks = list(data.stocks)
    px, div = executable_prices(data.close[stocks], data.actions)
    tr = S.attach_trades(panel.assign(in_index=True), px, data.low[stocks], data.close[stocks], div, POLICIES["barrier_20"])
    net = tr["net"].where(tr["complete"])
    sig_h = panel["ewma_vol_daily"] * np.sqrt(20)
    feats = P.features_for()
    edges = S.fold_edges(panel.index.get_level_values("date"))
    log("folds", [str(a.date()) for a, _ in edges], "net rows", int(net.notna().sum()))
    res = {"rows": len(panel), "coverage": cov, "survivorship": data.survivorship, "device": device}
    d = panel.index.get_level_values("date")
    label_end = pd.to_datetime(panel["label_end"])
    exit_end = pd.to_datetime(tr["exit_date"]).reindex(panel.index)

    def crossfit(fit_predict, label_col, end=None):
        end = label_end if end is None else end
        outs = []
        for start, stop in edges:
            test = (d >= start) & (d < stop)
            train = (end < start) & panel[label_col].notna()
            if not test.any() or panel.index[train].get_level_values("date").nunique() < 100:
                continue
            outs.append(fit_predict(panel[train], panel[test]))
        return pd.concat(outs)

    # --- 4 and 3: the ranker, by features and by label ---
    groups = {"current_54": feats,
              "+residual_momentum": [*feats, "res_mom_20", "res_mom_60"],
              "+earnings": [*feats, "days_since_earn", "days_to_earn_est", "earn_abret_2d"],
              "+geo_regime": [*feats, *X.REGIME_COLS, "geo_beta_126", "epu_beta_126", "geo_shock_int", "epu_shock_int"],
              "+all": [*feats, *X.STOCK_COLS, *X.REGIME_COLS]}
    res["ranker_features"] = {}
    oof = {}
    for name, fs in groups.items():
        def fp(train, test, fs=fs):
            s = Ranker(device).fit(train, fs).predict(test)
            return pd.Series(within_date_pct(test, s), index=test.index)
        oof[name] = crossfit(fp, "label")
        res["ranker_features"][name] = summarize_rank(oof[name], panel["label"], net)
        log("ranker", name, json.dumps(res["ranker_features"][name])[:300])
    # Ranker label: the barrier trade's net return over its volatility.
    panel["label_barrier_ir"] = (net / sig_h).where(net.notna())

    def fp_ir(train, test):
        s = Ranker(device).fit(train.assign(label=train["label_barrier_ir"]), feats).predict(test)
        return pd.Series(within_date_pct(test, s), index=test.index)
    oof["barrier_ir_label"] = crossfit(fp_ir, "label_barrier_ir", end=exit_end)
    res["ranker_label"] = {"return_over_cash (current)": res["ranker_features"]["current_54"],
                           "barrier_net_over_vol": summarize_rank(oof["barrier_ir_label"], panel["label"], net)}
    log("ranker label", json.dumps(res["ranker_label"])[:400])

    # --- 1: position risk ---
    res["risk"] = {}
    regime = [*X.REGIME_COLS, "geo_beta_126", "geo_shock_int"]
    for name, target, fs in (("own-volatility label (current)", "pos_large_loss", feats),
                             ("fixed 8% fall", "pos_dd8", feats),
                             ("fixed 8% fall + regime features", "pos_dd8", [*feats, *regime])):
        def fp_risk(train, test, target=target, fs=fs):
            m = XGBEvent(device).fit(train[train[target].notna()], fs, target)
            return pd.DataFrame({"p": m.predict_proba(test), "base": float(train[target].mean())}, index=test.index)
        pr = crossfit(fp_risk, target)
        res["risk"][name] = brier_summary(pr["p"], panel[target], pr["base"])
        # As a gate: barrier trades of stocks the model flags (top fifth of risk) vs the rest.
        j = pd.DataFrame({"p": pr["p"], "net": net}).dropna()
        hi = j.groupby(level="date")["p"].rank(pct=True) >= 0.8
        res["risk"][name]["net_flagged_minus_rest"] = float(j.loc[hi, "net"].mean() - j.loc[~hi, "net"].mean())
        log("risk", name, json.dumps(res["risk"][name])[:400])

    # --- 2: trade quality (on out-of-fold ranker percentiles, earlier folds only) ---
    rows = panel.assign(score_pct=oof["current_54"]).dropna(subset=["score_pct"])
    rows["win"] = (net.reindex(rows.index) > 0).astype(float).where(net.reindex(rows.index).notna())
    rows = rows[rows["win"].notna()]
    rd = rows.index.get_level_values("date")
    rend = exit_end.reindex(rows.index)
    q_feats_new = ["score_pct", "vix_term", "ewma_vol_daily", "rel_sector_1m", "days_to_earn_est", "res_mom_20",
                   "gpr_threat_z", "geo_shock_int"]
    q_feats_new = [f for f in q_feats_new if f in rows]

    def lgbm_mono(train, test):
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression

        cut = train.index.get_level_values("date").max() - pd.DateOffset(years=2)
        td = train.index.get_level_values("date")
        fit, cal = train[td < cut - pd.Timedelta(days=40)], train[td >= cut]
        mono = [1 if f == "score_pct" else 0 for f in q_feats_new]
        m = lgb.train({"objective": "binary", "num_leaves": 7, "learning_rate": 0.03, "min_data_in_leaf": 400,
                       "feature_fraction": 0.8, "lambda_l2": 10, "monotone_constraints": mono, "verbose": -1, "seed": 3},
                      lgb.Dataset(fit[q_feats_new], fit["win"]), 300)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(m.predict(cal[q_feats_new]), cal["win"])
        return iso.predict(m.predict(test[q_feats_new]))

    res["quality"] = {}
    for name, fn in (("logistic + sigmoid (current)",
                      lambda tr_, te_: TradeQuality("logit", "sigmoid", device).fit(tr_, [*feats, "score_pct"]).predict(te_)),
                     ("LightGBM monotone + isotonic", lgbm_mono)):
        outs = []
        for start, stop in edges:
            test = rows[(rd >= start) & (rd < stop)]
            train = rows[rend < start]
            if test.empty or train.index.get_level_values("date").nunique() < 150:
                continue
            outs.append(pd.DataFrame({"p": fn(train, test), "base": float(train["win"].mean())}, index=test.index))
        pq = pd.concat(outs)
        res["quality"][name] = brier_summary(pq["p"], rows["win"], pq["base"])
        log("quality", name, json.dumps(res["quality"][name])[:300])

    res["seconds"] = round(time.time() - t0)
    OUT.mkdir(exist_ok=True)
    (OUT / "quant_upgrades.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log("done", res["seconds"], "s")


if __name__ == "__main__":
    sys.exit(main())
