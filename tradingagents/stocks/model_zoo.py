"""Every stock forecasting model side by side: each fitted on all history to the last close, each scoring today's
S&P 500 members and your watchlist. Shown next to the internal model; none of them decides anything.

The models and data are the ones ``experiments/model_zoo_study.py`` compares on identical purged folds; its saved
results (``experiments/output/model_zoo.json``) are attached to each model as its tested record, so a forecast is
never shown without how that model did on data it had not seen.

Per stock and model:

- ``pct``: where the stock stands among today's index members by that model (1 = its top pick; members only, so
  your extra companies are placed among them without moving them);
- ``forecast``: for the models that predict a return, the 20-session log return over cash from the next close
  (rankers and baselines have no return units: ``None``).

Inputs (each known by the cutoff close): the 54 panel features, SEC 8-K news events (``news_archive``), Wall Street
analysts' consensus rebuilt per date (``consensus``; never the AI agents' ratings) and competitors' returns
(``peer_features``). Every run is saved (``model_zoo_runs``, append-only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "model-zoo-v1"
PEERS = 10
STUDY = Path(__file__).resolve().parents[2] / "experiments" / "output" / "model_zoo.json"
# name -> (label, kind, the study's key for its tested record, the section it is in)
MODELS = {
    "xgb_rank": ("XGBoost ranker, 54 features (the internal model's ranker)", "rank",
                 "xgb_rank (current, 54 features)", "models"),
    "xgb_rank_all": ("XGBoost ranker, all data", "rank", "xgb_rank (all features)", "models"),
    "lgbm_lambdarank": ("LightGBM LambdaRank, all data", "rank", "lgbm_lambdarank", "models"),
    "lgbm_regression": ("LightGBM regression, all data", "return", "lgbm_regression", "models"),
    "ridge_pooled": ("Ridge regression, all stocks together", "return", "ridge_pooled", "models"),
    "ridge_per_stock": ("Ridge regression for each stock (market, sector, competitors, economy, analysts, news)",
                        "return", "ridge_per_stock", "models"),
    "chronos2": ("Chronos-2, pretrained, prices only (zero-shot)", "return", "chronos2 (zero-shot, from 2023)", "models"),
    "mamba": ("Mamba state-space model: 64 days of prices, volume and sector, plus the 54 features", "rank",
              "mamba (64-day sequences + 54 features)", "models"),
    "chronos2_covariates": ("Chronos-2 with market covariates (SPY, sector, volume, VIX, 10-year yield), zero-shot",
                            "return", "chronos2 with covariates", "chronos"),
    "ensemble": ("Learned-weight ensemble of the distinct models (decides the ranking)", "return",
                 "ensemble (learned weights)", "models"),
    "momentum_12_1": ("Momentum, 12 months less the last", "rank", "momentum_12_1", "baselines"),
    "low_volatility": ("Low volatility", "rank", "low_volatility", "baselines"),
    "analysts_target_upside": ("Analysts' target upside", "rank", "analysts_target_upside", "baselines"),
    "eps_surprise": ("Last earnings surprise", "rank", "eps_surprise", "baselines"),
}
PER_STOCK_COLS = ["spy_mom_3m", "vix", "vix_slope", "tnx_chg_3m", "credit_spread", "curve_10y2y", "cpi_yoy", "nfci",
                  "rel_sector_1m", "rel_sector_3m", "rel_spy_3m", "mom_1m", "mom_3m", "mom_12_1", "vol_3m", "beta_1y"]
LGB = {"num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 500, "feature_fraction": 0.7, "bagging_fraction": 0.7,
       "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7}


# --- data ---

def peer_features(close: pd.DataFrame, sectors: dict, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """For each stock: the 10 same-sector stocks most correlated with it over the 252 sessions to the date, their
    average log return over 20 and 60 sessions, and the stock's 20-session return less theirs."""
    names_all = [t for t in tickers if t in close]
    lc = np.log(close[names_all])
    r, r20, r60 = lc.diff(), lc.diff(20), lc.diff(60)
    rows = []
    for d in dates:
        hist = r.loc[:d].iloc[-252:]
        for sec in {sectors.get(t) for t in names_all} - {None}:
            names = [t for t in names_all if sectors.get(t) == sec and hist[t].notna().sum() > 150]
            if len(names) < 3:
                continue
            corr = np.array(hist[names].corr().to_numpy(), copy=True)
            np.fill_diagonal(corr, -np.inf)
            a20, a60 = r20.loc[d, names].to_numpy(), r60.loc[d, names].to_numpy()
            for i, t in enumerate(names):
                top = [j for j in np.argsort(-np.nan_to_num(corr[i], nan=-np.inf)) if j != i][:PEERS]   # never itself
                p20, p60 = np.nanmean(a20[top]), np.nanmean(a60[top])
                rows.append((d, t, p20, p60, a20[i] - p20))
    f = pd.DataFrame(rows, columns=["date", "ticker", "peer_ret_20", "peer_ret_60", "own_minus_peer_20"])
    return f.set_index(["date", "ticker"])


def extra_features(cache_dir, data, dates: pd.DatetimeIndex, tickers: list[str], px: pd.DataFrame) -> pd.DataFrame:
    """News, analyst consensus and competitors for each (date, ticker)."""
    import sqlite3

    from . import consensus as C, news_archive as N

    sessions = data.close.index
    news = N.features(cache_dir, dates, tickers, sessions)
    with sqlite3.connect(Path(cache_dir) / "market_data.sqlite3") as con:   # only stocks whose Yahoo lists are stored
        listed = {r[0] for r in con.execute("SELECT DISTINCT ticker FROM snapshots WHERE kind = 'upgrades'")}
    cons = C.features(cache_dir, dates, [t for t in tickers if t in listed], sessions, px, data.actions)
    sectors = {t: (data.info.get(t) or {}).get("sector") for t in tickers}
    peers = peer_features(data.close, sectors, dates, tickers)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return news.reindex(idx).join(cons.reindex(idx)).join(peers.reindex(idx))


# --- the models: each fits on ``train`` and returns one score per row of ``test`` ---

def _clip(y: pd.Series, q=(0.01, 0.99)) -> pd.Series:
    return y.clip(*y.quantile(list(q)))


def fit_xgb_rank(train, test, feats, device="cpu"):
    from .adapters import Ranker

    return Ranker(device).fit(train, feats).predict(test)


def fit_lambdarank(train, test, feats, device="cpu"):
    import lightgbm as lgb

    tr = train.sort_index(level="date")
    y = tr["label"].groupby(level="date").rank(pct=True).mul(5).clip(upper=4.999).astype(int)
    grp = tr.groupby(level="date").size().to_numpy()
    m = lgb.train({**LGB, "objective": "lambdarank", "metric": "ndcg"}, lgb.Dataset(tr[feats], y, group=grp), 300)
    return m.predict(test[feats])


def fit_lgbm_regression(train, test, feats, device="cpu"):
    import lightgbm as lgb

    return lgb.train({**LGB, "objective": "regression"}, lgb.Dataset(train[feats], _clip(train["label"])), 300
                     ).predict(test[feats])


def _design(frame, cols):
    med = frame[cols].median()
    mu, sd = frame[cols].fillna(med).mean(), frame[cols].fillna(med).std().replace(0, 1).fillna(1)
    return lambda f: ((f[cols].fillna(med) - mu) / sd).fillna(0).to_numpy()


def fit_ridge_pooled(train, test, feats, device="cpu"):
    from sklearn.linear_model import Ridge

    z = _design(train, feats)
    return Ridge(alpha=1000.0).fit(z(train), _clip(train["label"])).predict(z(test))


def fit_ridge_per_stock(train, test, feats, device="cpu", min_rows: int = 80):
    """A ridge regression on each stock's own history; stocks with too little history get no forecast."""
    from sklearn.linear_model import Ridge

    cols = [c for c in feats if c in train]
    out = pd.Series(np.nan, index=test.index)
    have = set(train.index.get_level_values("ticker"))
    for t, te in test.groupby(level="ticker"):
        if t not in have:
            continue
        own = train.xs(t, level="ticker", drop_level=False)
        if len(own) < min_rows:
            continue
        z = _design(own, cols)
        out[te.index] = Ridge(alpha=50.0).fit(z(own), _clip(own["label"], (0.02, 0.98))).predict(z(te))
    return out.to_numpy()


FITTED = {"xgb_rank": (fit_xgb_rank, "base"), "xgb_rank_all": (fit_xgb_rank, "all"),
          "lgbm_lambdarank": (fit_lambdarank, "all"), "lgbm_regression": (fit_lgbm_regression, "all"),
          "ridge_pooled": (fit_ridge_pooled, "all"), "ridge_per_stock": (fit_ridge_per_stock, "per_stock")}
BASELINES = {"momentum_12_1": ("mom_12_1", 1), "low_volatility": ("vol_3m", -1),
             "analysts_target_upside": ("an_pt_upside", 1), "eps_surprise": ("eps_surprise_last", 1)}


def feature_sets(base: list[str], extra: list[str]) -> dict[str, list[str]]:
    return {"base": base, "all": base + extra, "per_stock": PER_STOCK_COLS + extra}


def tested_record() -> dict:
    """Each model's walk-forward record from the saved study, when it has been run."""
    if not STUDY.is_file():
        return {}
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    out = {}
    for name, (_, _, key, section) in MODELS.items():
        r = (study.get(section) or {}).get(key)
        if r and "development" in r:
            out[name] = {"development_ic": r["development"].get("ic"), "development_ic_ci": r["development"].get("ic_ci"),
                         "recent_ic": (r.get("2025_on") or {}).get("ic"),
                         "recent_ic_ci": (r.get("2025_on") or {}).get("ic_ci"),
                         "top_fifth_minus_all": r["development"].get("top_fifth_minus_all")}
    return out


# --- the live run ---

def run(store, cache_dir: Path, progress=None) -> dict:
    """Fit every model on history to the last close, score today's members and watchlist, save and return the run."""
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates

    from . import data as D, macro as M, panel as P
    from .adapters import member_pct
    from .daily import as_of_data
    from .positions import executable_prices

    say = progress or (lambda *a: None)
    t0 = time.time()
    device = "cuda" if cuda_usable() else "cpu"
    watch = sorted({str(t).upper() for t in store.get("global", "stock_watchlist") or []})
    say("Loading prices, filings and the economy", 0.02)
    data = D.load(watch, Path(cache_dir))
    spy = data.close["SPY"].dropna()
    cutoff = spy.index[-1]
    stocks = list(data.stocks)
    px, _ = executable_prices(data.close[stocks], data.actions)
    data = as_of_data(data, cutoff)
    px = px.loc[:cutoff]
    sessions = data.close.index
    say("Building the training panel (every 5 sessions since 2006)", 0.10)
    dates = decision_dates(sessions, start="2006-01-01", every=5)
    panel = P.build(data, 20, dates=dates, macro=M.features(dates, cache_dir))
    train = panel[panel["in_index"] & panel["label"].notna() & (pd.to_datetime(panel["label_end"]) <= cutoff)]
    today = P.build(data, 20, dates=pd.DatetimeIndex([cutoff]), macro=M.features(pd.DatetimeIndex([cutoff]), cache_dir))
    today = today[today["in_index"] | today.index.get_level_values("ticker").isin(watch)]
    tickers = sorted(set(train.index.get_level_values("ticker")) | set(today.index.get_level_values("ticker")))
    say("Adding news (SEC 8-K), analyst consensus and competitors", 0.30)
    all_dates = pd.DatetimeIndex(sorted(set(train.index.get_level_values("date")) | {cutoff}))
    extra = extra_features(cache_dir, data, all_dates, tickers, px)
    train, today = train.join(extra), today.join(extra)
    sets = feature_sets(P.features_for(), list(extra.columns))
    member = today["in_index"].to_numpy(dtype=bool)
    scores, notes = {}, {}
    steps = list(FITTED) + ["chronos2", "chronos2_covariates", "mamba"]
    for i, name in enumerate(steps):
        say(f"Fitting {MODELS[name][0]}", 0.40 + 0.55 * i / len(steps))
        try:
            if name == "mamba":
                from .seq_model import MambaRanker, SequenceBank

                s = MambaRanker(SequenceBank(data), device).fit(train, sets["base"]).predict(today)
            elif name == "chronos2_covariates":
                from tradingagents.research.foundation import Chronos2Covariates

                etf = {t: (data.info.get(t) or {}).get("sector_etf") for t in tickers}
                s = Chronos2Covariates(data.close, data.volume, etf, device).fit(None, None, 20).predict(today)
            elif name == "chronos2":
                from tradingagents.research.foundation import Chronos2

                s = Chronos2(data.close[[t for t in tickers if t in data.close]], device).fit(None, None, 20).predict(today)
            else:
                fn, which = FITTED[name]
                s = fn(train, today, sets[which], device)
            scores[name] = np.asarray(s, dtype=float)
        except Exception as exc:  # one model failing never stops the others
            notes[name] = f"failed: {str(exc)[:200]}"
    for name, (col, sign) in BASELINES.items():
        scores[name] = sign * today[col].to_numpy(dtype=float)
    tick = list(today.index.get_level_values("ticker"))
    stocks_out = {t: {} for t in tick}
    ens = ensemble_scores(scores, member, cache_dir)
    for name, s in scores.items():
        pct = member_pct(s, member)
        is_return = MODELS[name][1] == "return"
        for j, t in enumerate(tick):
            if np.isfinite(pct[j]):
                stocks_out[t][name] = {"pct": round(float(pct[j]), 4),
                                       "forecast": round(float(s[j]), 5) if is_return and np.isfinite(s[j]) else None}
    if ens is not None:
        for j, t in enumerate(tick):
            if np.isfinite(ens["pct"][j]):
                stocks_out[t]["ensemble"] = {"pct": round(float(ens["pct"][j]), 4),
                                             "forecast": round(float(ens["expected_net"][j]), 5),
                                             "excess_vs_spy": round(float(ens["expected_excess"][j]), 5),
                                             "net_sd": round(float(ens["net_sd"][j]), 5),
                                             "net_q10": round(float(ens["net_q10"][j]), 5) if np.isfinite(ens["net_q10"][j]) else None}
    if ens is None:
        notes["ensemble"] = ("not available: no learned-weight bundle (experiments/stacking_study.py) or it scored "
                             "nothing; the daily decision falls back to the XGBoost ranker")
    payload = {"version": VERSION, "info_cutoff": str(cutoff.date()), "device": device, "train_rows": int(len(train)),
               "ensemble": ens and {"weights": ens["weights"], "evaluation": ens["evaluation"], "version": ens["version"]},
               "members": int(member.sum()), "watchlist": [t for t in watch if t in stocks_out],
               "models": {k: {"label": v[0], "kind": v[1]} for k, v in MODELS.items()}, "notes": notes,
               "tested": tested_record(), "stocks": stocks_out, "seconds": round(time.time() - t0),
               "artifacts": _artifacts(scores, device)}
    run_id = f"zoo-{cutoff.date()}-{int(time.time())}"
    with transaction(store.con):
        store.con.execute("INSERT INTO model_zoo_runs VALUES (?, ?, ?, ?)",
                          (run_id, now_utc(), payload["info_cutoff"], json.dumps(payload)))
    say("Done", 1.0)
    return {"id": run_id, **{k: payload[k] for k in ("info_cutoff", "members", "train_rows", "seconds", "notes")}}


ENSEMBLE = "ensemble.json"


def _artifacts(scores: dict, device: str) -> dict:
    """The pretrained weights a run used, as loaded (the revision is recorded, not assumed)."""
    if not any(k.startswith("chronos2") for k in scores):
        return {}
    from tradingagents.research.foundation import loaded_revision

    return {"chronos2": loaded_revision(device)}


def output_key(name: str) -> str | None:
    """Which ``model_outputs`` entry a model's ``forecast`` number is: they differ in unit and clock."""
    if name == "ensemble":
        return "zoo.ensemble_expected_net"
    if name.startswith("chronos2"):
        return "zoo.chronos_log_return"
    return "zoo.forecast_log_excess" if MODELS.get(name, ("", ""))[1] == "return" else None


def describe(run: dict) -> dict:
    """What a saved run covers, read from the run itself (so older runs are described too): each model's stock
    count, its output, its lifecycle, and whether the learned ensemble really scored this run."""
    stocks = run.get("stocks") or {}
    n = len(stocks)
    coverage = {k: sum(1 for m in stocks.values() if k in m) for k in run.get("models") or {}}
    ens = coverage.get("ensemble", 0)
    models = {}
    for k, meta in (run.get("models") or {}).items():
        section = MODELS.get(k, ("", "", "", "models"))[3]
        lifecycle = ("active_decision" if k == "ensemble" and ens else "fallback_decision" if k == "xgb_rank" and not ens
                     else "baseline" if section == "baselines" else "challenger")
        models[k] = {**meta, "coverage": coverage.get(k, 0), "of": n, "output": output_key(k), "lifecycle": lifecycle,
                     "failed": (run.get("notes") or {}).get(k)}
    return {"models": models, "stocks_scored": n,
            "ranking_source": ("ensemble" if ens else "xgb_rank"),
            "ranking_note": (f"the learned-weight ensemble scored {ens} of {n} stocks and orders the daily decision"
                             if ens else "the learned ensemble has no scores in this run, so the daily decision is "
                             "ranked by the internal XGBoost ranker (fallback)")}


def ensemble_bundle(cache_dir) -> dict | None:
    """The learned weights and maps written by ``experiments/stacking_study.py``, when it has been run."""
    p = Path(cache_dir) / "decision_models" / ENSEMBLE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def ensemble_scores(scores: dict[str, np.ndarray], member: np.ndarray, cache_dir) -> dict | None:
    """The weighted ensemble: each member model's within-members rank (-0.5 to 0.5) times its learned weight, then
    ranked among members; its expected net P&L and excess over SPY from the bundle's out-of-sample maps."""
    from .adapters import apply_map, member_pct

    b = ensemble_bundle(cache_dir)
    if not b or any(m not in scores for m in b["members"]):
        return None
    z = sum(w * (member_pct(scores[m], member) - 0.5) for m, w in b["weights"].items())
    pct = member_pct(np.asarray(z, dtype=float), member)
    return {"pct": pct, "expected_net": apply_map(b["map_net"], pct), "expected_excess": apply_map(b["map_excess"], pct),
            "net_sd": apply_map(b["map_net"], pct, "sd"), "net_q10": apply_map(b["map_net"], pct, "q10"),
            "weights": b["weights"], "evaluation": b.get("evaluation"), "version": b.get("version")}


def latest(con) -> dict | None:
    try:
        row = con.execute("SELECT id, created_at, payload_json FROM model_zoo_runs ORDER BY created_at DESC LIMIT 1"
                          ).fetchone()
    except Exception:   # an older database without the table
        return None
    if not row:
        return None
    return {"id": row["id"], "created_at": row["created_at"], **json.loads(row["payload_json"]),
            "tested": tested_record() or json.loads(row["payload_json"]).get("tested", {})}
