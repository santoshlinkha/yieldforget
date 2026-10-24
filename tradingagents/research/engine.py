"""Today's forecasts: train on all the history that has resolved, forecast the latest close, save it.

A run:

1. Builds the leakage-safe panel (see ``features``) from the local price store.
2. Trains the chosen model (through ``forecast.Probabilistic``) on every row whose
   label had **ended** by the information cutoff, the last final close.
3. Forecasts that cutoff date for each ticker: mean, median, 5th and 95th
   percentiles, probability of a gain, and the bootstrap estimation spread.
4. Attaches the latest walk-forward trial of the same model, horizon and universe,
   which is the evidence for or against it, and flags a forecast with no such trial.
5. Saves the run (model file, its hash, versions, seeds, backend, time and peak
   memory) and every forecast in ``portfolio.sqlite3``. Runs are never changed;
   a new run is a new record.

Nothing here calls a paid model. The run is local numerical work.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import tracemalloc
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .compute import choose_device, run_on
from .forecast import FEATURE_SCHEMA, Probabilistic, bagged_mean, forecasts_for
from .models import XGB, ElasticNet, Momentum, TrainMean, VolPrior, Zero
from .prices import MARKET, bars

MODEL_VERSION = "forecast-v1"
SEED = 7


def factory(name: str, device: str = "cpu"):
    table = {"zero": Zero, "train_mean": TrainMean, "vol_prior": VolPrior, "momentum": Momentum,
             "elastic_net": ElasticNet,
             "xgb": lambda: XGB(device=device, seed=SEED), "xgb_rank": lambda: XGB("rank", device=device, seed=SEED),
             "xgb_quantile": lambda: XGB("quantile", device=device, seed=SEED)}
    if name not in table:
        raise ValueError(f"unknown model {name}; one of {', '.join(table)}")
    return table[name]


def latest_trial(con, model: str, horizon: int, universe: str) -> dict | None:
    for row in con.execute("SELECT * FROM research_trials WHERE horizon = ? AND universe = ? ORDER BY id DESC",
                           (horizon, universe)):
        scores = json.loads(row["scores_json"])
        if model in scores:
            s = scores[model]
            return {"trial": row["id"], "name": row["name"], "first_test": row["first_test"],
                    "last_test": row["last_test"], "mean_ic": s["mean_ic"], "ic_ci": s["ic_ci"],
                    "r2_oos": s["r2_oos"], "calibration": s["calibration"], "coverage_90": s["coverage_90"],
                    "calibration_xs": s.get("calibration_xs"), "timing_corr": s.get("timing_corr"),
                    "brier": s["brier"], "brier_null": s["brier_null"], "p_value_holm": s["p_value"],
                    "top_minus_all_net": s["top_minus_all_net"]}
    return None


def run_forecast(con, cache_dir: Path, tickers: list[str], horizon: int = 20, model: str = "elastic_net",
                 universe: str = "etf", device_setting: str = "auto", bags: int = 5,
                 artifact_dir: Path | None = None, cash: str = "BIL") -> dict:
    """Train, forecast the latest final close, and save the run; returns the run and its forecasts."""
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version

    started = time.time()
    tracemalloc.start()
    tickers = [t for t in dict.fromkeys(tickers) if t != cash]
    b = bars([*tickers, cash, *MARKET], cache_dir)
    close = b["Close"][[*tickers, cash]]
    close = close[close.index >= pd.Timestamp("2004-01-01")]
    high, low, volume = (b[f][[*tickers, cash]].reindex(close.index) for f in ("High", "Low", "Volume"))
    market = {m: b["Close"][m].dropna() for m in MARKET if b["Close"][m].notna().any()}
    cutoff = close.index[-1]
    dates = F.decision_dates(close.index, start="2006-01-01", every=min(horizon, 20))
    dates = dates.union(pd.DatetimeIndex([cutoff]))
    panel = F.build_panel(close, high, low, volume, market, cash, dates, horizon=horizon)
    features = list(F.PRICE_FEATURES) + list(F.MARKET_FEATURES)
    train = panel[(pd.to_datetime(panel["label_end"]) <= cutoff) & panel["label"].notna()].sort_index(level="date")
    rows = panel.xs(cutoff, level="date", drop_level=False)
    if rows.empty:
        raise ValueError(f"no ticker has a year of prices at {cutoff.date()}, so nothing can be forecast")
    device, why = choose_device(device_setting, len(train))
    artifact_dir = Path(artifact_dir or Path(cache_dir) / "forecast_models")

    def fit(dev):
        make = factory(model, dev)
        prob = Probabilistic(make).fit(train, features, horizon)
        spread = bagged_mean(make, train, rows, features, horizon, bags=bags, seed=SEED) if bags else None
        return prob, spread

    (prob, spread), used, note = run_on(device, artifact_dir, lambda: fit("cpu"), lambda: fit("cuda"))
    calibration = latest_trial(con, model, horizon, universe) or {}
    flags: dict[str, list[str]] = {}
    for (_, t), r in rows.iterrows():
        f = []
        if not calibration:
            f.append("no walk-forward trial backs this model at this horizon")
        if pd.isna(r.get("mom_12_1")):
            f.append("under a year of prices")
        if t not in train.index.get_level_values("ticker"):
            f.append("not in the training data")
        flags[t] = f
    run_id = uuid.uuid4().hex
    folder = artifact_dir / run_id
    folder.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(prob)
    (folder / "model.pkl").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    seconds = time.time() - started
    result = forecasts_for(prob, rows, str(cutoff.date()), MODEL_VERSION, used, seconds, spread, calibration, flags)
    config = {"model": model, "horizon": horizon, "features": features, "bags": bags, "seed": SEED,
              "train_mean_label": float(train["label"].clip(*train["label"].quantile([0.01, 0.99])).mean()),
              "device_setting": device_setting, "device_reason": why, "gpu_note": note, "tickers": tickers,
              "train_rows": len(train), "fit": prob.describe()}
    with transaction(con):
        con.execute("INSERT INTO forecast_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, now_utc(), model, MODEL_VERSION, horizon, str(cutoff.date()), universe, FEATURE_SCHEMA,
                     prob.train_start, prob.train_end, json.dumps(config, default=str), json.dumps(calibration),
                     used, seconds, peak / 2 ** 20, code_version(), str(folder / "model.pkl"), sha))
        for f in result:
            con.execute("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (run_id, f.ticker, f.mean, f.median, f.q05, f.q95, f.p_gain, f.sigma, f.estimation_sd,
                         f.cash_h, json.dumps(f.flags)))
    return {"run_id": run_id, "info_cutoff": str(cutoff.date()), "model": model, "horizon": horizon,
            "backend": used, "device_reason": why, "gpu_note": note, "seconds": round(seconds, 2),
            "peak_memory_mb": round(peak / 2 ** 20, 1), "calibration": calibration,
            "forecasts": [f.as_dict() for f in result]}


def load_run(con, run_id: str) -> dict:
    row = con.execute("SELECT * FROM forecast_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(run_id)
    run = dict(row)
    run["forecasts"] = [dict(r) for r in con.execute("SELECT * FROM forecasts WHERE run_id = ? ORDER BY mean DESC",
                                                     (run_id,))]
    return run


def latest_run(con, horizon: int | None = None, model: str | None = None) -> dict | None:
    sql, args = "SELECT id FROM forecast_runs WHERE 1=1", []
    if horizon:
        sql += " AND horizon = ?"
        args.append(horizon)
    if model:
        sql += " AND model = ?"
        args.append(model)
    row = con.execute(sql + " ORDER BY created_at DESC LIMIT 1", args).fetchone()
    return load_run(con, row["id"]) if row else None


def load_model(run: dict) -> Probabilistic:
    """The saved model of a run, checked against its hash."""
    blob = Path(run["artifact_path"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != run["artifact_sha256"]:
        raise ValueError(f"model file of run {run['id']} does not match its hash")
    return pickle.loads(blob)  # noqa: S301 (our own file, verified by hash)


def expected_returns(run: dict, tickers: list[str], conservative: bool = True, k: float = 1.0,
                     mode: str = "relative") -> dict[str, float]:
    """Per-ticker expected log return above cash over the run's horizon, for the optimizer.

    ``relative`` (the default): no model here has shown it can time the market (the
    walk-forward's ``timing_corr`` is near zero or negative), but some rank assets
    well, so the forecast moves money *between* assets only. Each gets the average
    return above cash seen in training, plus its forecast's difference from the
    day's average forecast, scaled by the backing trial's within-date calibration
    slope (kept between 0 and 1). ``absolute`` uses the forecast itself, scaled by the
    pooled slope. Either way the result is clipped to two predictive sigmas, so one
    extreme prediction cannot take over the plan. With ``conservative`` it is
    reduced by ``k`` times the estimation spread. A run no trial backs gets slope 0:
    its forecasts then move nothing. A ticker with no forecast gets none, which the
    caller treats as "do not buy".
    """
    cal = run.get("calibration") or {}
    if isinstance(cal, str):
        cal = json.loads(cal)
    key = "calibration_xs" if mode == "relative" else "calibration"
    shrink = float(np.clip(cal.get(key) or 0.0, 0.0, 1.0)) if cal else 0.0
    chosen = [f for f in run["forecasts"] if f["ticker"] in tickers]
    if not chosen:
        return {}
    centre = float(np.mean([f["mean"] for f in chosen]))
    config = run.get("config") or json.loads(run.get("config_json") or "{}")
    premium = float(run.get("train_mean_label") or config.get("train_mean_label") or 0.0)
    out = {}
    for f in chosen:
        mu = premium + shrink * (f["mean"] - centre) if mode == "relative" else shrink * f["mean"]
        mu = float(np.clip(mu, -2 * f["sigma"], 2 * f["sigma"]))
        if conservative and f.get("estimation_sd"):
            mu -= k * f["estimation_sd"]
        out[f["ticker"]] = mu
    return out
