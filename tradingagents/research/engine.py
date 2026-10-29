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
from .compute import choose_device, cuda_usable, run_on
from .forecast import FEATURE_SCHEMA, Probabilistic, bagged_mean, forecasts_for
from .models import XGB, ElasticNet, Momentum, TrainMean, VolPrior, Zero
from .prices import MARKET, bars

MODEL_VERSION = "forecast-v1"
SEED = 7
# What each universe's runs train on. ETF runs feed the plan; stock runs back the ticker page's
# forecast for single companies, and are survivorship-biased (today's S&P 500 members).
UNIVERSES = {"etf": "ETFs", "stock": "S&P 500 stocks"}
FLAT = "this run gives every ticker the same forecast (its model kept only market-wide inputs), so it ranks nothing"


def ranks(run: dict) -> bool:
    """Whether a saved run tells its tickers apart: one that gives them all the same forecast does not."""
    means = [f["mean"] for f in run.get("forecasts") or []]
    return len(means) < 2 or float(np.ptp(means)) >= 1e-9


CHRONOS = "chronos2"          # Chronos-2 on its own: pretrained, nothing fitted
STACKED = "+chronos"          # suffix: a model given Chronos-2's forecast as two more inputs


def factory(name: str, device: str = "cpu"):
    name = name.removesuffix(STACKED)
    if name == CHRONOS:
        return lambda: None   # built in run_forecast, which has the prices it needs
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


def _same_run(con, model: str, horizon: int, universe: str, cutoff, input_key: str) -> dict | None:
    """The saved run made from exactly these inputs, in the shape ``run_forecast`` returns, or None."""
    row = con.execute("SELECT * FROM forecast_runs WHERE model = ? AND horizon = ? AND universe = ? AND info_cutoff = ?"
                      " ORDER BY created_at DESC LIMIT 1", (model, horizon, universe, str(cutoff.date()))).fetchone()
    if not row or json.loads(row["config_json"]).get("input_key") != input_key:
        return None
    rows = con.execute("SELECT * FROM forecasts WHERE run_id = ?", (row["id"],)).fetchall()
    return {"run_id": row["id"], "info_cutoff": row["info_cutoff"], "model": model, "horizon": horizon,
            "backend": row["backend"], "seconds": row["seconds"], "peak_memory_mb": row["peak_memory_mb"],
            "calibration": json.loads(row["calibration_json"]), "reused": True,
            "forecasts": [{k: r[k] for k in r.keys() if k != "run_id"} for r in rows]}


def run_forecast(con, cache_dir: Path, tickers: list[str], horizon: int = 20, model: str = "elastic_net",
                 universe: str = "etf", device_setting: str = "auto", bags: int = 5,
                 artifact_dir: Path | None = None, cash: str = "BIL", force: bool = False) -> dict:
    """Train, forecast the latest final close, and save the run; returns the run and its forecasts. A run made
    from the same inputs (data to the same close with the same prices, tickers, model, horizon and code) is
    returned as saved, with ``reused``, unless ``force``."""
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version

    started = time.time()
    tracing = not tracemalloc.is_tracing()
    if tracing:
        tracemalloc.start()
    tickers = [t for t in dict.fromkeys(tickers) if t != cash]
    b = bars([*tickers, cash, *MARKET], cache_dir)
    close = b["Close"][[*tickers, cash]]
    close = close[close.index >= pd.Timestamp("2004-01-01")]
    high, low, volume = (b[f][[*tickers, cash]].reindex(close.index) for f in ("High", "Low", "Volume"))
    market = {m: b["Close"][m].dropna() for m in MARKET if b["Close"][m].notna().any()}
    cutoff = close.index[-1]
    from tradingagents.stocks import input_keys as K

    input_key = K.key({"tickers": sorted(tickers), "cash": cash, "model": model, "horizon": horizon, "bags": bags,
                       "universe": universe, "cutoff": str(cutoff.date()), "prices": K.prices_hash(close, cutoff),
                       "version": MODEL_VERSION, "schema": FEATURE_SCHEMA,
                       "code": K.code_hash("research/engine.py", "research/features.py", "research/forecast.py",
                                           "research/models.py", "research/foundation.py")})
    if not force:
        same = _same_run(con, model, horizon, universe, cutoff, input_key)
        if same is not None:
            if tracing:
                tracemalloc.stop()
            return same
    dates = F.decision_dates(close.index, start="2006-01-01", every=min(horizon, 20))
    dates = dates.union(pd.DatetimeIndex([cutoff]))
    panel = F.build_panel(close, high, low, volume, market, cash, dates, horizon=horizon)
    features = list(F.PRICE_FEATURES) + list(F.MARKET_FEATURES)
    chronos_device = "cuda" if cuda_usable() else "cpu"
    if model.endswith(STACKED):
        # Chronos-2's forecast for every training row and today's: about a minute on the GPU for the
        # ETFs; for 500 stocks it would be most of an hour, so stock runs do not offer it.
        if universe != "etf":
            raise ValueError("Chronos-2 inputs are offered for the ETF universe only: for 500 stocks they "
                             "would take most of an hour per run")
        from .foundation import FEATURES as CHRONOS_FEATURES
        from .foundation import chronos_features

        panel = panel.join(chronos_features(close[tickers], panel, horizon, chronos_device))
        features += CHRONOS_FEATURES
    if model == CHRONOS:
        bags = 0   # nothing is fitted, so refits on resampled data would all be the same
    train = panel[(pd.to_datetime(panel["label_end"]) <= cutoff) & panel["label"].notna()].sort_index(level="date")
    rows = panel.xs(cutoff, level="date", drop_level=False)
    if rows.empty:
        raise ValueError(f"no ticker has a year of prices at {cutoff.date()}, so nothing can be forecast")
    device, why = choose_device(device_setting, len(train))
    if model == CHRONOS and chronos_device == "cuda" and device_setting != "cpu":
        device, why = "cuda", "Chronos-2 is a neural network: the GPU is much faster"
    artifact_dir = Path(artifact_dir or Path(cache_dir) / "forecast_models")

    def fit(dev):
        if model == CHRONOS:
            from .foundation import Chronos2

            def make():
                return Chronos2(close[tickers], device=dev)
        else:
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
    _, peak = tracemalloc.get_traced_memory()   # in a parallel batch, the peak of all runs so far
    if tracing:
        tracemalloc.stop()
    seconds = time.time() - started
    result = forecasts_for(prob, rows, str(cutoff.date()), MODEL_VERSION, used, seconds, spread, calibration, flags)
    if len(result) > 1 and np.ptp([f.mean for f in result]) < 1e-9:
        # Early stopping can leave a tree model with market-wide splits only: one number for everyone.
        for f in result:
            f.flags.append(FLAT)
    config = {"input_key": input_key, "model": model, "horizon": horizon, "features": features, "bags": bags,
              "seed": SEED,
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


COMBINED = ("mean", "median", "q05", "q95", "p_gain")


def model_weight(run: dict) -> tuple[float, str]:
    """A run's weight in the combined forecast before normalizing, and why.

    Its walk-forward IC when its trial found real ranking skill (IC above 0, Holm p under
    0.05); otherwise none. Chronos-2 gets none either way: its pretraining may include
    these prices, so its trial scores could be flattered (its live record will tell). A run
    that gives every ticker the same forecast ranks nothing and gets none.
    """
    cal = run.get("calibration") or {}
    if isinstance(cal, str):
        cal = json.loads(cal)
    if not ranks(run):
        return 0.0, "ranks nothing"
    if run["model"] == CHRONOS:
        return 0.0, "pretraining may include these prices; waits on its live record"
    if not cal.get("trial"):
        return 0.0, "no walk-forward trial"
    ic, p = cal.get("mean_ic"), cal.get("p_value_holm")
    if ic is None or ic <= 0 or p is None or p >= 0.05:
        return 0.0, f"trial {cal['trial']} found no significant skill"
    return float(ic), f"IC {ic:.3f} (trial {cal['trial']}, p {p:.3f})"


def combine_runs(runs: list[dict]) -> dict:
    """Every model's forecast for each ticker, and their evidence-weighted combination.

    Weights come from ``model_weight``, normalized over the models that forecast each ticker.
    When no model has significant evidence, the models that rank are weighted equally and
    the result says so. The combined 5% and 95% are weighted averages of the models'
    (quantile averaging); ``spread`` is how far the models' means are apart (their weighted
    standard deviation), a measure of disagreement the single-model columns cannot show.
    """
    weights = {r["id"]: model_weight(r) for r in runs}
    basis = "evidence"
    if not any(w for w, _ in weights.values()):
        basis = "equal"
        weights = {r["id"]: ((1.0, "equal weight: no model has significant evidence") if ranks(r) and r["model"] != CHRONOS
                             else (0.0, weights[r["id"]][1])) for r in runs}
    by_ticker: dict[str, list[tuple[dict, dict]]] = {}
    for r in runs:
        for f in r["forecasts"]:
            by_ticker.setdefault(f["ticker"], []).append((r, f))
    rows = []
    for ticker, pairs in by_ticker.items():
        total = sum(weights[r["id"]][0] for r, _ in pairs)
        models = []
        for r, f in sorted(pairs, key=lambda x: -weights[x[0]["id"]][0]):
            w = weights[r["id"]][0] / total if total else 0.0
            models.append({"model": r["model"], "run_id": r["id"], "weight": w, "why": weights[r["id"]][1],
                           "estimation_sd": f.get("estimation_sd"),
                           "flags": json.loads(f.get("flags_json") or "[]") if "flags_json" in f else f.get("flags", []),
                           **{k: f[k] for k in COMBINED}})
        combined = None
        if total:
            used = [m for m in models if m["weight"] > 0]
            combined = {k: sum(m["weight"] * m[k] for m in used) for k in COMBINED}
            sds = [m for m in used if m["estimation_sd"] is not None]
            combined["estimation_sd"] = (sum(m["weight"] * m["estimation_sd"] for m in sds) / sum(m["weight"] for m in sds)
                                         if sds else None)
            combined["spread"] = float(np.sqrt(sum(m["weight"] * (m["mean"] - combined["mean"]) ** 2 for m in used)))
            combined["models"] = len(used)
        rows.append({"ticker": ticker, "combined": combined, "models": models})
    rows.sort(key=lambda x: -(x["combined"]["mean"] if x["combined"] else float("-inf")))
    return {"basis": basis, "rows": rows,
            "weights": [{"model": r["model"], "run_id": r["id"], "raw": weights[r["id"]][0], "why": weights[r["id"]][1]}
                        for r in runs]}


def runs_to_combine(con, horizon: int, universe: str = "etf") -> list[dict]:
    """The newest run of each model at ``horizon`` for the newest information cutoff."""
    row = con.execute("SELECT MAX(info_cutoff) FROM forecast_runs WHERE horizon = ? AND universe = ?",
                      (horizon, universe)).fetchone()
    if not row or not row[0]:
        return []
    ids = con.execute("SELECT model, id FROM forecast_runs WHERE horizon = ? AND universe = ? AND info_cutoff = ?"
                      " ORDER BY created_at", (horizon, universe, row[0])).fetchall()
    newest = {r["model"]: r["id"] for r in ids}
    out = []
    for rid in newest.values():
        run = load_run(con, rid)
        run["calibration"] = json.loads(run.get("calibration_json") or "{}")
        out.append(run)
    return out


def compare_runs(runs: list[dict], top: int = 3) -> list[dict]:
    """One row per run of a batch (``run_forecast`` results), for a side-by-side table.

    Each row has the walk-forward evidence behind the model (IC with its 95% range, the
    Holm p-value, within-date calibration, coverage, net return of the top fifth), whether
    it tells tickers apart at all, its highest forecasts, and how far it agrees with the
    batch's other models at the same horizon: the mean rank correlation of their forecasts.
    """
    means = {r["run_id"]: pd.Series({f["ticker"]: f["mean"] for f in r["forecasts"]}) for r in runs}
    rows = []
    for r in runs:
        c = r.get("calibration") or {}
        mine = means[r["run_id"]]
        others = [means[o["run_id"]] for o in runs if o["run_id"] != r["run_id"] and o["horizon"] == r["horizon"]]
        agree = [mine.corr(o.reindex(mine.index), method="spearman") for o in others]
        agree = [a for a in agree if a == a]
        flat = len(mine) > 1 and float(np.ptp(mine.to_numpy())) < 1e-9
        rows.append({"run_id": r["run_id"], "model": r["model"], "horizon": r["horizon"], "backend": r["backend"],
                     "seconds": r["seconds"], "ranks": not flat, "trial": c.get("trial"), "mean_ic": c.get("mean_ic"),
                     "ic_ci": c.get("ic_ci"), "p_value_holm": c.get("p_value_holm"),
                     "calibration_xs": c.get("calibration_xs"), "coverage_90": c.get("coverage_90"),
                     "top_minus_all_net": c.get("top_minus_all_net"),
                     "top": [] if flat else [{"ticker": t, "mean": float(v)} for t, v in mine.nlargest(top).items()],
                     "spread": float(mine.std()) if len(mine) > 1 else None,
                     "agreement": float(np.mean(agree)) if agree else None})
    return sorted(rows, key=lambda x: (x["horizon"], x["model"]))


def live_record(con, cache_dir: Path, model: str, horizon: int, universe: str = "etf", cash: str = "BIL") -> dict:
    """A model's saved forecasts scored against what happened: one run per information cutoff
    (the newest), counted once ``horizon`` sessions have closed since. Per date, the rank
    correlation (IC) of forecast with outcome, the outcome being the log return above cash;
    and the return of its top fifth above the average. This record cannot have been seen in
    any training, Chronos-2's included."""
    rows = con.execute("SELECT id, info_cutoff FROM forecast_runs WHERE model = ? AND horizon = ? AND universe = ?"
                       " ORDER BY created_at", (model, horizon, universe)).fetchall()
    by_cutoff = {r["info_cutoff"]: r["id"] for r in rows}
    if not by_cutoff:
        return {"model": model, "horizon": horizon, "made": 0, "resolved": 0, "dates": []}
    runs = {c: load_run(con, rid) for c, rid in by_cutoff.items()}
    tickers = sorted({f["ticker"] for r in runs.values() for f in r["forecasts"]})
    close = bars([*tickers, cash], cache_dir)["Close"]
    sessions = close.index
    out = []
    for cutoff, run in sorted(runs.items()):
        at = sessions.searchsorted(pd.Timestamp(cutoff))
        if at >= len(sessions) or sessions[at] != pd.Timestamp(cutoff) or at + horizon >= len(sessions):
            continue
        end = sessions[at + horizon]
        start = sessions[at]
        realized = np.log(close.loc[end] / close.loc[start])
        above = realized.drop(cash) - (realized.get(cash, 0.0) if pd.notna(realized.get(cash)) else 0.0)
        fc = pd.Series({f["ticker"]: f["mean"] for f in run["forecasts"]}).reindex(above.index).dropna()
        got = above.reindex(fc.index).dropna()
        fc = fc.reindex(got.index)
        if len(got) < 5:
            continue
        top = fc.nlargest(max(1, len(fc) // 5)).index
        out.append({"date": cutoff, "ended": str(end.date()), "ic": float(fc.corr(got, method="spearman")),
                    "top_minus_all": float(got.loc[top].mean() - got.mean()), "n": int(len(got))})
    ics = [d["ic"] for d in out if d["ic"] == d["ic"]]
    return {"model": model, "horizon": horizon, "made": len(runs), "resolved": len(out),
            "first": min(runs), "mean_ic": float(np.mean(ics)) if ics else None,
            "hit": float(np.mean([i > 0 for i in ics])) if ics else None,
            "top_minus_all": float(np.mean([d["top_minus_all"] for d in out])) if out else None,
            "dates": out}


def latest_run(con, horizon: int | None = None, model: str | None = None, universe: str = "etf") -> dict | None:
    """The newest run of a universe: ETF runs by default, the ones the plan may use.

    The daily Chronos-2 record is left out unless ``model`` asks for it: it is kept to
    score Chronos-2 on prices it cannot have seen, not to stand in for the chosen model.
    """
    sql, args = "SELECT id FROM forecast_runs WHERE universe = ?", [universe]
    if not model:
        sql += f" AND model != '{CHRONOS}'"
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


def forecast_one(con, cache_dir: Path, run: dict, symbol: str, cash: str = "BIL") -> dict | None:
    """A saved run's model applied to one more ticker, at the run's own information cutoff.

    The ticker's features are ranked against the run's own funds on that date, as in
    training, so its forecast is comparable with theirs. A ticker outside the run's
    universe (a single stock, when the model was trained on ETFs) is flagged: the model
    never saw anything like it, and its forecast deserves less trust. None when the
    ticker has no year of prices at the cutoff.
    """
    config = run.get("config") or json.loads(run.get("config_json") or "{}")
    tickers = [t for t in config.get("tickers", []) if t != symbol]
    cutoff = pd.Timestamp(run["info_cutoff"])
    b = bars([*tickers, symbol, cash, *MARKET], cache_dir)
    close = b["Close"][[*tickers, symbol, cash]]
    close = close[(close.index >= pd.Timestamp("2004-01-01")) & (close.index <= cutoff)]
    if symbol not in close or close[symbol].dropna().empty or close.index[-1] != cutoff:
        return None
    high, low, volume = (b[f][[*tickers, symbol, cash]].reindex(close.index) for f in ("High", "Low", "Volume"))
    market = {m: b["Close"][m].loc[:cutoff].dropna() for m in MARKET if b["Close"][m].notna().any()}
    panel = F.build_panel(close, high, low, volume, market, cash, pd.DatetimeIndex([cutoff]), horizon=run["horizon"])
    if (cutoff, symbol) not in panel.index:
        return None
    rows = panel.loc[[(cutoff, symbol)]]
    if run["model"].endswith(STACKED) or run["model"] == CHRONOS:
        from .foundation import chronos_features

        rows = rows.join(chronos_features(close[[symbol]], rows, run["horizon"], "cuda" if cuda_usable() else "cpu"))
    model = load_model(run)
    if run["model"] == CHRONOS:
        from .foundation import Chronos2

        model.model = Chronos2(close[[symbol]], device="cuda" if cuda_usable() else "cpu").fit(None, [], run["horizon"])
    row = model.predict(rows).iloc[0]
    flags = [] if symbol in config.get("tickers", []) else [
        f"this model was trained on {UNIVERSES.get(run['universe'], run['universe'])} and {symbol} was not among them"
        + ("; treat it as a forecast for an unfamiliar kind of asset" if run["universe"] == "etf" else "")]
    return {"horizon": run["horizon"], "model": run["model"], "universe": run["universe"],
            "info_cutoff": run["info_cutoff"], "on_demand": True,
            "flags": flags, **{k: float(row[k]) for k in ("mean", "median", "q05", "q95", "p_gain")},
            "estimation_sd": None}
