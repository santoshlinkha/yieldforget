"""A stock batch: many tickers, one information cutoff, every forecast saved with how it was made.

1. The tickers (pasted, CSV, watchlist, stock holdings) are checked: stocks only.
2. Shared data is loaded once (``data.load``); a stock whose data fails is recorded
   and the rest go on.
3. Per horizon (5, 20, 60 sessions): the panel is built on the S&P 500 universe plus
   the requested stocks, the models are fitted on every row whose outcome had ended
   by the cutoff (or loaded, when the same cutoff, settings and code were already
   fitted today: models are saved by that key), and every stock at the cutoff is
   forecast, so each requested stock also gets its percentile among the S&P 500.
4. News is collected for the requested stocks and summarized (``news``).
5. Each forecast carries its evidence (the latest ``stock_study`` trial for that
   horizon: development and final-test scores) and its reasons to be trusted less or
   not at all; a stock with no usable row at the cutoff **abstains** with the reason.
6. The run and every forecast are saved (``stock_runs``, ``stock_forecasts``),
   append-only, with code, feature, threshold and model versions and data provenance.

Nothing here calls a paid service. Numbers come from fitted models only.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import panel as P
from .targets import EVENTS, TREND_STATES, Thresholds

MODEL_VERSION = "stock-v1"
RETURN_MODELS = ("elastic_net", "xgb_quantile")


@dataclass
class BatchConfig:
    horizons: tuple[int, ...] = (5, 20, 60)
    groups: tuple[str, ...] = P.DEFAULT_GROUPS
    thresholds: Thresholds = field(default_factory=Thresholds)
    device: str = "auto"
    news: bool = True
    workers: int = 4

    def key(self) -> dict:
        return {"groups": list(self.groups), "thresholds": asdict(self.thresholds), "device": self.device}


class Progress:
    """Per-stock and per-stage progress the page (or the command line) can poll."""

    def __init__(self, callback=None):
        self.stage, self.tickers, self.callback = "starting", {}, callback

    def __call__(self, stage, **info):
        self.stage = stage + (f" ({info['done']}/{info['count']})" if "done" in info else "")
        if self.callback:
            self.callback(self.stage)

    def ticker(self, t, status, note=None):
        self.tickers[t] = {"status": status, **({"note": note} if note else {})}


def _fit(train: pd.DataFrame, feats: list[str], horizon: int, device: str) -> dict:
    from tradingagents.research.forecast import Probabilistic
    from tradingagents.research.models import XGB, ElasticNet

    from .models import Relative, Severity, Trend, XGBEvent

    fitted = {"elastic_net": Probabilistic(ElasticNet, name="elastic_net").fit(train, feats, horizon),
              "xgb_quantile": Probabilistic(lambda: XGB("quantile", device=device), name="xgb_quantile")
              .fit(train, feats, horizon),
              "relative": Relative(device).fit(train, feats),
              "trend": Trend("xgb", device).fit(train, feats),
              "severity": Severity(device).fit(train, feats)}
    for target in EVENTS:
        fitted[f"event:{target}"] = XGBEvent(device).fit(train, feats, target)
    rows = train[train["up_break"].notna()]
    fitted["base_rates"] = {t: float(rows[t].mean()) for t in EVENTS} | {
        "failed_up_given_break": float(rows.loc[rows["up_break"] == 1, "failed_up"].mean()),
        "failed_down_given_break": float(rows.loc[rows["down_break"] == 1, "failed_down"].mean()),
        "trend_mix": np.bincount(rows["trend_state"].dropna().astype(int), minlength=4).tolist()}
    return fitted


def models_for(panel: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, config: BatchConfig, folder: Path,
               device: str) -> tuple[dict, dict]:
    """The fitted models for this cutoff and settings: loaded when already saved, else fitted and saved."""
    from tradingagents.ledger.recommendations import code_version

    feats = P.features_for(config.groups)
    key = hashlib.sha256(json.dumps({"cutoff": str(cutoff.date()), "h": horizon, "features": feats,
                                     "config": config.key(), "version": MODEL_VERSION, "code": code_version()},
                                    sort_keys=True, default=str).encode()).hexdigest()[:20]
    path = Path(folder) / f"{key}.pkl"
    if path.exists():
        blob = path.read_bytes()
        return pickle.loads(blob), {"key": key, "sha256": hashlib.sha256(blob).hexdigest(), "reused": True}  # noqa: S301
    train = panel[panel["in_index"] & panel["label"].notna() & (pd.to_datetime(panel["label_end"]) <= cutoff)]
    t = time.time()
    fitted = _fit(train, feats, horizon, device)
    fitted["meta"] = {"train_rows": len(train), "train_dates": int(train.index.get_level_values("date").nunique()),
                      "train_start": str(train.index.get_level_values("date").min().date()),
                      "train_end": str(pd.to_datetime(train["label_end"]).max().date()),
                      "seconds": round(time.time() - t, 1), "features": feats}
    blob = pickle.dumps(fitted)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return fitted, {"key": key, "sha256": hashlib.sha256(blob).hexdigest(), "reused": False}


def evidence(con, horizon: int) -> dict:
    """What the latest stock study found at this horizon, per target: development and final test."""
    row = con.execute("SELECT id, scores_json FROM research_trials WHERE name = ? ORDER BY id DESC LIMIT 1",
                      (f"stock_study:h{horizon}",)).fetchone()
    if not row:
        return {"trial": None, "status": "untested"}
    s = json.loads(row["scores_json"])
    out = {"trial": row["id"], "choices": s.get("choices", {})}
    for part in ("development", "final_test"):
        p = s.get("parts", {}).get(part) or {}
        r = (p.get("returns") or {}).get("ensemble") or {}
        out[part] = {"return_ic": r.get("mean_ic"), "return_ic_ci": r.get("ic_ci"), "return_p_holm": r.get("p_value_holm"),
                     "relative_ic": (p.get("relative_ic") or {}).get("mean"),
                     "events": {t: {k: (v.get("xgb") or {}).get(k) for k in ("skill", "skill_ci", "auc")}
                                for t, v in (p.get("events") or {}).items()},
                     "trend_skill": (p.get("trend") or {}).get("xgb", {}).get("skill"),
                     "portfolio_minus_all": (p.get("portfolio") or {}).get("model_minus_all")}
    return out


def _status(ev: dict, part: str, key: str) -> str:
    """'backed' when the final test confirms skill (interval above zero), 'weak' when only development
    does, 'untested' when no study has run."""
    if ev.get("trial") is None:
        return "untested"
    test, dev = ev.get("final_test") or {}, ev.get("development") or {}
    if key == "return":
        ci = test.get("return_ic_ci") or [None, None]
        return "backed" if ci[0] is not None and ci[0] > 0 else "weak" if (dev.get("return_ic") or 0) > 0 else "none"
    t = (test.get("events") or {}).get(key) or {}
    ci = t.get("skill_ci") or [None, None]
    return "backed" if ci[0] is not None and ci[0] > 0 else "weak" if ((dev.get("events") or {}).get(key) or {}).get("skill", 0) > 0 \
        else "none"


def forecast_rows(fitted: dict, rows: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Every forecast figure for ``rows`` (one per stock at the cutoff)."""
    out = pd.DataFrame(index=rows.index.get_level_values("ticker"))
    per = {}
    for m in RETURN_MODELS:
        got = fitted[m].predict(rows)
        got.index = out.index
        per[m] = got
    w = {m: weights.get(m, 0.0) for m in RETURN_MODELS}
    if not sum(w.values()):
        w = {m: 1 / len(RETURN_MODELS) for m in RETURN_MODELS}
    tot = sum(w.values())
    for c in ("mean", "median", "q05", "q95", "p_gain"):
        out[c] = sum(w[m] / tot * per[m][c] for m in RETURN_MODELS)
    for m in RETURN_MODELS:
        out[f"{m}_mean"] = per[m]["mean"]
    out["model_spread"] = np.sqrt(sum(w[m] / tot * (per[m]["mean"] - out["mean"]) ** 2 for m in RETURN_MODELS))
    out["relative"] = fitted["relative"].predict(rows)
    out["cost_rt"] = rows["cost_rt"].to_numpy()
    out["net"] = out["mean"] - out["cost_rt"]
    p = fitted["trend"].predict_proba(rows)
    for k, s in enumerate(TREND_STATES):
        out[f"trend_{s}"] = p[:, k]
    for t in EVENTS:
        out[f"p_{t}"] = fitted[f"event:{t}"].predict_proba(rows)
    sev = fitted["severity"].predict(rows)
    out["dd_q05"], out["dd_q50"] = sev[:, 0], sev[:, 1]
    for c in ("upper", "lower", "sigma_h", "current_trend", "filing_age_days"):
        out[c] = rows[c].to_numpy() if c in rows else np.nan
    out["net_percentile"] = out["net"].rank(pct=True)
    return out


def run_batch(con, cache_dir: Path, requested: list[str], sources: dict[str, str] | None = None,
              config: BatchConfig | None = None, progress: Progress | None = None) -> dict:
    """Forecast every requested stock at the latest close and save the run; see the module docstring."""
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.sessions import next_session
    from tradingagents.research.compute import choose_device

    from . import data as D
    from . import macro as M
    from . import news as N

    config = config or BatchConfig()
    progress = progress or Progress()
    started = time.time()
    data = D.load(requested, cache_dir, progress=progress, workers=config.workers)
    for t, why in data.rejected.items():
        progress.ticker(t, "rejected", why)
    sessions = data.close.index
    cutoff = data.close["SPY"].dropna().index[-1]
    all_dates = pd.DatetimeIndex([])
    plans = {}
    for h in config.horizons:
        d = P.decision_dates(sessions, start="2006-01-01", every=P.every_for(h)).union(pd.DatetimeIndex([cutoff]))
        plans[h] = d
        all_dates = all_dates.union(d)
    progress("economic data as published")
    macro = M.features(all_dates, cache_dir)
    device, why = choose_device(config.device, 200_000)
    results, models, provenance = {}, {}, {}
    for h in config.horizons:
        progress(f"building the {h}-session panel")
        panel = P.build(data, h, dates=plans[h], th=config.thresholds, groups=config.groups, macro=macro)
        progress(f"fitting or loading the {h}-session models")
        fitted, art = models_for(panel, cutoff, h, config, Path(cache_dir) / "stock_models", device)
        ev = evidence(con, h)
        weights = (ev.get("choices") or {}).get("ensemble_weights") or {}
        at = panel.xs(cutoff, level="date", drop_level=False) if cutoff in panel.index.get_level_values(0) else panel.iloc[:0]
        table = forecast_rows(fitted, at, weights) if len(at) else pd.DataFrame()
        models[h] = {"artifact": art, "meta": fitted["meta"], "base_rates": fitted["base_rates"],
                     "evidence": ev, "weights": weights or "equal (no study yet)"}
        results[h] = (table, fitted, ev)
    news = {}
    if config.news and data.requested:
        progress("collecting news")
        collected = N.collect(con, {t: data.info[t] for t in data.requested}, workers=config.workers)
        when = datetime.now(UTC)
        for t in data.requested:
            news[t] = N.summary(con, t, when)
            if isinstance(collected.get(t), str):
                news[t]["error"] = collected[t]
    provenance = {"prices_through": str(cutoff.date()), "universe": len(data.stocks),
                  "sec_facts": sum(1 for f in data.facts.values() if f), "fred_key": bool(macro.notna().any().any()),
                  "news": {t: {"articles": v.get("articles"), "collected_since": v.get("collected_since"),
                               "model": v.get("model")} for t, v in news.items()},
                  "errors": data.errors}
    run_id = uuid.uuid4().hex
    rows = []
    for t in data.requested:
        quality = D.quality(data.close[t], data.volume[t], cutoff) if t in data.close else ["no prices"]
        for h in config.horizons:
            table, fitted, ev = results[h]
            reasons = list(quality) + [f"data: {e}" for e in data.errors.get(t, [])]
            if table.empty or t not in table.index:
                rows.append((run_id, t, h, "abstain", json.dumps(reasons or ["no usable row at the cutoff (a year of"
                                                                              " prices is needed)"]), "{}"))
                progress.ticker(t, "abstained", "; ".join(reasons)[:200])
                continue
            f = table.loc[t].to_dict()
            if np.isnan(f.get("filing_age_days", np.nan)):
                reasons.append("no SEC filings on record: fundamental inputs missing (treated as average)")
            if not data.info[t].get("sp500"):
                reasons.append("not in the S&P 500: forecast by models trained on S&P 500 stocks")
            status = {"return": _status(ev, "final_test", "return"),
                      **{e: _status(ev, "final_test", e) for e in EVENTS}}
            if status["return"] != "backed":
                reasons.append(f"return forecast evidence: {status['return']} (see the study)")
            payload = {"forecast": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in f.items()},
                       "evidence_status": status, "base_rates": fitted["base_rates"], "news": news.get(t),
                       "name": data.info[t]["name"], "sector": data.info[t]["sector"],
                       "source": (sources or {}).get(t, "pasted")}
            stale = any(r.startswith(("stale", "no price", "no trading volume")) for r in quality)
            rows.append((run_id, t, h, "abstain" if stale else "ok", json.dumps(reasons), json.dumps(payload, default=str)))
            progress.ticker(t, "abstained" if stale else "done", "; ".join(reasons)[:200] if reasons else None)
    th = config.thresholds
    with transaction(con):
        con.execute("INSERT INTO stock_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, now_utc(), str(cutoff.date()), str(next_session(cutoff.date())),
                     json.dumps({"tickers": data.requested, "sources": sources or {}}), json.dumps(data.rejected),
                     len(data.stocks), json.dumps(list(config.horizons)), code_version(),
                     hashlib.sha256(json.dumps(P.features_for(config.groups)).encode()).hexdigest()[:12],
                     th.version(), json.dumps(asdict(th)), json.dumps(models, default=str),
                     json.dumps(provenance, default=str), json.dumps(config.key(), default=str),
                     json.dumps({"device": device, "device_reason": why, "paid_ai_usd": 0.0}),
                     time.time() - started))
        con.executemany("INSERT INTO stock_forecasts VALUES (?, ?, ?, ?, ?, ?)", rows)
    return {"run_id": run_id, "info_cutoff": str(cutoff.date()), "earliest_execution": str(next_session(cutoff.date())),
            "requested": data.requested, "rejected": data.rejected, "tickers": progress.tickers,
            "seconds": round(time.time() - started, 1)}


def load_run(con, run_id: str | None = None) -> dict | None:
    """A saved run with its forecasts (the latest when ``run_id`` is None)."""
    row = con.execute("SELECT * FROM stock_runs WHERE id = ?" if run_id else
                      "SELECT * FROM stock_runs ORDER BY created_at DESC LIMIT 1", (run_id,) if run_id else ()).fetchone()
    if not row:
        return None
    run = dict(row)
    for k in ("requested_json", "rejected_json", "horizons_json", "thresholds_json", "models_json",
              "provenance_json", "config_json", "costs_json"):
        run[k.removesuffix("_json")] = json.loads(run.pop(k))
    run["forecasts"] = [{"ticker": r["ticker"], "horizon": r["horizon"], "status": r["status"],
                         "reasons": json.loads(r["reasons_json"]), **json.loads(r["payload_json"])}
                        for r in con.execute("SELECT * FROM stock_forecasts WHERE run_id = ? ORDER BY ticker, horizon",
                                             (run["id"],))]
    return run
