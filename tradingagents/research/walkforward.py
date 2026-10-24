"""Walk-forward evaluation of return forecasters: fit on the past, score on what came next.

The test period is cut into blocks (a year by default). For each block every
model is refitted on the rows whose labels had **ended** before the block began.
That purges the training rows whose label windows overlap the test period, and
the label horizon is the embargo. Training grows block by block. Models never
see a test row's label, and their settings are fixed before the run.

Scores, over every test date:

- **IC**: the rank correlation, across assets on one date, between forecast
  and outcome. It shows whether the ranking is right, which is what choosing
  between funds needs. The mean IC comes with a t-statistic and a
  moving-block bootstrap interval (blocks of 12 dates keep a year of
  dependence together).
- **R² out of sample** against the zero forecast (Campbell and Thompson): does
  the forecast's size beat saying "no return above cash"? Negative means worse.
- **Calibration slope**: outcome regressed on forecast; 1 means sizes are right,
  near 0 means forecasts are too confident.
- **Top fifth vs equal weight**: each date's top-fifth assets by forecast, equal
  weighted, against all assets equal weighted. Gross of costs; turnover is
  reported so costs can be judged.

Every run is recorded in ``research_trials`` with its settings and scores, so
the number of things tried is known; ``holm`` corrects their p-values for it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModelScore:
    name: str
    mean_ic: float
    ic_t: float
    ic_ci: tuple[float, float]
    ic_hit: float                 # share of dates with a positive IC
    r2_oos: float
    calibration: float
    top_minus_all: float          # per period, mean
    top_minus_all_ci: tuple[float, float]
    top_minus_all_annual: float
    turnover: float               # share of the top fifth replaced per period
    dates: int
    p_value: float = 1.0


@dataclass
class WalkForwardResult:
    horizon: int
    universe: str
    first_test: str
    last_test: str
    predictions: pd.DataFrame     # date, ticker, label, one column per model
    scores: dict[str, ModelScore]
    by_period: dict[str, dict[str, float]]   # mean IC by sub-period, per model
    fit_log: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def _ic_by_date(pred: pd.DataFrame, model: str) -> pd.Series:
    def one(g):
        if len(g) < 5 or g[model].nunique() < 2:
            return np.nan
        return g[model].rank().corr(g["label"].rank())
    return pred.groupby(level="date").apply(one).dropna()


def _block_bootstrap(values: np.ndarray, block: int = 12, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    n = len(values)
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    means = []
    for _ in range(reps):
        picks = rng.choice(starts, size=math.ceil(n / block))
        sample = np.concatenate([values[s:s + block] for s in picks])[:n]
        means.append(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _p_two_sided(t: float, n: int) -> float:
    # Normal approximation; with hundreds of dates the t distribution is close enough.
    return float(math.erfc(abs(t) / math.sqrt(2))) if n > 2 and np.isfinite(t) else 1.0


def score(pred: pd.DataFrame, model: str, horizon: int) -> ModelScore:
    ic = _ic_by_date(pred, model)
    n = len(ic)
    t = float(ic.mean() / ic.std() * math.sqrt(n)) if n > 2 and ic.std() > 0 else float("nan")
    y, f = pred["label"].to_numpy(), pred[model].to_numpy()
    r2 = 1 - np.sum((y - f) ** 2) / np.sum(y ** 2)
    calib = float(np.polyfit(f, y, 1)[0]) if np.std(f) > 1e-12 else float("nan")

    spreads, tops = [], []
    for _, g in pred.groupby(level="date"):
        if len(g) < 5:
            continue
        cut = g[model].quantile(0.8)
        top = g[g[model] >= cut]
        spreads.append(top["label"].mean() - g["label"].mean())
        tops.append(set(top.index.get_level_values("ticker")))
    spreads = np.array(spreads)
    turnover = float(np.mean([1 - len(a & b) / max(len(b), 1) for a, b in zip(tops, tops[1:], strict=False)])) \
        if len(tops) > 1 else 0.0
    periods = 252 / horizon
    return ModelScore(
        name=model, mean_ic=float(ic.mean()), ic_t=t, ic_ci=_block_bootstrap(ic.to_numpy()),
        ic_hit=float((ic > 0).mean()), r2_oos=float(r2), calibration=calib,
        top_minus_all=float(spreads.mean()) if len(spreads) else float("nan"),
        top_minus_all_ci=_block_bootstrap(spreads), top_minus_all_annual=float(spreads.mean() * periods)
        if len(spreads) else float("nan"), turnover=turnover, dates=n, p_value=_p_two_sided(t, n))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values: what a p-value means after trying this many things."""
    order = sorted(p_values, key=p_values.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[k]))
        out[k] = running
    return out


def walk_forward(panel: pd.DataFrame, factories, features: list[str], horizon: int, first_test: str,
                 block: str = "12MS", universe: str = "", min_train_dates: int = 36,
                 progress=None) -> WalkForwardResult:
    """Refit every block from ``first_test`` on, and score each model's test forecasts."""
    started = time.time()
    dates = panel.index.get_level_values("date")
    test_dates = np.sort(dates[(dates >= pd.Timestamp(first_test)) & panel["label"].notna().to_numpy()].unique())
    if not len(test_dates):
        raise ValueError("no test dates with outcomes after " + first_test)
    edges = list(pd.date_range(pd.Timestamp(test_dates[0]), pd.Timestamp(test_dates[-1]) + pd.Timedelta(days=1),
                               freq=block))
    if not edges or edges[0] > pd.Timestamp(test_dates[0]):
        edges = [pd.Timestamp(test_dates[0]), *edges]
    edges.append(pd.Timestamp(test_dates[-1]) + pd.Timedelta(days=1))
    chunks, log = [], []
    for n, (start, end) in enumerate(zip(edges, edges[1:], strict=False), 1):
        test = panel[(dates >= start) & (dates < end) & panel["label"].notna().to_numpy()]
        train = panel[pd.to_datetime(panel["label_end"]) < start]
        train = train[train["label"].notna()]
        if test.empty or train.index.get_level_values("date").nunique() < min_train_dates:
            continue
        if progress:
            progress(n, len(edges) - 1, start)
        out = test[["label"]].copy()
        for make in factories:
            model = make()
            t0 = time.time()
            model.fit(train, features, horizon)
            out[model.name] = model.predict(test)
            log.append({"block": start.strftime("%Y-%m-%d"), "model": model.name, "train_rows": len(train),
                        "train_last_label_end": str(pd.to_datetime(train["label_end"]).max().date()),
                        "seconds": round(time.time() - t0, 3), "fit": model.describe()})
        chunks.append(out)
    pred = pd.concat(chunks)
    models = [c for c in pred.columns if c != "label"]
    scores = {m: score(pred, m, horizon) for m in models}
    adjusted = holm({m: s.p_value for m, s in scores.items()})
    for m, s in scores.items():
        s.p_value = adjusted[m]
    by_period = {}
    pdates = pred.index.get_level_values("date")
    for a, b, name in (("1990", "2012-12-31", "to 2012"), ("2013", "2019-12-31", "2013-19"), ("2020", "2100", "2020 on")):
        part = pred[(pdates >= pd.Timestamp(a)) & (pdates <= pd.Timestamp(b))]
        if part.index.get_level_values("date").nunique() >= 12:
            by_period[name] = {m: float(_ic_by_date(part, m).mean()) for m in models}
    return WalkForwardResult(horizon, universe, str(pd.Timestamp(test_dates[0]).date()),
                             str(pd.Timestamp(test_dates[-1]).date()), pred, scores, by_period, log,
                             time.time() - started)


def record_trial(con, result: WalkForwardResult, name: str, config: dict, code_version: str) -> int:
    """Store a run in the trial registry; every run counts, whatever it found."""
    from tradingagents.ledger.db import now_utc, transaction

    with transaction(con):
        cur = con.execute(
            "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
            " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now_utc(), name, result.universe, result.horizon, result.first_test, result.last_test,
             json.dumps(config, default=str), code_version,
             json.dumps({m: s.__dict__ for m, s in result.scores.items()}, default=str),
             json.dumps(result.by_period), result.seconds))
        return cur.lastrowid


def trial_count(con) -> int:
    return con.execute("SELECT COUNT(*) FROM research_trials").fetchone()[0]
