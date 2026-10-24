"""Walk-forward evaluation of return forecasters: fit on the past, score on what came next.

Folds are chronological and shared by every security: the test period is cut
into blocks (a year by default), and for each block every model is refitted on
the rows whose labels **ended** before the block began. That purges training
rows whose label windows overlap the test period, with the label horizon as the
embargo. Training grows block by block. Models never see a test row's label, and
their settings are fixed before the run.

Thousands of rows on one date are one market state, not thousands of
independent ones, so every uncertainty below is computed over **dates**: the IC
is averaged per date, and intervals come from a moving-block bootstrap over
dates (12 dates per block).

Scores:

- **IC**: the rank correlation across assets on one date between forecast and
  outcome. It shows whether the ranking is right, which choosing funds needs.
- **R² out of sample** against the zero forecast (Campbell and Thompson).
  Negative means worse than "no return above cash".
- **Calibration slope**: outcome on forecast; 1 means the sizes are right, near 0
  means forecasts are far too confident. Also within each date (``calibration_xs``:
  are the *relative* sizes right?) and the correlation of the date-average forecast
  with the date-average outcome (``timing_corr``: can it time the whole market?).
- **Quantiles**: average pinball loss at the 5th, 50th and 95th percentiles,
  and the share of outcomes inside the 5-95 band (target 90%).
- **Probability of a gain**: Brier score against the null of the training base
  rate, and a reliability table (forecast probability bins against outcomes).
- **Top fifth vs all**: each date's top-fifth assets by forecast, equal
  weighted, minus all assets equal weighted, gross and net of ``cost_bps`` per
  side on the turnover.

Every run is recorded in ``research_trials`` with its settings and scores, so
the number of things tried is known; ``holm`` adjusts the IC p-values for it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .forecast import QUANTILES, Probabilistic

COLUMNS = ("mean", "median", "q05", "q95", "p_gain")


@dataclass
class ModelScore:
    name: str
    dates: int
    mean_ic: float
    ic_t: float
    ic_ci: tuple[float, float]
    ic_hit: float
    r2_oos: float
    calibration: float            # outcome on forecast, all rows pooled
    calibration_xs: float         # the same within each date: relative sizes
    timing_corr: float            # date-average forecast against date-average outcome: market timing
    pinball: dict[str, float]
    coverage_90: float
    brier: float
    brier_null: float
    reliability: list[tuple[float, float, int]]   # (mean forecast, share that gained, count) per bin
    top_minus_all: float          # per period, gross
    top_minus_all_net: float
    top_minus_all_ci: tuple[float, float]
    turnover: float
    p_value: float = 1.0


@dataclass
class WalkForwardResult:
    horizon: int
    universe: str
    first_test: str
    last_test: str
    predictions: pd.DataFrame     # label, cash_h, and <model>:<column> for every model
    scores: dict[str, ModelScore]
    by_period: dict[str, dict[str, float]]
    fit_log: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def _ic_by_date(pred: pd.DataFrame, col: str) -> pd.Series:
    def one(g):
        if len(g) < 5 or g[col].nunique() < 2:
            return np.nan
        return g[col].rank().corr(g["label"].rank())
    return pred.groupby(level="date").apply(one).dropna()


def block_bootstrap(values: np.ndarray, block: int = 12, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    n = len(values)
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    means = [np.concatenate([values[s:s + block] for s in rng.choice(starts, math.ceil(n / block))])[:n].mean()
             for _ in range(reps)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _p_two_sided(t: float, n: int) -> float:
    return float(math.erfc(abs(t) / math.sqrt(2))) if n > 2 and np.isfinite(t) else 1.0


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def score(pred: pd.DataFrame, model: str, cost_bps: float = 10.0, base_rate: float | None = None) -> ModelScore:
    col = lambda c: f"{model}:{c}"  # noqa: E731
    view = pred[["label", "cash_h"]].assign(**{c: pred[col(c)] for c in COLUMNS})
    ic = _ic_by_date(view, "mean")
    n = len(ic)
    t = float(ic.mean() / ic.std() * math.sqrt(n)) if n > 2 and ic.std() > 0 else float("nan")
    y, f = view["label"].to_numpy(), view["mean"].to_numpy()
    calib = float(np.polyfit(f, y, 1)[0]) if np.std(f) > 1e-12 else float("nan")
    fd = view["mean"] - view["mean"].groupby(level="date").transform("mean")
    yd = view["label"] - view["label"].groupby(level="date").transform("mean")
    calib_xs = float(np.polyfit(fd, yd, 1)[0]) if fd.std() > 1e-12 else float("nan")
    fa, ya = view["mean"].groupby(level="date").mean(), view["label"].groupby(level="date").mean()
    timing = float(np.corrcoef(fa, ya)[0, 1]) if fa.std() > 1e-12 else float("nan")
    gained = (view["label"] + view["cash_h"] > 0).to_numpy().astype(float)
    p = view["p_gain"].to_numpy()
    rate = gained.mean() if base_rate is None else base_rate
    bins = np.clip((p * 10).astype(int), 0, 9)
    reliability = [(float(p[bins == b].mean()), float(gained[bins == b].mean()), int((bins == b).sum()))
                   for b in range(10) if (bins == b).sum() >= 20]

    spreads, tops = [], []
    for _, g in view.groupby(level="date"):
        if len(g) < 5:
            continue
        top = g[g["mean"] >= g["mean"].quantile(0.8)]
        spreads.append(top["label"].mean() - g["label"].mean())
        tops.append(set(top.index.get_level_values("ticker")))
    spreads = np.array(spreads)
    changes = [1 - len(a & b) / max(len(b), 1) for a, b in zip(tops, tops[1:], strict=False)]
    turnover = float(np.mean(changes)) if changes else 0.0
    # Replacing a share of the top fifth sells that share and buys as much: two sides.
    net = spreads - 2 * turnover * cost_bps / 1e4 if len(spreads) else spreads
    return ModelScore(
        name=model, dates=n, mean_ic=float(ic.mean()), ic_t=t, ic_ci=block_bootstrap(ic.to_numpy()),
        ic_hit=float((ic > 0).mean()), r2_oos=float(1 - np.sum((y - f) ** 2) / np.sum(y ** 2)), calibration=calib,
        calibration_xs=calib_xs, timing_corr=timing,
        pinball={f"q{int(q * 100):02d}": pinball(y, view[c].to_numpy(), q)
                 for q, c in zip(QUANTILES, ("q05", "median", "q95"), strict=True)},
        coverage_90=float(np.mean((y >= view["q05"].to_numpy()) & (y <= view["q95"].to_numpy()))),
        brier=float(np.mean((p - gained) ** 2)), brier_null=float(np.mean((rate - gained) ** 2)),
        reliability=reliability,
        top_minus_all=float(spreads.mean()) if len(spreads) else float("nan"),
        top_minus_all_net=float(net.mean()) if len(spreads) else float("nan"),
        top_minus_all_ci=block_bootstrap(net) if len(spreads) else (float("nan"),) * 2,
        turnover=turnover, p_value=_p_two_sided(t, n))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values: what a p-value means after trying this many things."""
    order = sorted(p_values, key=p_values.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[k]))
        out[k] = running
    return out


def walk_forward(panel: pd.DataFrame, factories, features: list[str], horizon: int, first_test: str,
                 block: str = "12MS", universe: str = "", min_train_dates: int = 36, cost_bps: float = 10.0,
                 progress=None, should_stop=None) -> WalkForwardResult:
    """Refit every block from ``first_test`` on, and score each model's test forecasts.

    ``factories`` make fresh mean models (wrapped in ``Probabilistic`` here).
    ``should_stop()`` is checked between blocks, so a long run can be cancelled.
    """
    started = time.time()
    panel = panel.sort_index(level="date")
    dates = panel.index.get_level_values("date")
    has_label = panel["label"].notna().to_numpy()
    test_dates = np.sort(dates[(dates >= pd.Timestamp(first_test)) & has_label].unique())
    if not len(test_dates):
        raise ValueError("no test dates with outcomes after " + first_test)
    edges = sorted({pd.Timestamp(test_dates[0]), *pd.date_range(pd.Timestamp(test_dates[0]),
                                                                 pd.Timestamp(test_dates[-1]), freq=block)})
    edges.append(pd.Timestamp(test_dates[-1]) + pd.Timedelta(days=1))
    chunks, log = [], []
    for n, (start, end) in enumerate(zip(edges, edges[1:], strict=False), 1):
        if should_stop and should_stop():
            raise InterruptedError("walk-forward cancelled")
        test = panel[(dates >= start) & (dates < end) & has_label]
        train = panel[(pd.to_datetime(panel["label_end"]) < start) & has_label]
        if test.empty or train.index.get_level_values("date").nunique() < min_train_dates:
            continue
        if progress:
            progress(n, len(edges) - 1, start)
        out = test[["label", "cash_h"]].copy()
        for make in factories:
            model = Probabilistic(make)
            t0 = time.time()
            model.fit(train, features, horizon)
            got = model.predict(test)
            for c in COLUMNS:
                out[f"{model.name}:{c}"] = got[c].to_numpy()
            log.append({"block": start.strftime("%Y-%m-%d"), "model": model.name, "train_rows": len(train),
                        "train_dates": int(train.index.get_level_values("date").nunique()),
                        "train_last_label_end": str(pd.to_datetime(train["label_end"]).max().date()),
                        "seconds": round(time.time() - t0, 3), "fit": model.describe()})
        chunks.append(out)
    pred = pd.concat(chunks)
    models = sorted({c.split(":")[0] for c in pred.columns if ":" in c})
    scores = {m: score(pred, m, cost_bps) for m in models}
    adjusted = holm({m: s.p_value for m, s in scores.items()})
    for m, s in scores.items():
        s.p_value = adjusted[m]
    by_period = {}
    pdates = pred.index.get_level_values("date")
    for a, b, name in (("1990", "2012-12-31", "to 2012"), ("2013", "2019-12-31", "2013-19"), ("2020", "2100", "2020 on")):
        part = pred[(pdates >= pd.Timestamp(a)) & (pdates <= pd.Timestamp(b))]
        if part.index.get_level_values("date").nunique() >= 12:
            by_period[name] = {m: float(_ic_by_date(part[["label"]].assign(mean=part[f"{m}:mean"]), "mean").mean())
                               for m in models}
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
