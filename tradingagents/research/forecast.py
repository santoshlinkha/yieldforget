"""From a model's expected return to a full forecast: median, quantiles, probability of a gain.

Units: every number is a **log return over the horizon** (5, 20 or 60 sessions),
never annualized. ``mean`` and the quantiles are returns above cash; the
probability of a gain is for the **total** return (above cash plus cash's own
expected return over the horizon).

How the spread is found (``Probabilistic``). The model is first fitted on the
older four fifths of its training dates and scored on the newest fifth, which it
has not seen. There its errors are divided by each asset's EWMA volatility over
the horizon, giving standardized errors whose empirical quantiles are kept.
Then the model is refitted on all the training rows. A forecast is::

    quantile_q = mean + z_q * sigma,   sigma = EWMA daily vol * sqrt(horizon)

and the probability of a gain is the share of standardized errors above the
point where the total return is zero. Errors come from data the model had not
seen, so a model that fits its training data too well is not given intervals that
are too narrow. ``xgb_quantile`` fits the quantiles directly instead, as a
challenger.

Two uncertainties are kept apart:

- **predictive**: how far the outcome may land from the forecast (the quantiles);
- **estimation**: how much the forecast itself depends on which history the
  model saw (``estimation_sd``): the spread of the mean across refits on
  date-block bootstrap samples of the training data (``bagged_mean``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .models import XGB, Forecaster, inner_split

QUANTILES = (0.05, 0.5, 0.95)
FEATURE_SCHEMA = "features-v1"


@dataclass
class Forecast:
    ticker: str
    horizon: int                  # sessions
    info_cutoff: str              # last close the inputs include
    mean: float                   # expected log return above cash over the horizon
    median: float
    q05: float
    q95: float
    p_gain: float                 # probability the total return is above zero
    sigma: float                  # predictive scale used, log return over the horizon
    estimation_sd: float | None   # spread of the mean across bootstrap refits; None when not run
    cash_h: float                 # cash's expected log return over the horizon
    model: str
    model_version: str
    feature_schema: str
    train_start: str
    train_end: str                # last label end in training
    calibration: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    backend: str = "cpu"
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def horizon_sigma(rows: pd.DataFrame, horizon: int) -> np.ndarray:
    daily = rows["ewma_vol_daily"].astype(float)
    fallback = rows["raw_vol_3m"].astype(float) / math.sqrt(252)
    s = daily.fillna(fallback).fillna(daily.median())
    return np.maximum(s.to_numpy() * math.sqrt(horizon), 1e-4)


class Probabilistic:
    """A mean model plus residual-based quantiles and probabilities; see the module docstring."""

    def __init__(self, make, name: str | None = None):
        self.make = make
        probe = make()
        self.name = name or probe.name
        # Models that produce their own quantiles (XGB in quantile mode, Chronos) keep them.
        self.direct_quantiles = (isinstance(probe, XGB) and probe.mode == "quantile") or             bool(getattr(probe, "direct_quantiles", False))

    def fit(self, train: pd.DataFrame, features: list[str], horizon: int) -> Probabilistic:
        self.horizon = horizon
        before, valid = inner_split(train)
        first = self.make().fit(before, features, horizon)
        z = (valid["label"].to_numpy() - first.predict(valid)) / horizon_sigma(valid, horizon)
        z = z[np.isfinite(z)]
        self.z = np.sort(z)
        self.z_q = {q: float(np.quantile(self.z, q)) for q in QUANTILES}
        self.model = self.make().fit(train, features, horizon)
        self.train_start = str(train.index.get_level_values("date").min().date())
        self.train_end = str(pd.to_datetime(train["label_end"]).max().date())
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        mean = self.model.predict(rows)
        sigma = horizon_sigma(rows, self.horizon)
        cash = rows["cash_h"].fillna(0.0).to_numpy()
        if self.direct_quantiles:
            q = self.model.quantiles(rows)
            q05, q50, q95 = q[:, 0], q[:, 1], q[:, 2]
        else:
            q05, q50, q95 = (mean + self.z_q[p] * sigma for p in QUANTILES)
        # P(total > 0) = P(z > (-cash - mean) / sigma), from the held-back standardized errors.
        threshold = (-cash - mean) / sigma
        p_gain = 1 - np.searchsorted(self.z, threshold, side="right") / max(len(self.z), 1)
        return pd.DataFrame({"mean": mean, "median": q50, "q05": q05, "q95": q95, "p_gain": p_gain,
                             "sigma": sigma, "cash_h": cash}, index=rows.index)

    def describe(self) -> dict:
        return {**self.model.describe(), "z_quantiles": self.z_q, "z_count": int(len(self.z)),
                "train_start": self.train_start, "train_end": self.train_end}


def bagged_mean(make, train: pd.DataFrame, rows: pd.DataFrame, features: list[str], horizon: int,
                bags: int = 5, block_months: int = 12, seed: int = 0) -> np.ndarray:
    """Spread of the mean forecast across refits on date-block bootstrap samples of ``train``:
    estimation uncertainty, apart from the outcome's own spread."""
    rng = np.random.default_rng(seed)
    dates = np.sort(train.index.get_level_values("date").unique())
    blocks = [dates[i:i + block_months] for i in range(0, len(dates), block_months)]
    means = []
    for _ in range(bags):
        picked = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        # Resampled dates keep their own rows; a date drawn twice simply counts twice.
        sample = pd.concat([train.xs(d, level="date", drop_level=False) for d in picked])
        sample = sample.sort_index(level="date")
        means.append(make().fit(sample, features, horizon).predict(rows))
    return np.std(means, axis=0)


def forecasts_for(model: Probabilistic, rows: pd.DataFrame, info_cutoff: str, model_version: str,
                  backend: str, seconds: float, estimation_sd: np.ndarray | None = None,
                  calibration: dict | None = None, flags: dict[str, list[str]] | None = None) -> list[Forecast]:
    out = model.predict(rows)
    result = []
    for i, (date, ticker) in enumerate(rows.index):
        r = out.iloc[i]
        result.append(Forecast(
            ticker=ticker, horizon=model.horizon, info_cutoff=info_cutoff, mean=float(r["mean"]),
            median=float(r["median"]), q05=float(r["q05"]), q95=float(r["q95"]), p_gain=float(r["p_gain"]),
            sigma=float(r["sigma"]), estimation_sd=None if estimation_sd is None else float(estimation_sd[i]),
            cash_h=float(r["cash_h"]), model=model.name, model_version=model_version, feature_schema=FEATURE_SCHEMA,
            train_start=model.train_start, train_end=model.train_end, calibration=calibration or {},
            flags=(flags or {}).get(ticker, []), backend=backend, seconds=seconds))
    return result


__all__ = ["FEATURE_SCHEMA", "QUANTILES", "Forecast", "Forecaster", "Probabilistic", "bagged_mean",
           "forecasts_for", "horizon_sigma"]
