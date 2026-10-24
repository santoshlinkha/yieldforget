"""Return forecasters, from the simplest baseline to gradient-boosted trees.

Each one is fitted on training rows only and predicts the label: the log return
above cash over the horizon. They are compared, never assumed better:

- ``zero``: no return above cash, the null every forecast must beat.
- ``train_mean``: the average label in training, a constant risk premium.
- ``vol_prior``: the optimizer's current rule, 0.3 Sharpe times volatility,
  less half the variance. Kept as the incumbent: it makes a fund's expected
  return grow with its volatility, which is what the others are tested against.
- ``momentum``: the label on 12-1 month momentum rank, fitted by least squares.
- ``ridge``: all features, ridge regression; its penalty is chosen on the last
  fifth of the training dates, in time order.
- ``xgb``: gradient-boosted trees, shallow and slow-learning, stopped early on
  the same held-back fifth; runs on the CPU or a CUDA GPU.
- ``ensemble``: the average of ridge and xgb.

Labels are clipped at the training 1st and 99th percentiles before fitting, so
one crash month does not set every coefficient. Missing features count as 0,
the middle of a ranked feature's range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MARKET_FEATURES

TRADING_DAYS = 252


class Forecaster:
    name = "base"

    def fit(self, train: pd.DataFrame, features: list[str], horizon: int) -> Forecaster:
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


def _clip_label(y: pd.Series) -> pd.Series:
    lo, hi = y.quantile([0.01, 0.99])
    return y.clip(lo, hi)


def _inner_split(train: pd.DataFrame, share: float = 0.2):
    """The last ``share`` of training dates, held back in time order, and the rest before them."""
    dates = np.sort(train.index.get_level_values("date").unique())
    cut = dates[int(len(dates) * (1 - share))]
    fit_part = train[train.index.get_level_values("date") < cut]
    # Purge: a fitting label that ends after the validation starts would leak into it.
    fit_part = fit_part[pd.to_datetime(fit_part["label_end"]) < cut]
    return fit_part, train[train.index.get_level_values("date") >= cut]


class Zero(Forecaster):
    name = "zero"

    def predict(self, rows):
        return np.zeros(len(rows))


class TrainMean(Forecaster):
    name = "train_mean"

    def fit(self, train, features, horizon):
        self.mean = float(_clip_label(train["label"]).mean())
        return self

    def predict(self, rows):
        return np.full(len(rows), self.mean)


class VolPrior(Forecaster):
    """The optimizer's current rule, as a forecast over the horizon."""
    name = "vol_prior"

    def __init__(self, sharpe: float = 0.3):
        self.sharpe = sharpe

    def fit(self, train, features, horizon):
        self.h = horizon / TRADING_DAYS
        return self

    def predict(self, rows):
        vol = rows["raw_vol_3m"].fillna(rows["raw_vol_3m"].median()).to_numpy()
        return (self.sharpe * vol - 0.5 * vol ** 2) * self.h


class Momentum(Forecaster):
    name = "momentum"

    def fit(self, train, features, horizon):
        x = train["mom_12_1"].fillna(0.0).to_numpy()
        y = _clip_label(train["label"]).to_numpy()
        self.b, self.a = np.polyfit(x, y, 1)
        return self

    def predict(self, rows):
        return self.a + self.b * rows["mom_12_1"].fillna(0.0).to_numpy()

    def describe(self):
        return {"name": self.name, "slope": float(self.b), "intercept": float(self.a)}


class _Design:
    """Features as a matrix: missing as 0, market levels standardized on training rows."""

    def fit(self, train, features):
        self.features = list(features)
        x = train[self.features].astype(float)
        self.mu = x.mean().fillna(0.0)
        self.sd = x.std().replace(0, 1.0).fillna(1.0)
        # Ranked features are already centred and scaled; only market levels are standardized.
        self.ranked = [f for f in self.features if f not in MARKET_FEATURES]
        self.mu[self.ranked], self.sd[self.ranked] = 0.0, 1.0
        return self

    def __call__(self, rows) -> np.ndarray:
        x = (rows[self.features].astype(float) - self.mu) / self.sd
        return x.fillna(0.0).to_numpy()


class Ridge(Forecaster):
    name = "ridge"
    ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

    def _solve(self, x, y, alpha):
        xm, ym = x.mean(axis=0), y.mean()
        xc = x - xm
        w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (y - ym))
        return w, ym - xm @ w

    def fit(self, train, features, horizon):
        self.design = _Design().fit(train, features)
        fit_part, valid = _inner_split(train)
        x, y = self.design(fit_part), _clip_label(fit_part["label"]).to_numpy()
        xv, yv = self.design(valid), valid["label"].to_numpy()
        scores = {a: np.mean((yv - (xv @ w + b)) ** 2) for a in self.ALPHAS for w, b in [self._solve(x, y, a)]}
        self.alpha = min(scores, key=scores.get)
        self.w, self.b = self._solve(self.design(train), _clip_label(train["label"]).to_numpy(), self.alpha)
        return self

    def predict(self, rows):
        return self.design(rows) @ self.w + self.b

    def describe(self):
        return {"name": self.name, "alpha": self.alpha,
                "weights": dict(zip(self.design.features, np.round(self.w, 6).tolist(), strict=True))}


class XGB(Forecaster):
    name = "xgb"

    def __init__(self, device: str = "cpu", seed: int = 7, params: dict | None = None, rounds: int = 2000):
        self.device, self.seed, self.rounds = device, seed, rounds
        self.params = {"objective": "reg:pseudohubererror", "huber_slope": 0.05, "max_depth": 3, "eta": 0.02,
                       "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 200, "lambda": 10.0,
                       "tree_method": "hist", **(params or {})}

    def fit(self, train, features, horizon):
        import xgboost as xgb

        self.design = _Design().fit(train, features)
        fit_part, valid = _inner_split(train)
        params = {**self.params, "device": self.device, "seed": self.seed, "nthread": 0}
        dfit = xgb.DMatrix(self.design(fit_part), label=_clip_label(fit_part["label"]).to_numpy())
        dval = xgb.DMatrix(self.design(valid), label=valid["label"].to_numpy())
        probe = xgb.train(params, dfit, self.rounds, evals=[(dval, "valid")], early_stopping_rounds=100,
                          verbose_eval=False)
        self.best_rounds = max(int(probe.best_iteration) + 1, 10)
        dall = xgb.DMatrix(self.design(train), label=_clip_label(train["label"]).to_numpy())
        self.booster = xgb.train(params, dall, self.best_rounds, verbose_eval=False)
        return self

    def predict(self, rows):
        import xgboost as xgb

        return self.booster.predict(xgb.DMatrix(self.design(rows)))

    def describe(self):
        import xgboost as xgb

        gain = self.booster.get_score(importance_type="gain")
        names = {f"f{i}": n for i, n in enumerate(self.design.features)}
        return {"name": self.name, "device": self.device, "seed": self.seed, "rounds": self.best_rounds,
                "xgboost": xgb.__version__, "params": self.params,
                "gain": {names[k]: round(v, 6) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}}


class Ensemble(Forecaster):
    name = "ensemble"

    def __init__(self, members: list[Forecaster]):
        self.members = members

    def fit(self, train, features, horizon):
        for m in self.members:
            m.fit(train, features, horizon)
        return self

    def predict(self, rows):
        return np.mean([m.predict(rows) for m in self.members], axis=0)

    def describe(self):
        return {"name": self.name, "members": [m.describe() for m in self.members]}


def standard_models(device: str = "cpu", seed: int = 7) -> list:
    """Fresh model factories for one walk-forward step each."""
    return [Zero, TrainMean, VolPrior, Momentum, Ridge, lambda: XGB(device=device, seed=seed),
            lambda: Ensemble([Ridge(), XGB(device=device, seed=seed)])]
