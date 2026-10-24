"""Return forecasters, from the simplest baseline to gradient-boosted trees.

Every model predicts the same quantity: the log return **above cash** over the
horizon (5, 20 or 60 sessions), in horizon units, never annualized. They are
compared, never assumed better:

- ``zero``: no return above cash, the null every forecast must beat.
- ``train_mean``: the average label in training, a constant risk premium.
- ``vol_prior``: the optimizer's current rule, 0.3 Sharpe times volatility, less
  half the variance. It is the incumbent, and it gives a volatile fund a higher
  expected return just for being volatile; the others are tested against it.
- ``momentum``: the label on 12-1 month momentum rank, by least squares (transparent baseline).
- ``elastic_net``: all features, L1 + L2 penalized linear regression; penalty
  and mix chosen on the last fifth of the training dates, in time order.
- ``xgb``: gradient-boosted trees on the label (pseudo-Huber loss, so a crash
  month does not dominate); shallow, slow-learning, stopped early on the same
  held-back fifth.
- ``xgb_rank``: the same trees with a pairwise ranking objective within each
  date, which learns order rather than size; its scores are mapped to returns
  by a line fitted on the held-back fifth.
- ``xgb_quantile``: trees fitted to the 5th, 50th and 95th percentiles directly,
  a challenger to the residual-based intervals of ``forecast.Probabilistic``.

Labels are clipped at the training 1st and 99th percentiles before fitting.
Missing features count as 0, the middle of a ranked feature's range; market
levels are standardized with training means and deviations only.
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


def clip_label(y: pd.Series) -> pd.Series:
    lo, hi = y.quantile([0.01, 0.99])
    return y.clip(lo, hi)


def inner_split(train: pd.DataFrame, share: float = 0.2):
    """The last ``share`` of training dates held back in time order, and the rows before them
    whose labels ended before it began (purged)."""
    dates = np.sort(train.index.get_level_values("date").unique())
    cut = dates[int(len(dates) * (1 - share))]
    before = train[train.index.get_level_values("date") < cut]
    before = before[pd.to_datetime(before["label_end"]) < cut]
    return before, train[train.index.get_level_values("date") >= cut]


class Zero(Forecaster):
    name = "zero"

    def predict(self, rows):
        return np.zeros(len(rows))


class TrainMean(Forecaster):
    name = "train_mean"

    def fit(self, train, features, horizon):
        self.mean = float(clip_label(train["label"]).mean())
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
        self.b, self.a = np.polyfit(x, clip_label(train["label"]).to_numpy(), 1)
        return self

    def predict(self, rows):
        return self.a + self.b * rows["mom_12_1"].fillna(0.0).to_numpy()

    def describe(self):
        return {"name": self.name, "slope": float(self.b), "intercept": float(self.a)}


class Design:
    """Features as a matrix: missing as 0, market levels standardized on training rows only."""

    def fit(self, train, features):
        self.features = list(features)
        x = train[self.features].astype(float)
        self.levels = [f for f in self.features if f in MARKET_FEATURES]
        self.mu = x[self.levels].mean().fillna(0.0)
        self.sd = x[self.levels].std().replace(0, 1.0).fillna(1.0)
        return self

    def __call__(self, rows) -> np.ndarray:
        x = rows[self.features].astype(float).copy()
        if self.levels:
            x[self.levels] = (x[self.levels] - self.mu) / self.sd
        return x.fillna(0.0).to_numpy(dtype=np.float32)

    def state(self) -> dict:
        return {"features": self.features, "levels": self.levels, "mu": self.mu.to_dict(), "sd": self.sd.to_dict()}


class ElasticNet(Forecaster):
    name = "elastic_net"
    ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    L1_RATIOS = (0.2, 0.5, 0.9)

    def fit(self, train, features, horizon):
        from sklearn.linear_model import ElasticNet as EN

        self.design = Design().fit(train, features)
        before, valid = inner_split(train)
        x, y = self.design(before), clip_label(before["label"]).to_numpy()
        xv, yv = self.design(valid), valid["label"].to_numpy()
        best = None
        for a in self.ALPHAS:
            for r in self.L1_RATIOS:
                m = EN(alpha=a, l1_ratio=r, max_iter=5000).fit(x, y)
                err = float(np.mean((yv - m.predict(xv)) ** 2))
                if best is None or err < best[0]:
                    best = (err, a, r)
        _, self.alpha, self.l1_ratio = best
        self.model = EN(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000).fit(
            self.design(train), clip_label(train["label"]).to_numpy())
        return self

    def predict(self, rows):
        return self.model.predict(self.design(rows))

    def describe(self):
        return {"name": self.name, "alpha": self.alpha, "l1_ratio": self.l1_ratio,
                "weights": {f: round(float(w), 6) for f, w in zip(self.design.features, self.model.coef_, strict=True)
                            if abs(w) > 0}}


XGB_PARAMS = {"max_depth": 3, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 200,
              "lambda": 10.0, "tree_method": "hist"}


class XGB(Forecaster):
    """Boosted trees: ``mode`` is 'regression', 'rank' or 'quantile'."""

    def __init__(self, mode: str = "regression", device: str = "cpu", seed: int = 7, rounds: int = 2000,
                 params: dict | None = None):
        self.mode, self.device, self.seed, self.rounds = mode, device, seed, rounds
        self.name = {"regression": "xgb", "rank": "xgb_rank", "quantile": "xgb_quantile"}[mode]
        objective = {"regression": {"objective": "reg:pseudohubererror", "huber_slope": 0.05},
                     "rank": {"objective": "rank:pairwise"},
                     "quantile": {"objective": "reg:quantileerror", "quantile_alpha": np.array([0.05, 0.5, 0.95])}}
        self.params = {**XGB_PARAMS, **objective[mode], **(params or {})}

    def _matrix(self, rows, labels=True):
        import xgboost as xgb

        y = clip_label(rows["label"]).to_numpy() if labels else None
        if self.mode == "rank":
            # Groups are dates (rows contiguous by date); relevance is the fifth each
            # outcome falls in on its date, 0 to 4, since the objective wants whole grades.
            codes = pd.factorize(rows.index.get_level_values("date"))[0]
            if labels:
                pct = rows["label"].groupby(level="date").rank(pct=True).to_numpy()
                y = np.clip(np.floor(pct * 5), 0, 4)
            return xgb.DMatrix(self.design(rows), label=y, qid=codes)
        return xgb.DMatrix(self.design(rows), label=y)

    def fit(self, train, features, horizon):
        import xgboost as xgb

        train = train.sort_index(level="date")
        self.design = Design().fit(train, features)
        before, valid = inner_split(train)
        params = {**self.params, "device": self.device, "seed": self.seed, "nthread": 0}
        probe = xgb.train(params, self._matrix(before), self.rounds, evals=[(self._matrix(valid), "valid")],
                          early_stopping_rounds=100, verbose_eval=False)
        self.best_rounds = max(int(probe.best_iteration) + 1, 10)
        if self.mode == "rank":
            # Ranking scores have no units: a line on the held-back fifth turns them into returns.
            score = probe.predict(xgb.DMatrix(self.design(valid)), iteration_range=(0, self.best_rounds))
            self.slope, self.intercept = np.polyfit(score, valid["label"].to_numpy(), 1)
        self.booster = xgb.train(params, self._matrix(train), self.best_rounds, verbose_eval=False)
        return self

    def raw(self, rows) -> np.ndarray:
        import xgboost as xgb

        return self.booster.predict(xgb.DMatrix(self.design(rows)))

    def predict(self, rows):
        out = self.raw(rows)
        if self.mode == "rank":
            return self.intercept + self.slope * out
        if self.mode == "quantile":
            return out[:, 1]
        return out

    def quantiles(self, rows) -> np.ndarray:
        """5th, 50th and 95th percentiles (quantile mode only), sorted so they never cross."""
        return np.sort(self.raw(rows), axis=1)

    def describe(self):
        import xgboost as xgb

        gain = self.booster.get_score(importance_type="gain")
        names = {f"f{i}": n for i, n in enumerate(self.design.features)}
        return {"name": self.name, "device": self.device, "seed": self.seed, "rounds": self.best_rounds,
                "xgboost": xgb.__version__,
                "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.params.items()},
                "gain": {names[k]: round(v, 6) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}}
