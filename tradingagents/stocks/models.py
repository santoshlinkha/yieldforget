"""Event and trend classifiers, drawdown severity, and the relative-return model, for stocks.

Return forecasts reuse ``tradingagents.research.models`` (Elastic Net, boosted trees,
quantile trees) wrapped in ``forecast.Probabilistic``. Here:

- ``BaseRate``: the event's frequency in training, the null every classifier must beat.
- ``Logit``: L2-regularized logistic regression; the penalty is chosen on the last
  fifth of the training dates (in time order, purged) by log loss.
- ``XGBEvent``: boosted trees with a logistic objective, stopped early on the same
  held-back fifth. Shallow and slow-learning, like the return trees.
- ``Trend``: the four trend states (up, down, sideways, unclear) as one multinomial
  model, logistic or boosted trees.
- ``Severity``: boosted quantile trees on the drawdown from entry (5th and 50th
  percentiles): how deep the bad case goes, beside how likely it is.
- ``Relative``: boosted trees on the return above SPY, so benchmark-relative and
  absolute forecasts stay separate.

All use ``research.models.Design``: ranked features as they are, market and economic
levels standardized on training rows only, missing as 0. Rows without the target
(the future not yet happened) are dropped from training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.models import XGB, XGB_PARAMS, Design, inner_split

from .targets import TREND_STATES

EVENT_PARAMS = {**XGB_PARAMS, "eta": 0.05}


def _rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return frame[frame[target].notna()]


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class BaseRate:
    kind = "base_rate"

    def fit(self, train, features, target):
        self.p = float(_rows(train, target)[target].mean())
        return self

    def predict_proba(self, rows):
        return np.full(len(rows), self.p)


class Logit:
    kind = "logit"
    CS = (0.01, 0.1, 1.0)

    def fit(self, train, features, target):
        from sklearn.linear_model import LogisticRegression

        train = _rows(train, target).sort_index(level="date")
        self.design = Design().fit(train, features)
        before, valid = inner_split(train)
        best = None
        for c in self.CS:
            m = LogisticRegression(C=c, max_iter=2000).fit(self.design(before), before[target].astype(int))
            loss = log_loss(valid[target].to_numpy(), m.predict_proba(self.design(valid))[:, 1])
            if best is None or loss < best[0]:
                best = (loss, c)
        self.c = best[1]
        self.model = LogisticRegression(C=self.c, max_iter=2000).fit(self.design(train), train[target].astype(int))
        return self

    def predict_proba(self, rows):
        return self.model.predict_proba(self.design(rows))[:, 1]


class XGBEvent:
    kind = "xgb"

    def __init__(self, device: str = "cpu", seed: int = 7, rounds: int = 1500):
        self.device, self.seed, self.rounds = device, seed, rounds

    def _fit(self, train, features, y_of, params):
        import xgboost as xgb

        train = train.sort_index(level="date")
        self.design = Design().fit(train, features)
        before, valid = inner_split(train)
        params = {**EVENT_PARAMS, **params, "device": self.device, "seed": self.seed, "nthread": 0}
        probe = xgb.train(params, xgb.DMatrix(self.design(before), label=y_of(before)), self.rounds,
                          evals=[(xgb.DMatrix(self.design(valid), label=y_of(valid)), "valid")],
                          early_stopping_rounds=100, verbose_eval=False)
        self.best_rounds = max(int(probe.best_iteration) + 1, 10)
        self.booster = xgb.train(params, xgb.DMatrix(self.design(train), label=y_of(train)), self.best_rounds)
        return self

    def fit(self, train, features, target):
        return self._fit(_rows(train, target), features, lambda f: f[target].astype(int).to_numpy(),
                         {"objective": "binary:logistic", "eval_metric": "logloss"})

    def predict_proba(self, rows):
        import xgboost as xgb

        return self.booster.predict(xgb.DMatrix(self.design(rows)))

    def gain(self) -> dict:
        names = {f"f{i}": n for i, n in enumerate(self.design.features)}
        g = self.booster.get_score(importance_type="gain")
        return {names[k]: round(v, 4) for k, v in sorted(g.items(), key=lambda kv: -kv[1])[:12]}


class Trend:
    """The four trend states at once; ``kind`` is 'logit' or 'xgb'. ``predict_proba`` gives
    ``(rows, 4)`` in ``TREND_STATES`` order."""

    def __init__(self, kind: str = "xgb", device: str = "cpu"):
        self.kind, self.device = kind, device

    def fit(self, train, features, target="trend_state"):
        train = _rows(train, target).sort_index(level="date")
        self.prior = np.bincount(train[target].astype(int), minlength=4) / len(train)
        if self.kind == "logit":
            from sklearn.linear_model import LogisticRegression

            self.design = Design().fit(train, features)
            self.model = LogisticRegression(C=0.1, max_iter=3000).fit(self.design(train), train[target].astype(int))
            return self
        self.model = XGBEvent(self.device)._fit(train, features, lambda f: f[target].astype(int).to_numpy(),
                                                {"objective": "multi:softprob", "num_class": len(TREND_STATES),
                                                 "eval_metric": "mlogloss"})
        return self

    def predict_proba(self, rows):
        if self.kind == "logit":
            p = self.model.predict_proba(self.design(rows))
            out = np.zeros((len(rows), len(TREND_STATES)))
            out[:, self.model.classes_.astype(int)] = p
            return out
        return self.model.predict_proba(rows).reshape(len(rows), len(TREND_STATES))


class Severity:
    """Drawdown from entry at its 5th and 50th percentiles: how bad the bad case is."""

    def __init__(self, device: str = "cpu"):
        self.model = XGB("quantile", device=device, params={"quantile_alpha": np.array([0.05, 0.5, 0.95])})

    def fit(self, train, features, target="drawdown"):
        rows = _rows(train, target).assign(label=lambda f: f[target])
        self.model.fit(rows, features, 1)
        return self

    def predict(self, rows) -> np.ndarray:
        """``(rows, 2)``: the 5th and 50th percentile drawdowns (negative numbers)."""
        return self.model.quantiles(rows)[:, :2]


class Relative:
    """Expected log return above SPY over the horizon (entering at the next close)."""

    def __init__(self, device: str = "cpu"):
        self.model = XGB("regression", device=device)

    def fit(self, train, features, target="label_rel"):
        self.model.fit(_rows(train, target).assign(label=lambda f: f[target]), features, 1)
        return self

    def predict(self, rows):
        return self.model.predict(rows)


def event_model(kind: str, device: str = "cpu"):
    return {"base_rate": BaseRate, "logit": Logit, "xgb": lambda: XGBEvent(device)}[kind]()
