"""One interface for every stock model, from cash to pretrained forecasters.

An adapter has a ``name``, a ``kind`` ("score": a within-date ranking score; "prob": a
probability; "distribution": quantiles of the forecast return), ``fit(train, features)``,
``predict(rows) -> np.ndarray`` (one value or row per input row, aligned), ``describe()``, and
``status()``: ``("available", "")`` or ``("unavailable", the precise reason)``. Optional
dependencies load lazily inside ``fit``/``predict``; a missing one makes ``status`` say so and
never breaks the application.

Every ranking adapter sees the same rows and the same training rows (chosen by the caller,
purged by label end); ties and constant predictions are reported by the scorer, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.models import XGB, Design, ElasticNet, inner_split


class Adapter:
    name = "base"
    kind = "score"

    def status(self) -> tuple[str, str]:
        return "available", ""

    def fit(self, train: pd.DataFrame, features: list[str]):
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind}


class Cash(Adapter):
    """No trade: the same score for everyone (it ranks nothing, and says so)."""
    name = "cash"

    def predict(self, rows):
        return np.zeros(len(rows))


class Momentum(Adapter):
    """12-1 month momentum rank: the transparent baseline."""
    name = "momentum"

    def predict(self, rows):
        return rows["mom_12_1"].fillna(0.0).to_numpy(dtype=float)


class WithinDateLinear(Adapter):
    """Elastic Net on the label **minus its date's average**: only differences between stocks are
    learned, so the market's move on a date (which no model here can time) cannot dominate."""
    name = "linear_xs"

    def fit(self, train, features):
        rows = train[train["label"].notna()]
        demeaned = rows["label"] - rows.groupby(level="date")["label"].transform("mean")
        self.model = ElasticNet().fit(rows.assign(label=demeaned), features, 1)
        return self

    def predict(self, rows):
        return self.model.predict(rows)

    def describe(self):
        return self.model.describe()


class Ranker(Adapter):
    """XGBoost learning-to-rank: pairwise objective, groups = decision dates, relevance = the
    fifth each outcome fell in on its date."""
    name = "xgb_rank"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fit(self, train, features):
        self.model = XGB("rank", device=self.device).fit(train[train["label"].notna()], features, 1)
        return self

    def predict(self, rows):
        return self.model.raw(rows)

    def describe(self):
        return self.model.describe()


class TradeQuality(Adapter):
    """P(the policy's trade nets positive): logistic or boosted trees on the features plus the
    primary model's cross-fitted score percentile. ``calibration``: 'none' or 'sigmoid' (a
    regularized Platt fit on the last fifth of training dates, purged, never random folds)."""
    kind = "prob"

    def __init__(self, model: str = "logit", calibration: str = "none", device: str = "cpu"):
        self.model_kind, self.calibration, self.device = model, calibration, device
        self.name = f"quality_{model}" + ("_sig" if calibration == "sigmoid" else "")

    def _fit_core(self, rows, features):
        y = rows["win"].astype(int).to_numpy()
        self.design = Design().fit(rows, features)
        if self.model_kind == "logit":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=0.1, max_iter=3000).fit(self.design(rows), y)
        import xgboost as xgb

        params = {"max_depth": 3, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 200,
                  "lambda": 10.0, "tree_method": "hist", "objective": "binary:logistic", "device": self.device,
                  "seed": 7, "nthread": 0}
        before, valid = inner_split(rows)
        probe = xgb.train(params, xgb.DMatrix(self.design(before), label=before["win"].astype(int)), 1500,
                          evals=[(xgb.DMatrix(self.design(valid), label=valid["win"].astype(int)), "v")],
                          early_stopping_rounds=100, verbose_eval=False)
        self.rounds = max(int(probe.best_iteration) + 1, 10)
        return xgb.train(params, xgb.DMatrix(self.design(rows), label=y), self.rounds)

    def _raw(self, model, rows):
        if self.model_kind == "logit":
            return model.predict_proba(self.design(rows))[:, 1]
        import xgboost as xgb

        # binary:logistic predicts probabilities (not margins) by default.
        return model.predict(xgb.DMatrix(self.design(rows)))

    def fit(self, train, features):
        rows = train[train["win"].notna()].sort_index(level="date")
        self.features = features
        if self.calibration == "sigmoid":
            before, valid = inner_split(rows)
            core = self._fit_core(before, features)
            p = np.clip(self._raw(core, valid), 1e-6, 1 - 1e-6)
            from sklearn.linear_model import LogisticRegression

            logit = np.log(p / (1 - p)).reshape(-1, 1)
            self.platt = LogisticRegression(C=1.0).fit(logit, valid["win"].astype(int))
        self.core = self._fit_core(rows, features)
        return self

    def predict(self, rows):
        p = self._raw(self.core, rows)
        if self.calibration == "sigmoid":
            p = np.clip(p, 1e-6, 1 - 1e-6)
            p = self.platt.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return p


def score_to_net(scores_pct: np.ndarray, nets: np.ndarray, bins: int = 10, k: float = 200.0) -> dict:
    """A shrunken map from within-date score percentile to expected net arithmetic P&L per trade,
    learned only on held-out (cross-fitted) predictions: each bin's mean, pulled toward the overall
    mean in proportion k / (n + k)."""
    ok = np.isfinite(scores_pct) & np.isfinite(nets)
    s, n_ = scores_pct[ok], nets[ok]
    overall = float(n_.mean()) if len(n_) else 0.0
    edges = np.linspace(0, 1, bins + 1)
    table = []
    for a, b in zip(edges, edges[1:], strict=False):
        sel = (s >= a) & ((s < b) if b < 1 else (s <= b))
        m = int(sel.sum())
        mean = float(n_[sel].mean()) if m else overall
        sd = float(n_[sel].std()) if m > 1 else float(n_.std()) if len(n_) > 1 else 0.0
        table.append({"lo": a, "hi": b, "n": m, "raw": mean, "shrunk": (m * mean + k * overall) / (m + k), "sd": sd,
                      "q10": float(np.quantile(n_[sel], 0.1)) if m >= 10 else None})
    return {"overall": overall, "bins": table, "k": k}


def apply_map(mapping: dict, scores_pct: np.ndarray, field: str = "shrunk") -> np.ndarray:
    """The mapped value of each percentile: ``shrunk`` (expected net), ``sd`` (its spread) or ``q10``."""
    out = np.full(len(scores_pct), np.nan if field != "shrunk" else mapping["overall"])
    for b in mapping["bins"]:
        sel = (scores_pct >= b["lo"]) & ((scores_pct < b["hi"]) if b["hi"] < 1 else (scores_pct <= b["hi"]))
        out[sel] = b[field] if b.get(field) is not None else np.nan
    return out


def within_date_pct(rows: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Each score's percentile among its date's rows (ties share the average; constant scores are 0.5)."""
    s = pd.Series(scores, index=rows.index)
    pct = s.groupby(level="date").rank(pct=True, method="average")
    flat = s.groupby(level="date").transform(lambda x: x.nunique() < 2)
    return np.where(flat, 0.5, pct.to_numpy())
