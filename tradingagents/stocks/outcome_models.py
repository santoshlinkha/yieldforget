"""Outcome models for the candidate policies: what a policy's trade will do, not whether a level is touched.

For one policy, from the panel features at the decision close:

- class probabilities of the trade's outcome: ``target`` first, ``stop`` first, ``expiry`` (neither by
  the deadline) and, for a limit entry, ``nofill``. A delisting counts as ``stop`` (a loss exit);
- the expected excess return over cash of the whole policy from the decision (no-fill = 0);
- the 10th percentile of that excess return (downside).

Two implementations behind one interface, so they are compared on identical rows:

- ``lgbm``: LightGBM (multiclass; L2 on the excess clipped to its 1st-99th training percentiles;
  quantile 0.1), small trees and strong regularization.
- ``linear``: multinomial logistic regression and ridge on standardized features; its 10th percentile
  is the prediction plus the calibration window's 10th percentile residual.

**Calibration** is fitted on a later, separate slice of the training window (``fit`` takes ``train`` and
``cal``; the caller makes the split chronological and purged): per-class isotonic regression when the
slice has at least ``ISOTONIC_MIN`` rows, Platt (sigmoid) otherwise, renormalized to sum to one; the
expected excess gets a linear recalibration on the same slice. Nothing is fitted on the rows it scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ISOTONIC_MIN = 5000
LGB_PARAMS = {"num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 500, "feature_fraction": 0.7,
              "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7,
              "num_threads": 0}
ROUNDS = 300


def classes_for(policy: str) -> list[str]:
    return ["nofill", "target", "stop", "expiry"] if policy == "limit_dip" else ["target", "stop", "expiry"]


def outcome_class(outcomes: pd.Series) -> pd.Series:
    return outcomes.replace({"delisted": "stop"})


class _Design:
    """Median fill and standardization, learned on the training rows only."""

    def fit(self, X: pd.DataFrame):
        self.med = X.median()
        Z = X.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1.0).fillna(1.0)
        return self

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        return ((X.fillna(self.med) - self.mu) / self.sd).fillna(0.0).to_numpy(dtype=float)


class OutcomeModel:
    def __init__(self, kind: str, policy: str):
        if kind not in ("lgbm", "lgbm_seq", "linear"):
            raise ValueError(kind)
        self.kind, self.policy, self.classes = kind, policy, classes_for(policy)

    # --- fitting ---
    def fit(self, train: pd.DataFrame, cal: pd.DataFrame, features: list[str]) -> OutcomeModel:
        self.features = features
        # The last decision date learned from (fit or calibration): predicting it or earlier is refused.
        dates = [f.index.get_level_values("date").max() for f in (train, cal)
                 if "date" in (f.index.names or []) and len(f)]
        self.trained_through = str(max(dates).date()) if dates else None
        y = outcome_class(train["outcome"]).map({c: i for i, c in enumerate(self.classes)})
        keep = y.notna()
        train, y = train[keep], y[keep].astype(int)
        ex = train["excess"].to_numpy(dtype=float)
        self.clip = (float(np.nanquantile(ex, 0.01)), float(np.nanquantile(ex, 0.99)))
        exc = np.clip(ex, *self.clip)
        self.base = np.bincount(y, minlength=len(self.classes)) / len(y)
        if self.kind == "lgbm_seq" and "nofill" in self.classes:
            # Two steps: does the limit fill (short-term moves decide), then how the filled trade ends.
            import lightgbm as lgb

            X = train[features]
            nofill = self.classes.index("nofill")
            filled = (y != nofill).to_numpy()
            self.fill = lgb.train({**LGB_PARAMS, "objective": "binary"}, lgb.Dataset(X, filled.astype(int)), ROUNDS)
            rest = [c for c in self.classes if c != "nofill"]
            yc = y[filled].map({self.classes.index(c): i for i, c in enumerate(rest)}).to_numpy()
            self.clf = lgb.train({**LGB_PARAMS, "objective": "multiclass", "num_class": len(rest)},
                                 lgb.Dataset(X[filled], yc), ROUNDS)
            self.reg = lgb.train({**LGB_PARAMS, "objective": "regression"}, lgb.Dataset(X, exc), ROUNDS)
            self.q10 = lgb.train({**LGB_PARAMS, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(X, exc), ROUNDS)
        elif self.kind in ("lgbm", "lgbm_seq"):
            import lightgbm as lgb

            X = train[features]
            self.clf = lgb.train({**LGB_PARAMS, "objective": "multiclass", "num_class": len(self.classes)},
                                 lgb.Dataset(X, y.to_numpy()), ROUNDS)
            self.reg = lgb.train({**LGB_PARAMS, "objective": "regression"}, lgb.Dataset(X, exc), ROUNDS)
            self.q10 = lgb.train({**LGB_PARAMS, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(X, exc), ROUNDS)
        else:
            from sklearn.linear_model import LogisticRegression, Ridge

            self.design = _Design().fit(train[features])
            Z = self.design(train[features])
            self.clf = LogisticRegression(C=0.1, max_iter=2000).fit(Z, y.to_numpy())
            self.reg = Ridge(alpha=100.0).fit(Z, exc)
        self._calibrate(cal)
        return self

    def _raw(self, rows: pd.DataFrame):
        if getattr(self, "fill", None) is not None:
            X = rows[self.features]
            pf, pc = self.fill.predict(X), self.clf.predict(X)
            nofill = self.classes.index("nofill")
            p = np.zeros((len(rows), len(self.classes)))
            p[:, nofill] = 1 - pf
            others = [i for i in range(len(self.classes)) if i != nofill]
            p[:, others] = pc * pf[:, None]
            return p, self.reg.predict(X), self.q10.predict(X)
        if self.kind in ("lgbm", "lgbm_seq"):
            X = rows[self.features]
            p = self.clf.predict(X)
            return p, self.reg.predict(X), self.q10.predict(X)
        Z = self.design(rows[self.features])
        p = np.zeros((len(rows), len(self.classes)))
        p[:, self.clf.classes_] = self.clf.predict_proba(Z)
        mean = self.reg.predict(Z)
        return p, mean, mean + getattr(self, "resid_q10", 0.0)

    def _calibrate(self, cal: pd.DataFrame) -> None:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        y = outcome_class(cal["outcome"]).map({c: i for i, c in enumerate(self.classes)})
        cal = cal[y.notna()]
        y = y[y.notna()].astype(int).to_numpy()
        self.n_cal = len(cal)
        self.cal_method = "isotonic" if len(cal) >= ISOTONIC_MIN else "sigmoid"
        self.calibrators = []
        if len(cal) < 200:
            self.calibrators, self.cal_method = None, "none (too few calibration rows)"
            self.net_line = (1.0, 0.0)
            return
        p, mean, _ = self._raw(cal)
        ex = np.clip(cal["excess"].to_numpy(dtype=float), *self.clip)
        if self.kind == "linear":
            self.resid_q10 = float(np.quantile(ex - mean, 0.1))
        for j in range(len(self.classes)):
            t = (y == j).astype(float)
            if self.cal_method == "isotonic":
                self.calibrators.append(IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[:, j], t))
            elif t.min() != t.max():
                self.calibrators.append(LogisticRegression(C=1.0).fit(p[:, [j]], t))
            else:
                self.calibrators.append(None)
        slope, intercept = np.polyfit(mean, ex, 1) if np.std(mean) > 0 else (1.0, 0.0)
        self.net_line = (float(slope), float(intercept))

    # --- prediction ---
    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        from .training_store import guard

        if "date" in (rows.index.names or []):
            guard(getattr(self, "trained_through", None), rows.index.get_level_values("date"))
        p, mean, q10 = self._raw(rows)
        raw = p.copy()
        if self.calibrators:
            cols = []
            for j, c in enumerate(self.calibrators):
                if c is None:
                    cols.append(raw[:, j])
                elif self.cal_method == "isotonic":
                    cols.append(c.predict(raw[:, j]))
                else:
                    cols.append(c.predict_proba(raw[:, [j]])[:, 1])
            p = np.clip(np.column_stack(cols), 1e-4, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
        a, b = self.net_line
        out = pd.DataFrame({f"p_{c}": p[:, j] for j, c in enumerate(self.classes)}, index=rows.index)
        out["mean_excess"] = a * mean + b
        out["q10_excess"] = np.minimum(q10, out["mean_excess"].to_numpy())
        return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        sel = (p >= a) & (p <= b)
        if sel.sum():
            out.append({"lo": round(float(a), 4), "hi": round(float(b), 4), "n": int(sel.sum()),
                        "forecast": round(float(p[sel].mean()), 4), "observed": round(float(hit[sel].mean()), 4)})
    return out
