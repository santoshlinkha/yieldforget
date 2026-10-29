"""A pretrained forecasting challenger: Chronos-2 (Amazon), run locally and zero-shot.

Checked 2026-10-24 on its Hugging Face card (amazon/chronos-2): Apache-2.0, 120M
parameters, encoder-only, context up to 8,192 steps, CPU or GPU, package
``chronos-forecasting>=2.0`` (``Chronos2Pipeline``). Its training data is "a subset
of Chronos Datasets, a subset of GIFT-Eval Pretrain, and synthetic data", with no
cutoff date given and no statement on financial series. It may therefore have seen
some of the very prices it is tested on: **contamination risk unknown**. Its
results are reported over the whole test period, with this label, and separately
for test dates after its public release (October 2025, arXiv 2510.15821).
TimesFM-3 was not used: its weights are licensed for non-commercial,
non-production use only.

How it forecasts here: for each (date, ticker) row, the context is that asset's last
1,024 daily log closes up to the date, and Chronos-2 forecasts the next ``horizon``
of them. The forecast log return is the forecast's final step minus the last log
close, less cash's expected return over the horizon. Its own 5th, 50th and 95th
percentiles become the forecast's quantiles; the median serves as the mean. Nothing
is fitted: ``fit`` only records the horizon. One series per row, in bounded batches.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .models import Forecaster

CONTEXT = 1024
RELEASE = "2025-10-17"
_PIPELINES: dict[str, object] = {}


REPO = "amazon/chronos-2"
# The weights are not pinned unless this names a Hugging Face commit; what was loaded is recorded either way.
REVISION_ENV = "TRADINGAGENTS_CHRONOS_REVISION"


def pipeline(device: str):
    if device not in _PIPELINES:
        from chronos import Chronos2Pipeline

        pinned = os.environ.get(REVISION_ENV) or None
        kwargs = {"revision": pinned} if pinned else {}
        _PIPELINES[device] = Chronos2Pipeline.from_pretrained(REPO, device_map=device, **kwargs)
    return _PIPELINES[device]


def loaded_revision(device: str) -> dict:
    """What a run used: the requested pin (or none), and the commit the weights resolved to when known."""
    pipe = _PIPELINES.get(device)
    config = getattr(getattr(pipe, "model", None), "config", None)
    return {"repo": REPO, "requested_revision": os.environ.get(REVISION_ENV) or None,
            "resolved_commit": getattr(config, "_commit_hash", None),
            "pinned": bool(os.environ.get(REVISION_ENV))}


class Chronos2(Forecaster):
    name = "chronos2"
    direct_quantiles = True

    def __init__(self, close: pd.DataFrame, device: str = "cpu", batch: int = 256):
        self.logp = np.log(close)
        self.pos = {d: i for i, d in enumerate(close.index)}
        self.device, self.batch = device, batch
        self._cache: dict = {}

    def fit(self, train, features, horizon):
        self.horizon = horizon
        return self

    def _forecast(self, rows: pd.DataFrame) -> np.ndarray:
        """``(rows, 3)``: forecast log return to the horizon at the 5th, 50th and 95th percentiles."""
        import torch

        todo = [k for k in rows.index if (k, self.horizon) not in self._cache]
        for start in range(0, len(todo), self.batch):
            keys = todo[start:start + self.batch]
            contexts, last = [], []
            for d, t in keys:
                i = self.pos[d] + 1
                series = self.logp[t].iloc[max(0, i - CONTEXT):i].dropna().to_numpy(np.float32)
                contexts.append(torch.tensor(series))
                last.append(series[-1])
            quantiles, _ = pipeline(self.device).predict_quantiles(
                contexts, prediction_length=self.horizon, quantile_levels=[0.05, 0.5, 0.95])
            for k, q, base in zip(keys, quantiles, last, strict=True):
                q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
                self._cache[(k, self.horizon)] = q.reshape(-1, self.horizon, 3)[0, -1, :] - base
        return np.array([self._cache[(k, self.horizon)] for k in rows.index])

    def quantiles(self, rows):
        out = self._forecast(rows) - rows["cash_h"].fillna(0.0).to_numpy()[:, None]
        return np.sort(out, axis=1)

    def predict(self, rows):
        return self.quantiles(rows)[:, 1]

    def describe(self):
        return {"name": self.name, "checkpoint": "amazon/chronos-2", "license": "Apache-2.0",
                "context": CONTEXT, "device": self.device, "contamination": "unknown (no training cutoff published)",
                "released": RELEASE}


class Chronos2Covariates(Chronos2):
    """Chronos-2 with past covariates: beside the stock's log closes it reads SPY's and its sector fund's log closes,
    its log volume, the VIX and the 10-year yield over the same window, so the forecast can lean on the market. Still
    zero-shot (nothing fitted); the same contamination caveat."""

    name = "chronos2_covariates"
    COVARIATES = ("spy", "sector", "volume", "vix", "tnx")

    def __init__(self, close: pd.DataFrame, volume: pd.DataFrame, sector_etf: dict[str, str | None],
                 device: str = "cpu", batch: int = 128):
        super().__init__(close, device, batch)
        self.logv = np.log(volume.reindex(index=close.index).where(volume > 0))
        self.sector_etf = sector_etf

    def _series(self, frame: pd.DataFrame, col: str | None, lo: int, hi: int) -> np.ndarray:
        if col is None or col not in frame:
            return np.zeros(hi - lo, dtype=np.float32)
        v = frame[col].iloc[lo:hi].ffill().bfill()
        return v.fillna(0.0).to_numpy(np.float32)

    def _forecast(self, rows: pd.DataFrame) -> np.ndarray:
        todo = [k for k in rows.index if (k, self.horizon) not in self._cache]
        for start in range(0, len(todo), self.batch):
            keys = todo[start:start + self.batch]
            inputs, last = [], []
            for d, t in keys:
                hi = self.pos[d] + 1
                target = self.logp[t].iloc[max(0, hi - CONTEXT):hi]
                first = target.first_valid_index()
                lo = self.pos[first] if first is not None else hi - 1
                y = self.logp[t].iloc[lo:hi].ffill().to_numpy(np.float32)
                etf = self.sector_etf.get(t)
                cov = {"spy": self._series(self.logp, "SPY", lo, hi),
                       "sector": self._series(self.logp, etf if etf in self.logp else "SPY", lo, hi),
                       "volume": self._series(self.logv, t, lo, hi),
                       "vix": self._series(self.logp, "^VIX", lo, hi), "tnx": self._series(self.logp, "^TNX", lo, hi)}
                inputs.append({"target": y, "past_covariates": cov})
                last.append(y[-1])
            quantiles, _ = pipeline(self.device).predict_quantiles(
                inputs, prediction_length=self.horizon, quantile_levels=[0.05, 0.5, 0.95])
            for k, q, base in zip(keys, quantiles, last, strict=True):
                q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
                self._cache[(k, self.horizon)] = q.reshape(-1, self.horizon, 3)[0, -1, :] - base
        return np.array([self._cache[(k, self.horizon)] for k in rows.index])

    def describe(self):
        return {**super().describe(), "name": self.name, "covariates": list(self.COVARIATES)}


FEATURES = ["chronos_mean", "chronos_range"]


def chronos_features(close: pd.DataFrame, rows: pd.DataFrame, horizon: int, device: str = "cpu") -> pd.DataFrame:
    """Chronos-2's forecast as two inputs for another model (stacking), one row per panel row.

    ``chronos_mean``: its median log return above cash over ``horizon``; ``chronos_range``:
    its 95th minus 5th percentile, how unsure it is. Chronos-2 is not fitted, so each row's
    value uses only the prices up to that row's date: no leakage beyond Chronos-2's own
    pretraining, whose contamination risk is unknown (see the module docstring).
    """
    model = Chronos2(close, device=device).fit(None, [], horizon)
    q = model.quantiles(rows)
    return pd.DataFrame({"chronos_mean": q[:, 1], "chronos_range": q[:, 2] - q[:, 0]}, index=rows.index)
