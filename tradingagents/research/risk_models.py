"""Volatility and covariance forecasts, scored on what the next period actually did.

Realized variance here is the sum of squared **daily close-to-close** log returns
over the horizon. That is a noisy measure, not intraday realized volatility
(there is no intraday data), so losses are compared between models, never read
as absolute accuracy. HAR-RV needs intraday data and is not attempted.

Volatility (per asset, over the next ``horizon`` sessions):

- ``sample_10y``: the optimizer's current figure, ten years of daily returns.
- ``trailing_3m``: the last 63 sessions.
- ``ewma``: RiskMetrics, lambda 0.94.
- ``garch``: GARCH(1,1) with Student-t errors (``arch``), refitted on each
  date on up to 2,000 sessions, its variance forecast summed over the horizon.

Scored by QLIKE (robust to the noise in realized variance; lower is better)
and by how often the outcome lands inside a ±1.645 sigma band (90% if right). A
GARCH fit that does not converge, or forecasts a variance more than 25 times away
from the series' own, counts as no forecast for that row.

Covariance (for the optimizer):

- ``sample_10y``: the current one.
- ``ledoit_wolf``: shrinkage toward a scaled identity on one year of daily returns.
- ``factor``: three principal components plus each asset's own variance (for large universes).
- ``ewma``: exponentially weighted, lambda 0.97.

Scored by the realized volatility of the long-only minimum-variance portfolio
each one builds (lower means the risk was better understood), and by the ratio of
realized to forecast variance of an equal-weight portfolio (1 is unbiased).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    ok = (realized > 0) & (forecast > 0) & np.isfinite(realized) & np.isfinite(forecast)
    r = realized[ok] / forecast[ok]
    return float(np.mean(r - np.log(r) - 1))


def garch_variance(returns: pd.Series, horizon: int) -> float:
    """GARCH(1,1)-t variance of the log return summed over ``horizon`` sessions; NaN if it cannot fit."""
    from arch import arch_model

    r = returns.dropna().iloc[-2000:] * 100
    if len(r) < 500:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(
                disp="off", show_warning=False)
            var = float(fit.forecast(horizon=horizon, reindex=False).variance.values[-1].sum() / 1e4)
        except Exception:
            return float("nan")
    # A fit that did not converge, or a forecast far outside the series' own history, is no forecast.
    sample = float((r / 100).var() * horizon)
    if fit.convergence_flag != 0 or not np.isfinite(var) or not sample / 25 < var < sample * 25:
        return float("nan")
    return var


def vol_forecasts(close: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int, garch: bool = True) -> pd.DataFrame:
    """Each model's forecast variance over the horizon, with the realized variance, per (date, ticker)."""
    r = np.log(close).diff()
    ewma = r.pow(2).ewm(alpha=0.06, min_periods=60).mean()
    rows = []
    pos = {d: i for i, d in enumerate(close.index)}
    for d in dates:
        i = pos[d]
        if i + horizon >= len(close.index):
            continue
        future = r.iloc[i + 1:i + 1 + horizon]
        past = r.iloc[:i + 1]
        for t in close.columns:
            hist = past[t].dropna()
            if len(hist) < 252:
                continue
            row = {"date": d, "ticker": t, "realized": float(future[t].pow(2).sum()),
                   "return": float(future[t].sum()),
                   "sample_10y": float(hist.iloc[-2520:].var() * horizon),
                   "trailing_3m": float(hist.iloc[-63:].var() * horizon),
                   "ewma": float(ewma[t].loc[d] * horizon)}
            if garch:
                row["garch"] = garch_variance(hist, horizon)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def score_vol(frame: pd.DataFrame, models=("sample_10y", "trailing_3m", "ewma", "garch")) -> dict:
    out = {}
    real = frame["realized"].to_numpy()
    for m in models:
        if m not in frame:
            continue
        f = frame[m].to_numpy()
        z = frame["return"].to_numpy() / np.sqrt(np.where(f > 0, f, np.nan))
        ok = np.isfinite(f) & (f > 0)
        out[m] = {"qlike": qlike(real, f), "coverage_90": float(np.mean(np.abs(z[ok]) <= 1.645)),
                  # Median of realized over forecast: robust to the few huge realized months.
                  "bias": float(np.median(real[ok] / f[ok])), "rows": int(ok.sum())}
    return out


# --- covariance ---

def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(returns).covariance_


def factor_cov(returns: np.ndarray, k: int = 3) -> np.ndarray:
    x = returns - returns.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(k, len(s))
    loadings = vt[:k].T * (s[:k] / math.sqrt(len(x) - 1))
    common = loadings @ loadings.T
    resid = np.clip(np.var(x, axis=0, ddof=1) - np.diag(common), 1e-10, None)
    return common + np.diag(resid)


def ewma_cov(returns: np.ndarray, lam: float = 0.97) -> np.ndarray:
    w = lam ** np.arange(len(returns))[::-1]
    w /= w.sum()
    x = returns - (w[:, None] * returns).sum(axis=0)
    return (x * w[:, None]).T @ x


def min_variance(cov: np.ndarray, iters: int = 3000) -> np.ndarray:
    """Long-only, fully invested minimum-variance weights, by projected gradient on the simplex."""
    n = len(cov)
    w = np.full(n, 1 / n)
    step = 1 / max(np.linalg.eigvalsh(cov).max(), 1e-12)
    for _ in range(iters):
        v = w - step * (cov @ w)
        # Euclidean projection onto the simplex.
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1
        rho = np.nonzero(u - css / np.arange(1, n + 1) > 0)[0][-1]
        w_next = np.maximum(v - css[rho] / (rho + 1), 0)
        if np.abs(w_next - w).max() < 1e-10:
            return w_next
        w = w_next
    return w


def covariance_study(close: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int = 20) -> dict:
    """Realized volatility of each estimator's minimum-variance portfolio, and equal-weight bias."""
    r = np.log(close).diff()
    pos = {d: i for i, d in enumerate(close.index)}
    results = {m: {"gmv_realized": [], "ew_ratio": []} for m in ("sample_10y", "ledoit_wolf", "factor", "ewma")}
    for d in dates:
        i = pos[d]
        if i + horizon >= len(close.index):
            continue
        past = r.iloc[max(0, i - 2520):i + 1]
        names = [t for t in close.columns if past[t].iloc[-252:].notna().all() and past[t].notna().sum() >= 504]
        if len(names) < 5:
            continue
        future = r.iloc[i + 1:i + 1 + horizon][names].fillna(0.0).to_numpy()
        year = past[names].iloc[-252:].to_numpy()
        estimates = {
            "sample_10y": past[names].cov(min_periods=252).fillna(0.0).to_numpy(),
            "ledoit_wolf": ledoit_wolf(year),
            "factor": factor_cov(year),
            "ewma": ewma_cov(year),
        }
        ew = np.full(len(names), 1 / len(names))
        ew_real = float((future @ ew) @ (future @ ew))
        for m, cov in estimates.items():
            w = min_variance(cov)
            port = future @ w
            results[m]["gmv_realized"].append(float(port @ port))
            results[m]["ew_ratio"].append(ew_real / max(float(ew @ cov @ ew) * horizon, 1e-12))
    periods = TRADING_DAYS / horizon
    return {m: {"gmv_vol_annual": float(math.sqrt(np.mean(v["gmv_realized"]) * periods)) if v["gmv_realized"] else None,
                "ew_variance_ratio": float(np.mean(v["ew_ratio"])) if v["ew_ratio"] else None,
                "dates": len(v["gmv_realized"])}
            for m, v in results.items()}
