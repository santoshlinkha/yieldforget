"""How to split the trading budget across the 1st pick and the runners-up.

An even blend of two standard methods, each robust where the other is weak:

- **Mean-variance with shrinkage.** Expected returns are the decision's 20-session net estimates, shrunk halfway to
  their average (the model is not validated, so its differences are trusted only in part). Risk is the covariance of
  the stocks' daily log returns over the last year, Ledoit-Wolf shrunk (a stable estimate from few observations)
  and scaled to 20 sessions. Long-only, solved by projected gradient on the simplex with a per-stock cap.
- **Hierarchical Risk Parity** (Lopez de Prado, 2016). Stocks are clustered by correlation and the budget is split
  down the cluster tree in inverse proportion to each branch's variance. It uses no return forecast at all.

Blended 50/50, capped at ``CAP`` per stock and renormalized. With one candidate the weight is 100%; when prices are
missing for a candidate it falls back to equal weights and says so.

This is a display allocation of the trading budget. The decision's tested sizing rule (equal dollars, at most 25% of
the sleeve, five names) is unchanged; the weights are untested and labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHOD = "allocation-v1: 50% mean-variance (Ledoit-Wolf, returns shrunk 50%) + 50% hierarchical risk parity"
CAP = 0.40
SHRINK_MU = 0.5
RISK_AVERSION = 4.0
LOOKBACK = 252
H = 20


def _project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {w : 0 <= w <= cap, sum w = 1} by bisection on the shift."""
    lo, hi = v.min() - 1.0, v.max()
    for _ in range(100):
        t = (lo + hi) / 2
        s = np.clip(v - t, 0.0, cap).sum()
        lo, hi = (lo, t) if s < 1.0 else (t, hi)
    return np.clip(v - (lo + hi) / 2, 0.0, cap)


def mean_variance(mu: np.ndarray, cov: np.ndarray, cap: float = CAP, gamma: float = RISK_AVERSION) -> np.ndarray:
    """max mu'w - gamma/2 w'Cw over the capped simplex (projected gradient ascent)."""
    n = len(mu)
    w = np.full(n, 1.0 / n)
    step = 1.0 / (gamma * max(np.linalg.eigvalsh(cov).max(), 1e-12))
    for _ in range(2000):
        w_new = _project_capped_simplex(w + step * (mu - gamma * cov @ w), cap)
        if np.abs(w_new - w).max() < 1e-10:
            break
        w = w_new
    return w


def hrp(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity: single-linkage clusters on the correlation distance, then recursive bisection."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = len(cov)
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), "single")))
    w = np.ones(n)

    def cluster_var(idx):
        c = cov[np.ix_(idx, idx)]
        iv = 1 / np.diag(c)
        iv /= iv.sum()
        return float(iv @ c @ iv)

    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        vl, vr = cluster_var(left), cluster_var(right)
        alpha = 1 - vl / (vl + vr)
        w[left] *= alpha
        w[right] *= 1 - alpha
        stack += [left, right]
    return w / w.sum()


def weights(tickers: list[str], expected_net: list[float | None], close: pd.DataFrame | None) -> dict:
    """``{"weights": {ticker: share}, "method", "note"}``; shares sum to 1."""
    n = len(tickers)
    if n == 0:
        return {"weights": {}, "method": METHOD, "note": "no candidates"}
    if n == 1:
        return {"weights": {tickers[0]: 1.0}, "method": METHOD, "note": "one candidate: the whole budget"}
    have = close is not None and all(t in close.columns for t in tickers)
    rets = np.log(close[tickers]).diff().iloc[-LOOKBACK:].dropna() if have else pd.DataFrame()
    if len(rets) < 60:
        eq = {t: 1.0 / n for t in tickers}
        return {"weights": eq, "method": "equal weights", "note": "not enough price history for the risk model"}
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(rets.to_numpy()).covariance_ * H
    mu = np.array([np.nan if v is None else float(v) for v in expected_net])
    mu = np.where(np.isfinite(mu), mu, np.nanmean(mu) if np.isfinite(mu).any() else 0.0)
    mu = (1 - SHRINK_MU) * mu + SHRINK_MU * mu.mean()
    cap = max(CAP, 1.0 / n)
    w = 0.5 * mean_variance(mu, cov, cap) + 0.5 * hrp(cov)
    w = _project_capped_simplex(w, cap)
    return {"weights": {t: round(float(x), 4) for t, x in zip(tickers, w, strict=True)}, "method": METHOD,
            "note": f"risk from {len(rets)} sessions of daily returns; capped at {cap:.0%} a stock"}
