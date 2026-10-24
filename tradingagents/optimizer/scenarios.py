"""Historical crash scenarios: the stress loss of a mix as the worst past crash replayed on it.

The worst-case stress loss assumes every holding repeats its own deepest drop on
the same day. Real crashes are not like that: in 2008 stocks fell 55% while
long Treasuries rose, and in 2022 both fell. Replaying each crash on the mix
keeps what actually fell together and what did not.

A scenario is any stretch in which one of the market references fell by more than
its threshold within three years of a peak: stocks (SPY), long Treasuries (TLT),
gold (GLD) and commodities (DBC). It runs from the peak to the trough, or to the
three-year mark, so a slow decade-long slide does not count as one crash. The
scenarios are found in the price history, so a backtest dated 2015 knows 2008
but not 2020 or 2022.

The loss of a mix in a scenario is that of buying it at the scenario's start and
holding it, measured on every day to the trough; the worst day counts.

A fund with no prices at the start takes its longer-lived proxy's path if it has
one. Otherwise its path is estimated from how it moved on the reference's worst
days in the history it does have (its downside beta): beta times the reference's
loss, never a gain, and never more than the fund's own worst drop. With too few
shared worst days there is nothing to estimate from, and the fund is taken at
its own worst drop throughout, the worst-case treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reference → how far it must fall, peak to trough, for the fall to be a scenario.
REFERENCES = {"SPY": 0.15, "TLT": 0.15, "GLD": 0.20, "DBC": 0.25}
# A scenario ends at the trough or this long after the peak, whichever comes first.
MAX_YEARS = 3
# Downside beta is measured on the reference's worst days, this share of them.
TAIL = 0.10
# Shared worst days needed before a fund's downside beta is trusted (about a year's worth).
MIN_TAIL_DAYS = 25


@dataclass(frozen=True)
class Scenario:
    reference: str
    start: pd.Timestamp     # the reference's peak
    end: pd.Timestamp       # its trough (the last day known, if the fall is still going on)
    depth: float            # the reference's fall, a positive fraction

    @property
    def name(self) -> str:
        return (f"{self.reference} fell {self.depth:.0%}, "
                f"{self.start:%b %Y} to {self.end:%b %Y}")


def find_scenarios(closes: pd.DataFrame, references: dict[str, float] = REFERENCES) -> list[Scenario]:
    """Every peak-to-trough fall of a reference deeper than its threshold, oldest first."""
    found = []
    for ref, threshold in references.items():
        if ref not in closes:
            continue
        prices = closes[ref].dropna()
        if prices.empty:
            continue
        peak = prices.cummax()
        drawdown = 1 - prices / peak
        # Each new high starts an episode; its deepest point within the cap is the trough.
        episode = (prices >= peak).cumsum()
        for _, dd in drawdown.groupby(episode):
            dd = dd[dd.index <= dd.index[0] + pd.DateOffset(years=MAX_YEARS)]
            depth = float(dd.max())
            if depth >= threshold:
                found.append(Scenario(ref, dd.index[0], dd.idxmax(), depth))
    return sorted(found, key=lambda s: s.start)


def _prune_index(rows: np.ndarray) -> list[int]:
    """The rows no other row is at least as bad as, asset by asset, in their order.

    With weights of zero or more, a dominated row can never be the one that binds,
    so dropping it leaves the constraint set unchanged; a crash's days collapse to
    a handful of rows.
    """
    # A row can only be dominated by one with a larger sum, so one pass in that order does.
    order = np.argsort(-rows.sum(axis=1), kind="stable")
    kept = np.empty((len(rows), rows.shape[1]))
    index: list[int] = []
    for i in order:
        if not index or not (kept[:len(index)] >= rows[i] - 1e-12).all(axis=1).any():
            kept[len(index)] = rows[i]
            index.append(int(i))
    return sorted(index)


def downside_betas(closes: pd.DataFrame, tickers: list[str], reference: str) -> np.ndarray:
    """Each ticker's beta to ``reference`` on the reference's worst days; NaN without enough of them.

    Assets fall together more in a crash than on an ordinary day, so an ordinary
    day's beta would understate a crash; the worst days are the ones to learn from.
    """
    returns = closes.pct_change(fill_method=None)
    ref = returns[reference].dropna()
    tail = ref[ref <= ref.quantile(TAIL)]
    assets = returns.reindex(columns=tickers).loc[tail.index]
    shared = assets.notna()
    beta = assets.mul(tail, axis=0).sum() / shared.mul(tail ** 2, axis=0).sum().replace(0.0, np.nan)
    return np.where(shared.sum().to_numpy() >= MIN_TAIL_DAYS, beta.to_numpy(), np.nan)


def scenario_losses(closes: pd.DataFrame, tickers: list[str], scenarios: list[Scenario],
                    fallback: np.ndarray, proxies: dict[str, str] | None = None,
                    prune: bool = True) -> tuple[np.ndarray, list[int]]:
    """Loss rows, one per scenario day kept, and the scenario each row belongs to.

    Row ``r`` holds, for every ticker, the loss since the scenario's start (negative
    for a gain); the mix ``w`` loses ``r @ w`` that day. A ticker without prices at
    the start takes its proxy's path, else its downside-beta estimate, else
    ``fallback`` (its own worst drop) every day; see the module docstring.
    """
    rows, owners, betas = [], [], {}
    fallback = np.asarray(fallback, dtype=float)
    for s, scenario in enumerate(scenarios):
        window = closes.loc[scenario.start:scenario.end]
        if window.empty:
            continue
        ref = window[scenario.reference].ffill()
        ref_loss = (1 - ref / ref.iloc[0]).to_numpy()
        block = np.empty((len(window), len(tickers)))
        for i, t in enumerate(tickers):
            names = [t, *([proxies[t]] if proxies and proxies.get(t) in closes else [])]
            path = None
            for name in names:
                series = window[name] if name in window else None
                if series is not None and pd.notna(series.iloc[0]):
                    filled = series.ffill()
                    path = (1 - filled / filled.iloc[0]).to_numpy()
                    break
            if path is None:
                if scenario.reference not in betas:
                    betas[scenario.reference] = downside_betas(closes, tickers, scenario.reference)
                beta = betas[scenario.reference][i]
                path = (np.clip(beta * ref_loss, 0.0, fallback[i]) if np.isfinite(beta)
                        else np.full(len(window), fallback[i]))
            block[:, i] = path
        block = block[_prune_index(block)] if prune else block
        rows.append(block)
        owners += [s] * len(block)
    if not rows:
        return np.empty((0, len(tickers))), []
    matrix = np.vstack(rows)
    if prune and len(matrix) > 1:
        # Across scenarios too; keep each surviving row's owner.
        keep = _prune_index(matrix)
        return matrix[keep], [owners[i] for i in keep]
    return matrix, owners


def worst_scenario(rows: np.ndarray, owners: list[int], scenarios: list[Scenario],
                   weights: np.ndarray) -> tuple[float, str | None]:
    """The mix's loss in its worst scenario, and that scenario's name."""
    if not len(rows):
        return 0.0, None
    losses = rows @ weights
    i = int(np.argmax(losses))
    return max(float(losses[i]), 0.0), scenarios[owners[i]].name
