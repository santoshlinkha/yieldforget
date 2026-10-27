"""Candidate trading policies and their leakage-safe triple-barrier outcomes.

The policy set is small and fixed before any test (``POLICY_SET``); it is not searched.

For a decision at the close of session t (information cutoff), on as-traded split-adjusted prices:

- ``next_close``: buy at the close of t+1 (the first session after the cutoff).
- ``limit_dip``: a buy limit at  L = close_t * exp(-k * sigma * sqrt(d))  (k = 0.5, d = 5), good for
  sessions t+1 .. t+d. **Filled** on the first session whose low is at least ``fill_buffer`` (0.1%)
  *below* L: a touch is not a fill (queue position is unknown). The fill price is L. The data has no
  opening prices, so a gap down through L, which would really fill at the lower open, is priced at L:
  conservative for the buyer. Not filled by t+d: **nofill**, the money stays in cash (excess 0).
- ``hold_now`` (stocks already held): the position from the close of t, no entry cost: the economics of
  keeping it from now on, whatever it was bought at.

Every entry then follows the same exit rule, from the **actual entry price** E (not the signal price):
target E * exp(+pk * sigma_H), stop E * exp(-rk * sigma_H), sigma_H = sigma * sqrt(H), H = 20 sessions,
pk = rk = 1. Exits are **close-confirmed**: a close at or through a level on session k (k >= 1 after
entry; never the entry session, so no day trade) sells at the close of k+1, so an overnight gap or a
further fall is in the loss; otherwise the position is sold at the close of entry + H. Because each
session has one close, both levels cannot trigger on one bar: there is no intrabar ordering to assume.

Outcomes: ``target``, ``stop``, ``expiry``, ``nofill``, ``delisted`` (prices stop before the exit: sold
at the last price seen, flagged, never dropped), ``incomplete`` (the window has not finished: no label).
Each carries the net return (price change plus dividends less the round-trip cost ``cost_rt``, half on
each side), the excess over cash (the cash fund's return while the money was in the stock), holding
sessions, maximum adverse excursion (from executable lows), the entry price and the session the label
is known (``label_end``), used to purge training rows that overlap a test period.

Research prices are split-adjusted as traded (``positions.executable_prices``); dividends are added on
ex-dates. Orders shown on the page are priced from the verified as-traded close.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

OUTCOMES = ("target", "stop", "expiry", "nofill", "delisted", "incomplete")


@dataclass(frozen=True)
class PolicySpec:
    name: str
    entry: str                       # next_close | limit | hold_now
    horizon: int = 20
    profit_k: float = 1.0
    risk_k: float = 1.0
    limit_k: float = 0.5
    limit_days: int = 5
    fill_buffer: float = 0.001

    def window(self) -> int:
        """Sessions from the decision to the latest possible exit."""
        return self.horizon + (self.limit_days if self.entry == "limit" else 1 if self.entry == "next_close" else 0)


POLICY_SET = {
    "next_close": PolicySpec("next_close", "next_close"),
    "limit_dip": PolicySpec("limit_dip", "limit"),
    "hold_now": PolicySpec("hold_now", "hold_now"),
}


def limit_price(close_t: np.ndarray, sigma: np.ndarray, spec: PolicySpec) -> np.ndarray:
    return close_t * np.exp(-spec.limit_k * sigma * math.sqrt(spec.limit_days))


def simulate(px: np.ndarray, low: np.ndarray, div: np.ndarray, cash_ret: np.ndarray, t_idx: np.ndarray,
             cols: np.ndarray, sigma: np.ndarray, cost_rt: np.ndarray, spec: PolicySpec) -> pd.DataFrame:
    """The policy's outcome for each (decision session index, column). Arrays are sessions x tickers."""
    n = px.shape[0]
    m = len(t_idx)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    H = spec.horizon
    out = {k: np.full(m, np.nan) for k in ("entry_idx", "entry_px", "net", "excess", "held", "mae", "label_end",
                                           "limit_px")}
    outcome = np.full(m, "incomplete", dtype=object)
    close_t = px[t_idx, cols]
    # --- entry ---
    if spec.entry == "next_close":
        e = t_idx + 1
        ok = e < n
        entry_px = np.where(ok, px[np.minimum(e, n - 1), cols], np.nan)
        entry_cost = cost_rt / 2
    elif spec.entry == "hold_now":
        e = t_idx.copy()
        entry_px = close_t.copy()
        entry_cost = np.zeros(m)
    elif spec.entry == "limit":
        lim = limit_price(close_t, sigma, spec)
        out["limit_px"] = lim
        e = np.full(m, -1)
        decided = np.zeros(m, dtype=bool)
        for s in range(1, spec.limit_days + 1):
            idx = t_idx + s
            inside = idx < n
            lo = np.where(inside, low[np.minimum(idx, n - 1), cols], np.nan)
            hit = inside & ~decided & np.isfinite(lo) & (lo <= lim * (1 - spec.fill_buffer))
            e = np.where(hit, idx, e)
            decided |= hit
        window_done = t_idx + spec.limit_days < n
        nofill = ~decided & window_done & np.isfinite(lim)
        outcome[nofill] = "nofill"
        out["net"][nofill] = out["excess"][nofill] = out["held"][nofill] = 0.0
        out["label_end"][nofill] = (t_idx + spec.limit_days)[nofill]
        entry_px = np.where(decided, lim, np.nan)
        entry_cost = cost_rt / 2
    else:
        raise ValueError(f"unknown entry {spec.entry}")
    live = np.isfinite(entry_px) & (e >= 0) & np.isfinite(sigma)
    # --- the path after entry ---
    sig_h = sigma * math.sqrt(H)
    profit, risk = np.expm1(spec.profit_k * sig_h), np.expm1(-spec.risk_k * sig_h)
    steps = np.arange(1, H + 1)
    rows = e[:, None] + steps[None, :]
    inside = rows < n
    r = np.minimum(rows, n - 1)
    fwd = np.where(inside, px[r, cols[:, None]], np.nan)
    fwd_low = np.where(inside, low[r, cols[:, None]], np.nan)
    fwd_div = np.where(inside, div[r, cols[:, None]], 0.0)
    fwd_cash = np.where(inside, cash_ret[r], np.nan)
    with np.errstate(all="ignore"):
        ret = fwd / entry_px[:, None] - 1
    hit_t, hit_s = ret >= profit[:, None], ret <= risk[:, None]
    first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1) + 1, H + 1)
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1) + 1, H + 1)
    trig = np.minimum(first_t, first_s)
    exit_k = np.minimum(trig + 1, H)
    reason = np.where(trig + 1 <= H, np.where(first_s <= first_t, "stop", "target"), "expiry")
    # Prices that stop before the exit (a delisting or a halt): sold at the last price seen, flagged.
    held_mask = steps[None, :] <= exit_k[:, None]
    missing = held_mask & inside & ~np.isfinite(fwd)
    gone = missing.any(axis=1)
    first_gap = np.where(gone, missing.argmax(axis=1), H)          # steps before the gap that are valid
    exit_k = np.where(gone, np.maximum(first_gap, 1), exit_k)
    reason = np.where(gone, "delisted", reason)
    finished = (e + exit_k < n) | gone
    held_mask = steps[None, :] <= exit_k[:, None]
    k = np.arange(m)
    exit_px = np.where(gone & (first_gap == 0), entry_px, fwd[k, np.clip(exit_k - 1, 0, H - 1)])
    divs = np.nansum(np.where(held_mask, fwd_div, 0.0), axis=1)
    cash_held = np.nanprod(np.where(held_mask, 1 + np.nan_to_num(fwd_cash), 1.0), axis=1) - 1
    with np.errstate(all="ignore"):
        gross = (exit_px + divs) / entry_px - 1
        mae = np.minimum(np.nanmin(np.where(held_mask, fwd_low, np.nan), axis=1) / entry_px - 1, 0.0)
    done = live & finished
    net = gross - entry_cost - cost_rt / 2
    out["entry_idx"] = np.where(live, e, out["entry_idx"])
    out["entry_px"] = np.where(live, entry_px, out["entry_px"])
    for key, val in (("net", net), ("excess", net - cash_held), ("held", exit_k), ("mae", mae),
                     ("label_end", e + exit_k)):
        out[key] = np.where(done, val, out[key])
    outcome = np.where(done, reason, outcome)
    frame = pd.DataFrame(out)
    frame["outcome"] = outcome
    frame["policy"] = spec.name
    return frame


def label(panel_rows: pd.DataFrame, px: pd.DataFrame, low_adj: pd.DataFrame, close_adj: pd.DataFrame,
          div: pd.DataFrame, cash_close: pd.Series, spec: PolicySpec) -> pd.DataFrame:
    """Outcomes for panel rows (index date, ticker; columns ``ewma_vol_daily`` and ``cost_rt``)."""
    tickers = list(px.columns)
    sessions = px.index
    P = px.to_numpy(dtype=float)
    ratio = P / close_adj[tickers].to_numpy(dtype=float)
    L = low_adj[tickers].to_numpy(dtype=float) * ratio                 # lows on the executable basis
    D = div[tickers].reindex(sessions).fillna(0.0).to_numpy(dtype=float)
    cash = cash_close.reindex(sessions).ffill()
    cash_ret = cash.pct_change().fillna(0.0).to_numpy(dtype=float)
    dates = panel_rows.index.get_level_values("date")
    names = panel_rows.index.get_level_values("ticker")
    t_idx, cols = sessions.get_indexer(dates), pd.Index(tickers).get_indexer(names)
    keep = (t_idx >= 0) & (cols >= 0)
    res = simulate(P, L, D, cash_ret, t_idx[keep], cols[keep], panel_rows["ewma_vol_daily"].to_numpy(dtype=float)[keep],
                   panel_rows["cost_rt"].fillna(0.004).to_numpy(dtype=float)[keep], spec)
    res.index = panel_rows.index[keep]
    end = res["label_end"]
    res["label_end_date"] = pd.NaT
    ok = end.notna()
    res.loc[ok, "label_end_date"] = sessions[end[ok].astype(int).to_numpy()]
    res["entry_date"] = pd.NaT
    ent = res["entry_idx"].notna()
    res.loc[ent, "entry_date"] = sessions[res.loc[ent, "entry_idx"].astype(int).to_numpy()]
    return res


def spec_dict() -> dict:
    return {k: asdict(v) for k, v in POLICY_SET.items()}
