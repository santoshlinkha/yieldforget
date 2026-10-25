"""The one position and event engine: trade outcomes for labels, the historical simulation and the
prospective shadow account all go through it, so they follow the same execution contract.

Execution contract (daily, after the close):

- A decision at the close of session t uses only what was known by that close.
- The earliest entry is the **close of t+1** (an after-close decision cannot trade at its own close).
- Exit levels are **close-confirmed**: when the close of session j crosses a level, the exit is
  scheduled for the **next executable close**, j+1. No fill is ever placed at an intraday high,
  low, stop or target read from a daily bar; a gap past a level is lost at the next close, so a
  loss can exceed the stated risk level.
- A maximum holding deadline of H sessions means the **exit date** is entry + H: the deadline
  decision is taken at the close of entry + H - 1.
- No lot is sold on the session it was bought (no day trades): the earliest exit is entry + 1.
- Sale proceeds settle the next session (T+1); purchases need settled cash.

Prices. Trades happen at the **executable** price (the close as traded, split-adjusted only),
and dividends are credited as cash on their ex-dates while the lot is held. The research
series (``label`` in ``targets``) is the dividend-adjusted total return; the two are validated
against each other in the tests. The position's adverse excursion uses the lows of sessions
entry+1 .. exit, relative to the entry close, and is never above zero.

Costs. The estimated round-trip cost (``panel.spread_cost``) is charged half at entry and half
at exit. Results are arithmetic, net of cost: P&L, not log returns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

REASONS = ("target", "stop", "deadline", "incomplete")


@dataclass(frozen=True)
class Policy:
    """An exit policy. ``kind``: 'fixed' (exit at the deadline only) or 'barrier' (close-confirmed
    profit and risk levels, with the deadline). Levels are multiples of sigma_H, the daily EWMA
    volatility at the decision times sqrt(H), turned into arithmetic levels with expm1."""
    name: str = "barrier_20"
    kind: str = "barrier"
    horizon: int = 20               # H: the maximum holding period, in sessions
    profit_k: float = 1.0
    risk_k: float = 1.0

    def version(self) -> str:
        return "pol-" + hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def levels(self, sigma_daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(profit level, risk level) as arithmetic returns from the entry close."""
        s = np.asarray(sigma_daily, dtype=float) * np.sqrt(self.horizon)
        if self.kind == "fixed":
            return np.full(s.shape, np.inf), np.full(s.shape, -np.inf)
        return np.expm1(self.profit_k * s), np.expm1(-self.risk_k * s)


POLICIES = {
    "fixed_5": Policy("fixed_5", "fixed", 5), "fixed_20": Policy("fixed_20", "fixed", 20),
    "fixed_60": Policy("fixed_60", "fixed", 60),
    "barrier_20": Policy("barrier_20", "barrier", 20, 1.0, 1.0),
    "barrier_60": Policy("barrier_60", "barrier", 60, 1.0, 1.0),
}


def executable_prices(adjusted: pd.DataFrame, actions: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(price as traded, split-adjusted; dividends per split-adjusted share on ex-dates) per ticker."""
    from tradingagents.research.quality import raw_closes

    px, div = {}, {}
    for t in adjusted.columns:
        a = actions.get(t) or {}
        s = adjusted[t].dropna()
        if s.empty:
            continue
        px[t] = raw_closes(s, a.get("dividends") or [])
        d = pd.Series({pd.Timestamp(day): amt for day, amt in (a.get("dividends") or [])}, dtype=float)
        div[t] = d.groupby(level=0).sum().reindex(adjusted.index).fillna(0.0) if len(d) else \
            pd.Series(0.0, index=adjusted.index)
    return (pd.DataFrame(px).reindex(index=adjusted.index, columns=adjusted.columns),
            pd.DataFrame(div).reindex(index=adjusted.index, columns=adjusted.columns).fillna(0.0))


def _forward(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, steps: int) -> np.ndarray:
    """``(n, steps)``: values at rows+1 .. rows+steps for each (row, col); NaN past the end."""
    n = values.shape[0]
    idx = rows[:, None] + np.arange(1, steps + 1)[None, :]
    out = values[np.minimum(idx, n - 1), cols[:, None]]
    out[idx >= n] = np.nan
    return out


def trades(px: pd.DataFrame, low_adj: pd.DataFrame, close_adj: pd.DataFrame, div: pd.DataFrame,
           decisions: pd.DataFrame, policy: Policy, cost_rt: np.ndarray | None = None) -> pd.DataFrame:
    """The outcome of entering each decision row under ``policy`` (vectorized).

    ``decisions``: rows with columns ``date`` (decision close t), ``ticker`` and ``sigma`` (daily
    EWMA volatility at t). Returns entry and exit dates, the exit reason, holding sessions, gross
    and net arithmetic return (price change plus dividends, less ``cost_rt``), the adverse
    excursion during the position, and whether the outcome is complete.
    """
    sessions = px.index
    cols = px.columns.get_indexer(decisions["ticker"])
    t_idx = sessions.get_indexer(pd.to_datetime(decisions["date"]))
    if (cols < 0).any() or (t_idx < 0).any():
        raise ValueError("every decision needs its ticker and date in the price frame")
    H = policy.horizon
    P, L, C, D = (f.to_numpy(dtype=float) for f in (px, low_adj, close_adj, div))
    entry = t_idx + 1
    n = len(sessions)
    entry_px = np.where(entry < n, P[np.minimum(entry, n - 1), cols], np.nan)
    # Path from the session after entry: closes (for decisions), lows (excursion), dividends.
    fwd_px = _forward(P, entry, cols, H)                       # sessions entry+1 .. entry+H
    fwd_div = _forward(D, entry, cols, H)
    # Lows on the executable basis: the adjusted low times that day's executable/adjusted ratio.
    ratio = P / C
    fwd_low = _forward(L * ratio, entry, cols, H)
    profit, risk = policy.levels(decisions["sigma"].to_numpy(dtype=float))
    ret = fwd_px / entry_px[:, None] - 1
    hit_t = ret >= profit[:, None]
    hit_s = ret <= risk[:, None]
    # A close-confirmed trigger at session k (1-based after entry) exits at k+1; the deadline exit is
    # at H whatever happens. A trigger at H-1 or H therefore exits at H.
    first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1) + 1, H + 1)
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1) + 1, H + 1)
    trigger = np.minimum(first_t, first_s)
    exit_k = np.minimum(trigger + 1, H)                         # sessions after entry
    reason = np.where(trigger + 1 <= H, np.where(first_s <= first_t, "stop", "target"), "deadline")
    rows = np.arange(len(decisions))
    exit_px = fwd_px[rows, exit_k - 1]
    held = np.arange(1, H + 1)[None, :] <= exit_k[:, None]
    divs = np.nansum(np.where(held, fwd_div, 0.0), axis=1)
    lows = np.where(held, fwd_low, np.nan)
    with np.errstate(all="ignore"):
        mae = np.minimum(np.nanmin(lows, axis=1) / entry_px - 1, 0.0)
        gross = (exit_px + divs) / entry_px - 1
    complete = np.isfinite(entry_px) & np.isfinite(exit_px) & (entry + exit_k < n)
    # A path with a missing price before the exit (a delisting, a halt) has no known outcome.
    gap = np.isnan(np.where(held, fwd_px, 0.0)).any(axis=1)
    complete &= ~gap
    cost = np.zeros(len(decisions)) if cost_rt is None else np.asarray(cost_rt, dtype=float)
    entry_dates = np.where(entry < n, sessions[np.minimum(entry, n - 1)], pd.NaT)
    exit_dates = np.where(entry + exit_k < n, sessions[np.minimum(entry + exit_k, n - 1)], pd.NaT)
    out = pd.DataFrame({
        "date": pd.to_datetime(decisions["date"]).to_numpy(), "ticker": decisions["ticker"].to_numpy(),
        "entry_date": entry_dates, "exit_date": exit_dates,
        "reason": np.where(complete, reason, "incomplete"), "held_sessions": np.where(complete, exit_k, np.nan),
        "entry_px": entry_px, "exit_px": np.where(complete, exit_px, np.nan),
        "dividends": np.where(complete, divs, np.nan),
        "gross": np.where(complete, gross, np.nan), "net": np.where(complete, gross - cost, np.nan),
        "mae": np.where(complete, mae, np.nan), "profit_level": profit, "risk_level": risk,
        "cost_rt": cost, "complete": complete})
    out["policy"] = policy.name
    return out


def benchmark(close_adj: pd.Series, trades_frame: pd.DataFrame) -> np.ndarray:
    """A total-return benchmark (SPY, or the cash fund) over each trade's own entry and exit closes."""
    s = close_adj.reindex(close_adj.index)
    a = s.reindex(pd.to_datetime(trades_frame["entry_date"])).to_numpy(dtype=float)
    b = s.reindex(pd.to_datetime(trades_frame["exit_date"])).to_numpy(dtype=float)
    return b / a - 1


# --- a capital-constrained book: the historical simulation and the prospective shadow account ---

@dataclass
class Lot:
    ticker: str
    shares: float
    entry_date: pd.Timestamp
    entry_px: float
    deadline: pd.Timestamp
    profit_level: float
    risk_level: float
    cost_rt: float
    source: str = ""


@dataclass
class Book:
    """Cash (settled and settling), lots, and a dated log. ``sessions`` is the trading calendar."""
    sessions: pd.DatetimeIndex
    cash: float
    settling: dict = field(default_factory=dict)        # settle date -> amount
    lots: list[Lot] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    costs: float = 0.0

    def _next(self, day: pd.Timestamp, k: int = 1) -> pd.Timestamp:
        """The ``k``-th session after ``day``. Past the end of the loaded prices it comes from the exchange
        calendar (``ledger.sessions``), never clamped to the last row: a lot bought on the latest session
        must get a deadline ``k`` sessions later, not that same session."""
        i = self.sessions.get_loc(day)
        if i + k < len(self.sessions):
            return self.sessions[i + k]
        from tradingagents.ledger.sessions import next_session

        return pd.Timestamp(next_session(self.sessions[-1].date(), i + k - (len(self.sessions) - 1)))

    def settle(self, day: pd.Timestamp):
        for d in [d for d in self.settling if d <= day]:
            self.cash += self.settling.pop(d)

    def buy(self, day, ticker, dollars, price, policy: Policy, sigma, cost_rt, source="") -> Lot | None:
        """Spend settled cash only; ``dollars`` includes half the round-trip cost."""
        if dollars > self.cash + 1e-9 or dollars <= 0 or not np.isfinite(price) or price <= 0:
            return None
        fee = dollars * cost_rt / 2
        profit, risk = policy.levels(np.array([sigma]))
        lot = Lot(ticker, (dollars - fee) / price, day, price, self._next(day, policy.horizon), float(profit[0]),
                  float(risk[0]), cost_rt, source)
        self.cash -= dollars
        self.costs += fee
        self.lots.append(lot)
        self.log.append({"date": day, "action": "buy", "ticker": ticker, "dollars": dollars, "price": price})
        return lot

    def sell(self, day, lot: Lot, price, reason, fraction: float = 1.0):
        if day <= lot.entry_date:
            raise ValueError(f"{lot.ticker}: a lot cannot be sold on the session it was bought (no day trades)")
        shares = lot.shares * fraction
        value = shares * price
        fee = value * lot.cost_rt / 2
        self.settling[self._next(day)] = self.settling.get(self._next(day), 0.0) + value - fee
        self.costs += fee
        lot.shares -= shares
        if lot.shares <= 1e-12:
            self.lots.remove(lot)
        self.log.append({"date": day, "action": "sell" if fraction >= 1 else "reduce", "ticker": lot.ticker,
                         "value": value, "price": price, "reason": reason})

    def dividends(self, day, div_row: pd.Series):
        for lot in self.lots:
            amount = float(div_row.get(lot.ticker, 0.0) or 0.0)
            if amount and day > lot.entry_date:
                self.cash += amount * lot.shares        # credited on the ex-date (pay dates are not in the data)

    def value(self, prices: pd.Series) -> float:
        held = sum(l.shares * float(prices.get(l.ticker, np.nan)) for l in self.lots)
        return self.cash + sum(self.settling.values()) + (held if np.isfinite(held) else np.nan)

    def exits_due(self, day, prices: pd.Series) -> list[tuple[Lot, str]]:
        """Close-confirmed checks at this close: lots to exit at the **next** close, with the reason."""
        due = []
        for lot in self.lots:
            p = float(prices.get(lot.ticker, np.nan))
            if self._next(day) >= lot.deadline:
                due.append((lot, "deadline"))
            elif np.isfinite(p) and p / lot.entry_px - 1 <= lot.risk_level:
                due.append((lot, "stop"))
            elif np.isfinite(p) and p / lot.entry_px - 1 >= lot.profit_level:
                due.append((lot, "target"))
        return due
