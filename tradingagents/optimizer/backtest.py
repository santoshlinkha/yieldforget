"""Walk-forward backtest of the optimizer: would its rules have kept their promise?

At each rebalance date the optimizer sees only prices up to that day's close: the
crash estimates, crash scenarios, returns and correlations it uses are rebuilt
from that history alone. It trades at that close, the mix drifts with the market
from the next day on, and the next rebalance starts from wherever that left it.

Two checks of the promise:
- the strategy's own worst drop over the whole run, against its loss limit;
- every rebalance's mix held unchanged for the following year: how deep it fell,
  against the limit it was chosen under. A model that is right about risk breaks
  this rarely; one that is wrong breaks it in every crash it did not foresee.

What it cannot test:
- AI ratings. Re-running the agents for every fund on every past date would cost
  thousands of runs, and the models already know how those years turned out.
- Survivorship. The scan list is today's; a fund that closed in 2015 is not in it.
- The daily-buy path. Rebalances are instant; ``cost_bps`` charges each dollar
  traded (spreads; a Roth IRA at Robinhood pays no commissions or tax).

T-bills (BIL) stand in for SGOV throughout, since SGOV only exists from 2020.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TRADING_DAYS, compute_stats, max_drawdown
from .plan import OptimizerSettings, forecast_excess, growth_excess, plan_weights, scan_list, stress_models


# What the strategy is shown against. A fund that did not exist yet holds cash (BIL)
# until it did, so "SGOV" is T-bills before mid-2020.
BENCHMARKS = {"VTI": {"VTI": 1.0}, "SPY": {"SPY": 1.0}, "SGOV": {"SGOV": 1.0}}


@dataclass
class Metrics:
    final: float          # growth of $1
    cagr: float
    max_drawdown: float
    volatility: float
    worst_month: float    # worst 21-trading-day return
    sharpe: float = 0.0   # annualized, over cash
    worst_year: float = 0.0   # worst 252-trading-day return


@dataclass
class Rebalance:
    date: str
    invested: float
    limit: float | None       # the loss limit the mix was chosen under
    predicted: float          # the mix's stress loss under the risk model, as chosen
    expected: float           # the mix's expected growth per year, as chosen
    next_year_drop: float | None = None   # worst drop of the mix held a year; None near the end


@dataclass
class BacktestResult:
    start: str
    end: str
    loss_limit: float | None
    dates: list[str]                       # weekly samples of the curves
    curves: dict[str, list[float]]         # strategy and benchmarks, each starting at 1.0
    metrics: dict[str, Metrics]
    invested: list[tuple[str, float]]      # (rebalance date, share invested)
    last_mix: dict[str, float]
    rebalances: int
    notes: list[str] = field(default_factory=list)
    risk_model: str = "worst_case"
    target_growth: float | None = None
    cost_bps: float = 0.0
    turnover: float = 0.0                  # share of the portfolio traded per year
    history: list[Rebalance] = field(default_factory=list)

    @property
    def breaches(self) -> list[Rebalance]:
        """Rebalances whose mix, held a year, fell further than the limit it was chosen under."""
        return [r for r in self.history if r.limit is not None and r.next_year_drop is not None
                and r.next_year_drop > r.limit + 1e-9]


def _metrics(curve: pd.Series, cash: pd.Series | None = None) -> Metrics:
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    daily = curve.pct_change().dropna()
    vol = float(daily.std() * np.sqrt(TRADING_DAYS))
    excess = daily - (cash.reindex(daily.index).fillna(0.0) if cash is not None else 0.0)
    return Metrics(
        final=float(curve.iloc[-1] / curve.iloc[0]),
        cagr=float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0,
        max_drawdown=max_drawdown(curve),
        volatility=vol,
        worst_month=float((curve / curve.shift(21) - 1).min()),
        sharpe=float(excess.mean() * TRADING_DAYS / vol) if vol > 0 else 0.0,
        worst_year=float((curve / curve.shift(TRADING_DAYS) - 1).min()) if len(curve) > TRADING_DAYS else 0.0,
    )


def _simulate(returns: pd.DataFrame, cash: pd.Series, schedule: list[tuple[pd.Timestamp, dict[str, float]]],
              cost: float = 0.0) -> tuple[pd.Series, float]:
    """Value of $1 traded to each scheduled mix at that day's close, drifting with prices
    in between, and the total share of the portfolio traded. ``cost`` is charged per
    dollar of fund traded; moving money in and out of cash is free."""
    value, holdings, out, traded = 1.0, {"__cash__": 1.0}, {}, 0.0
    plan = dict(schedule)
    for day in returns.index:
        for t in holdings:
            r = cash[day] if t == "__cash__" else returns.at[day, t]
            holdings[t] *= 1 + (0.0 if np.isnan(r) else r)
        value = sum(holdings.values())
        if day in plan:
            mix = plan[day]
            trade = sum(abs(w * value - holdings.get(t, 0.0)) for t, w in mix.items())
            trade += sum(v for t, v in holdings.items() if t != "__cash__" and t not in mix)
            traded += trade / value if value else 0.0
            value -= cost * trade
            holdings = {t: w * value for t, w in mix.items()}
            holdings["__cash__"] = value * (1 - sum(mix.values()))
        out[day] = value
    return pd.Series(out), traded


def _held_drop(returns: pd.DataFrame, cash: pd.Series, day: pd.Timestamp, mix: dict[str, float],
               days: int = TRADING_DAYS) -> float | None:
    """Worst drop of ``mix`` bought at ``day``'s close and left alone for ``days``; None if
    the history ends sooner."""
    after = returns.loc[returns.index > day].iloc[:days]
    if len(after) < days:
        return None
    value = pd.Series(1 - sum(mix.values()), index=after.index) * (1 + cash.loc[after.index]).cumprod()
    for t, w in mix.items():
        value += w * (1 + after[t].fillna(0.0)).cumprod()
    return max_drawdown(pd.concat([pd.Series([1.0]), value.reset_index(drop=True)]))


def rebalance_plan(closes: pd.DataFrame, candidates: list[str], settings: OptimizerSettings,
                   days: list[pd.Timestamp], progress: Callable[[int, int], None] | None = None,
                   mu_for: Callable[[pd.Timestamp], dict[str, float]] | None = None
                   ) -> tuple[list[tuple[pd.Timestamp, dict[str, float]]], list[Rebalance]]:
    """The optimizer's mix at each of ``days``, from prices up to that day's close only.

    ``mu_for(day)``, when given, supplies forecast returns made with data up to ``day``
    (the walk-forward's own out-of-sample predictions), for ``return_model="forecast"``.
    """
    cash_ticker = settings.cash_proxy
    schedule, history = [], []
    for n, day in enumerate(days, 1):
        if progress:
            progress(n, len(days))
        past = closes.loc[:day]
        usable, _ = scan_list(past, [t for t in candidates if t in past], settings.min_history_years)
        if not usable:
            schedule.append((day, {}))
            history.append(Rebalance(day.strftime("%Y-%m-%d"), 0.0, settings.loss_limit, 0.0, 0.0))
            continue
        proxies = {t: p for t, p in settings.proxies.items() if t in usable}
        stats = compute_stats(past, [cash_ticker, *usable], cash_ticker, settings.lookback_years,
                              settings.short_history_years, settings.short_history_floor, proxies)
        model, _ = stress_models(stats, settings, past, proxies)
        if mu_for is not None:
            settings.forecast_mu = mu_for(day)
            excess, _ = forecast_excess(stats, settings, set())
        else:
            excess = growth_excess(stats, settings)
        weights, limit, _ = plan_weights(stats, settings, excess, model)
        mix = {t: float(w) for t, w in zip(stats.tickers, weights, strict=True) if w > 1e-4}
        schedule.append((day, mix))
        history.append(Rebalance(day.strftime("%Y-%m-%d"), float(weights.sum()), limit,
                                 model.loss(weights), float(stats.cash_return + excess @ weights)))
    return schedule, history


def run_backtest(closes: pd.DataFrame, candidates: list[str], settings: OptimizerSettings,
                 start: str = "2012-01-03", every: int = 21, current_mix: dict[str, float] | None = None,
                 progress: Callable[[int, int], None] | None = None, cost_bps: float = 0.0,
                 schedule: list | None = None, history: list[Rebalance] | None = None,
                 benchmarks: dict[str, dict[str, float]] | None = None) -> BacktestResult:
    """Rebalance every ``every`` trading days from ``start``, as the optimizer would have.

    ``schedule`` and ``history`` from an earlier ``rebalance_plan`` over the same
    days skip the optimizing, to score one plan at several costs.
    """
    cash_ticker = settings.cash_proxy
    if cash_ticker not in closes:
        raise ValueError(f"no price history for {cash_ticker}, the backtest's cash")
    candidates = [t for t in dict.fromkeys(candidates) if t not in (cash_ticker, settings.cash_ticker)]
    all_returns = closes.pct_change(fill_method=None)
    returns = all_returns[all_returns.index >= pd.Timestamp(start)]
    if returns.empty:
        raise ValueError(f"no prices after {start}")
    cash = all_returns[cash_ticker].fillna(0.0)
    days = list(returns.index)
    rebalance_days = days[::every]

    if schedule is None:
        schedule, history = rebalance_plan(closes, candidates, settings, rebalance_days, progress)
    for (day, mix), r in zip(schedule, history, strict=True):
        r.next_year_drop = _held_drop(all_returns, cash, day, mix)

    cost = cost_bps / 1e4
    curve, traded = _simulate(returns, cash, schedule, cost)
    curves = {"Optimizer": curve}
    for name, mix in {**(BENCHMARKS if benchmarks is None else benchmarks),
                      "Your mix now": current_mix or {}}.items():
        if not mix or not all(t in returns for t in mix):
            continue
        # A fund that did not exist yet holds its share in cash until it does.
        fixed = [(day, {t: w for t, w in mix.items() if returns[t].loc[:day].notna().any()})
                 for day in rebalance_days]
        curves[name] = _simulate(returns, cash, fixed, cost)[0]

    weekly = curve.index[::5]
    years = (days[-1] - days[0]).days / 365.25
    notes = []
    if current_mix and "Your mix now" in curves:
        late = [t for t in current_mix if returns[t].first_valid_index() > returns.index[0]]
        if late:
            notes.append("Your mix now holds cash in place of " + ", ".join(late)
                         + " until each fund existed.")
    return BacktestResult(
        start=days[0].strftime("%Y-%m-%d"), end=days[-1].strftime("%Y-%m-%d"),
        loss_limit=settings.loss_limit,
        dates=[d.strftime("%Y-%m-%d") for d in weekly],
        curves={k: [round(float(v), 5) for v in c.loc[weekly]] for k, c in curves.items()},
        metrics={k: _metrics(c, cash) for k, c in curves.items()},
        invested=[(r.date, r.invested) for r in history], last_mix=schedule[-1][1] if schedule else {},
        rebalances=len(rebalance_days), notes=notes,
        risk_model=settings.risk_model, target_growth=settings.target_growth, cost_bps=cost_bps,
        turnover=traded / years if years > 0 else 0.0, history=history,
    )


def render_backtest(result: BacktestResult) -> str:
    limit = "none" if result.loss_limit is None else f"{result.loss_limit:.1%}"
    model = "crash scenarios" if result.risk_model == "scenarios" else "worst case"
    goal = f", growth target {result.target_growth:.1%}" if result.target_growth is not None else ""
    lines = [f"# Backtest {result.start} to {result.end}", "",
             f"Loss limit {limit} ({model}){goal}, rebalanced {result.rebalances} times, each time from "
             f"prices known then; {result.turnover:.0%} of the portfolio traded a year"
             + (f" at {result.cost_bps:g} bp a trade." if result.cost_bps else "."), "",
             "| | Growth of $1 | Per year | Worst drop | Volatility | Sharpe | Worst month | Worst year |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in result.metrics.items():
        lines.append(f"| {name} | ${m.final:,.2f} | {m.cagr:.1%} | {m.max_drawdown:.1%} "
                     f"| {m.volatility:.1%} | {m.sharpe:.2f} | {m.worst_month:.1%} | {m.worst_year:.1%} |")
    opt = result.metrics["Optimizer"]
    if result.loss_limit is not None:
        held = opt.max_drawdown <= result.loss_limit
        lines += ["", f"The optimizer's worst drop was {opt.max_drawdown:.1%} against a {limit} limit: "
                  + ("the limit held." if held else "the limit did NOT hold over this period.")]
    scored = [r for r in result.history if r.limit is not None and r.next_year_drop is not None]
    if scored:
        broken = result.breaches
        lines.append(f"Each month's mix held for a year fell further than its limit {len(broken)} times "
                     f"in {len(scored)}" + (f", worst {max(r.next_year_drop - r.limit for r in broken):.1%} "
                                             f"past it ({max(broken, key=lambda r: r.next_year_drop - r.limit).date})."
                                             if broken else "."))
    lines += [f"- {n}" for n in result.notes]
    return "\n".join(lines) + "\n"
