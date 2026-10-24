"""From holdings, a buy schedule and a loss limit to a target mix and the trades that reach it.

The loss limit is a hard constraint on the *stress loss*, measured one of two ways
(``risk_model``):

- ``worst_case``: what the portfolio would lose if every holding repeated the
  deepest drop of its history at the same time. Crashes are when assets fall
  together, so this bounds the case where they all do, and gives no credit for
  diversification.
- ``scenarios``: the worst of the real past crashes replayed on the mix (see
  scenarios.py), so holdings that did not fall together count as not falling
  together. A fund never seen in a crash is still taken at its worst drop.

Either way it is not a guarantee; a future crash can be deeper than any past one.

With a ``target_growth`` the plan aims for that expected growth at the least
stress loss that reaches it, and never goes past the loss limit of the risk level.

No asset may use more than ``max_budget_share`` of the loss limit. Without it the
whole budget goes to the asset with the most expected return per unit of stress
loss, which is always a leveraged fund: its loss stops at 100% while its
volatility, and so its expected return, keeps scaling.

Returns are shown as compound growth (mean minus half the variance): that is what
a holding actually grows at, and the average return overstates it badly for a
volatile fund.

AI ratings, when given, move only the expected returns. They can shift money
between assets but never past the loss limit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.portfolio import PortfolioContext

from .data import TRADING_DAYS, AssetStats, compute_stats, fetch_closes, max_drawdown
from .risk import DEFAULT_LEVEL, RISK_LEVELS, RISK_MODELS, risk_level
from .scenarios import REFERENCES, Scenario, find_scenarios, scenario_losses
from .solve import solve_weights
from .universe import describe

# Rating → notches of Sharpe tilt; one notch is OptimizerSettings.tilt_per_notch.
RATING_NOTCHES = {"Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2}


@dataclass
class OptimizerSettings:
    # Stress loss cap, as a fraction of the whole portfolio; set from a risk level (see risk.py).
    loss_limit: float | None = risk_level(DEFAULT_LEVEL).loss_limit
    cash_ticker: str = "SGOV"         # where uninvested money waits
    max_weight: float = 0.35          # cap on any one risky asset
    max_budget_share: float = 0.25    # cap on the share of the loss limit any one asset may use
    risk_aversion: float = 3.0
    prior_sharpe: float = 0.3         # what every asset is assumed to earn per unit of risk
    history_weight: float = 0.5       # how far each asset's own past Sharpe moves it off the prior
    tilt_per_notch: float = 0.15      # Sharpe added per rating notch above Hold
    lookback_years: float = 10
    short_history_years: float = 10
    short_history_floor: float = 0.5
    # A scanned fund needs this much history to be trusted; held funds are always kept.
    min_history_years: float = 3
    # Scanned funds taken into the optimizer, best scores first; the rest are screened out.
    max_candidates: int = 60
    # Trades smaller than this share of the portfolio are skipped, so the plan does not churn.
    rebalance_band: float = 0.005
    # Longer-lived funds on the same market, for the crash history of a young one.
    proxies: dict = field(default_factory=lambda: {"QQQM": "QQQ", "AVUV": "VBR"})
    cash_proxy: str = "BIL"           # stands in for the cash fund before it existed
    frontier: tuple = tuple(r.loss_limit for r in RISK_LEVELS)
    risk_model: str = "worst_case"    # or "scenarios"; see the module docstring
    # Expected growth per year to aim for (0.08 is 8%), cash included; None: the most
    # growth the loss limit allows.
    target_growth: float | None = None
    # Market references whose past falls are the crash scenarios, and how deep a fall counts.
    scenario_references: dict = field(default_factory=lambda: dict(REFERENCES))

    def __post_init__(self):
        if self.risk_model not in RISK_MODELS:
            raise ValueError(f"risk model must be one of {', '.join(RISK_MODELS)}, not {self.risk_model}")


@dataclass
class Row:
    ticker: str
    price: float
    value: float
    current: float
    target: float
    worst_loss: float
    expected: float
    rating: str | None = None
    floored: bool = False
    trade: float = 0.0
    daily_buy: float = 0.0
    name: str = ""
    held: bool = False
    score: float = 0.0     # growth above cash per unit of worst drop
    reason: str = ""
    screened: bool = False  # scored cheaply and left out before the optimizer ran


@dataclass
class FrontierPoint:
    loss_limit: float | None
    invested: float
    stress_loss: float
    expected_return: float


@dataclass
class GrowthTarget:
    target: float              # expected growth per year aimed for
    needed: float | None       # least stress-loss limit that reaches it; None: only no limit does
    reachable: bool            # False: no mix reaches it, not even with no limit
    level_limit: float | None  # the risk level's limit, which the plan never goes past

    @property
    def capped(self) -> bool:
        """Whether the risk level stops the plan short of the target."""
        if not self.reachable:
            return True
        if self.level_limit is None:
            return False
        return self.needed is None or self.needed > self.level_limit + 1e-9


@dataclass
class Plan:
    total: float
    rows: list[Row]
    cash: Row
    settings: OptimizerSettings
    daily_budget: float
    days_to_target: float | None
    current_stress: float            # under settings.risk_model
    target_stress: float
    current_history_dd: tuple[float, str]
    target_history_dd: tuple[float, str]
    expected_current: float
    expected_target: float
    frontier: list[FrontierPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scanned: int = 0
    skipped: list[str] = field(default_factory=list)
    # Both measures, (current, target), whichever one is in force; scenarios is None without any.
    stress_worst_case: tuple[float, float] = (0.0, 0.0)
    stress_scenarios: tuple[float, float] | None = None
    worst_scenario: tuple[str | None, str | None] = (None, None)   # (current, target)
    scenarios: list[str] = field(default_factory=list)
    goal: GrowthTarget | None = None


def load_schedule(path: str | Path) -> dict[str, float]:
    """Dollars bought per trading day, by ticker, from ``{"daily": {...}, "weekly": {...}}``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"schedule file {path} is not usable: {exc}") from exc
    per_day: dict[str, float] = {}
    for period, divisor in (("daily", 1), ("weekly", 5)):
        for ticker, amount in (data.get(period) or {}).items():
            per_day[ticker.upper()] = per_day.get(ticker.upper(), 0.0) + float(amount) / divisor
    return per_day


def expected_excess(stats: AssetStats, settings: OptimizerSettings,
                    ratings: dict[str, str] | None = None) -> np.ndarray:
    """Expected annual return above cash: volatility times a shrunk, tilted Sharpe.

    The less history an asset has, the less its own Sharpe counts: four great years
    of a stock listed in 2022 are weak evidence next to a common prior, and taking
    them at face value would make the optimizer chase whatever just ran up.
    """
    tilt = np.array([RATING_NOTCHES.get((ratings or {}).get(t) or "", 0) for t in stats.tickers])
    trust = settings.history_weight * np.clip(stats.years / settings.lookback_years, 0.0, 1.0)
    blended = (trust * np.clip(stats.sharpe, -1.0, 1.5)
               + (1 - trust) * settings.prior_sharpe
               + settings.tilt_per_notch * tilt)
    return stats.vol * blended


def history_drawdown(closes: pd.DataFrame, weights: dict[str, float], cash_ticker: str,
                     cash_return: float, lookback_years: float) -> tuple[float, str]:
    """Deepest drop of a daily-rebalanced mix over the lookback, and when its window starts.

    Days before the cash fund existed earn today's cash yield. The window starts
    when every held asset has prices, so a recently listed holding shortens it.
    """
    returns = closes.pct_change(fill_method=None)
    returns = returns[returns.index > returns.index[-1] - pd.DateOffset(years=lookback_years)]
    held = [t for t, w in weights.items() if w > 1e-6 and t != cash_ticker]
    cash = returns[cash_ticker].fillna(cash_return / TRADING_DAYS)
    frame = returns[held].dropna() if held else returns[[]]
    if held and frame.empty:
        return 0.0, "n/a"
    daily = cash.loc[frame.index] * weights.get(cash_ticker, 0.0)
    for t in held:
        daily = daily + frame[t] * weights[t]
    curve = (1 + daily).cumprod()
    return max_drawdown(curve), frame.index[0].strftime("%Y-%m-%d") if len(frame) else "n/a"


def growth_excess(stats: AssetStats, settings: OptimizerSettings,
                  ratings: dict[str, str] | None = None) -> np.ndarray:
    """Expected compound growth above cash.

    Each holding compounds at its own growth rate between buys, so it is ranked by
    that, not by its average return: at a small weight the portfolio's variance
    term is negligible and would leave a volatile fund's drag unpriced.
    """
    return expected_excess(stats, settings, ratings) - 0.5 * stats.vol ** 2


def upper_bounds(stats: AssetStats, settings: OptimizerSettings, limit: float | None) -> np.ndarray:
    """Per-fund caps. The budget share always uses the fund's own worst drop, under either
    risk model: a fund can crash on its own, outside any market-wide scenario."""
    upper = np.full(len(stats.tickers), settings.max_weight)
    if limit is not None:
        upper = np.minimum(upper, settings.max_budget_share * limit / np.maximum(stats.worst_loss, 1e-9))
    return upper


# A fund that gained in every scenario still counts as able to lose this much, for its score.
CRASH_FLOOR = 0.01


@dataclass
class StressModel:
    """What the loss limit is measured on: rows of per-asset losses, the limit holding for each."""
    rows: np.ndarray               # one row per constraint, one column per asset
    owners: list[int]              # the scenario of each row; empty for the worst case
    scenarios: list[Scenario]
    crash: np.ndarray              # each asset's own crash estimate under this model
    word: str                      # what ``crash`` is, for the explanations

    def loss(self, w: np.ndarray) -> float:
        return max(float((self.rows @ w).max()), 0.0) if len(self.rows) else 0.0

    def binding(self, w: np.ndarray) -> str | None:
        """The scenario the mix loses most in."""
        if not self.owners or not np.any(w > 1e-9):
            return None
        return self.scenarios[self.owners[int(np.argmax(self.rows @ w))]].name


def worst_case_model(stats: AssetStats) -> StressModel:
    return StressModel(stats.worst_loss[None, :], [], [], stats.worst_loss, "worst drop")


def scenario_model(stats: AssetStats, closes: pd.DataFrame, scenarios: list[Scenario],
                   proxies: dict[str, str] | None) -> StressModel | None:
    """The crash scenarios as loss rows; None when the history holds no scenario."""
    rows, owners = scenario_losses(closes, stats.tickers, scenarios, stats.worst_loss, proxies)
    if not len(rows):
        return None
    return StressModel(rows, owners, scenarios, np.maximum(rows.max(axis=0), CRASH_FLOOR),
                       "loss in its worst past crash")


def solver_for(stats: AssetStats, settings: OptimizerSettings, excess: np.ndarray,
               model: StressModel) -> Callable[[float | None], np.ndarray]:
    last = {}

    def solve(limit):
        # Starts from the answer at the nearest limit solved so far: a bisection's
        # limits close in on each other, so the answers do too.
        near = min(last, key=lambda k: abs(k - (limit if limit is not None else 1.0)), default=None)
        x = solve_weights(excess, stats.cov, model.rows, limit, upper_bounds(stats, settings, limit),
                          settings.risk_aversion, start=last.get(near))
        last[limit if limit is not None else 1.0] = x
        return x
    return solve


def growth_target(solve: Callable[[float | None], np.ndarray], growth: Callable[[np.ndarray], float],
                  target: float, level_limit: float | None, tol: float = 5e-4) -> GrowthTarget:
    """The least stress-loss limit whose plan expects at least ``target`` growth, by bisection.

    A higher limit never lowers the plan's expected growth, so the limits that reach
    the target are an interval and its lower end is found to within ``tol``. A limit
    of 100% no longer binds any loss row; past it only "no limit at all" remains,
    which also lifts the per-fund loss-budget caps.
    """
    def reaches(limit):
        return growth(solve(limit)) >= target - 1e-9

    if not reaches(1.0):
        return GrowthTarget(target, None, reaches(None), level_limit)
    if reaches(tol):
        return GrowthTarget(target, tol, True, level_limit)
    lo, hi = tol, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if reaches(mid) else (mid, hi)
    return GrowthTarget(target, hi, True, level_limit)


def plan_weights(stats: AssetStats, settings: OptimizerSettings, excess: np.ndarray,
                 model: StressModel) -> tuple[np.ndarray, float | None, GrowthTarget | None]:
    """Target weights, the loss limit they were solved at, and the growth target if one is set."""
    solve = solver_for(stats, settings, excess, model)
    if settings.target_growth is None:
        return solve(settings.loss_limit), settings.loss_limit, None
    goal = growth_target(solve, lambda w: float(stats.cash_return + excess @ w),
                         settings.target_growth, settings.loss_limit)
    limit = goal.level_limit if goal.capped else goal.needed
    return solve(limit), limit, goal


def stress_models(stats: AssetStats, settings: OptimizerSettings, closes: pd.DataFrame,
                  proxies: dict[str, str] | None) -> tuple[StressModel, StressModel | None]:
    """The model in force, and the scenario model (None without scenarios) for comparison."""
    scenarios = find_scenarios(closes, settings.scenario_references)
    scen = scenario_model(stats, closes, scenarios, proxies) if scenarios else None
    model = scen if settings.risk_model == "scenarios" and scen is not None else worst_case_model(stats)
    return model, scen


def target_weights(stats: AssetStats, settings: OptimizerSettings, excess: np.ndarray,
                   limit: float | None) -> np.ndarray:
    """Worst-case weights at ``limit``, without a growth target."""
    return solver_for(stats, settings, excess, worst_case_model(stats))(limit)


def scan_list(closes: pd.DataFrame, candidates, min_years: float) -> tuple[list[str], list[str]]:
    """The candidates with enough price history to trust, and the ones skipped."""
    last = closes.index[-1]
    keep, skip = [], []
    for t in candidates:
        first = closes[t].first_valid_index() if t in closes else None
        (keep if first is not None and (last - first).days / 365.25 >= min_years else skip).append(t)
    return keep, skip


def _explain(i: int, stats: AssetStats, excess: np.ndarray, target: np.ndarray, upper: np.ndarray,
             settings: OptimizerSettings, ratings: dict | None, model: StressModel) -> str:
    """Why this fund gets the weight it gets, in the numbers that decided it."""
    t, w, g, d = stats.tickers[i], target[i], excess[i], stats.worst_loss[i]
    c = model.crash[i]
    limit = settings.loss_limit
    sd = np.sqrt(np.diag(stats.cov))
    chosen = [j for j in range(len(target)) if target[j] > 1e-4 and j != i]
    overlap = max(chosen, key=lambda j: stats.cov[i, j] / (sd[i] * sd[j]), default=None)
    corr = stats.cov[i, overlap] / (sd[i] * sd[overlap]) if overlap is not None else 0.0
    scores = excess / np.maximum(model.crash, 1e-9)

    if g <= 0:
        text = (f"Expected growth is not above cash ({g:+.1%} a year), so no crash risk is worth "
                f"taking on it.")
    elif w >= upper[i] - 1e-4 and upper[i] < settings.max_weight - 1e-6:
        text = (f"Earns {g:+.1%} a year above cash for a {d:.0%} worst drop. Held at the cap: one fund "
                f"may use at most {settings.max_budget_share:.0%} of the loss budget.")
    elif w >= settings.max_weight - 1e-4:
        text = f"Earns {g:+.1%} a year above cash. Held at the {settings.max_weight:.0%} single-fund cap."
    elif w > 1e-4:
        text = f"Earns {g:+.1%} a year above cash for a {c:.0%} {model.word}."
        if overlap is not None and corr > 0.7:
            text += (f" Not more, because it moves with {stats.tickers[overlap]} (correlation "
                     f"{corr:.2f}), which the plan also holds.")
        elif limit is not None:
            text += " Not more, because the rest of the loss budget earns more elsewhere."
    elif overlap is not None and corr > 0.85:
        text = (f"Earns {g:+.1%} a year above cash, but it largely duplicates {stats.tickers[overlap]} "
                f"(correlation {corr:.2f}), which earns more per unit of crash risk.")
    else:
        best = int(np.argmax(np.where(target > 1e-4, scores, -np.inf))) if (target > 1e-4).any() else None
        text = (f"Earns {g:+.1%} a year above cash, but its {c:.0%} {model.word} makes that "
                f"{scores[i]:.2f} per unit of crash risk")
        text += (f", below the funds chosen (best: {stats.tickers[best]} at {scores[best]:.2f})."
                 if best is not None else ".")
    rating = (ratings or {}).get(t)
    if rating and RATING_NOTCHES.get(rating):
        tilt = settings.tilt_per_notch * RATING_NOTCHES[rating] * stats.vol[i]
        text += f" The AI rating {rating} moved its expected growth {tilt:+.1%}."
    if stats.floored[i]:
        text += f" Under {settings.short_history_years:g} years of history, so a {d:.0%} crash is assumed."
    return text


def build_plan(portfolio: PortfolioContext, schedule: dict[str, float], settings: OptimizerSettings,
               ratings: dict[str, str] | None = None, closes: pd.DataFrame | None = None,
               universe: list[str] | None = None, cache_dir=None, summary: bool = True) -> Plan:
    """The target mix and the trades to it. ``summary=False`` skips the frontier and the
    historical drawdowns, which a day-by-day replay does not need."""
    cash_ticker = settings.cash_ticker.upper()
    quantities = {p.ticker.upper(): p.quantity for p in portfolio.positions}
    owned = list(dict.fromkeys([*quantities, *schedule, cash_ticker]))
    candidates = [t for t in dict.fromkeys(universe or []) if t not in owned]
    proxies = {t: p for t, p in settings.proxies.items() if t in owned or t in candidates}
    if closes is None:
        closes = fetch_closes([*owned, *candidates, *proxies.values(), settings.cash_proxy,
                               *settings.scenario_references], required=owned, cache_dir=cache_dir)
    candidates, skipped = scan_list(closes, candidates, settings.min_history_years)

    # Screen: score every candidate cheaply, optimize over your holdings and the best scorers.
    def stats_for(names, with_cov=True):
        return compute_stats(closes, names, cash_ticker, settings.lookback_years,
                             settings.short_history_years, settings.short_history_floor, proxies,
                             settings.cash_proxy, with_cov=with_cov)

    scenarios = find_scenarios(closes, settings.scenario_references)
    screened: list[Row] = []
    if len(candidates) > settings.max_candidates:
        light = stats_for([cash_ticker, *candidates], with_cov=False)
        light_excess = growth_excess(light, settings, ratings)
        light_crash, word = light.worst_loss, "worst drop"
        if settings.risk_model == "scenarios" and scenarios:
            # Each fund alone: its worst row, so no pruning is needed.
            alone, _ = scenario_losses(closes, light.tickers, scenarios, light.worst_loss, proxies, prune=False)
            if len(alone):
                light_crash, word = np.maximum(alone.max(axis=0), CRASH_FLOOR), "loss in its worst past crash"
        light_scores = light_excess / np.maximum(light_crash, 1e-9)
        order = np.argsort(-light_scores)
        kept = {light.tickers[i] for i in order[:settings.max_candidates]}
        for rank, i in enumerate(order, 1):
            t = light.tickers[i]
            if t in kept:
                continue
            g, d = light_excess[i], light.worst_loss[i]
            why = (f"Expected growth is not above cash ({g:+.1%} a year), so no crash risk is worth taking on it."
                   if g <= 0 else
                   f"Screened out: earns {g:+.1%} a year above cash for a {light_crash[i]:.0%} {word}, "
                   f"{light_scores[i]:.2f} per unit of crash risk, ranked {rank} of {len(order)}; only the "
                   f"top {settings.max_candidates} go to the optimizer.")
            screened.append(Row(t, light.prices[t], 0.0, 0.0, 0.0, float(d),
                                float(light.cash_return + g), floored=bool(light.floored[i]),
                                name=describe(t), score=float(light_scores[i]), reason=why, screened=True))
        candidates = [t for t in candidates if t in kept]

    tickers = [*owned, *candidates]
    stats = stats_for(tickers)

    values = {t: quantities.get(t, 0.0) * stats.prices[t] for t in tickers}
    values[cash_ticker] += portfolio.cash or 0.0
    total = sum(values.values())
    if total <= 0:
        raise ValueError("the portfolio has no value to allocate")

    current = np.array([values[t] / total for t in stats.tickers])
    excess = growth_excess(stats, settings, ratings)
    scen = scenario_model(stats, closes, scenarios, proxies) if scenarios else None
    worst = worst_case_model(stats)
    model = scen if settings.risk_model == "scenarios" and scen is not None else worst
    solve = solver_for(stats, settings, excess, model)

    def growth(w):
        return float(stats.cash_return + excess @ w)

    level_settings = settings
    target, limit, goal = plan_weights(stats, settings, excess, model)
    # From here on the limit is the one the target was solved at, the growth target's if set.
    settings = replace(settings, loss_limit=limit)
    expected = stats.cash_return + excess
    upper = upper_bounds(stats, settings, limit)
    scores = excess / np.maximum(model.crash, 1e-9)

    rows = [Row(t, stats.prices[t], values[t], float(current[i]), float(target[i]),
                float(stats.worst_loss[i]), float(expected[i]), (ratings or {}).get(t),
                bool(stats.floored[i]), trade=float(target[i] * total - values[t]),
                name=describe(t), held=t in quantities, score=float(scores[i]),
                reason=_explain(i, stats, excess, target, upper, settings, ratings, model))
            for i, t in enumerate(stats.tickers)]
    cash_row = Row(cash_ticker, stats.prices[cash_ticker], values[cash_ticker],
                   values[cash_ticker] / total, float(1 - target.sum()), 0.0, stats.cash_return,
                   trade=float((1 - target.sum()) * total - values[cash_ticker]),
                   name=describe(cash_ticker), held=True,
                   reason=f"Whatever the loss limit does not let the plan invest waits here, earning "
                          f"{stats.cash_return:.1%} a year with no meaningful crash risk.")

    # Daily buys go to the assets still under target, in proportion to the gap,
    # funded from cash. Nothing under target means the buys should pause.
    budget = sum(schedule.values())
    gaps = {r.ticker: r.trade for r in rows if r.trade > 0.005 * total}
    days = None
    if gaps and budget > 0:
        need = sum(gaps.values())
        for r in rows:
            r.daily_buy = round(budget * gaps.get(r.ticker, 0.0) / need, 2)
        days = need / budget

    def mix(weights):
        return {**dict(zip(stats.tickers, weights, strict=True)), cash_ticker: float(1 - weights.sum())}

    plan = Plan(
        total=total, rows=rows, cash=cash_row, settings=settings, daily_budget=budget,
        days_to_target=days,
        current_stress=model.loss(current),
        target_stress=model.loss(target),
        current_history_dd=history_drawdown(closes, mix(current), cash_ticker, stats.cash_return,
                                            settings.lookback_years) if summary else (0.0, "n/a"),
        target_history_dd=history_drawdown(closes, mix(target), cash_ticker, stats.cash_return,
                                           settings.lookback_years) if summary else (0.0, "n/a"),
        expected_current=growth(current),
        expected_target=growth(target),
        scanned=len(candidates) + len(screened),
        skipped=skipped,
        stress_worst_case=(worst.loss(current), worst.loss(target)),
        stress_scenarios=(scen.loss(current), scen.loss(target)) if scen else None,
        worst_scenario=(scen.binding(current), scen.binding(target)) if scen else (None, None),
        scenarios=[s.name for s in scenarios],
        goal=goal,
    )
    plan.rows += screened
    for level_limit in (settings.frontier if summary else ()):
        w = solve(level_limit)
        plan.frontier.append(FrontierPoint(level_limit, float(w.sum()), model.loss(w), growth(w)))
    if level_settings.risk_model == "scenarios" and model is worst:
        plan.notes.append("No crash scenario is in the price history (none of "
                          + ", ".join(settings.scenario_references) + " fell far enough), "
                          "so the worst-case stress loss is used instead.")
    for r in rows:
        if r.floored and r.ticker in stats.tickers and (r.held or r.target > 1e-4):
            plan.notes.append(f"{r.ticker} has {stats.years[stats.tickers.index(r.ticker)]:.1f} years of "
                              f"history, so its stress loss is assumed to be {settings.short_history_floor:.0%}.")
    if gaps and budget <= 0:
        plan.notes.append("No buy schedule was given, so the plan has trades but no daily amounts.")
    if skipped:
        plan.notes.append(f"Not scanned, under {settings.min_history_years:g} years of prices or none at all: "
                          + ", ".join(skipped) + ".")
    return plan


@dataclass
class Order:
    side: str       # "buy" or "sell"
    ticker: str
    amount: float   # dollars
    why: str


@dataclass
class DailySuggestion:
    orders: list[Order]                  # to place today
    settled_cash: float
    cash_left: float                     # settled cash still unspent after today's buys
    settling_tomorrow: float             # proceeds of today's sells, spendable next trading day
    tomorrow: list[Order] = field(default_factory=list)   # buys those proceeds pay for
    notes: list[str] = field(default_factory=list)


def _split(amount: float, gaps: dict[str, float]) -> dict[str, float]:
    """``amount`` shared across the gaps in proportion, never more than a gap."""
    total = sum(gaps.values())
    if total <= 0 or amount <= 0:
        return {}
    share = min(1.0, amount / total)
    return {t: round(g * share, 2) for t, g in gaps.items()}


def daily_orders(plan: Plan, settled_cash: float, min_order: float = 1.0) -> DailySuggestion:
    """Today's orders, and tomorrow's buys, for a cash account that settles sales T+1.

    Buys spend only settled cash, so nothing bought rests on an unsettled sale and
    selling it later can never be a good-faith violation. The rest follows from
    where the money comes from:

    - Settled cash is spent today on whatever is under target.
    - Money from trimming a holding is already invested, so it is reinvested in
      full the next trading day rather than left idle; what is not needed waits in
      the cash fund.
    - New money comes out of the cash fund at the daily buy pace, sold today so
      it has settled by tomorrow.

    Orders under ``min_order`` are dropped, as brokers refuse fractional orders that small.
    """
    cash = plan.cash.ticker
    by_ticker = {r.ticker: r for r in plan.rows}

    def why(ticker):
        r = by_ticker[ticker]
        return f"now {r.current:.1%}, target {r.target:.1%}. {r.reason}"

    # Within the band a holding is close enough to target; trading it would only churn.
    band = max(min_order, plan.settings.rebalance_band * plan.total)
    gaps = {r.ticker: r.trade for r in plan.rows if r.trade >= band}
    orders, tomorrow, notes = [], [], []

    # Today: settled cash goes to the gaps.
    for ticker, amount in _split(settled_cash, gaps).items():
        if amount >= min_order:
            orders.append(Order("buy", ticker, amount, why(ticker)))
            gaps[ticker] -= amount
    left = round(settled_cash - sum(o.amount for o in orders), 2)

    # Today: trims, whose proceeds settle tomorrow.
    trims = [(r.ticker, round(-r.trade, 2)) for r in plan.rows if -r.trade >= band]
    for ticker, amount in trims:
        orders.append(Order("sell", ticker, amount, why(ticker)))
    proceeds = round(sum(a for _, a in trims), 2)

    # Today: new money from the cash fund, at the daily pace, for what trims will not cover.
    remaining = max(sum(gaps.values()) - proceeds, 0.0)
    from_cash = round(min(remaining, plan.daily_budget, plan.cash.value), 2)
    if from_cash >= min_order:
        orders.append(Order("sell", cash, from_cash,
                            "new money for tomorrow's buys, at your daily pace; it settles overnight"))
    else:
        from_cash = 0.0
    settling = round(proceeds + from_cash, 2)

    # Tomorrow: what settles is spent on the remaining gaps; the rest waits in cash.
    for ticker, amount in _split(settling, gaps).items():
        if amount >= min_order:
            tomorrow.append(Order("buy", ticker, amount, why(ticker)))
    idle = round(settling - sum(o.amount for o in tomorrow), 2)
    if idle >= min_order:
        tomorrow.append(Order("buy", cash, idle, "sale proceeds the plan does not need yet wait here"))
    # Settled cash beyond every gap is not needed; it waits in the cash fund.
    if left >= min_order:
        orders.append(Order("buy", cash, left, "nothing is under target; cash waits here"))
        left = 0.0

    if settled_cash < min_order and any(g >= min_order for g in gaps.values()):
        notes.append("No settled cash entered, so nothing can be bought today. If your account shows "
                     "settled cash, enter it under Settings > Holdings and the buys move to today.")
    if tomorrow:
        notes.append("Tomorrow's buys are paid for by today's sales once they settle (T+1).")
    return DailySuggestion(orders, settled_cash, left, settling, tomorrow, notes)


def render_suggestion(s: DailySuggestion, day: str) -> str:
    lines = [f"# Orders for {day}", ""]
    if not s.orders:
        lines.append("No orders today.")
    for o in sorted(s.orders, key=lambda o: o.side != "sell"):
        lines.append(f"- **{o.side.upper()} ${o.amount:,.2f} {o.ticker}**: {o.why}")
    if s.tomorrow:
        lines += ["", "## Next trading day, once today's sales settle", ""]
        lines += [f"- **BUY ${o.amount:,.2f} {o.ticker}**: {o.why}" for o in s.tomorrow]
    lines += ["", f"Settled cash ${s.settled_cash:,.2f}, left after buys ${s.cash_left:,.2f}; "
              f"${s.settling_tomorrow:,.2f} settles next trading day."]
    lines += [f"- {n}" for n in s.notes]
    return "\n".join(lines) + "\n"


def _pct(x: float | None) -> str:
    return "none" if x is None else f"{x:.1%}"


MODEL_TEXT = {
    "worst_case": "in a stress test where every holding repeats its worst historical drop at once",
    "scenarios": "in the worst of the past crashes replayed on the mix",
}


def _goal_text(goal: GrowthTarget) -> str:
    level = next((r for r in RISK_LEVELS if r.loss_limit is None
                  or (goal.needed is not None and r.loss_limit >= goal.needed - 1e-9)), RISK_LEVELS[-1])
    if not goal.reachable:
        return (f"Growth target **{goal.target:.1%}** a year is more than any mix is expected to reach, "
                f"so the plan takes the most growth your risk level allows.")
    need = "no stress-loss limit at all" if goal.needed is None else f"a stress loss of {goal.needed:.1%}"
    if goal.capped:
        return (f"Growth target **{goal.target:.1%}** a year needs {need} (level {level.level} or higher), "
                f"more than your level's {_pct(goal.level_limit)} limit, so the plan stops at that limit.")
    return (f"Growth target **{goal.target:.1%}** a year: the least risky mix that reaches it takes {need}, "
            f"within your level's {_pct(goal.level_limit)} limit.")


def render_plan(plan: Plan) -> str:
    """The plan as markdown."""
    s = plan.settings
    other = (("Worst-case stress loss (every holding at its worst drop)", plan.stress_worst_case)
             if s.risk_model == "scenarios" else
             ("Crash-scenario stress loss (worst past crash replayed)", plan.stress_scenarios))
    lines = [
        "# Portfolio optimization",
        "",
        f"Portfolio value **${plan.total:,.2f}**. "
        + (f"Loss limit **{_pct(s.loss_limit)}** (${s.loss_limit * plan.total:,.2f}) of the whole portfolio "
           f"{MODEL_TEXT[s.risk_model]}."
           if s.loss_limit is not None else
           "**No stress-loss limit**: only the per-fund caps hold the mix back."),
        *(["", _goal_text(plan.goal)] if plan.goal else []),
        "",
        "| | Current | Target |",
        "|---|---|---|",
        f"| Invested (not in {plan.cash.ticker}) | {1 - plan.cash.current:.1%} | {1 - plan.cash.target:.1%} |",
        f"| Stress loss | {plan.current_stress:.1%} (${plan.current_stress * plan.total:,.0f}) "
        f"| {plan.target_stress:.1%} (${plan.target_stress * plan.total:,.0f}) |",
        *([f"| {other[0]} | {other[1][0]:.1%} | {other[1][1]:.1%} |"] if other[1] else []),
        *([f"| Worst past crash for the mix | {plan.worst_scenario[0] or '-'} | {plan.worst_scenario[1] or '-'} |"]
          if s.risk_model == "scenarios" else []),
        f"| Worst drop over the last {s.lookback_years:g} years | {plan.current_history_dd[0]:.1%} "
        f"(since {plan.current_history_dd[1]}) | {plan.target_history_dd[0]:.1%} (since {plan.target_history_dd[1]}) |",
        f"| Expected growth per year | {plan.expected_current:.1%} | {plan.expected_target:.1%} |",
        "",
        "## Target mix",
        "",
        "| Ticker | Value | Current | Target | Trade | Worst drop | Exp. growth | AI rating |",
        "|---|---|---|---|---|---|---|---|",
    ]
    shown = sorted((r for r in plan.rows if r.value > 0 or r.target > 1e-4), key=lambda r: -r.target)
    for r in shown:
        drop = f"{r.worst_loss:.0%}" + ("*" if r.floored else "")
        trade = f"{'buy' if r.trade > 0 else 'sell'} ${abs(r.trade):,.2f}" if abs(r.trade) >= 0.01 else "-"
        lines.append(f"| {r.ticker} | ${r.value:,.2f} | {r.current:.1%} | {r.target:.1%} | {trade} "
                     f"| {drop} | {r.expected:.1%} | {r.rating or '-'} |")
    c = plan.cash
    lines.append(f"| {c.ticker} (cash) | ${c.value:,.2f} | {c.current:.1%} | {c.target:.1%} "
                 f"| {'buy' if c.trade > 0 else 'sell'} ${abs(c.trade):,.2f} | 0% | {c.expected:.1%} | - |")

    lines += ["", "## Why", ""]
    lines += [f"- **{r.ticker}**{f' ({r.name})' if r.name else ''}: {r.reason}" for r in shown]
    lines.append(f"- **{c.ticker}** (cash): {c.reason}")
    if plan.scanned:
        passed = sorted((r for r in plan.rows if not r.held and r.target <= 1e-4 and r.expected > c.expected),
                        key=lambda r: -r.score)[:5]
        lines += ["", f"Scanned {plan.scanned} funds besides yours. Strongest ones left out:", ""]
        lines += [f"- **{r.ticker}** ({r.name}): {r.reason}" for r in passed]

    lines += ["", "## Daily buys", ""]
    if plan.days_to_target is not None:
        lines += [f"Spread your ${plan.daily_budget:,.2f} a day like this, funded from {c.ticker}; "
                  f"about {plan.days_to_target:,.0f} trading days to reach the target.", "",
                  "| Ticker | Per trading day |", "|---|---|"]
        lines += [f"| {r.ticker} | ${r.daily_buy:,.2f} |" for r in plan.rows if r.daily_buy > 0]
    else:
        lines.append(f"Nothing is under target, so the daily buys should pause and new money wait in "
                     f"{c.ticker}. Every extra dollar invested adds to the stress loss.")
    sells = [r for r in plan.rows if r.trade < -0.01]
    if sells:
        lines += ["", "To reach the target now, sell: "
                  + ", ".join(f"{r.ticker} ${-r.trade:,.2f}" for r in sells)
                  + f", into {c.ticker}. In a Roth IRA this is not taxed."]

    lines += ["", "## What the loss limit costs", "",
              "| Loss limit | Invested | Stress loss | Exp. growth per year | Exp. gain per year |",
              "|---|---|---|---|---|"]
    for p in plan.frontier:
        lines.append(f"| {_pct(p.loss_limit)} | {p.invested:.1%} | {p.stress_loss:.1%} "
                     f"| {p.expected_return:.1%} | ${p.expected_return * plan.total:,.0f} |")

    lines += ["", "## Notes", "",
              "- Expected growth figures are estimates: each asset's own history, shrunk toward a common "
              f"Sharpe of {s.prior_sharpe:g}, plus the AI tilt. They are not forecasts.",
              "- The stress loss bounds a repeat of history, not every future; a crash can be deeper.",
              *([f"- Crash scenarios ({len(plan.scenarios)}): " + "; ".join(plan.scenarios) + "."]
                if s.risk_model == "scenarios" and plan.scenarios else []),
              "- Nothing here places an order."]
    lines += [f"- {n}" for n in plan.notes]
    if any(r.floored for r in plan.rows):
        lines.append("- \\* worst drop assumed, not observed.")
    return "\n".join(lines) + "\n"
