"""The portfolio optimizer on synthetic prices: loss limit, tilts, daily orders under T+1."""

import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.optimizer.plan import (
    OptimizerSettings,
    build_plan,
    daily_orders,
    load_schedule,
    render_plan,
)
from tradingagents.optimizer.solve import solve_weights
from tradingagents.portfolio import PortfolioContext, Position

pytestmark = pytest.mark.unit


def _closes(years=12, seed=0):
    """Two equity funds, a volatile one with a deep crash, and a cash fund."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end="2026-10-21", periods=252 * years)
    n = len(days)
    steady = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    other = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    wild_r = rng.normal(0.001, 0.04, n)
    wild_r[n // 2: n // 2 + 120] = -0.015      # a 120-day crash
    wild = 100 * np.cumprod(1 + wild_r)
    cash = 100 * np.cumprod(np.full(n, 1 + 0.04 / 252))
    return pd.DataFrame({"AAA": steady, "BBB": other, "WILD": wild, "CASH": cash}, index=days)


def _book(cash=0.0):
    return PortfolioContext(cash=cash, positions=[
        Position(ticker="AAA", quantity=10), Position(ticker="BBB", quantity=10),
        Position(ticker="WILD", quantity=1), Position(ticker="CASH", quantity=200)])


def _settings(**kw):
    return OptimizerSettings(cash_ticker="CASH", proxies={}, **kw)


def test_solver_respects_loss_limit_and_caps():
    excess = np.array([0.08, 0.06, 0.20])
    cov = np.diag([0.04, 0.03, 0.5])
    worst = np.array([0.5, 0.4, 0.9])
    x = solve_weights(excess, cov, worst, 0.05, 0.35, 3.0)
    assert worst @ x <= 0.05 + 1e-9
    assert (x >= 0).all() and (x <= 0.35 + 1e-9).all() and x.sum() <= 1 + 1e-9


def test_solver_without_limit_prefers_better_return_per_risk():
    x = solve_weights(np.array([0.10, 0.02]), np.diag([0.04, 0.04]), np.array([0.5, 0.5]), None, 1.0, 3.0)
    assert x[0] > x[1]


def test_plan_stays_within_loss_limit():
    plan = build_plan(_book(), {"AAA": 5.0}, _settings(loss_limit=0.02), closes=_closes())
    assert plan.target_stress <= 0.02 + 1e-9
    assert plan.cash.target > 0.9


def test_no_asset_takes_more_than_its_share_of_the_budget():
    settings = _settings(loss_limit=0.10, max_budget_share=0.25)
    plan = build_plan(_book(), {}, settings, closes=_closes())
    for row in plan.rows:
        assert row.target * row.worst_loss <= 0.25 * 0.10 + 1e-9


def test_buy_rating_raises_and_sell_rating_lowers_weight():
    # Caps off, so the tilt is not hidden behind a binding cap.
    closes, settings = _closes(), _settings(loss_limit=0.20, max_budget_share=1.0, max_weight=1.0)
    base = {r.ticker: r.target for r in build_plan(_book(), {}, settings, closes=closes).rows}
    up = {r.ticker: r.target for r in build_plan(_book(), {}, settings, {"BBB": "Buy"}, closes).rows}
    down = {r.ticker: r.target for r in build_plan(_book(), {}, settings, {"AAA": "Sell"}, closes).rows}
    assert up["BBB"] > base["BBB"]
    assert down["AAA"] < base["AAA"]


def test_ratings_never_breach_the_loss_limit():
    ratings = {"AAA": "Buy", "BBB": "Buy", "WILD": "Buy"}
    plan = build_plan(_book(), {}, _settings(loss_limit=0.05), ratings, _closes())
    assert plan.target_stress <= 0.05 + 1e-9


def test_daily_buys_go_to_underweight_assets_and_keep_the_budget():
    plan = build_plan(_book(), {"AAA": 10.0, "BBB": 5.0}, _settings(loss_limit=None), closes=_closes())
    under = {r.ticker for r in plan.rows if r.trade > 0.005 * plan.total}
    buys = {r.ticker: r.daily_buy for r in plan.rows if r.daily_buy > 0}
    assert set(buys) <= under
    assert sum(buys.values()) == pytest.approx(15.0, abs=0.05)


def test_orders_spend_only_settled_cash_and_sell_cash_fund_for_tomorrow():
    plan = build_plan(_book(), {"AAA": 10.0, "BBB": 5.0}, _settings(loss_limit=None), closes=_closes())
    none_settled = daily_orders(plan, settled_cash=0.0)
    assert not [o for o in none_settled.orders if o.side == "buy"]
    assert any(o.side == "sell" and o.ticker == "CASH" for o in none_settled.orders)

    some = daily_orders(plan, settled_cash=6.0)
    assert sum(o.amount for o in some.orders if o.side == "buy") <= 6.0 + 1e-9


def test_orders_under_a_dollar_are_dropped():
    plan = build_plan(_book(), {"AAA": 10.0}, _settings(loss_limit=None), closes=_closes())
    assert all(o.amount >= 1.0 for o in daily_orders(plan, settled_cash=100.0).orders)


def test_short_history_gets_floored_stress_loss():
    closes = _closes()
    closes.loc[closes.index[:-500], "BBB"] = np.nan    # two years of history
    plan = build_plan(_book(), {}, _settings(loss_limit=0.05), closes=closes)
    bbb = next(r for r in plan.rows if r.ticker == "BBB")
    assert bbb.floored and bbb.worst_loss == pytest.approx(0.5)


def test_schedule_turns_weekly_into_per_day(tmp_path):
    path = tmp_path / "buys.json"
    path.write_text(json.dumps({"daily": {"vti": 10}, "weekly": {"JEPI": 5}}))
    assert load_schedule(path) == {"VTI": 10.0, "JEPI": 1.0}


def test_report_renders():
    plan = build_plan(_book(), {"AAA": 5.0}, _settings(), closes=_closes())
    text = render_plan(plan)
    assert "Target mix" in text and "What the loss limit costs" in text


def test_scanned_funds_are_considered_and_young_ones_skipped():
    closes = _closes()
    closes["NEW"] = closes["BBB"] * 1.01
    closes.loc[closes.index[:-200], "NEW"] = np.nan   # under a year of prices
    book = PortfolioContext(positions=[Position(ticker="AAA", quantity=10), Position(ticker="CASH", quantity=200)])
    plan = build_plan(book, {}, _settings(loss_limit=0.10), closes=closes, universe=["BBB", "WILD", "NEW"])
    assert {r.ticker for r in plan.rows} == {"AAA", "BBB", "WILD"}
    assert plan.skipped == ["NEW"] and plan.scanned == 2
    assert not next(r for r in plan.rows if r.ticker == "BBB").held


def test_every_fund_gets_a_reason():
    plan = build_plan(_book(), {}, _settings(loss_limit=0.05), closes=_closes())
    assert all(r.reason for r in plan.rows) and plan.cash.reason
    orders = daily_orders(plan, settled_cash=50.0).orders
    assert all("target" in o.why for o in orders if o.ticker != "CASH")


def test_walk_forward_backtest_keeps_its_loss_limit():
    from tradingagents.optimizer.backtest import run_backtest

    closes = _closes(years=12)
    settings = _settings(loss_limit=0.05)
    settings.cash_proxy = "CASH"
    result = run_backtest(closes, ["AAA", "BBB", "WILD"], settings, start=str(closes.index[1500].date()), every=63)
    assert result.metrics["Optimizer"].max_drawdown <= 0.05
    assert result.rebalances > 5 and result.curves["Optimizer"][0] == pytest.approx(1.0, abs=0.01)


def test_with_no_settled_cash_trims_today_pay_for_buys_tomorrow():
    book = PortfolioContext(positions=[Position(ticker="AAA", quantity=60), Position(ticker="CASH", quantity=10)])
    plan = build_plan(book, {"AAA": 5.0}, _settings(loss_limit=0.10), closes=_closes(), universe=["BBB"])
    s = daily_orders(plan, settled_cash=0.0)
    assert not [o for o in s.orders if o.side == "buy"]
    sold = sum(o.amount for o in s.orders if o.side == "sell")
    assert s.tomorrow and sum(o.amount for o in s.tomorrow) == pytest.approx(sold, abs=0.05)
    assert any(o.ticker == "BBB" for o in s.tomorrow)


def test_screening_keeps_the_best_scores_and_explains_the_rest():
    closes = _closes()
    for k in range(6):
        closes[f"S{k}"] = closes["BBB"] * (1 + 0.0001 * k)
    plan = build_plan(_book(), {}, _settings(loss_limit=0.10, max_candidates=2), closes=closes,
                      universe=[f"S{k}" for k in range(6)])
    screened = [r for r in plan.rows if r.reason.startswith("Screened out")]
    assert len(screened) == 4 and plan.scanned == 6
    assert all(r.target == 0 for r in screened)


def test_paper_tracker_follows_the_plan_day_by_day():
    from datetime import datetime

    from tradingagents.optimizer.tracker import start_ledger, update_ledger

    closes = _closes()
    settings = _settings(loss_limit=0.05)
    settings.cash_proxy = "CASH"
    book = PortfolioContext(positions=[Position(ticker="AAA", quantity=40), Position(ticker="CASH", quantity=10)])
    ledger = start_ledger(book, settings, closes, start=str(closes.index[-10].date()))
    after_close = datetime(2030, 1, 1, 17, 0)
    ledger = update_ledger(ledger, {"AAA": 2.0}, settings, closes, ["BBB"], now=after_close)
    days = ledger["history"]
    assert len(days) == 10 and days[0]["plan"] == pytest.approx(days[0]["hold"])
    first_orders = days[0]["orders"]["plan"]
    assert first_orders and all(side == "sell" for side, _, _ in first_orders)   # nothing settled on day one
    assert any(side == "buy" for d in days[1:] for side, _, _ in d["orders"]["plan"])
    # caught up: a second update adds nothing
    assert len(update_ledger(ledger, {"AAA": 2.0}, settings, closes, ["BBB"], now=after_close)["history"]) == 10


def test_tracker_waits_for_the_close():
    from datetime import datetime

    from tradingagents.optimizer.tracker import _final_days

    closes = _closes()
    last = closes.index[-1]
    before = datetime(last.year, last.month, last.day, 11, 0)
    after = datetime(last.year, last.month, last.day, 17, 0)
    assert last not in _final_days(closes, "AAA", before)
    assert last in _final_days(closes, "AAA", after)


def test_universe_config_reads_old_and_new_forms(tmp_path):
    from tradingagents.optimizer.universe import read_config

    old = tmp_path / "old.json"
    old.write_text(json.dumps({"tickers": ["vti", "gld"]}))
    assert read_config(old) == {"groups": [], "extra": ["VTI", "GLD"], "exclude": []}
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"groups": ["themes", "bogus"], "extra": ["pltr"], "exclude": ["soxl"]}))
    assert read_config(new) == {"groups": ["themes"], "extra": ["PLTR"], "exclude": ["SOXL"]}
    from tradingagents.optimizer.universe import etf_list, tickers_for

    assert "SOXL" not in etf_list(read_config(new)) and "SOXL" not in tickers_for({"groups": ["etfs"], "exclude": ["SOXL"]})


def test_market_overview_from_prices():
    from tradingagents.optimizer.market import market_overview

    closes = _closes().rename(columns={"AAA": "SPY", "BBB": "QQQ", "WILD": "IONQ", "CASH": "^VIX"})
    closes["MSFT"] = closes["SPY"] * 2
    overview = market_overview(closes, {"MSFT": "Microsoft (Information Technology)"})
    spy = next(r for r in overview["rows"] if r["ticker"] == "SPY")
    expected = closes["SPY"].iloc[-1] / closes["SPY"].iloc[-22] - 1
    assert spy["changes"]["1M"] == pytest.approx(expected)
    vix = next(r for r in overview["rows"] if r["ticker"] == "^VIX")
    assert vix["changes"]["1D"] == pytest.approx(closes["^VIX"].iloc[-1] - closes["^VIX"].iloc[-2])   # points
    assert overview["sp500"]["count"] == 1 and overview["sp500"]["leaders_1m"][0]["ticker"] == "MSFT"
    assert any(t["theme"] == "Quantum computing" for t in overview["themes"])


def test_uncited_trend_is_low_confidence():
    from tradingagents.optimizer.market import Trend, _checked

    headlines = [{"title": "Stocks fall as yields jump", "publisher": "Wire", "link": "http://x"}]
    cited = _checked(Trend(title="t", direction="down", what="w", why="y", sources=[1], data=[], confidence="high"),
                     headlines)
    assert cited["confidence"] == "high" and cited["sources"][0]["title"] == "Stocks fall as yields jump"
    bogus = _checked(Trend(title="t", direction="down", what="w", why="y", sources=[7], data=[], confidence="high"),
                     headlines)
    assert bogus["confidence"] == "low" and bogus["sources"] == []


def test_brief_falls_back_to_gemini_only_when_the_policy_allows(monkeypatch):
    from tradingagents.optimizer.market import _models

    config = {"llm_provider": "openai", "quick_think_llm": "m"}
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert len(_models(config)) == 1   # a key alone is not permission to spend on another provider
    assert [p for p, _ in _models(config, ("google",))[1:]] == ["google", "google"]
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert len(_models(config, ("google",))) == 1


def test_risk_levels_rise_and_are_saved(tmp_path):
    from tradingagents.optimizer.risk import DEFAULT_LEVEL, RISK_LEVELS, read_level, risk_level, write_level

    limits = [r.loss_limit for r in RISK_LEVELS[:-1]]
    assert limits == sorted(limits) and RISK_LEVELS[-1].loss_limit is None
    path = tmp_path / "risk.json"
    assert read_level(path) == DEFAULT_LEVEL
    write_level(path, 7)
    assert read_level(path) == 7
    with pytest.raises(ValueError):
        risk_level(11)


def test_higher_risk_level_invests_more():
    from tradingagents.optimizer.risk import risk_level

    invested = [1 - build_plan(_book(), {}, _settings(loss_limit=risk_level(n).loss_limit),
                               closes=_closes()).cash.target for n in (1, 5, 10)]
    assert invested[0] < invested[1] <= invested[2] + 1e-9


def test_tracker_keeps_its_record_when_the_risk_level_changes(tmp_path):
    from datetime import datetime

    from tradingagents.optimizer.tracker import archive_ledger, start_ledger, update_ledger, write_ledger

    closes = _closes()
    settings = _settings(loss_limit=0.05)
    settings.cash_proxy = "CASH"
    book = PortfolioContext(positions=[Position(ticker="AAA", quantity=40), Position(ticker="CASH", quantity=10)])
    ledger = start_ledger(book, settings, closes, start=str(closes.index[-10].date()))
    after_close = datetime(2030, 1, 1, 17, 0)
    ledger = update_ledger(ledger, {}, settings, closes.iloc[:-5], [], now=after_close)
    settings.loss_limit = 0.20
    ledger = update_ledger(ledger, {}, settings, closes, [], now=after_close)
    days = ledger["history"]
    assert len(days) == 10 and days[0]["loss_limit"] == 0.05 and days[-1]["loss_limit"] == 0.20
    assert all(0 <= d["invested"] <= 1 for d in days)
    path = tmp_path / "paper_tracker.json"
    write_ledger(path, ledger)
    archived = archive_ledger(path)
    assert not path.exists() and archived.exists()


def _two_crashes(seed=1):
    """A market reference that crashes twice; AAA falls in the first crash, BBB in the second."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end="2026-10-21", periods=252 * 12)
    n = len(days)
    first, second = slice(1000, 1100), slice(2000, 2100)

    def path(drift, crashes):
        r = rng.normal(drift, 0.004, n)
        for window, total in crashes:
            r[window] = (1 - total) ** (1 / 100) - 1
        return 100 * np.cumprod(1 + r)

    return pd.DataFrame({
        "REF": path(0.0004, [(first, 0.30), (second, 0.30)]),
        "AAA": path(0.0005, [(first, 0.40)]),
        "BBB": path(0.0005, [(second, 0.40)]),
        "CASH": 100 * np.cumprod(np.full(n, 1 + 0.03 / 252)),
    }, index=days)


def _book_ab():
    return PortfolioContext(positions=[Position(ticker="AAA", quantity=1), Position(ticker="BBB", quantity=1),
                                       Position(ticker="CASH", quantity=10)])


def test_solver_keeps_every_loss_row():
    rows = np.array([[0.5, -0.1, 0.2], [-0.05, 0.5, 0.2], [0.1, 0.1, 0.6]])
    x = solve_weights(np.array([0.08, 0.08, 0.10]), np.diag([0.04, 0.04, 0.09]), rows, 0.05, 1.0, 3.0)
    assert (rows @ x).max() <= 0.05 + 1e-9 and (x >= 0).all() and x.sum() <= 1 + 1e-9


def test_scenarios_are_found_in_the_past_only():
    from tradingagents.optimizer.scenarios import find_scenarios

    closes = _two_crashes()
    both = find_scenarios(closes, {"REF": 0.15})
    assert len(both) == 2 and all(0.25 < s.depth < 0.35 for s in both)
    assert len(find_scenarios(closes.iloc[:1500], {"REF": 0.15})) == 1


def test_crashes_that_did_not_coincide_let_the_mix_invest_more():
    settings = dict(loss_limit=0.10, max_budget_share=1.0, scenario_references={"REF": 0.15})
    worst = build_plan(_book_ab(), {}, _settings(**settings), closes=_two_crashes())
    replay = build_plan(_book_ab(), {}, _settings(risk_model="scenarios", **settings), closes=_two_crashes())
    assert replay.target_stress <= 0.10 + 1e-6 and worst.target_stress <= 0.10 + 1e-6
    assert 1 - replay.cash.target > 1.3 * (1 - worst.cash.target)
    # Measured the worst-case way, the scenario mix takes more risk: that is the trade.
    assert replay.stress_worst_case[1] > 0.10
    assert replay.worst_scenario[1] and "REF fell" in replay.worst_scenario[1]


def test_a_fund_without_crash_history_is_estimated_from_its_downside_beta():
    from tradingagents.optimizer.scenarios import find_scenarios, scenario_losses

    closes = _two_crashes()
    young = closes["REF"].pct_change() * 1.5
    closes["YNG"] = 100 * (1 + young.where(closes.index >= closes.index[1500], np.nan)).cumprod()
    scenarios = find_scenarios(closes, {"REF": 0.15})
    rows, owners = scenario_losses(closes, ["YNG"], scenarios, np.array([0.6]), prune=False)
    first = rows[[i for i, o in enumerate(owners) if o == 0]]
    assert 0.40 < first.max() <= 0.6   # about 1.5 × the reference's 30%, never past its own worst
    unknown, _ = scenario_losses(closes.iloc[:1600], ["YNG"], scenarios[:1], np.array([0.6]), prune=False)
    assert unknown.max() == pytest.approx(0.6)   # too little history: its own worst drop


def test_growth_target_takes_the_least_risk_that_reaches_it():
    base = _settings(loss_limit=0.30)
    full = build_plan(_book(), {}, base, closes=_closes())
    goal = (full.cash.expected + full.expected_target) / 2
    plan = build_plan(_book(), {}, _settings(loss_limit=0.30, target_growth=goal), closes=_closes())
    assert plan.goal.reachable and not plan.goal.capped
    assert plan.expected_target == pytest.approx(goal, abs=0.002)
    assert plan.target_stress < full.target_stress and plan.settings.loss_limit < 0.30


def test_growth_target_never_passes_the_risk_level():
    low = build_plan(_book(), {}, _settings(loss_limit=0.01, target_growth=0.15), closes=_closes())
    assert low.goal.capped and low.settings.loss_limit == 0.01 and low.target_stress <= 0.01 + 1e-9
    assert "more than your level" in render_plan(low) or "more than any mix" in render_plan(low)
    out_of_reach = build_plan(_book(), {}, _settings(loss_limit=None, target_growth=0.90), closes=_closes())
    assert not out_of_reach.goal.reachable and out_of_reach.settings.loss_limit is None


def test_risk_choice_keeps_the_level_and_clears_the_target(tmp_path):
    from tradingagents.optimizer.risk import read_choice, write_choice, write_level

    path = tmp_path / "risk.json"
    path.write_text('{"level": 4}', encoding="utf-8")
    assert read_choice(path).model == "worst_case" and read_choice(path).target_growth is None
    write_choice(path, model="scenarios", target_growth=0.08)
    write_level(path, 6)
    choice = read_choice(path)
    assert (choice.level, choice.model, choice.target_growth) == (6, "scenarios", 0.08)
    write_choice(path, target_growth=None)
    assert read_choice(path).target_growth is None and read_choice(path).model == "scenarios"
    with pytest.raises(ValueError):
        write_choice(path, model="vibes")


def test_backtest_trades_at_the_close_and_earns_from_the_next_day():
    from tradingagents.optimizer.backtest import _simulate

    days = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.DataFrame({"AAA": [0.0, 0.5, 0.0, 0.0, 0.0]}, index=days)
    cash = pd.Series(0.0, index=days)
    curve, traded = _simulate(returns, cash, [(days[1], {"AAA": 1.0})])
    assert curve.iloc[-1] == pytest.approx(1.0)   # bought at day 1's close, after its jump
    curve, _ = _simulate(returns, cash, [(days[0], {"AAA": 1.0})])
    assert curve.iloc[-1] == pytest.approx(1.5) and traded == pytest.approx(1.0)


def test_walk_forward_backtest_with_scenarios_and_a_target_reports_its_promise():
    from tradingagents.optimizer.backtest import render_backtest, run_backtest

    closes = _two_crashes()
    settings = _settings(loss_limit=0.20, risk_model="scenarios", target_growth=0.05,
                         scenario_references={"REF": 0.15})
    settings.cash_proxy = "CASH"
    result = run_backtest(closes, ["AAA", "BBB", "REF"], settings, start=str(closes.index[1200].date()),
                          every=126, cost_bps=10)
    assert all(r.limit is None or r.predicted <= r.limit + 1e-6 for r in result.history)
    assert all(r.limit is None or r.limit <= 0.20 + 1e-9 for r in result.history)
    assert result.turnover > 0 and "held for a year" in render_backtest(result)


def test_auto_invest_follows_the_target_mix_and_keeps_the_budget():
    from tradingagents.optimizer.auto_invest import recommend

    plan = build_plan(_book(), {"AAA": 20.0}, _settings(loss_limit=0.10), closes=_closes())
    rec = recommend(plan, 20.0, {"AAA": 20.0})
    assert sum(r.per_day for r in rec.schedule) == pytest.approx(20.0, abs=0.01)
    # Each order is at least $1, at Robinhood's slowest pace that keeps the share.
    assert all(r.amount >= 1.0 - 1e-9 for r in rec.schedule)
    weights = {r.ticker: r.target for r in plan.rows if r.target > 1e-4} | {plan.cash.ticker: plan.cash.target}
    for r in rec.schedule:
        assert r.per_day / 20.0 == pytest.approx(weights[r.ticker] / sum(weights.values()), rel=0.05)
    assert any(c["ticker"] == "AAA" for c in rec.changes) or "AAA" in {r.ticker for r in rec.schedule}


def test_a_small_share_is_bought_less_often_not_dropped():
    from tradingagents.optimizer.auto_invest import _frequency

    assert _frequency(5.0, 1.0) == ("every market day", 5.0)
    assert _frequency(0.30, 1.0) == ("every week", 1.5)
    assert _frequency(0.06, 1.0) == ("every month", 1.26)


def test_an_account_without_contributions_funds_its_gaps_from_the_cash_fund():
    book = PortfolioContext(positions=[Position(ticker="CASH", quantity=1000)])
    plan = build_plan(book, {}, _settings(loss_limit=0.10), closes=_closes(), universe=["AAA", "BBB", "WILD"])
    assert plan.daily_budget == 0 and any(r.trade > 0 for r in plan.rows)
    s = daily_orders(plan, settled_cash=0.0)
    sold = sum(o.amount for o in s.orders if o.side == "sell" and o.ticker == "CASH")
    bought = sum(o.amount for o in s.tomorrow if o.side == "buy" and o.ticker != "CASH")
    assert sold > 0 and bought == pytest.approx(sum(r.trade for r in plan.rows if r.trade >= max(1, 0.005 * plan.total)), abs=0.05)
