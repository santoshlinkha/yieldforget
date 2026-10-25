"""The per-stock verdict: BUY only when every test passes, AVOID on bearish evidence, SELL for a held stock."""

from datetime import date

import pytest

from tradingagents.stocks.verdict import verdict

pytestmark = pytest.mark.unit
TODAY = date(2026, 10, 24)
GOOD = {"quality_score": 90, "analyst_mean": 1.8, "target_upside": 0.25}
CALM = {"status": "ok", "warnings": []}
BREAKDOWN = {"status": "ok", "warnings": [{"kind": "breakdown", "text": "breakdown risk: 80% vs 42% usual"}]}


def test_buy_needs_every_test():
    assert verdict(False, None, CALM, GOOD, None, TODAY)["verdict"] == "BUY"
    v = verdict(False, None, BREAKDOWN, GOOD, None, TODAY)
    assert v["verdict"] == "WATCH" and v["summary"] == "waiting for the breakdown or crash warning to clear"
    assert verdict(False, None, None, GOOD, None, TODAY)["verdict"] == "WATCH"      # no signals: not enough data


def test_bearish_evidence_means_avoid_and_old_ratings_expire():
    bad = {"rating": "Underweight", "date": "2026-10-23"}
    assert verdict(False, None, CALM, GOOD, bad, TODAY)["verdict"] == "AVOID"
    assert verdict(False, None, CALM, GOOD, {"rating": "Sell", "date": "2026-08-01"}, TODAY)["verdict"] == "BUY"
    assert verdict(False, None, CALM, {**GOOD, "analyst_mean": 3.1}, None, TODAY)["verdict"] == "AVOID"
    assert verdict(False, None, CALM, {**GOOD, "quality_score": 40}, None, TODAY)["verdict"] == "AVOID"


def test_a_held_stock_is_sold_only_on_the_decision_or_two_bearish_signs():
    assert verdict(True, {"action": "EXIT"}, CALM, GOOD, None, TODAY)["verdict"] == "SELL"
    bad = {"rating": "Sell", "date": "2026-10-20"}
    assert verdict(True, None, BREAKDOWN, GOOD, bad, TODAY)["verdict"] == "SELL"
    one = verdict(True, None, BREAKDOWN, GOOD, None, TODAY)
    assert one["verdict"] == "HOLD" and "caution" in one["summary"]


def test_the_daily_decision_enter_is_a_buy():
    assert verdict(False, {"action": "ENTER"}, BREAKDOWN, {}, None, TODAY)["verdict"] == "BUY"
