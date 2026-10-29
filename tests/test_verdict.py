"""The per-stock verdict: decided by the internal 20-session model alone; AI ratings and screens are context."""

from datetime import date

import pytest

from tradingagents.stocks.verdict import ai_decision, verdict

pytestmark = pytest.mark.unit
TODAY = date(2026, 10, 24)
TOP = {"rank": 12, "of": 499, "score_pct": 0.95, "expected_net": 0.007, "p_win": 0.56, "p_risk": 0.07, "cost_rt": 0.001}
MID = {**TOP, "rank": 250, "score_pct": 0.5, "expected_net": 0.002}
LOW = {**TOP, "rank": 480, "score_pct": 0.05, "expected_net": -0.004}
BREAKDOWN = {"status": "ok", "warnings": [{"kind": "breakdown", "text": "breakdown risk: 80% vs 42% usual"}]}


def test_buy_comes_from_the_model_not_from_ai_or_screens():
    bad_ai = {"rating": "Sell", "date": "2026-10-23"}
    v = verdict(False, None, BREAKDOWN, {"quality_score": 30, "analyst_mean": 3.5}, bad_ai, TODAY, score=TOP)
    assert v["verdict"] == "BUY"                                    # bearish AI, warnings and screens: context only
    assert {c["name"] for c in v["context"]} >= {"Agents' rating", "Breakdown or crash warning"}
    assert all(not c["decides"] for c in v["context"])
    good_ai = {"rating": "Buy", "date": "2026-10-23"}
    assert verdict(False, None, None, {}, good_ai, TODAY, score=LOW)["verdict"] == "AVOID"


def test_middle_ranks_wait_and_failed_gates_block_a_buy():
    assert verdict(False, None, None, {}, None, TODAY, score=MID)["verdict"] == "WATCH"
    risky = verdict(False, None, None, {}, None, TODAY, score={**TOP, "p_risk": 0.3})
    assert risky["verdict"] == "WATCH" and "chance of a large loss" in risky["summary"]
    assert verdict(False, None, None, {}, None, TODAY, score={**TOP, "expected_net": -0.001})["verdict"] == "AVOID"
    assert verdict(False, None, None, {}, None, TODAY)["verdict"] == "WATCH"       # not scored


def test_a_held_stock_is_sold_on_the_decision_or_a_bottom_rank_with_a_loss():
    assert verdict(True, {"action": "EXIT"}, None, {}, None, TODAY, score=TOP)["verdict"] == "SELL"
    assert verdict(True, None, None, {}, None, TODAY, score=LOW)["verdict"] == "SELL"
    assert verdict(True, None, BREAKDOWN, {}, {"rating": "Sell", "date": "2026-10-23"}, TODAY, score=MID)["verdict"] == "HOLD"


def test_the_daily_decision_enter_is_a_buy():
    assert verdict(False, {"action": "ENTER"}, None, {}, None, TODAY, score=MID)["verdict"] == "BUY"


def test_the_agents_rating_becomes_an_action_for_display():
    today = date.today().isoformat()
    assert ai_decision({"rating": "Overweight", "date": today}, held=False)["action"] == "BUY"
    assert ai_decision({"rating": "Underweight", "date": today}, held=True)["action"] == "SELL"
    assert ai_decision({"rating": "Underweight", "date": today}, held=False)["action"] == "AVOID"
    assert ai_decision({"rating": "Hold", "date": today}, held=False)["action"] == "WATCH"
    assert ai_decision({"rating": "Hold", "date": "2020-01-02"}, held=True)["stale"]


def test_a_fund_is_judged_by_its_rank_among_funds_without_a_large_loss_gate():
    fund = {"kind": "fund", "rank": 3, "of": 41, "rank_pct": 0.95, "expected_net": 0.01, "p_win": 0.55, "p_risk": None}
    v = verdict(False, None, None, {}, None, TODAY, score=fund)
    assert v["verdict"] == "BUY" and "funds" in v["checks"][0]["detail"]
    assert [c["name"] for c in v["checks"]] == ["Model rank", "Expected return", "Chance of a positive total return (before costs)"]
    assert verdict(True, None, None, {}, None, TODAY, score={**fund, "rank": 40, "rank_pct": 0.03, "expected_net": -0.01})["verdict"] == "SELL"
    assert "no model view" in verdict(True, None, None, {}, None, TODAY)["summary"]


def test_an_unvalidated_buy_is_labelled_and_its_action_waits():
    v = verdict(False, {"action": "WAIT", "paper_action": "PAPER ENTER"}, None, {}, None, TODAY, score=TOP)
    assert v["verdict"] == "BUY" and v["action"] == "WAIT" and v["label"] == "BUY (model view, not validated)"
    ok = verdict(False, None, None, {}, None, TODAY, score=TOP, validated=True)
    assert ok["action"] == "BUY" and ok["label"] == "BUY"
    fund = {"kind": "fund", "rank": 1, "of": 41, "rank_pct": 1.0, "expected_net": 0.01, "p_win": 0.55, "p_risk": None}
    assert verdict(False, None, None, {}, None, TODAY, score=fund, validated=True)["action"] == "WAIT"   # funds never validated
    assert verdict(True, {"action": "EXIT"}, None, {}, None, TODAY, score=TOP)["action"] == "SELL"     # rule-based exit stays


def test_a_stock_without_a_risk_estimate_never_passes_the_gate():
    for missing in (None, float("nan")):
        v = verdict(False, None, None, {}, None, TODAY, score={**TOP, "p_risk": missing}, validated=True)
        assert v["verdict"] == "WATCH" and "risk estimate" in v["summary"].replace("chance of a large loss at the exit",
                                                                                   "risk estimate")


def test_an_abstaining_analysis_is_shown_as_abstain_not_as_its_placeholder_rating():
    from tradingagents.stocks.verdict import ABSTAIN_MARK, ai_decision, rating_on_record

    entry = {"rating": "Hold", "date": "2026-10-23", "decision": f"**Rating**: Hold\n\n{ABSTAIN_MARK}: abstained"}
    assert rating_on_record(entry)["abstained"]
    assert ai_decision(rating_on_record(entry), held=True)["action"] == "ABSTAIN"
    assert ai_decision({"rating": "Hold", "date": "2026-10-23"}, held=True)["action"] == "HOLD"
