"""Stock signals: lifts against base rates, the peer test, flat models, the trend label, and actions."""

import pytest

from tradingagents.stocks import signals

pytestmark = pytest.mark.unit

BASE = {"up_break": 0.5, "down_break": 0.4, "large_loss": 0.06, "deep_drawdown": 0.15, "trend_mix": [20, 10, 20, 50]}


def _f(**kw):
    f = {"p_up_break": 0.5, "p_down_break": 0.4, "p_large_loss": 0.06, "p_deep_drawdown": 0.15,
         "trend_up": 0.2, "trend_down": 0.1, "trend_sideways": 0.2, "trend_unclear": 0.5}
    return {**f, **kw}


def test_without_peers_the_lift_alone_decides():
    d = signals.describe(_f(p_down_break=0.6, p_large_loss=0.12), BASE)
    assert [w["kind"] for w in d["warnings"]] == ["breakdown", "crash"]
    assert d["lifts"]["down_break"] == pytest.approx(1.5)
    assert signals.describe(_f(), BASE)["warnings"] == []          # an ordinary stock raises nothing


def test_a_market_wide_shift_flags_only_the_top_fifth():
    crowd = [_f(p_down_break=0.55 + i / 1000) for i in range(100)]
    peer = signals.peers(crowd)
    low, high = signals.describe(crowd[10], BASE, peer), signals.describe(crowd[95], BASE, peer)
    assert low["lifts"]["down_break"] > 1.2 and not low["warnings"]   # high lift, but so is everyone's
    assert [w["kind"] for w in high["warnings"]] == ["breakdown"]


def test_a_flat_model_raises_no_warning_and_says_so():
    crowd = [_f(p_large_loss=0.2) for _ in range(60)]                  # 3.3x usual for every stock alike
    d = signals.describe(crowd[0], BASE, signals.peers(crowd))
    assert "large_loss" in d["flat"] and not [w for w in d["warnings"] if w["kind"] == "crash"]


def test_the_trend_needs_a_lift_over_its_usual_share():
    assert signals.describe(_f(trend_sideways=0.3, trend_unclear=0.4), BASE)["trend"] == "sideways"  # 1.5x of 20%
    assert signals.describe(_f(), BASE)["trend"] == "no clear trend"


def test_actions_cover_candidates_and_holdings():
    dec = {"id": "d1", "mode": "experimental", "info_cutoff": "2026-10-23", "execution_date": "2026-10-26",
           "entries": [{"ticker": "AAA", "action": "WAIT", "paper_action": "PAPER ENTER", "blocked": ["settled cash"]}],
           "lots": [{"ticker": "BBB", "action": "EXIT", "reasons": ["deadline"]}]}
    a = signals.actions(dec)
    assert a["AAA"]["paper_action"] == "PAPER ENTER" and a["BBB"] == {"action": "EXIT", "reasons": ["deadline"], "kind": "held"}
    assert signals.actions(None) == {}


def test_the_entry_plan_orders_limits_and_puts_exits_one_volatility_away():
    import math

    plan = signals.entry_plan({"sigma_h": 0.10, "lower": 95.0, "p_down_break": 0.6, "dd_q50": -0.04, "dd_q05": -0.15}, 100.0)
    names = [r["name"] for r in plan["rows"]]
    assert names == ["Buy at the next close", "Median dip", "At the 20-day low", "Deep dip"]
    now = plan["rows"][0]
    assert now["stop"] == pytest.approx(100 * (1 + math.expm1(-0.10))) and now["profit"] == pytest.approx(100 * (1 + math.expm1(0.10)))
    assert plan["rows"][2]["chance"] == 0.6 and plan["rows"][1]["price"] == pytest.approx(96.0)
    assert signals.entry_plan({}, 100.0) is None
