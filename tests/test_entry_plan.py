"""Entry plans: the schema stays compatible, the three agents carry them, the app checks and keeps them."""

import json
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingagents.agents import entry_plan as EP
from tradingagents.agents.schemas import (
    EntryPlan,
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_trader_proposal,
)

pytestmark = pytest.mark.unit

TODAY = date(2026, 10, 26)        # a Monday
REF = {"ticker": "AAA", "cutoff": "2026-10-23", "session": "2026-10-23", "price": 100.0, "adjusted": 100.0,
       "currency": "USD", "verified": True, "how": "the store's latest session", "basis": "as traded"}
ACCOUNT = {"sleeve": "trading", "available": 600.0, "reserved": 100.0, "sleeve_value": 1000.0, "held": 0.0,
           "names_held": 1, "ira_total": 6000.0, "ira_in_ticker": 0.0, "sold_on": [], "bought_on": [],
           "rules": {"max_position_share": 0.25, "max_names": 5, "min_order": 1.0, "combined_name_cap": 0.10},
           "settings": {"version": 7, "hash": "abc"}}


def plan(**kw):
    base = {"status": "ready", "entry_type": "limit", "entry_price": 98.0, "entry_range_low": 97.0,
            "entry_range_high": 99.0, "currency": "USD", "trigger": None, "earliest_execution": "2026-10-26",
            "expires_at": "2026-10-30", "evidence_ids": ["REF1", "PX1"], "rationale": "a pullback to support"}
    return {**base, **kw}


# --- schema ---

def test_answers_without_an_entry_plan_still_parse_and_render_as_before():
    old = TraderProposal.model_validate({"action": "Buy", "reasoning": "r", "entry_price": 189.5})
    assert old.entry_plan is None and "Entry Plan" not in render_trader_proposal(old)
    pm = PortfolioDecision.model_validate({"rating": "Hold", "executive_summary": "s", "investment_thesis": "t"})
    assert pm.entry_plan is None and pm.entry_plan_review is None and "Entry Plan" not in render_pm_decision(pm)


def test_an_entry_plan_parses_nested_with_formatted_prices_cleaned():
    pm = PortfolioDecision.model_validate({"rating": "Buy", "executive_summary": "s", "investment_thesis": "t",
                                           "entry_plan_review": "revised",
                                           "entry_plan": plan(entry_price="$98.00", entry_range_low="N/A")})
    assert pm.entry_plan.entry_price == 98.0 and pm.entry_plan.entry_range_low is None
    md = render_pm_decision(pm)
    assert "**Entry Plan Review**: revised" in md and "status ready" in md and "price 98.0" in md


# --- the local check ---

def test_a_sound_limit_plan_is_feasible_and_sized_on_settled_cash():
    v = EP.validate(plan(), "Buy", REF, ACCOUNT, TODAY, stop_loss=94.0, price_target=110.0)
    assert v["state"] == "actionable" and v["feasible"] and not v["blocks"]
    assert v["sizing"]["dollars"] == 150.0                     # min(25% of $1,000, $600 / 4 free names)
    assert v["levels"]["reward_to_risk"] == 3.0 and v["levels"]["loss_at_stop_usd"] == pytest.approx(6.12, abs=0.01)
    assert any("no exit on the entry session" in w for w in v["warnings"])


@pytest.mark.parametrize("change, action, account, needle", [
    ({}, "Sell", {}, "long-only"),
    ({"entry_range_low": 101.0, "entry_range_high": 99.0}, "Buy", {}, "not ordered"),
    ({"entry_price": -5.0, "entry_range_low": None, "entry_range_high": None}, "Buy", {}, "not a positive price"),
    ({"entry_price": 105.0}, "Buy", {}, "outside its range"),
    ({"earliest_execution": "2026-10-23"}, "Buy", {}, "before 2026-10-26"),
    ({"earliest_execution": "2026-10-25"}, "Buy", {}, "not a trading session"),
    ({"expires_at": "2026-10-20"}, "Buy", {}, "before its earliest execution"),
    ({"entry_price": 60.0, "entry_range_low": 59.0, "entry_range_high": 61.0}, "Buy", {}, "from the reference close"),
    ({"currency": "EUR"}, "Buy", {}, "currency EUR"),
    ({"entry_type": "conditional", "status": "conditional"}, "Buy", {}, "needs its trigger"),
    ({}, "Buy", {"available": 0.0}, "settled cash"),
    ({}, "Buy", {"sold_on": ["2026-10-26"]}, "day trade"),
    ({}, "Buy", {"ira_in_ticker": 590.0}, "IRA concentration"),
])
def test_an_infeasible_plan_is_blocked_with_its_reason(change, action, account, needle):
    v = EP.validate(plan(**change), action, REF, {**ACCOUNT, **account}, TODAY)
    assert v["state"] == "blocked" and not v["feasible"]
    assert any(needle in b for b in v["blocks"]), v["blocks"]


def test_an_unverified_or_adjusted_basis_blocks_a_priced_plan():
    unverified = {**REF, "verified": False, "how": "no corporate actions fetched"}
    assert any("verified" in b for b in EP.validate(plan(), "Buy", unverified, ACCOUNT, TODAY)["blocks"])
    # The close is 100 as traded but 96 adjusted: a plan at 96 was priced on the wrong basis.
    adjusted = {**REF, "adjusted": 96.0}
    v = EP.validate(plan(entry_price=96.0, entry_range_low=95.5, entry_range_high=96.5), "Buy", adjusted, ACCOUNT, TODAY)
    assert any("adjusted close" in b for b in v["blocks"])


def test_wait_keeps_its_condition_and_an_old_plan_expires():
    v = EP.validate(plan(status="wait", entry_type=None, trigger=None), "Hold", REF, ACCOUNT, TODAY)
    assert v["state"] == "waiting" and any("reconsideration" in w for w in v["warnings"])
    later = EP.validate(plan(), "Buy", REF, ACCOUNT, date(2026, 11, 2))
    assert later["state"] == "expired" and not later["feasible"]
    assert EP.validate(None, "Buy", REF, ACCOUNT, TODAY)["state"] == "unavailable"


def test_missing_exits_come_from_the_trader_then_the_internal_model():
    import math

    v = EP.validate(plan(), "Buy", REF, ACCOUNT, TODAY, stop_loss=94.0, price_target=None, sigma_h=0.1)
    used = 98.0
    assert v["levels"]["stop_loss"] == 94.0 and v["levels"]["source"]["stop_loss"] == "agents"
    assert v["levels"]["price_target"] == round(used * math.exp(0.1), 2) and v["levels"]["source"]["price_target"] == "model"
    none = EP.validate(plan(), "Buy", REF, ACCOUNT, TODAY)          # no volatility known: nothing invented
    assert none["levels"]["stop_loss"] is None and none["levels"]["price_target"] is None


def test_a_sellers_exits_are_not_used_for_a_buy():
    import math

    # After a Sell / Underweight call the agents' stop sits above and their target below: the model's instead.
    v = EP.validate(plan(), "Underweight", REF, ACCOUNT, TODAY, stop_loss=101.95, price_target=93.0, sigma_h=0.1)
    lv = v["levels"]
    assert lv["dropped"] == {"stop_loss": 101.95, "price_target": 93.0}
    assert lv["stop_loss"] == round(98.0 * math.exp(-0.1), 2) and lv["source"]["stop_loss"] == "model"
    assert lv["price_target"] == round(98.0 * math.exp(0.1), 2) and lv["source"]["price_target"] == "model"
    assert any("seller's level" in w for w in v["warnings"])
    # A Buy with one level on the wrong side drops only that one; no volatility known: nothing in its place.
    w = EP.validate(plan(), "Buy", REF, ACCOUNT, TODAY, stop_loss=94.0, price_target=96.0)
    assert w["levels"]["stop_loss"] == 94.0 and w["levels"]["price_target"] is None
    assert w["levels"]["dropped"] == {"price_target": 96.0}


def test_a_plan_saved_with_a_sellers_exits_reads_back_without_them(tmp_path):
    import json

    from tradingagents.ledger.store import Store

    store = Store(tmp_path / "book")
    EP.record(store, "AAA", "2026-10-23", "run-1", "gemini", REF, ACCOUNT, None,
              {"rating": "Underweight", "plan": plan()}, today=TODAY)
    # As saved before the check: the agents' levels kept on the wrong side, Buy at the entry price.
    row = store.con.execute("SELECT id, validation_json FROM entry_plans").fetchone()
    old = json.loads(row[1])
    old["levels"] = {"stop_loss": 101.95, "price_target": 93.0, "base": 98.0,
                     "source": {"stop_loss": "agents", "price_target": "agents"}}
    old["buy_at"] = {"price": 98.0, "source": "agents"}
    runs = EP.plans_for(_with_validation(store, row[0], old), "AAA", TODAY)
    v = runs[0]["final"]["validation"]
    assert v["levels"]["stop_loss"] is None and v["levels"]["price_target"] is None
    assert v["levels"]["dropped"] == {"stop_loss": 101.95, "price_target": 93.0}
    assert v["buy_at"]["price"] == 97.0


def _with_validation(store, plan_id, validation):
    """A copy of the plans table with one row's saved validation replaced (the table itself is append-only)."""
    import json
    import sqlite3

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    cols = [r[1] for r in store.con.execute("PRAGMA table_info(entry_plans)")]
    con.execute(f"CREATE TABLE entry_plans ({', '.join(cols)})")
    for r in store.con.execute("SELECT * FROM entry_plans"):
        r = dict(zip(cols, r))
        if r["id"] == plan_id:
            r["validation_json"] = json.dumps(validation)
        con.execute(f"INSERT INTO entry_plans VALUES ({', '.join('?' * len(cols))})", [r[c] for c in cols])
    return con


def test_buy_at_is_one_price_the_low_end_of_a_range_else_the_internal_models():
    import math

    assert EP.validate(plan(), "Buy", REF, ACCOUNT, TODAY)["buy_at"] == {"price": 97.0, "source": "agents"}  # 97-99, not 98
    single = EP.validate(plan(entry_range_low=None, entry_range_high=None), "Buy", REF, ACCOUNT, TODAY)
    assert single["buy_at"] == {"price": 98.0, "source": "agents"}                 # no range: the entry price
    bare = plan(entry_price=None, entry_range_low=None, entry_range_high=None, entry_type="limit")
    v = EP.validate(bare, "Buy", REF, ACCOUNT, TODAY, sigma_h=0.1, model_buy=96.4)
    assert v["buy_at"] == {"price": 96.4, "source": "model"}
    assert v["levels"]["price_target"] == round(96.4 * math.exp(0.1), 2) and v["levels"]["source"]["stop_loss"] == "model"
    assert any("needs a price" in b for b in v["blocks"])           # still flagged: the agents gave no price


def test_the_record_gives_the_manager_the_traders_exits_it_left_out(tmp_path):
    from tradingagents.ledger.store import Store

    store = Store(tmp_path / "book")
    trader = {"action": "Buy", "plan": plan(), "stop_loss": 94.0, "price_target": 112.0}
    final = {"rating": "Buy", "plan": plan(), "review": "approved", "stop_loss": None, "price_target": None}
    out = EP.record(store, "AAA", "2026-10-23", None, "test", REF, ACCOUNT, trader, final, today=TODAY, sigma_h=0.1)
    lv = out["plans"][1]["validation"]["levels"]
    assert (lv["stop_loss"], lv["price_target"]) == (94.0, 112.0)
    assert lv["source"] == {"stop_loss": "trader", "price_target": "trader"}


def test_the_trader_states_a_sell_target():
    p = TraderProposal.model_validate({"action": "Buy", "reasoning": "r", "price_target": "$112.50"})
    assert p.price_target == 112.5 and "**Sell Target**: 112.5" in render_trader_proposal(p)


# --- the reference quote ---

def _closes(days, values):
    return {"Close": pd.DataFrame({"AAA": values}, index=pd.DatetimeIndex(days))}


def test_the_reference_is_the_as_traded_close_and_its_basis_is_verified(monkeypatch):
    import tradingagents.research.prices as prices

    days = pd.bdate_range("2026-10-19", "2026-10-23")
    monkeypatch.setattr(prices, "bars", lambda *a, **k: _closes(days, [95.0, 96.0, 97.0, 98.0, 100.0]))
    monkeypatch.setattr(EP, "_currency", lambda *a: "USD")
    latest = EP.reference_quote("AAA", "2026-10-23", "cache")
    assert latest["verified"] and latest["price"] == 100.0 and latest["session"] == "2026-10-23"

    monkeypatch.setattr(EP, "_cached_actions", lambda *a: (None, None))
    assert not EP.reference_quote("AAA", "2026-10-21", "cache")["verified"]     # later sessions, no actions

    # A 2-for-1 split on 10-22: the stored (adjusted) 97 on 10-21 traded at 194.
    monkeypatch.setattr(EP, "_cached_actions", lambda *a: ({"dividends": [], "splits": [["2026-10-22", 2.0]]},
                                                           "2026-10-24T01:00:00"))
    past = EP.reference_quote("AAA", "2026-10-21", "cache")
    assert past["verified"] and past["price"] == pytest.approx(194.0) and past["adjusted"] == 97.0
    assert "1 later split" in past["how"]


# --- the agents ---

def _structured_llm(captured, result):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def test_the_trader_is_given_the_reference_and_its_plan_reaches_the_state():
    from tradingagents.agents.trader.trader import create_trader

    captured = {}
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="r", stop_loss=94.0,
                              entry_plan=EntryPlan.model_validate(plan()))
    out = create_trader(_structured_llm(captured, proposal))(
        {"company_of_interest": "AAA", "investment_plan": "Buy", "market_report": "",
         "entry_reference": EP.render_reference(REF)})
    text = " ".join(m["content"] for m in captured["prompt"])
    assert "closed at 100.0 USD on 2026-10-23" in text and "entry_plan" in text
    assert out["trader_entry"]["plan"]["entry_price"] == 98.0 and out["trader_entry"]["stop_loss"] == 94.0
    assert "**Entry Plan**: status ready" in out["trader_investment_plan"]


def test_the_manager_reviews_the_plan_and_the_risk_analysts_challenge_it():
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator

    risk = {"history": "", "aggressive_history": "", "conservative_history": "", "neutral_history": "",
            "latest_speaker": "", "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 0}
    state = {"company_of_interest": "AAA", "investment_plan": "Buy", "trader_investment_plan": "Buy at 98",
             "risk_debate_state": risk, "entry_reference": EP.render_reference(REF), "market_report": "",
             "sentiment_report": "", "news_report": "", "fundamentals_report": ""}
    captured = {}
    decision = PortfolioDecision(rating=PortfolioRating.OVERWEIGHT, executive_summary="s", investment_thesis="t",
                                 price_target=110.0, stop_loss=93.0, entry_plan_review="revised",
                                 entry_plan=EntryPlan.model_validate(plan(entry_price=97.5)))
    out = create_portfolio_manager(_structured_llm(captured, decision))(state)
    assert "approve the trader's entry plan, revise it, or reject it" in captured["prompt"]
    assert "always entry_price (buy), price_target (sell) and stop_loss" in captured["prompt"]
    assert out["final_entry"] == {"rating": "Overweight", "price_target": 110.0, "stop_loss": 93.0, "review": "revised",
                                  "plan": EntryPlan.model_validate(plan(entry_price=97.5)).model_dump(mode="json")}
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: captured.__setitem__("risk", prompt) or MagicMock(content="careful")
    create_conservative_debator(llm)(state)
    assert EP.RISK_INSTRUCTION in captured["risk"] and "closed at 100.0" in captured["risk"]


def test_the_instructions_reach_only_the_agents_that_need_them():
    import inspect

    from tradingagents.agents.analysts import market_analyst, news_analyst
    from tradingagents.agents.managers import research_manager
    from tradingagents.agents.researchers import bear_researcher, bull_researcher

    for module in (market_analyst, news_analyst, research_manager, bull_researcher, bear_researcher):
        src = inspect.getsource(module)
        assert "entry_reference" not in src and "entry_plan" not in src, module.__name__


# --- persistence ---

def test_plans_are_kept_with_their_approval_and_later_ones_supersede_them(tmp_path):
    from tradingagents.ledger.store import Store

    store = Store(tmp_path / "book")
    trader = {"action": "Buy", "plan": plan(), "stop_loss": 94.0}
    first = EP.record(store, "AAA", "2026-10-23", "run-1", "gemini", REF, ACCOUNT, trader,
                      {"rating": "Buy", "plan": plan(entry_price=97.5), "review": "revised", "price_target": 110.0},
                      {"model_decision": "d1"}, today=TODAY)
    assert [p["stage"] for p in first["plans"]] == ["trader", "portfolio_manager"]
    EP.record(store, "AAA", "2026-10-26", "run-2", "chatgpt", REF, ACCOUNT, None,
              {"rating": "Hold", "plan": plan(status="wait", entry_type=None, trigger="a close above 105")}, today=TODAY)
    runs = EP.plans_for(store.con, "AAA", TODAY)
    assert runs[0]["current"] and runs[0]["trader"] is None and runs[0]["final"]["validation"]["state"] == "waiting"
    old = runs[1]
    assert not old["current"] and old["final"]["superseded_by"] == runs[0]["final"]["id"]
    assert old["final"]["review"] == "revised" and old["final"]["parent_id"] == old["trader"]["id"]
    assert old["final"]["settings"]["settings"]["version"] == 7 and old["final"]["reference"]["price"] == 100.0
    assert old["final"]["evidence"] == {"model_decision": "d1", "evidence_ids": ["REF1", "PX1"]}
    assert EP.plans_for(store.con, "AAA", date(2026, 11, 2))[1]["final"]["expired"]
    with pytest.raises(Exception, match="append-only"):
        store.con.execute("UPDATE entry_plans SET review = 'approved'")


def test_the_measured_prompt_additions_stay_small():
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    added = {name: len(enc.encode(text)) for name, text in
             (("trader", EP.TRADER_INSTRUCTION), ("risk", EP.RISK_INSTRUCTION), ("pm", EP.PM_INSTRUCTION),
              ("reference", EP.render_reference(REF)))}
    assert added["trader"] < 100 and added["risk"] < 40 and added["pm"] < 100 and added["reference"] < 60, added
    print(json.dumps(added))
