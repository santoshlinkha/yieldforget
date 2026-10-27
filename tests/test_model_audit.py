"""The 2026-10 model audit's fixes, on hand-made cases: member-only percentiles, shared ranks for tied scores, an
uninformative risk model exposed rather than passed, the execution spec the page shows, and the agent-run record."""

import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import daily, trade, verdict
from tradingagents.stocks.adapters import member_pct, within_date_pct

pytestmark = pytest.mark.unit


def test_members_keep_their_percentiles_whatever_else_is_scored():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=40), 1)                         # rounded: plenty of exact ties
    member = np.r_[np.ones(30, bool), np.zeros(10, bool)]
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2026-10-26")], [f"T{i}" for i in range(30)]], names=["date", "ticker"])
    alone = within_date_pct(pd.DataFrame(index=idx), s[:30])
    got = member_pct(s, member)
    assert np.allclose(got[:30], alone)                          # the 10 extra rows do not move the members
    top = member_pct(np.r_[s[:30], [99.0]], np.r_[member[:30], [False]])
    assert top[-1] == 1.0 and np.allclose(top[:30], alone)       # above every member: rank 31 of 31
    assert np.all(member_pct(np.ones(5), np.ones(5, bool)) == 0.5)


def _today(scores, p_risk=0.07, informative=True):
    n = len(scores)
    return pd.DataFrame({"expected_net": [0.007] * n, "net_sd": [0.08] * n, "net_q10": [-0.1] * n,
                         "expected_excess": [0.0] * n, "p_win": np.linspace(0.6, 0.5, n), "p_risk": [p_risk] * n,
                         "sigma": [0.02] * n, "cost_rt": [0.003] * n, "score_pct": scores, "in_index": [True] * n,
                         "risk_informative": [informative] * n}, index=[f"S{i}" for i in range(n)][::-1])


R = {**daily.DEFAULT_RULES, "max_names": 5, "max_position_share": 0.25, "min_order": 1.0, "max_large_loss_prob": 0.15}
IRA = {"total": 6000.0, "by_ticker": {}}


def test_tied_scores_are_listed_alphabetically_not_by_the_unskilled_p_profit():
    today = _today([0.9, 0.9, 0.5])                               # S2 and S1 tied; S2 has the higher P(profit)
    lines = daily.entries(today, set(), set(), R, 1000.0, 1000.0, IRA, pd.Series(10.0, index=today.index), True, None, "x")
    assert [x["ticker"] for x in lines] == ["S1", "S2", "S0"]
    assert [x["score_tied_with"] for x in lines] == [1, 1, 0]


def test_an_uninformative_risk_model_blocks_with_its_reason_or_passes_only_by_setting():
    today = _today([0.9, 0.5], informative=False)
    prices = pd.Series(10.0, index=today.index)
    lines = daily.entries(today, set(), set(), {**R, "risk_gate_when_uninformative": "block"}, 1000.0, 1000.0, IRA,
                          prices, True, None, "x")
    assert all(any("risk gate not assessed" in b for b in x["blocked"]) for x in lines)
    assert not any(x.get("paper_action") for x in lines)
    skip = daily.entries(today, set(), set(), R, 1000.0, 1000.0, IRA, prices, True, None, "x")   # the default
    assert skip[0]["paper_action"] == "PAPER ENTER" and "passed anyway" in skip[0]["notes"][0]
    assert not daily.risk_informative(np.full(10, 0.0714)) and daily.risk_informative(np.array([0.05, 0.2]))
    score = {"rank": 1, "of": 499, "rank_pct": 0.99, "expected_net": 0.01, "p_win": 0.55, "p_risk": 0.07,
             "p_risk_informative": False, "cost_rt": 0.003}
    v = verdict.verdict(False, None, None, {}, None, score=score)
    risk = next(c for c in v["checks"] if c["name"].startswith("Chance of a large loss"))
    assert risk["ok"] is True and "the same for every stock" in risk["detail"] and v["verdict"] == "BUY"


def test_the_page_shows_the_tested_entry_policy_and_exits_from_the_fill():
    from tradingagents.stocks.positions import POLICIES, execution_spec

    spec = execution_spec(POLICIES["barrier_20"])
    assert "next session's official close" in spec["order"] and "no limit price" in spec["order"]
    assert "20 sessions" in spec["exit"] and spec["version"] == POLICIES["barrier_20"].version()
    decision = {"id": "d", "info_cutoff": "2026-10-26", "execution_date": "2026-10-27", "mode": "experimental",
                "horizon": 20, "lots": [], "model": {"policy": {"name": "barrier_20"}},
                "entries": [{"ticker": "AAA", "action": "WAIT", "paper_action": "PAPER ENTER", "price": 100.0,
                             "sigma": 0.02, "execute": "2026-10-27", "score_tied_with": 2}]}
    best = trade.plan(decision, 50.0)["best"]
    assert best["entry"]["text"].startswith("at the close of 2026-10-27") and best["score_tied_with"] == 2
    assert best["target_pct"] == pytest.approx(np.expm1(0.02 * np.sqrt(20)))


def test_every_routed_agent_run_is_recorded_failures_included_and_never_edited(tmp_path, monkeypatch):
    import sqlite3

    from tradingagents.webapp import server as srv

    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    app = srv.App(tmp_path / "my_portfolio.json", tmp_path / "my_buys.json", tmp_path / "u.json",
                  {**srv.DEFAULT_CONFIG, "data_cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r")})
    app.store.set("global", "ai_fallbacks", ["openrouter"])
    app.store.set("global", "ai_openrouter_models", ["a:free"])
    g = app.store.settings("global")
    monkeypatch.setattr("tradingagents.optimizer.ai_views.analyze",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Error code: 429 - free-models-per-day")))
    job = srv.Job()
    with pytest.raises(ValueError, match="No AI provider"):
        app._run_agents(job, "GEN", "2026-10-26", ["market"], g)
    row = dict(app.store.con.execute("SELECT * FROM agent_runs").fetchone())
    assert row["status"] == "failed" and row["ticker"] == "GEN" and row["actual_provider"] is None
    tried = json.loads(row["attempts_json"])
    assert tried[-1] == {"provider": "openrouter", "model": "a:free", "state": "failed",
                         "why": "OpenRouter free daily limit reached"}
    assert row["requested_provider"] == tried[0]["provider"] and len(row["route_version"]) > 8
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        app.store.con.execute("UPDATE agent_runs SET status = 'ok'")


def test_the_runners_up_cell_reads_the_agents_own_call_not_the_models_verdict(tmp_path, monkeypatch):
    from tradingagents.webapp import server as srv

    app = srv.App(tmp_path / "my_portfolio.json", tmp_path / "my_buys.json", tmp_path / "u.json",
                  {**srv.DEFAULT_CONFIG, "data_cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r")})
    folder = tmp_path / "r" / "FICO" / "2026-10-23" / "reports"
    folder.mkdir(parents=True)
    (folder / "complete_report.md").write_text("# Report", encoding="utf-8")
    (folder / "0_FINAL_DECISION.md").write_text(
        "# FINAL DECISION: BUY\n\n- **The agents' view** (shown, not used): rating Hold, i.e. WATCH; differs\n"
        "- **Agent models**: google/gemini-3.5-flash-lite (after a fallback)\n", encoding="utf-8")
    r = app.latest_agent_report("fico")
    assert (r["agent_rating"], r["agent_action"], r["agent_models"]) == ("Hold", "WATCH", "google/gemini-3.5-flash-lite")
    assert r["last_run"] is None and app.latest_agent_report("NONE")["available"] is False
