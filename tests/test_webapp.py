"""The web app's API on synthetic prices: the trade view, the books, and page loads that never pay for AI.

Prices and holdings are labelled synthetic fixtures; no network and no model is used.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _synthetic_closes(tickers, required=(), cache_dir=None):
    rng = np.random.default_rng(7)
    days = pd.bdate_range(end="2026-10-21", periods=252 * 12)
    out = {}
    for i, t in enumerate(dict.fromkeys(tickers)):
        if t in ("SGOV", "BIL"):
            out[t] = 100 * np.cumprod(np.full(len(days), 1 + 0.04 / 252))
        else:
            out[t] = 100 * np.cumprod(1 + rng.normal(0.0003 + 0.0001 * (i % 3), 0.01 + 0.003 * (i % 4), len(days)))
    return pd.DataFrame(out, index=days)


@pytest.fixture
def server(tmp_path, monkeypatch):
    from tradingagents.webapp import server as srv

    (tmp_path / "my_portfolio.json").write_text(json.dumps({"positions": [
        {"ticker": "AAA", "quantity": 10}, {"ticker": "BBB", "quantity": 5}, {"ticker": "SGOV", "quantity": 20}]}))
    (tmp_path / "my_buys.json").write_text(json.dumps({"daily": {"AAA": 5}}))
    monkeypatch.setattr(srv, "fetch_closes", _synthetic_closes)
    monkeypatch.setattr(srv.App, "universe", lambda self: [])
    monkeypatch.setattr(srv, "latest_prices", lambda cache, tickers: {t: 100.0 for t in tickers})

    def no_model(*a, **k):
        raise AssertionError("a model client was created")

    import tradingagents.llm_clients.factory as factory
    monkeypatch.setattr(factory, "create_llm_client", no_model)
    monkeypatch.setenv("TRADINGAGENTS_NO_PAID_AI", "1")
    config = {"data_cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "results")}
    httpd = srv.serve(str(tmp_path / "my_portfolio.json"), str(tmp_path / "my_buys.json"), port=0, config=config)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, tmp_path, httpd.RequestHandlerClass
    httpd.shutdown()


def _call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def _ai_rows(tmp_path):
    import sqlite3
    con = sqlite3.connect(tmp_path / "portfolio.sqlite3")
    return con.execute("SELECT (SELECT COUNT(*) FROM ai_usage) + (SELECT COUNT(*) FROM ai_jobs)").fetchone()[0]


def test_page_loads_and_polling_never_call_a_model(server):
    base, tmp_path, _ = server
    for path in ("/api/state", "/api/ledger", "/api/ai", "/api/recommendations", "/api/ratings", "/api/capital",
                 "/api/market/explain", "/api/trade", "/api/income", "/api/model-record", "/api/decisions"):
        status, _ = _call(base, path)
        assert status == 200, path
    for gone in ("/api/tracker", "/api/backtest"):
        assert _call(base, gone)[0] == 404
    assert _call(base, "/api/plan", {"scan": False})[0] == 404
    assert _ai_rows(tmp_path) == 0


def test_the_trade_view_shows_the_budget_and_waits_for_a_decision(server, monkeypatch):
    from datetime import date

    from tradingagents.ledger import income

    base, tmp_path, _ = server
    monkeypatch.setattr(income, "_dividends", lambda cache, t: [(date(2026, 10, 1), 0.30), (date(2100, 1, 1), 0.30)])
    status, t = _call(base, "/api/trade")
    assert status == 200 and not t["ready"] and "decision" in t["note"]
    assert t["income"]["budget"]["available"] == 0.0 and "No trading budget" in t["income"]["budget"]["note"]
    status, c = _call(base, "/api/capital", {"values": {"trading_budget_usd": 250}})
    assert status == 200 and c["settings"]["trading_budget_usd"] == 250.0 and c["budget"]["available"] == 250.0
    assert _call(base, "/api/trade")[1]["income"]["budget"]["set"] == 250.0
    assert _call(base, "/api/capital", {"values": {"trading_capital_usd": 1}})[0] == 400


def test_ledger_writes_go_through_the_books_and_keep_the_old_file_in_step(server):
    base, tmp_path, _ = server
    books = _call(base, "/api/ledger")[1]["sleeves"]
    assert set(books) == {"trading", "combined"} and books["combined"]["positions"]["SGOV"]["quantity"] == 20
    status, out = _call(base, "/api/ledger/contribution", {"amount": 29})
    assert status == 200, out
    _call(base, "/api/portfolio", {"positions": [{"ticker": "AAA", "quantity": 12}, {"ticker": "BBB", "quantity": 5},
                                                 {"ticker": "SGOV", "quantity": 20}]})
    exported = json.loads((tmp_path / "my_portfolio.json").read_text())
    assert {p["ticker"]: p["quantity"] for p in exported["positions"]}["AAA"] == 12


def test_a_paid_job_is_refused_without_prices_and_says_why(server):
    base, tmp_path, _ = server
    status, out = _call(base, "/api/ratings", {})
    assert status == 202
    for _ in range(100):
        state = _call(base, "/api/ratings")[1]
        if not state["running"]:
            break
    assert _ai_rows(tmp_path) == 0
    assert any("switched off" in line or "no price" in line for line in state["log"])


def test_a_ticker_page_answers_even_when_its_sources_fail(server):
    base, _, _ = server
    status, out = _call(base, "/api/ticker?symbol=AAA")
    assert status == 200 and out["symbol"] == "AAA"
    # Prices come from the stubbed feed; Yahoo's other sources are unreachable in tests.
    assert "error" in out["news"] and "error" in out["analysts"]
    assert out["mine"]["quantity"]["trading"] == 10
    assert _call(base, "/api/ticker?symbol=../etc")[0] == 400
    assert _call(base, "/api/report?path=C:/Windows/win.ini")[0] == 400


def test_a_pasted_answer_is_kept_with_its_prompt_and_a_second_one_archives_the_first(server, monkeypatch):
    from tests.test_manual import offline

    base, tmp_path, _ = server
    offline(monkeypatch)
    status, prompt = _call(base, "/api/manual-prompt?ticker=AAA&date=2026-10-23")
    assert status == 200 and "API request 1 of 12: Market Analyst report" in prompt["prompt"]
    day = tmp_path / "results" / "AAA" / "2026-10-23"
    assert (day / "prompts" / f"{prompt['prompt_id']}.txt").is_file()

    answer = "## Step 1 ...\n```json\n" + json.dumps(
        {"rating": "Hold", "executive_summary": "Flat.", "investment_thesis": "None.",
         "trader": {"action": "Hold", "entry_price": 101.0, "stop_loss": 95.0}}) + "\n```"
    status, first = _call(base, "/api/manual-answer", {"ticker": "AAA", "date": "2026-10-23", "text": answer})
    assert status == 200 and first["prompt_id"] == prompt["prompt_id"] and first["trader"]["entry_price"] == 101.0
    prov = json.loads((day / "reports" / "provenance.json").read_text())
    assert prov["prompt_id"] == prompt["prompt_id"] and prov["trader"]["stop_loss"] == 95.0
    assert (day / "reports" / "prompt.txt").read_text(encoding="utf-8") == prompt["prompt"]
    assert "**Trader Proposal**" in (day / "reports" / "5_portfolio" / "decision.md").read_text()

    status, _ = _call(base, "/api/manual-answer", {"ticker": "AAA", "date": "2026-10-23", "text": answer.replace("Hold", "Sell")})
    assert status == 200
    archived = list((day / "archive").iterdir())
    assert len(archived) == 1 and '"Hold"' in (archived[0] / "chat_answer.md").read_text()
    assert '"Sell"' in (day / "reports" / "chat_answer.md").read_text()


def test_failed_gemini_flash_moves_on_to_free_openrouter_models_then_other_gemini_models(tmp_path, monkeypatch):
    from tradingagents.webapp import server as srv

    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    app = srv.App(tmp_path / "my_portfolio.json", tmp_path / "my_buys.json", tmp_path / "u.json",
                  {**srv.DEFAULT_CONFIG, "data_cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r")})
    app.store.set("global", "ai_fallbacks", ["google", "openrouter"])
    app.store.set("global", "ai_primary", {"provider": "google", "quick": "gemini-3.8-flash", "deep": "gemini-3.8-flash"})
    g = app.store.settings("global")
    chain = [(c["llm_provider"], c["deep_think_llm"]) for c in app.agent_attempts(g)]
    assert chain[0] == ("google", "gemini-3.8-flash")
    assert [p for p, _ in chain[1:]] == ["openrouter"] * 3 + ["google"] * (len(chain) - 1 - 3)
    assert all(m.endswith(":free") for p, m in chain if p == "openrouter")
    prices = app.gateway().policy.prices
    assert prices["openrouter/" + chain[1][1]].input == 0.0           # free models are priced at $0
    assert srv._short_error(RuntimeError("Error code: 429 - rate limited upstream")) == "rate limited"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert all(c["llm_provider"] != "openrouter" for c in app.agent_attempts(g))   # no key, no OpenRouter


def test_keys_are_saved_to_the_env_file_masked_on_the_page_and_limited_to_known_names(tmp_path, monkeypatch):
    from tradingagents.webapp import keys

    env = tmp_path / ".env"
    env.write_text("OTHER=1\nOPENROUTER_API_KEY=old\n", encoding="utf-8")
    monkeypatch.setattr(keys, "env_path", lambda: env)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = keys.save("OPENROUTER_API_KEY", "sk-or-v1-abcdef1234567890")
    assert out["masked"] == "sk-or-…7890" and "abcdef" not in str(keys.listing())
    assert env.read_text().splitlines() == ["OTHER=1", "OPENROUTER_API_KEY=sk-or-v1-abcdef1234567890"]
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-abcdef1234567890"     # used at once
    keys.save("OPENROUTER_API_KEY", "")
    assert env.read_text().splitlines() == ["OTHER=1"] and "OPENROUTER_API_KEY" not in os.environ
    with pytest.raises(ValueError):
        keys.save("PATH", "x")                                              # only known key names
    with pytest.raises(ValueError):
        keys.save("GROQ_API_KEY", "a b")


def test_a_stuck_job_can_be_given_up_and_its_late_result_is_ignored(monkeypatch):
    from tradingagents.webapp import server as srv

    gate = threading.Event()
    job = srv.Job()
    job.start(lambda j: (gate.wait(5), "late")[1])
    assert job.running and not job.stop_waiting()                        # it reported progress just now
    job.touched_at -= srv.Job.STUCK_AFTER + 1
    assert job.state()["stuck"] and job.stop_waiting() and not job.running
    assert job.start(lambda j: "fresh")                                   # a new run can start
    gate.set()
    for _ in range(50):
        if not job.running:
            break
        time.sleep(0.05)
    assert job.result == "fresh"                                          # the abandoned run never overwrites it


def test_live_status_names_the_agent_and_the_tool():
    import uuid

    from tradingagents.llm_clients.live_status import LiveStatus

    lines = []
    s = LiveStatus(lines.append)
    rid = uuid.uuid4()
    s.on_chat_model_start({}, [[]], run_id=rid, metadata={"langgraph_node": "Market Analyst"})
    s.on_tool_start({"name": "get_stock_data"}, "", metadata={"langgraph_node": "tools_market"})
    s.on_llm_end(None, run_id=rid, metadata={"langgraph_node": "Bull Researcher"})
    assert lines[0] == "Market Analyst: thinking" and lines[1] == "Market Analyst: fetching get_stock_data"
    assert lines[2].startswith("Bull Researcher: answered in")


def test_a_recorded_buy_shows_in_the_portfolio_with_its_value_but_sgov_does_not(server):
    base, tmp_path, _ = server
    status, out = _call(base, "/api/ledger/fill", {"sleeve": "trading", "side": "buy", "ticker": "AAA", "quantity": 2,
                                                   "price": 90.0, "date": "2026-10-20", "broker_record": True})
    assert status == 200, out
    status, p = _call(base, "/api/portfolio-view")
    held = {h["ticker"]: h for h in p["holdings"]}
    assert status == 200 and "SGOV" not in held and set(held) == {"AAA", "BBB"}       # SGOV is never listed
    assert held["AAA"]["units"] == 12 and held["AAA"]["value"] == 1200.0               # 10 held + 2 bought, at 100
    assert p["trades"][0]["ticker"] == "AAA" and p["trades"][0]["quantity"] == 2 and p["bought"] == 180.0
