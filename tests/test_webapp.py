"""The web app's API on synthetic prices: the trade view, the books, and page loads that never pay for AI.

Prices and holdings are labelled synthetic fixtures; no network and no model is used.
"""

import json
from pathlib import Path
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


_TOKENS = {}


def _token(base):
    """The write token the server prints into its page, read the way the browser gets it."""
    if base not in _TOKENS:
        with urllib.request.urlopen(base + "/", timeout=30) as res:
            page = res.read().decode()
        _TOKENS[base] = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]
    return _TOKENS[base]


def _call(base, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    sent = {"Content-Type": "application/json"}
    if data is not None:
        sent["X-CSRF-Token"] = _token(base)
    req = urllib.request.Request(base + path, data=data, headers={**sent, **(headers or {})})
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
                 "/api/market/explain", "/api/trade", "/api/income", "/api/model-record", "/api/decisions",
                 "/api/model-outputs", "/api/schedule"):
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
    n_or = len(g["ai_openrouter_models"])
    assert [p for p, _ in chain[1:]] == ["openrouter"] * n_or + ["google"] * (len(chain) - 1 - n_or)
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


def _raw_post(base, path, data: bytes, headers: dict):
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return res.status
    except urllib.error.HTTPError as err:
        return err.code


def test_writes_from_another_site_or_without_the_page_token_are_refused(server):
    base, tmp_path, _ = server
    body = json.dumps({"values": {"ai_shortlist": 7}}).encode()
    token = _token(base)
    good = {"Content-Type": "application/json", "X-CSRF-Token": token}
    port = base.rsplit(":", 1)[1]
    refused = [
        {**good, "Origin": "https://evil.example"},                         # another site
        {**good, "Host": "evil.example:" + port},                           # DNS rebinding
        {**good, "Content-Type": "text/plain"},                             # a form-like simple request
        {"Content-Type": "application/json"},                               # no token
        {**good, "X-CSRF-Token": "stale"},                                  # a token from another run
        {**good, "Sec-Fetch-Site": "cross-site"},
    ]
    for path in ("/api/settings", "/api/ledger/contribution", "/api/portfolio", "/api/keys", "/api/decisions/run"):
        for headers in refused:
            assert _raw_post(base, path, body, headers) == 403, (path, headers)
    from tradingagents.ledger.store import Store
    assert Store(tmp_path).get("global", "ai_shortlist") == 10            # nothing was written
    assert _raw_post(base, "/api/settings", body, {**good, "Origin": base}) == 200
    assert _raw_post(base, "/api/settings", b"[1, 2]", good) == 400        # not a JSON object


def test_two_identical_manual_fills_are_two_trades_and_a_retry_is_one(server):
    base, tmp_path, _ = server
    _call(base, "/api/ledger/contribution", {"amount": 1000, "date": "2026-10-19"})
    fill = {"sleeve": "trading", "side": "buy", "ticker": "SYNTH", "quantity": 1, "price": 80, "date": "2026-10-19"}
    s1, a = _call(base, "/api/ledger/fill", fill)
    s2, b = _call(base, "/api/ledger/fill", fill)
    assert s1 == s2 == 200 and a["event"] != b["event"] and a["applied"] and b["applied"]
    s3, again = _call(base, "/api/ledger/fill", {**fill, "op_id": b["op_id"]})
    assert s3 == 200 and again["event"] == b["event"] and again["applied"] is False
    books = _call(base, "/api/ledger")[1]["sleeves"]["trading"]
    assert books["positions"]["SYNTH"]["quantity"] == 2
    assert _call(base, "/api/ledger/fill", {**fill, "price": "nan"})[0] == 400
    assert _call(base, "/api/ledger/fill", {**fill, "op_id": "x"})[0] == 400


def test_a_cash_edit_is_saved_reloaded_and_must_say_what_it_is(server):
    base, tmp_path, _ = server
    state = _call(base, "/api/state")[1]
    positions = state["portfolio"]["positions"]
    before = state["cash"]["settled"]
    status, out = _call(base, "/api/portfolio", {"positions": positions, "cash": before + 1234})
    assert status == 400 and "correction" in out["error"]
    assert _call(base, "/api/state")[1]["cash"]["settled"] == before          # refused means nothing saved
    status, out = _call(base, "/api/portfolio", {"positions": positions, "cash": before + 1234, "cash_mode": "flow"})
    assert status == 200 and out["saved"]["cash_settled"] == pytest.approx(before + 1234)
    assert out["cash_change"]["mode"] == "flow"
    assert _call(base, "/api/state")[1]["cash"]["settled"] == pytest.approx(before + 1234)   # survives a reload
    status, out = _call(base, "/api/portfolio", {"positions": positions, "cash": before + 1000,
                                                 "cash_mode": "correction"})
    assert status == 200 and out["saved"]["cash_settled"] == pytest.approx(before + 1000)


def test_an_abandoned_run_cannot_overwrite_the_new_runs_progress_or_log():
    from tradingagents.webapp import server as srv

    gate, wrote = threading.Event(), threading.Event()

    def slow(j):
        gate.wait(5)
        j.progress = "old run still talking"
        j.log.append("old line")
        j.log += ["old extra"]
        j.attempts.append({"provider": "old"})
        j.fraction = 0.99
        j.say("old run says")
        wrote.set()
        return "late"

    job = srv.Job()
    job.start(slow)
    job.touched_at -= srv.Job.STUCK_AFTER + 1
    assert job.stop_waiting()
    hold = threading.Event()
    assert job.start(lambda j: (j.say("new run working"), j.log.append("new line"), hold.wait(5), "fresh")[-1])
    time.sleep(0.05)
    gate.set()
    assert wrote.wait(5)
    s = job.state()
    assert s["progress"] == "new run working" and s["log"] == ["new line"] and s["attempts"] == []
    assert s["fraction"] is None and s["running"]
    hold.set()
    for _ in range(50):
        if not job.running:
            break
        time.sleep(0.05)
    assert job.result == "fresh"


def test_a_key_that_cannot_be_written_is_not_used_either(tmp_path, monkeypatch):
    import os
    import pathlib

    from tradingagents.webapp import keys

    env = tmp_path / ".env"
    env.write_text("OTHER=1\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def locked(self, target):
        raise PermissionError("the .env file is open elsewhere")

    monkeypatch.setattr(pathlib.Path, "replace", locked)
    with pytest.raises(PermissionError):
        keys.save("OPENROUTER_API_KEY", "sk-or-v1-synthetic123456", path=env)
    assert "OPENROUTER_API_KEY" not in os.environ and env.read_text(encoding="utf-8") == "OTHER=1\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_the_portfolio_view_separates_stocks_sgov_cash_and_reservations(server):
    base, tmp_path, _ = server
    _call(base, "/api/ledger/contribution", {"amount": 50})
    status, d = _call(base, "/api/portfolio-view")
    a = d["account"]
    assert status == 200 and a["reserve_fund"] == "SGOV" and a["reserve_value"] == pytest.approx(20 * 100.0)
    assert a["stocks"] == pytest.approx(d["value"]) and a["cash_settled"] >= 50
    assert a["total"] == pytest.approx(a["stocks"] + a["reserve_value"] + a["cash_settled"] + a["cash_unsettled"])
    assert a["missing_prices"] == []


def _agent_app(tmp_path, monkeypatch):
    from tradingagents.webapp import server as srv

    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("TRADINGAGENTS_NO_PAID_AI", "1")
    import tradingagents.events.sources as event_sources        # no network in tests: the event sync is a no-op
    monkeypatch.setattr(event_sources, "sync", lambda cache, **k: [])
    import tradingagents.research.sector_model as sector_model    # nor the morning's signals and sector steps
    import tradingagents.stocks.pipeline as pipeline
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **k: {"reused": True})
    monkeypatch.setattr(sector_model, "run", lambda *a, **k: {})
    return srv, srv.App(tmp_path / "my_portfolio.json", tmp_path / "my_buys.json", tmp_path / "u.json",
                        {**srv.DEFAULT_CONFIG, "data_cache_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r")})


def _agent_row(app, ticker, day, status, finished, **kw):
    import uuid as _uuid
    row = {"run_id": _uuid.uuid4().hex, "ticker": ticker, "as_of": day, "started_at": finished, "finished_at": finished,
           "status": status, "route_version": "r", "requested_provider": "google", "requested_model": "m",
           "actual_provider": "google" if status == "ok" else None, "actual_model": "m" if status == "ok" else None,
           "fell_back": 0, "attempts_json": "[]", "roles_json": None, "analysts_json": '["market"]',
           "evidence_hash": "e", "how": "paid", "rating": "Buy" if status == "ok" else None,
           "final_action": "BUY" if status == "ok" else None, "error": None if status == "ok" else "402 out of credit",
           "code_version": "test", **kw}
    app.store.con.execute(f"INSERT INTO agent_runs ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                          tuple(row.values()))
    return row["run_id"]


def test_the_same_agent_inputs_are_not_run_twice(tmp_path, monkeypatch):
    from tradingagents.ledger.ai_gateway import stable_hash

    srv, app = _agent_app(tmp_path, monkeypatch)
    g = app.store.settings("global")
    attempts = app.agent_attempts(g)
    day = "2026-10-23"
    _agent_row(app, "AAA", day, "ok", "2026-10-24T01:00:00.000000Z",
               route_version=stable_hash(app._agent_chain(attempts)), evidence_hash=stable_hash({}))
    folder = tmp_path / "r" / "AAA" / day / "reports"
    folder.mkdir(parents=True)
    (folder / "complete_report.md").write_text("# synthetic report", encoding="utf-8")
    out = app._run_agents(srv.Job(), "AAA", day, ["market"], g)
    assert out["reused"] and out["how"] == "reused" and out["rating"] == "Buy"
    with pytest.raises(Exception):                                   # forced: it really tries (and is refused here)
        app._run_agents(srv.Job(), "AAA", day, ["market"], g, force=True)
    with pytest.raises(Exception):                                   # other analysts are other inputs
        app._run_agents(srv.Job(), "AAA", day, ["market", "news"], g)


def test_a_failed_run_message_can_be_dismissed_and_a_saved_analysis_deleted(tmp_path, monkeypatch):
    srv, app = _agent_app(tmp_path, monkeypatch)
    _agent_row(app, "AAA", "2026-10-22", "ok", "2026-10-22T23:00:00.000000Z")
    _agent_row(app, "AAA", "2026-10-23", "failed", "2026-10-24T01:00:00.000000Z")
    app.analysis.error, app.analysis.attempts = "No AI provider could run the agents", [{"provider": "google"}]
    app.analysis._progress = "AAA as of 2026-10-23: try 2 of 2 on openrouter"
    assert app._last_agent_run("AAA")["status"] == "failed"
    out = app.dismiss_agent_failures("aaa")
    assert out == {"ticker": "AAA", "job_error_cleared": True, "runs_hidden": 1}
    assert app.analysis.error is None and app._last_agent_run("AAA")["status"] == "ok"
    assert app.store.con.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0] == 2     # records kept

    day = tmp_path / "r" / "AAA" / "2026-10-23"
    (day / "reports").mkdir(parents=True)
    (day / "reports" / "complete_report.md").write_text("# synthetic", encoding="utf-8")
    moved = app.delete_agent_report("AAA", "2026-10-23", None)["moved_to"]
    assert not (day / "reports").exists() and (Path(moved[0]) / "complete_report.md").is_file()
    assert "_deleted" in moved[0]
    for bad in (("../etc", "2026-10-23", None), ("AAA", "2026-10-23/../..", None), ("AAA", "2026-10-23", "../x")):
        with pytest.raises(ValueError):
            app.delete_agent_report(*bad)


def test_a_provider_out_of_credit_or_at_its_free_limit_is_skipped_until_it_can_work(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    srv, app = _agent_app(tmp_path, monkeypatch)
    now = datetime(2026, 10, 29, 3, 0, tzinfo=UTC)
    key, until, why = srv._cooldown("openrouter", "m:free", "Error code: 429 - Rate limit exceeded: free-models-per-day", now)
    assert key == "openrouter/*:free" and until == datetime(2026, 10, 30, tzinfo=UTC) and "free daily limit" in why
    key, until, _ = srv._cooldown("google", "gemini-3.8-flash", "402 RESOURCE_EXHAUSTED prepayment credits are depleted", now)
    assert key == "google" and until.hour == 9
    assert srv._cooldown("google", "x", "429 rate limit", now)[0] == "google/x"
    assert srv._cooldown("google", "x", "connection reset", now) is None

    calls = []
    import tradingagents.optimizer.ai_views as views
    monkeypatch.setattr(views, "analyze", lambda *a, **k: calls.append(a[2]["llm_provider"]) or (_ for _ in ()).throw(
        RuntimeError("402 RESOURCE_EXHAUSTED: prepayment credits are depleted")))
    g = app.store.settings("global")
    attempts = [{"llm_provider": "google", "deep_think_llm": "a"}, {"llm_provider": "google", "deep_think_llm": "b"}]
    job = srv.Job()
    with pytest.raises(ValueError) as err:
        app._try_agents(job, "AAA", "2026-10-28", ["market"], tmp_path / "f", 0.0, {}, attempts, [])
    assert calls == ["google"]                                         # the second Google model was not tried
    assert [a["state"] for a in job.attempts] == ["failed", "skipped"] and len(str(err.value)) < 300
    assert "google" in app.store.get("global", "ai_cooldowns")


def test_the_morning_run_is_due_once_per_close_after_its_time_and_runs_each_step(tmp_path, monkeypatch):
    from datetime import date as d, datetime as dt

    srv, app = _agent_app(tmp_path, monkeypatch)
    monkeypatch.setattr(srv.local_store, "last_final_session", lambda now=None: d(2026, 10, 28))
    assert not app.morning_due(dt(2026, 10, 29, 7, 0))                     # before 07:30
    assert app.morning_due(dt(2026, 10, 29, 8, 0))
    calls = []
    import tradingagents.research.engine as engine
    from tradingagents.stocks import daily, model_zoo
    monkeypatch.setattr(engine, "run_forecast", lambda *a, **k: calls.append(("forecast", k["model"], k["horizon"])) or {})
    monkeypatch.setattr(model_zoo, "run", lambda *a, **k: calls.append(("zoo",)) or {"reused": True})
    monkeypatch.setattr(daily, "run", lambda *a, **k: calls.append(("decision", a[4])) or {"id": "x"})
    import tradingagents.research.sector_model as sector_model
    import tradingagents.stocks.pipeline as pipeline
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **k: calls.append(("signals", a[4].horizons)) or {"reused": True})
    monkeypatch.setattr(sector_model, "run", lambda *a, **k: calls.append(("sector",)) or {})
    monkeypatch.setattr(srv, "get_sp500_members", lambda cache: ["AAA"])
    import tradingagents.events.sources as event_sources
    monkeypatch.setattr(event_sources, "sync", lambda cache: calls.append(("events",)) or [{"source": "bls", "status": "ok"}])
    assert app.start_morning()
    for _ in range(100):
        if not app.morning.running:
            break
        time.sleep(0.05)
    state = app.store.get("global", "schedule_state")
    assert state["cutoff"] == "2026-10-28" and state["error"] is None and len(state["steps"]) == 8
    assert ("zoo",) in calls and ("decision", True) in calls and sum(c[0] == "forecast" for c in calls) == 3
    assert ("signals", (5, 20, 60)) in calls and ("sector",) in calls
    assert [s.get("reused") for s in state["steps"] if s["step"].startswith("Screener")] == [True]
    assert not app.morning_due(dt(2026, 10, 29, 9, 0))                     # done for this close
    app.store.set("global", "schedule", {"enabled": False, "morning_at": "07:30", "retrain_every_sessions": 20})
    monkeypatch.setattr(srv.local_store, "last_final_session", lambda now=None: d(2026, 10, 29))
    assert not app.morning_due(dt(2026, 10, 30, 9, 0))                     # switched off


def test_training_inside_the_morning_run_moves_its_bar_and_shows_on_the_models_tab(tmp_path, monkeypatch):
    from datetime import date as d

    srv, app = _agent_app(tmp_path, monkeypatch)
    monkeypatch.setattr(srv.local_store, "last_final_session", lambda now=None: d(2026, 10, 28))
    import tradingagents.research.engine as engine
    from tradingagents.stocks import daily, model_zoo
    seen, gate = {}, threading.Event()

    def zoo(store, cache, say):
        say("Fitting Mamba", 0.5)
        seen["fraction"] = app.morning.fraction
        seen["models_tab"] = app._job_or_morning(app.zoo, "Every model")
        gate.wait(5)
        return {}
    monkeypatch.setattr(engine, "run_forecast", lambda *a, **k: {})
    monkeypatch.setattr(model_zoo, "run", zoo)
    monkeypatch.setattr(daily, "run", lambda *a, **k: {"id": "x"})
    assert app.start_morning()
    for _ in range(100):
        if "models_tab" in seen:
            break
        time.sleep(0.05)
    gate.set()
    assert seen["fraction"] == pytest.approx(0.40 + 0.25 * 0.5)          # the models take 40-65% of the bar
    tab = seen["models_tab"]
    assert tab["running"] and tab["via"] == "morning" and tab["progress"] == "Every model: Fitting Mamba"
    assert not app._job_or_morning(app.decisions, "Daily decision")["running"]     # not that tab's work


def test_dismiss_clears_a_failure_from_another_stock_too(tmp_path, monkeypatch):
    srv, app = _agent_app(tmp_path, monkeypatch)
    _agent_row(app, "GDDY", "2026-10-28", "failed", "2026-10-29T06:00:00.000000Z")
    app.analysis.error, app.analysis.attempts = "No AI provider could run the agents", [{"provider": "google"}]
    app.analysis._progress = "GDDY · openrouter · agent: model error (OpenAIRateLimitError)"
    out = app.dismiss_agent_failures("DVA")                     # the Trade page shows another stock
    assert out["job_error_cleared"] and app.analysis.error is None and out["runs_hidden"] == 1
    assert app._last_agent_run("GDDY") is None


def test_a_per_step_chat_answer_is_saved_as_the_api_runs_report_files(server):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_manual import ANSWER

    base, tmp_path, _ = server
    status, out = _call(base, "/api/manual-answer", {"ticker": "AMD", "date": "2026-10-23", "text": ANSWER})
    assert status == 200 and out["rating"] == "Overweight" and out["trader"]["entry_price"] == 570.5
    rep = tmp_path / "results" / "AMD" / "2026-10-23" / "reports"
    for f in ("1_analysts/market.md", "1_analysts/sentiment.md", "1_analysts/news.md", "1_analysts/fundamentals.md",
              "2_research/bull.md", "2_research/bear.md", "2_research/manager.md", "3_trading/trader.md",
              "4_risk/aggressive.md", "4_risk/conservative.md", "4_risk/neutral.md", "5_portfolio/decision.md",
              "complete_report.md", "complete_report.html"):
        assert (rep / f).is_file(), f
    assert (rep / "5_portfolio" / "decision.md").read_text(encoding="utf-8").startswith("**Rating**: Overweight")
    assert "FINAL TRANSACTION PROPOSAL: **BUY**" in (rep / "3_trading" / "trader.md").read_text(encoding="utf-8")
    prov = json.loads((rep / "provenance.json").read_text(encoding="utf-8"))
    assert prov["answer_form"] == "per-step (API form)" and len(prov["steps_answered"]) == 12


def test_non_agent_ai_work_uses_its_own_gemini_key_and_agents_do_not(tmp_path, monkeypatch):
    from tradingagents.llm_clients.factory import build_llm_kwargs

    srv, app = _agent_app(tmp_path, monkeypatch)
    monkeypatch.delenv("GEMINI_CONTEXT_API_KEY", raising=False)
    assert app.context_config() is app.config                                  # no key: the usual config
    monkeypatch.setenv("GEMINI_CONTEXT_API_KEY", "synthetic-context-key")
    cfg = app.context_config()
    assert cfg["llm_provider"] == "google" and build_llm_kwargs(cfg)["api_key"] == "synthetic-context-key"
    for attempt in app.agent_attempts(app.store.settings("global")):         # the agents never carry it
        assert "context_google_api_key" not in attempt and "api_key" not in build_llm_kwargs(attempt)
    from tradingagents.webapp import keys
    assert "GEMINI_CONTEXT_API_KEY" in keys.known()                           # settable from Settings


def test_ask_the_ai_to_pick_runs_once_a_calendar_day(tmp_path, monkeypatch):
    srv, app = _agent_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "_run_agents", lambda job, t, day, analysts, g: {"rating": "Hold", "how": "agents"})
    decided = {"AAA": True, "BBB": False}                                       # BBB failed: it may be asked again
    monkeypatch.setattr(app, "_agents_ran_today", lambda t: decided.get(t))
    body = {"tickers": ["AAA", "BBB"], "date": app._last_session(), "ai_pick": True}
    assert app.start_ai_decisions(body)
    for _ in range(100):
        if not app.analysis.running:
            break
        time.sleep(0.05)
    assert app.store.get("global", "ai_pick_state")["tickers"] == ["AAA", "BBB"]
    assert app.start_ai_decisions(body) is not None                              # a stock failed: allowed again
    for _ in range(100):
        if not app.analysis.running:
            break
        time.sleep(0.05)
    decided["BBB"] = True
    with pytest.raises(ValueError, match="already picked today"):
        app.start_ai_decisions(body)
    assert app.start_ai_decisions({**body, "ai_pick": False}) is not None     # a plain agent run is not limited


def test_the_agents_analyse_at_most_the_daily_limit_of_stocks(tmp_path, monkeypatch):
    from tradingagents.ledger.ai_gateway import AIBlocked

    srv, app = _agent_app(tmp_path, monkeypatch)
    g = {**app.store.settings("global"), "ai_daily_scans": 2}
    monkeypatch.setattr(app, "_scanned_today", lambda: {"AAA", "BBB"})
    monkeypatch.setattr(app, "_agents_ran_today", lambda t: None)
    monkeypatch.setattr(app, "_same_agent_run", lambda record: None)
    monkeypatch.setattr(app, "_try_agents", lambda *a, **k: {"rating": "Buy", "how": "agents"})
    with pytest.raises(AIBlocked, match="daily limit of 2"):
        app._run_agents(srv.Job(), "CCC", "2026-10-28", ["market"], g)


def test_the_agents_run_at_most_once_a_day_per_stock(tmp_path, monkeypatch):
    from datetime import UTC, datetime as dt

    srv, app = _agent_app(tmp_path, monkeypatch)
    g = app.store.settings("global")
    now = dt.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _agent_row(app, "AAA", "2026-10-20", "ok", now)                     # earlier today, another data date
    out = app._run_agents(srv.Job(), "AAA", "2026-10-23", ["market"], g)
    assert out["reused"] and out["once_a_day"] and out["rating"] == "Buy"
    _agent_row(app, "BBB", "2026-10-23", "failed", now)                 # a failure today does not count
    assert app._agents_ran_today("BBB") is None


def test_a_limit_stops_sending_to_that_model_until_it_resets_and_a_second_key_takes_over(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    srv, app = _agent_app(tmp_path, monkeypatch)
    now = datetime(2026, 10, 29, 3, 0, tzinfo=UTC)
    key, until, why = srv._cooldown("google", "g", "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel", now)
    assert key == "google/g" and "daily quota" in why and until == datetime(2026, 10, 29, 7, 0, tzinfo=UTC)  # midnight PDT
    key, until, _ = srv._cooldown("openrouter", "m", "429 Rate limit exceeded: requests per day", now)
    assert until == datetime(2026, 10, 30, tzinfo=UTC)
    assert srv._cooldown("google", "g", "429 rate limit, retry in 37s", now)[1] == now + srv.RATE_LIMIT_HOLD
    assert srv._cooldown("google", "g", "429 rate limit; retryDelay: 7200s", now)[1] == now + timedelta(hours=2)

    sent = []
    import tradingagents.optimizer.ai_views as views

    def analyze(ticker, day, cfg, *a, **k):
        sent.append((cfg.get("alt_api_key"), cfg["llm_max_retries"]))
        if not cfg.get("alt_api_key"):
            raise RuntimeError("Error code: 429 - Rate limit exceeded")
        return {"rating": "Buy"}, "agents"

    monkeypatch.setattr(views, "analyze", analyze)
    monkeypatch.setattr(app, "_write_decision", lambda *a, **k: {})
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "sk-or-second")
    cfg = {"llm_provider": "openrouter", "deep_think_llm": "d", "quick_think_llm": "q"}
    out = app._try_agents(srv.Job(), "AAA", "2026-10-28", ["market"], tmp_path / "f", 0.0, {}, [cfg], [])
    assert sent == [(None, 1), ("sk-or-second", 1)] and out["rating"] == "Buy"
    cooling = app.store.get("global", "ai_cooldowns")
    assert {"openrouter/d", "openrouter/q"} <= set(cooling)                # both of the run's models are held
    sent.clear()
    app._try_agents(srv.Job(), "BBB", "2026-10-28", ["market"], tmp_path / "f", 0.0, {}, [cfg], [])
    assert sent == [("sk-or-second", 1)]                                   # the limited key is not sent to again


def test_a_free_daily_limit_pauses_only_the_free_models_of_that_key(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    srv, app = _agent_app(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)                # one key: its free models are paused
    until = (datetime.now(UTC) + timedelta(hours=3)).isoformat()
    app.store.set("global", "ai_cooldowns", {"openrouter/*:free": {"until": until, "why": "free daily limit reached"}})
    sent = []
    import tradingagents.optimizer.ai_views as views
    monkeypatch.setattr(views, "analyze", lambda t, d, cfg, *a, **k: (sent.append(cfg["deep_think_llm"]), ({"rating": "Buy"}, "agents"))[1])
    monkeypatch.setattr(app, "_write_decision", lambda *a, **k: {})
    attempts = [{"llm_provider": "openrouter", "deep_think_llm": "x:free", "quick_think_llm": "x:free"},
                {"llm_provider": "openrouter", "deep_think_llm": "paid", "quick_think_llm": "paid"}]
    out = app._try_agents(srv.Job(), "AAA", "2026-10-28", ["market"], tmp_path / "f", 0.0, {}, attempts, [])
    assert sent == ["paid"] and out["model"] == "paid"                     # the free model skipped, the paid one ran


def test_a_watched_entry_plan_is_met_by_its_price_condition(tmp_path, monkeypatch):
    srv, app = _agent_app(tmp_path, monkeypatch)
    app.store.set("global", "entry_alerts", ["AAA", "BBB", "CCC"])
    plans = {"AAA": {"entry_type": "limit", "entry_range_low": 95, "entry_range_high": 100},
             "BBB": {"entry_type": "limit", "entry_price": 50},
             "CCC": {"entry_type": "market", "entry_price": 10}}
    import tradingagents.agents.entry_plan as ep
    monkeypatch.setattr(ep, "plans_for", lambda con, t: [{"stage": "portfolio_manager", "plan": plans[t], "analysis_date": "2026-10-28"}])
    monkeypatch.setattr(srv, "latest_prices", lambda cache, tickers: {"AAA": 98.0, "BBB": 51.0, "CCC": 11.0})
    got = {x["ticker"]: x["met"] for x in app.entry_alerts()["alerts"]}
    assert got == {"AAA": True, "BBB": False, "CCC": True}       # inside the range; above the limit; market entry
