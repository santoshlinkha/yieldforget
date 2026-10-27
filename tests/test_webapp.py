"""The web app's API on synthetic prices: previews are not stored, saves are, page loads never pay for AI.

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
    for path in ("/api/state", "/api/ledger", "/api/ai", "/api/recommendations", "/api/tracker", "/api/ratings",
                 "/api/backtest", "/api/market/explain"):
        status, _ = _call(base, path)
        assert status == 200, path
    status, _ = _call(base, "/api/plan", {"sleeve": "investing", "scan": False})
    assert status == 200
    assert _ai_rows(tmp_path) == 0


def test_a_preview_is_not_stored_and_a_save_is_official(server):
    base, tmp_path, _ = server
    status, preview = _call(base, "/api/plan", {"scan": False})
    assert status == 200 and preview["preview"] and "saved" not in preview
    assert _call(base, "/api/recommendations")[1] == []
    assert not (tmp_path / "results" / "optimizer").exists()
    status, saved = _call(base, "/api/recommendations/save", {"scan": False})
    assert status == 200 and saved["saved"]["id"]
    recs = _call(base, "/api/recommendations")[1]
    assert len(recs) == 1 and recs[0]["status"] == "official"
    status, rec = _call(base, "/api/recommendation?id=" + saved["saved"]["id"])
    assert status == 200 and rec["artifact"]["price_tickers"]


def test_settings_are_per_sleeve(server):
    base, _, _ = server
    _call(base, "/api/risk", {"sleeve": "trading", "level": 8})
    assert _call(base, "/api/state?sleeve=trading")[1]["risk"]["level"] == 8
    assert _call(base, "/api/state?sleeve=investing")[1]["risk"]["level"] == 2


def test_ledger_writes_go_through_the_books_and_keep_the_old_file_in_step(server):
    base, tmp_path, _ = server
    status, out = _call(base, "/api/ledger/transfer",
                        {"from": "investing", "to": "trading", "ticker": "SGOV", "quantity": 5, "price": 100})
    assert status == 200, out
    books = _call(base, "/api/ledger")[1]["sleeves"]
    assert books["trading"]["positions"]["SGOV"]["quantity"] == 5
    assert books["combined"]["positions"]["SGOV"]["quantity"] == 20
    status, out = _call(base, "/api/ledger/transfer",
                        {"from": "investing", "to": "trading", "ticker": "SGOV", "quantity": 50, "price": 100})
    assert status == 400 and "holds" in out["error"]
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
    assert out["mine"]["quantity"]["investing"] == 10
    assert _call(base, "/api/ticker?symbol=../etc")[0] == 400
    assert _call(base, "/api/report?path=C:/Windows/win.ini")[0] == 400


def test_the_same_plan_is_not_computed_twice(server):
    base, _, handler = server
    t0 = time.time()
    _call(base, "/api/plan", {"scan": False})
    first = time.time() - t0
    t0 = time.time()
    _call(base, "/api/plan", {"scan": False})
    assert time.time() - t0 < max(first / 2, 0.05)


def test_an_existing_paper_record_is_never_started_over_by_accident(server):
    base, tmp_path, _ = server
    (tmp_path / "paper_tracker.json").write_text(json.dumps({"start": "2026-10-22", "history": [], "books": {},
                                                             "last_day": None, "loss_limit": 0.05}))
    status, out = _call(base, "/api/tracker/start", {})
    assert status == 400 and "already exists" in out["error"]
    assert json.loads((tmp_path / "paper_tracker.json").read_text())["start"] == "2026-10-22"


def test_a_pasted_answer_is_kept_with_its_prompt_and_a_second_one_archives_the_first(server, monkeypatch):
    from tests.test_manual import offline

    base, tmp_path, _ = server
    offline(monkeypatch)
    status, prompt = _call(base, "/api/manual-prompt?ticker=AAA&date=2026-10-23")
    assert status == 200 and "## Step 1: Market Analyst report" in prompt["prompt"]
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
