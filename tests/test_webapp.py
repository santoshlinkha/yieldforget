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


def test_a_pasted_answer_is_tied_to_its_packet_and_a_second_one_archives_the_first(server, monkeypatch):
    from tradingagents.agents import packet

    base, tmp_path, _ = server
    idx = pd.bdate_range(end="2026-10-23", periods=260)
    close = pd.Series(range(100, 360), index=idx, dtype=float)
    inputs = {"ticker": "AAA", "cutoff": "2026-10-23", "ohlcv": None, "facts": None, "macro": {}, "news": [],
              "bars": pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Volume": 1e6}),
              "profile": {"quoteType": "EQUITY", "longName": "Aaa Inc."}, "profile_fetched": None}
    monkeypatch.setattr(packet, "load_inputs", lambda *a, **k: inputs)
    status, prompt = _call(base, "/api/manual-prompt?ticker=AAA&date=2026-10-23")
    assert status == 200 and prompt["packet_id"] in prompt["prompt"].splitlines()[0]
    day = tmp_path / "results" / "AAA" / "2026-10-23"
    assert (day / "packets" / f"{prompt['packet_id']}.json").is_file()

    answer = ("Analysis [PX1] and [ZZ9].\n```json\n" + json.dumps(
        {"rating": "Hold", "executive_summary": "Flat.", "investment_thesis": "None.", "packet_id": prompt["packet_id"],
         "evidence_ids": ["PX1"], "insufficient_evidence": True}) + "\n```")
    status, first = _call(base, "/api/manual-answer", {"ticker": "AAA", "date": "2026-10-23", "text": answer})
    assert status == 200 and first["packet_id"] == prompt["packet_id"] and first["unknown_cited_ids"] == ["ZZ9"]
    prov = json.loads((day / "reports" / "provenance.json").read_text())
    assert prov["abstained"] and prov["packet_id"] == prompt["packet_id"]
    assert (day / "reports" / "evidence_packet.json").is_file()
    assert "ZZ9" in (day / "reports" / "chat_answer.md").read_text()

    status, _ = _call(base, "/api/manual-answer", {"ticker": "AAA", "date": "2026-10-23", "text": answer.replace("Hold", "Sell")})
    assert status == 200
    archived = list((day / "archive").iterdir())
    assert len(archived) == 1 and '"Hold"' in (archived[0] / "chat_answer.md").read_text()
    assert '"Sell"' in (day / "reports" / "chat_answer.md").read_text()
    status, page = _call(base, "/api/ticker?symbol=AAA")
    runs = [r for r in (page.get("reports") or {}).get("runs", []) if r["date"] == "2026-10-23"]
    assert sorted({r["run"] or "" for r in runs}) == ["", f"archive/{archived[0].name}"]   # roots may repeat a run
    rated = [r for r in page["reports"]["ratings"] if r["date"] == "2026-10-23"]
    assert rated and rated[0]["abstained"]


def test_an_entry_plan_from_a_pasted_answer_is_checked_kept_and_shown(server, monkeypatch):
    from tradingagents.agents import entry_plan, packet

    base, tmp_path, _ = server
    idx = pd.bdate_range(end="2026-10-23", periods=260)
    close = pd.Series(range(100, 360), index=idx, dtype=float)
    monkeypatch.setattr(packet, "load_inputs", lambda *a, **k: {
        "ticker": "AAA", "cutoff": "2026-10-23", "ohlcv": None, "facts": None, "macro": {}, "news": [],
        "bars": pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Volume": 1e6}),
        "profile": {"quoteType": "EQUITY"}, "profile_fetched": None})
    monkeypatch.setattr(entry_plan, "reference_quote", lambda t, d, c: {
        "ticker": t, "cutoff": d, "session": "2026-10-23", "price": 359.0, "adjusted": 359.0, "currency": "USD",
        "verified": True, "how": "the store's latest session", "basis": "as traded"})
    status, prompt = _call(base, "/api/manual-prompt?ticker=AAA&date=2026-10-23")
    assert status == 200 and '"REF1"' in prompt["prompt"] and '"close_as_traded":359.0' in prompt["prompt"]

    plan = {"status": "ready", "entry_type": "limit", "entry_price": 355.0, "entry_range_low": None,
            "entry_range_high": None, "currency": "USD", "trigger": None, "earliest_execution": "2026-10-26",
            "expires_at": "2026-10-30", "evidence_ids": ["REF1"], "rationale": "a pullback"}
    answer = "Analysis [PX1].\n```json\n" + json.dumps(
        {"rating": "Buy", "executive_summary": "s", "investment_thesis": "t", "packet_id": prompt["packet_id"],
         "evidence_ids": ["PX1"], "entry_plan": plan}) + "\n```"
    status, saved = _call(base, "/api/manual-answer", {"ticker": "AAA", "date": "2026-10-23", "text": answer})
    assert status == 200 and saved["entry_plans"]["plans"][0]["stage"] == "portfolio_manager"
    status, shown = _call(base, "/api/entry-plans?ticker=AAA")
    run = shown["runs"][0]
    assert status == 200 and run["current"] and run["trader"] is None
    final = run["final"]
    assert final["plan"]["entry_price"] == 355.0 and final["reference"]["source"].startswith("evidence packet")
    assert final["evidence"]["packet_id"] == prompt["packet_id"] and final["action"] == "Buy"
    # The fixture's Trading sleeve holds no settled cash: the plan is kept, blocked, with that reason.
    assert final["validation"]["state"] == "blocked" and not final["validation"]["feasible"]
    assert any(b.startswith("settled cash: $0.00") for b in final["validation"]["blocks"])
