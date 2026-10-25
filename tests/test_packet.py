"""The evidence packet for LLM analyses: cutoff first, canonical values, routing, missing stays missing."""

import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.agents import packet as P

pytestmark = pytest.mark.unit
CUTOFF = "2026-10-23"


def _bars(n=320, end="2026-10-30"):
    idx = pd.bdate_range(end=end, periods=n)
    close = pd.Series(np.linspace(100, 160, n), index=idx)
    close.iloc[-5:] = 999.0                       # after the cutoff: must never reach the packet
    return pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99, "Volume": 1e6})


def _macro():
    cpi, unrate = [], []
    for i, m in enumerate(pd.date_range("2025-08-01", "2026-09-01", freq="MS")):
        release = (m + pd.DateOffset(months=1, days=12)).date().isoformat()
        cpi.append({"date": m.date().isoformat(), "value": 320.0 + i, "from": release})
        unrate.append({"date": m.date().isoformat(), "value": 4.0 + 0.1 * (i % 3), "from": release})
    # A revision of September published after the cutoff: never used.
    cpi.append({"date": "2026-09-01", "value": 999.0, "from": "2026-11-01"})
    dff = [{"date": d.date().isoformat(), "value": 3.5} for d in pd.bdate_range("2025-09-01", "2026-10-30")]
    return {"CPIAUCSL": cpi, "UNRATE": unrate, "DFF": dff}


def _news():
    base = {"publisher": "Wire", "link": "https://example.com/a", "summary": "", "first_seen": "2026-10-24T01:00:00+00:00"}
    return [
        {**base, "title": "Acme beats estimates", "pub_date": "2026-10-22T14:00:00+00:00"},
        {**base, "title": "Acme beats estimates!", "pub_date": "2026-10-22T15:00:00+00:00"},   # duplicate story
        {**base, "title": "Acme cuts guidance", "pub_date": "2026-10-28T14:00:00+00:00"},      # after the cutoff
        {**base, "title": "Acme undated rumour", "pub_date": None},                            # unknown time
    ]


def _facts():
    return {"revenue": [["2025-01-01", "2025-12-31", "2026-02-20", "10-K", 1000.0],
                        ["2026-04-01", "2026-06-30", "2026-07-30", "10-Q", 300.0],
                        ["2026-07-01", "2026-09-30", "2026-10-23", "10-Q", 350.0]],   # filed on the cutoff: next day
            "shares": [[None, "2026-06-30", "2026-07-30", "10-Q", 50.0]]}


def _inputs(kind="EQUITY", **kw):
    profile = {"quoteType": kind, "longName": "Acme Corp"} if kind else None
    if kind == "ETF":
        profile = {"quoteType": "ETF", "longName": "ProShares UltraPro QQQ", "category": "Trading--Leveraged Equity",
                   "netExpenseRatio": 0.84, "totalAssets": 2.2e10, "longBusinessSummary": "Seeks 3x the daily return."}
    return {"ticker": "ACME", "cutoff": CUTOFF, "bars": _bars(), "ohlcv": None, "profile": profile,
            "profile_fetched": "2026-10-25T10:00:00", "facts": _facts(), "macro": _macro(), "news": _news(), **kw}


def test_everything_is_cut_at_the_cutoff_before_it_is_computed():
    p = P.assemble(_inputs())
    assert p["cutoff"]["latest_session"] == CUTOFF
    assert p["snapshot"]["close"] < 200 and all(b["close"] < 200 for b in p["recent_bars"])   # no post-cutoff 999s
    assert p["snapshot"]["sma_200"] < 200
    heads = [n["headline"] for n in p["news"]]
    assert heads == ["Acme beats estimates"]                                  # duplicate and future dropped
    reasons = " ".join(m["reason"] for m in p["missing"])
    assert "no publication time" in reasons and "published after the cutoff" in reasons


def test_dates_are_named_for_what_they_are():
    p = P.assemble(_inputs())
    n = p["news"][0]
    assert n["published_at"].startswith("2026-10-22") and n["first_seen_at"].startswith("2026-10-24")
    assert n["text_available"] == "headline only"
    rev = [i for i in p["fundamentals"]["items"] if i["concept"] == "revenue"]
    q = next(i for i in rev if i["period"] == "quarter")
    assert q["period_end"] == "2026-06-30" and q["filed"] == "2026-07-30" and q["available_from"] == "2026-07-31"


def test_stocks_and_funds_get_their_own_sections():
    stock, fund = P.assemble(_inputs("EQUITY")), P.assemble(_inputs("ETF"))
    assert "fundamentals" in stock and "fund" not in stock
    assert "fund" in fund and "fundamentals" not in fund
    f = fund["fund"]
    assert f["leveraged_or_inverse"] is True and f["net_expense_ratio_pct"] == 0.84
    assert f["holdings"] is None and f["premium_discount_to_nav"] is None
    assert {m["field"] for m in fund["missing"]} >= {"fund.holdings", "fund.premium_discount_to_nav"}


def test_one_canonical_value_and_a_flag_when_sources_disagree():
    bars = _bars()
    cut = bars[bars.index <= CUTOFF]
    other = pd.DataFrame({"open": cut["Close"], "close": cut["Close"] * 1.02}, index=cut.index)   # 2% apart
    p = P.assemble(_inputs(ohlcv=other))
    assert p["snapshot"]["close"] == round(float(cut["Close"].iloc[-1]), 2)    # the local store wins
    flag = next(c for c in p["checks"] if c["check"].startswith("close cross-check"))
    assert "DISCREPANCY" in flag["result"]


def test_missing_values_stay_unknown():
    short = _bars(n=60)
    p = P.assemble(_inputs(kind=None, bars=short, facts=None, news=[], macro={}))
    assert p["instrument"]["type"] == "unknown" and p["snapshot"]["sma_200"] is None      # null, never 0
    fields = {m["field"] for m in p["missing"]}
    assert {"instrument.type", "snapshot.sma_200", "news", "macro.CPIAUCSL"} <= fields
    assert p["portfolio"]["holding"] == "unknown"


def test_essential_evidence_survives_compression():
    p = P.assemble(_inputs(), {"max_chars": 1500})
    for part in ("instrument", "cutoff", "snapshot", "checks", "missing"):
        assert p[part] is not None
    assert len(p["recent_bars"]) >= 3 and p["trimmed"]
    json.loads(P.render(p))                                                   # still whole JSON
    assert p["snapshot"]["close"] is not None and p["snapshot"]["sma_50"] is not None


def test_macro_comparisons_use_the_right_periods_and_units():
    p = P.assemble(_inputs())
    cpi = next(m for m in p["macro"] if m["series"] == "CPIAUCSL")
    assert cpi["observation_date"] == "2026-09-01" and cpi["value"] == 333.0  # the post-cutoff revision unused
    yoy = next(c for c in cpi["comparisons"] if c["label"] == "12 months earlier")
    assert yoy["observation_date"] == "2025-09-01" and yoy["change_unit"] == "percent change in the level"
    assert cpi["inflation_12m_pct"] == pytest.approx(100 * (333 / 321 - 1), abs=0.01)
    un = next(m for m in p["macro"] if m["series"] == "UNRATE")
    assert all(c["change_unit"] == "percentage points" for c in un["comparisons"])
    dff = next(m for m in p["macro"] if m["series"] == "DFF")
    assert dff["observation_date"] <= "2026-10-22"                             # one business day's lag


def test_the_packet_is_deterministic():
    assert P.render(P.assemble(_inputs())) == P.render(P.assemble(_inputs()))


def test_the_prompt_is_versioned_grounded_and_free_of_app_conclusions(monkeypatch):
    monkeypatch.setattr(P, "load_inputs", lambda *a, **k: _inputs())
    text, packet = P.build_prompt("ACME", CUTOFF, cache_dir=None)
    assert text.startswith(f"[{P.PACKET_VERSION}]") and text.index("Rules:") < text.index("EVIDENCE PACKET")
    assert "commit to" not in text.lower() and "evidence IDs" in text
    for leak in ("verdict", "model rank", "model_forecasts"):
        assert leak not in P.render(packet).lower()


def test_answers_in_both_formats_still_parse():
    from tradingagents.agents.manual import decision_markdown, parse_answer

    new = parse_answer('```json\n{"rating": "Hold", "executive_summary": "x", "investment_thesis": "y [PX1]", '
                       '"price_target": null, "time_horizon": "3-6 months", "evidence_ids": ["PX1", "N1"], '
                       '"insufficient_evidence": true}\n```')
    assert new["rating"] == "Hold" and new["insufficient_evidence"] is True
    md = decision_markdown(new)
    assert "abstained" in md and "PX1, N1" in md
    old = parse_answer('```json\n{"rating": "Sell", "executive_summary": "a", "investment_thesis": "b"}\n```')
    assert old["rating"] == "Sell" and old["insufficient_evidence"] is False
