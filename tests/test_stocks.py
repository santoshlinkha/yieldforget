"""The stock module: no look-ahead, shared folds, failure isolation, caching, persistence, cost controls,
and recommendations that keep to the account's rules. All offline, on synthetic data."""

import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.features import MACRO_FEATURES
from tradingagents.stocks import data as D
from tradingagents.stocks import fundamentals as FU
from tradingagents.stocks import macro as M
from tradingagents.stocks import panel as P

pytestmark = pytest.mark.unit

SUPPORT = ("SPY", "BIL", "XLK", "^VIX", "^VIX3M", "^IRX", "^TNX")


def _stock_data(tmp_path, n=16, years=7, seed=1, requested=("S00", "S01", "NEWCO", "GONE")):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", periods=252 * years)
    names = [f"S{i:02d}" for i in range(n)]
    cols = {}
    for i, t in enumerate(names):
        cols[t] = 50 * np.exp(np.cumsum(rng.normal(0.0003 + 0.0001 * i, 0.015 + 0.001 * i, len(days))))
    cols["NEWCO"] = np.r_[np.full(len(days) - 120, np.nan), 20 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))]
    cols["SPY"] = 300 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(days))))
    cols["XLK"] = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(days))))
    cols["BIL"] = 90 * np.exp(np.arange(len(days)) * 0.00008)
    cols["^VIX"] = 18 + rng.normal(0, 2, len(days)).cumsum() * 0.05
    cols["^VIX3M"] = cols["^VIX"] * 1.05
    cols["^IRX"] = np.full(len(days), 4.0)
    cols["^TNX"] = 4 + rng.normal(0, 0.02, len(days)).cumsum()
    close = pd.DataFrame(cols, index=days)
    high, low = close * (1 + rng.uniform(0, 0.01, close.shape)), close * (1 - rng.uniform(0, 0.01, close.shape))
    volume = pd.DataFrame(1e6, index=days, columns=close.columns)
    info = {t: {"name": t, "sector": "Information Technology", "sector_etf": "XLK", "cik": None,
                "added": "2010-01-01", "sp500": True} for t in names}
    info["NEWCO"] = {"name": "Newco", "sector": "Information Technology", "sector_etf": "XLK", "cik": None,
                     "added": None, "sp500": False}
    info["GONE"] = {"name": "Gone", "sector": None, "sector_etf": None, "cik": None, "added": None, "sp500": False}
    return D.StockData(tmp_path, info, list(requested), {"VTI": "a fund (ETF): this module forecasts stocks only"},
                       close, high, low, volume, facts={}, actions={},
                       errors={"GONE": ["no prices from Yahoo"]})


# --- information known at the time --------------------------------------------------------------------

def _fact(start, end, filed, value, form="10-K"):
    return [start, end, filed, form, value]


def test_a_filing_counts_from_the_day_after_it_was_filed_and_restatements_wait_their_turn():
    facts = {"revenue": [_fact("2019-01-01", "2019-12-31", "2020-02-10", 100.0),
                         _fact("2020-01-01", "2020-12-31", "2021-02-15", 120.0),
                         _fact("2020-01-01", "2020-12-31", "2022-02-10", 90.0)]}   # FY2020 restated a year later
    h = FU.history(facts, pd.DatetimeIndex(["2021-02-15", "2021-02-16", "2021-12-01", "2022-02-11"]))
    assert h.loc["2021-02-15", "revenue_y"] == 100.0           # filed that day: not yet usable
    assert h.loc["2021-02-16", "revenue_y"] == 120.0
    assert h.loc["2021-12-01", "rev_growth_y"] == pytest.approx(0.2)   # the original, not today's restated value
    assert h.loc["2022-02-11", "revenue_y"] == 90.0 and h.loc["2022-02-11", "rev_growth_y"] == pytest.approx(-0.1)


def test_a_period_end_is_never_its_publication_date():
    facts = {"revenue": [_fact("2019-01-01", "2019-12-31", "2020-03-01", 100.0)]}
    h = FU.history(facts, pd.DatetimeIndex(["2020-01-15"]))   # after the period ended, before it was filed
    assert np.isnan(h.loc["2020-01-15", "revenue_y"])


def test_a_split_does_not_read_as_dilution():
    facts = {"shares": [[None, "2019-10-01", "2019-10-30", "10-K", 100.0], [None, "2020-10-01", "2020-10-30", "10-K", 404.0]]}
    snap = FU.snapshot(facts, "2020-12-31", splits=[["2020-08-31", 4.0]])
    assert snap["dilution_1y"] == pytest.approx(0.01)


def test_economic_data_is_used_as_first_published_from_the_next_day():
    rows = [{"date": "2020-04-01", "value": 4.4, "from": "2020-05-08"},
            {"date": "2020-04-01", "value": 14.7, "from": "2020-06-05"},
            {"date": "2020-03-01", "value": 4.4, "from": "2020-04-03"}]
    known = M.as_known(rows, pd.DatetimeIndex(["2020-05-08", "2020-05-09", "2020-06-06"]))
    assert known[pd.Timestamp("2020-05-08")].index.max() == pd.Timestamp("2020-03-01")   # released that day: not yet
    assert known[pd.Timestamp("2020-05-09")].loc["2020-04-01"] == 4.4
    assert known[pd.Timestamp("2020-06-06")].loc["2020-04-01"] == 14.7                   # the revision, once published


def test_panel_features_at_a_date_ignore_everything_after_it(tmp_path):
    data = _stock_data(tmp_path)
    dates = pd.DatetimeIndex([data.close.index[800]])
    a = P.build(data, 20, dates=dates, groups=("technical", "market"))
    later = data.close.index > data.close.index[805]      # after the cutoff and after the next-close entry
    stocks = [c for c in data.close.columns if c.startswith("S") and c != "SPY"]
    for frame in (data.close, data.high, data.low):
        frame.loc[later, stocks] *= 3
    b = P.build(data, 20, dates=dates, groups=("technical", "market"))
    feats = P.features_for(("technical", "market"))
    pd.testing.assert_frame_equal(a[feats], b[feats])
    assert not np.allclose(a["label"].to_numpy(), b["label"].to_numpy())   # only the outcome moved


def test_every_stock_shares_the_same_decision_dates(tmp_path):
    data = _stock_data(tmp_path)
    panel = P.build(data, 20, groups=("technical",))
    by = panel.reset_index().groupby("ticker")["date"].apply(set)
    full = [t for t in by.index if t.startswith("S")]
    assert all(by[t] == by[full[0]] for t in full)


# --- inputs ----------------------------------------------------------------------------------------

def test_tickers_are_read_from_pasted_text_and_csv():
    assert D.parse_tickers("aapl, $msft brk.b\nAAPL;  nvda") == ["AAPL", "MSFT", "BRK-B", "NVDA"]
    assert D.parse_tickers(csv_text="Name,Symbol\nApple,AAPL\nJPMorgan,JPM\n") == ["AAPL", "JPM"]


def test_funds_crypto_and_unknown_symbols_are_turned_away_and_one_failure_stays_local(monkeypatch, tmp_path):
    from tradingagents.dataflows.vendors.yahoo import history as yh

    profiles = {"VTI": {"quoteType": "ETF"}, "BTC-USD": {"quoteType": "CRYPTOCURRENCY"}, "ZZZZ": {},
                "SHOP": {"quoteType": "EQUITY", "sector": "Technology", "longName": "Shopify"}}

    def profile(t, cache_dir=None):
        if t == "BOOM":
            raise RuntimeError("Yahoo timed out")
        return profiles[t]

    monkeypatch.setattr(yh, "profile", profile)
    table = pd.DataFrame({"Security": ["Apple"], "GICS Sector": ["Information Technology"], "CIK": [320193],
                          "added": [pd.Timestamp("1982-11-30")]}, index=["AAPL"])
    stocks, rejected = D.classify(["AAPL", "VTI", "BTC-USD", "ZZZZ", "SHOP", "BOOM"], tmp_path, table)
    assert set(stocks) == {"AAPL", "SHOP"} and stocks["SHOP"]["sector_etf"] == "XLK"
    assert rejected["VTI"].startswith("a fund") and rejected["BTC-USD"].startswith("crypto")
    assert rejected["ZZZZ"] == "not found on Yahoo Finance" and rejected["BOOM"].startswith("lookup failed")


# --- a batch ---------------------------------------------------------------------------------------

@pytest.fixture
def batch_env(tmp_path, monkeypatch):
    from tradingagents.ledger.db import connect
    from tradingagents.stocks import pipeline

    data = _stock_data(tmp_path)
    monkeypatch.setattr(D, "load", lambda *a, **k: data)
    monkeypatch.setattr(M, "features", lambda dates, cache_dir: pd.DataFrame(index=dates, columns=MACRO_FEATURES,
                                                                              dtype=float))

    def no_paid(*a, **k):
        raise AssertionError("a batch must never call a paid model")

    monkeypatch.setattr("tradingagents.llm_clients.create_llm_client", no_paid)
    con = connect(tmp_path / "p.sqlite3")
    config = pipeline.BatchConfig(horizons=(20,), news=False, device="cpu")
    return con, tmp_path, config, data


def test_a_batch_isolates_failures_saves_everything_and_never_pays(batch_env):
    from tradingagents.stocks.pipeline import load_run, run_batch

    con, cache, config, data = batch_env
    out = run_batch(con, cache, data.requested, {t: "pasted" for t in data.requested}, config)
    assert out["rejected"] == {"VTI": "a fund (ETF): this module forecasts stocks only"}
    run = load_run(con, out["run_id"])
    by = {f["ticker"]: f for f in run["forecasts"]}
    assert by["S00"]["status"] == "ok" and by["S01"]["status"] == "ok"
    assert by["NEWCO"]["status"] == "abstain" and by["GONE"]["status"] == "abstain"   # too short; no prices
    fc = by["S00"]["forecast"]
    for k in ("mean", "median", "q05", "q95", "p_gain", "net", "cost_rt", "relative", "trend_up", "trend_down",
              "trend_sideways", "trend_unclear", "p_up_break", "p_down_break", "p_large_loss", "p_deep_drawdown",
              "dd_q05", "upper", "lower"):
        assert fc[k] is not None, k
    assert abs(sum(fc[f"trend_{s}"] for s in ("up", "down", "sideways", "unclear")) - 1) < 1e-6
    assert fc["net"] == pytest.approx(fc["mean"] - fc["cost_rt"])
    assert run["earliest_execution"] > run["info_cutoff"]
    assert by["S00"]["evidence_status"]["return"] == "untested"
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE stock_forecasts SET status = 'ok' WHERE ticker = 'GONE'")


def test_models_fitted_for_a_cutoff_are_reused_not_refitted(batch_env):
    from tradingagents.stocks.pipeline import load_run, run_batch

    con, cache, config, data = batch_env
    first = load_run(con, run_batch(con, cache, ["S00"], None, config)["run_id"])
    second = load_run(con, run_batch(con, cache, ["S01"], None, config)["run_id"])
    assert first["models"]["20"]["artifact"]["reused"] is False
    assert second["models"]["20"]["artifact"]["reused"] is True
    assert second["models"]["20"]["artifact"]["sha256"] == first["models"]["20"]["artifact"]["sha256"]


def test_company_facts_are_fetched_once_then_read_from_the_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(FU, "_get", lambda url: calls.append(url) or {"facts": {"us-gaap": {}}})
    FU.facts_for("AAA", tmp_path, cik="0000000001")
    FU.facts_for("AAA", tmp_path, cik="0000000001")
    assert len(calls) == 1


def test_the_paid_news_reading_is_refused_without_a_price_and_drops_invented_citations(tmp_path):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ledger.ai_gateway import AIBlocked, Budget, Gateway, Policy
    from tradingagents.ledger.db import connect
    from tradingagents.stocks.llm import checked, read_news

    gateway = Gateway(connect(tmp_path / "p.sqlite3"), Policy(Budget(1.0, 10.0, 100, 1000)))
    with pytest.raises(AIBlocked):
        read_news("AAA", "Aaa", "2026-10-23", [{"title": "Aaa raises guidance"}], "", DEFAULT_CONFIG, gateway,
                  allow_paid=True)
    got = checked({"events": [{"sources": [1, 7]}, {"sources": [9]}], "summary": ""}, n_items=2)
    assert got["events"] == [{"sources": [1]}] and got["dropped"] == 1


# --- recommendations -------------------------------------------------------------------------------

def test_recommendations_never_short_stay_within_settled_cash_and_limits(tmp_path, monkeypatch):
    from tradingagents.ledger.db import now_utc
    from tradingagents.ledger.sessions import next_session
    from tradingagents.ledger.store import Store
    from tradingagents.stocks import ideas

    store = Store(tmp_path)
    day = date(2026, 10, 1)
    store.ledger.contribute(1000.0, day, sleeve="trading", settle_date=day)
    store.ledger.fill("trading", "buy", "HELD", 2.0, 50.0, day, settle_date=day)
    tickers = ["AAA", "BBB", "CCC", "HELD", "WEAK"]
    close = pd.DataFrame({t: np.linspace(40, 47.3, 260) for t in [*tickers, "SPY", "BIL"]},
                         index=pd.bdate_range(end="2026-10-23", periods=260))
    monkeypatch.setattr(ideas, "_prices", lambda cache_dir, names: close)
    con = store.con
    con.execute("INSERT INTO stock_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("r1", now_utc(), "2026-10-23", "2026-10-26", "{}", "{}", 5, "[20]", "x", "x", "thr", "{}", "{}",
                 "{}", "{}", "{}", 1.0))

    def forecast(t, net, evidence="backed", p_loss=0.05):
        payload = {"forecast": {"net": net, "mean": net + 0.003, "cost_rt": 0.003, "p_large_loss": p_loss},
                   "evidence_status": {"return": evidence}, "name": t}
        con.execute("INSERT INTO stock_forecasts VALUES (?, ?, ?, ?, ?, ?)", ("r1", t, 20, "ok", "[]", json.dumps(payload)))

    forecast("AAA", 0.03)
    forecast("BBB", 0.02)
    forecast("CCC", 0.01, p_loss=0.4)          # too likely to lose big
    forecast("HELD", -0.02)                     # negative, and held
    forecast("WEAK", 0.05, evidence="weak")     # positive, but not confirmed by the final test
    out = ideas.make(con, store, tmp_path, "r1", 20, as_of=date(2026, 10, 24))
    lines = {line["ticker"]: line for line in out["lines"]}
    assert lines["HELD"]["action"].startswith("consider reducing") and not lines["HELD"].get("order")
    assert lines["CCC"]["action"].startswith("watch") and lines["WEAK"]["action"].startswith("watch")
    orders = {t: line["order"] for t, line in lines.items() if line.get("order")}
    assert set(orders) == {"AAA", "BBB"} and all(o["side"] == "buy" for o in orders.values())
    assert sum(o["dollars"] for o in orders.values()) <= out["settled_cash"] + 1e-6
    assert all(o["dollars"] <= 0.25 * out["sleeve_value"] + 1e-6 for o in orders.values())
    assert any(o["shares"] != int(o["shares"]) for o in orders.values())   # fractional shares
    assert out["hold_until"] == str(next_session(date(2026, 10, 26), 20))
    ideas.decide(con, out["id"], "AAA", "accepted")
    with pytest.raises(ValueError):
        ideas.decide(con, out["id"], "AAA", "maybe")


def test_the_brier_skill_interval_runs_low_to_high_and_brackets_a_real_improvement():
    from tradingagents.stocks.evaluate import event_scores

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=60)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(40)]], names=["date", "ticker"])
    p = rng.uniform(0.05, 0.95, len(idx))
    y = (rng.uniform(size=len(idx)) < p).astype(float)       # the forecast is the true probability
    frame = pd.DataFrame({"y": y, "p": p, "base": y.mean()}, index=idx)
    s = event_scores(frame, "y", "p", "base")
    lo, hi = s["skill_ci"]
    assert lo < hi and lo > 0 and s["skill"] > 0
