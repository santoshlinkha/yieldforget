"""The local market-data store: download once, then only what is new; re-download after a re-adjustment.

Synthetic bars and articles; the "source" is a fake that counts what it was asked.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import pytz

from tradingagents.dataflows.local_store import MarketStore, last_final_session

pytestmark = pytest.mark.unit
NY = pytz.timezone("America/New_York")


class FakeYahoo:
    def __init__(self, days):
        self.days = days
        self.scale = 1.0
        self.calls = []

    def __call__(self, tickers, start, end):
        self.calls.append((tuple(tickers), start))
        idx = self.days[(self.days >= pd.Timestamp(start or "1990-01-01")) & (self.days < pd.Timestamp(end))]
        base = pd.Series(range(1, len(self.days) + 1), index=self.days, dtype=float).loc[idx] * self.scale
        return {f: pd.DataFrame({t: base * (2 if f == "High" else 1) for t in tickers}, index=idx)
                for f in ("Close", "High", "Low", "Volume")}


def _at(y, m, d, hour):
    return NY.localize(datetime(y, m, d, hour)).astimezone(UTC)


def test_only_final_sessions_count():
    assert last_final_session(_at(2026, 10, 21, 12)).isoformat() == "2026-10-20"   # Wednesday noon
    assert last_final_session(_at(2026, 10, 21, 17)).isoformat() == "2026-10-21"   # after the close
    assert last_final_session(_at(2026, 10, 24, 12)).isoformat() == "2026-10-23"   # Saturday


def test_history_is_downloaded_once_then_only_new_sessions(tmp_path):
    days = pd.bdate_range("2026-01-01", "2026-10-30")
    yahoo, store = FakeYahoo(days), MarketStore(tmp_path)
    first = store.bars(["AAA"], yahoo, now=_at(2026, 10, 20, 17))["Close"]["AAA"]
    assert first.index[-1] == pd.Timestamp("2026-10-20") and yahoo.calls == [(("AAA",), None)]
    store.bars(["AAA"], yahoo, now=_at(2026, 10, 20, 18))          # same final session: no download
    assert len(yahoo.calls) == 1
    later = store.bars(["AAA"], yahoo, now=_at(2026, 10, 22, 17))["Close"]["AAA"]
    assert yahoo.calls[-1][1] == "2026-10-10"                      # only the new sessions, with overlap
    assert later.index[-1] == pd.Timestamp("2026-10-22") and later.is_monotonic_increasing
    assert later.loc["2026-10-21"] == first.iloc[-1] + 1


def test_a_readjusted_history_is_downloaded_again_in_full(tmp_path):
    days = pd.bdate_range("2026-01-01", "2026-10-30")
    yahoo, store = FakeYahoo(days), MarketStore(tmp_path)
    store.bars(["AAA"], yahoo, now=_at(2026, 10, 20, 17))
    yahoo.scale = 0.99                                             # a dividend rewrote every past close
    closes = store.bars(["AAA"], yahoo, now=_at(2026, 10, 22, 17))["Close"]["AAA"]
    assert yahoo.calls[-1] == (("AAA",), None)
    assert closes.iloc[0] == pytest.approx(0.99)


def test_a_ticker_with_no_data_is_asked_again_at_most_daily(tmp_path):
    calls = []

    def empty(tickers, start, end):
        calls.append(tickers)
        return {f: pd.DataFrame(columns=tickers, dtype=float) for f in ("Close", "High", "Low", "Volume")}

    store = MarketStore(tmp_path)
    store.bars(["GONE"], empty, now=_at(2026, 10, 20, 17))
    store.bars(["GONE"], empty, now=_at(2026, 10, 20, 18))
    assert len(calls) == 1
    store.bars(["GONE"], empty, now=_at(2026, 10, 21, 17) + timedelta(days=1))
    assert len(calls) == 2


def test_news_is_reused_within_its_ttl_and_accumulates_after(tmp_path):
    store, asked = MarketStore(tmp_path), []

    def page(titles):
        def fetch():
            asked.append(titles)
            return [{"title": t, "publisher": "Wire", "link": f"http://x/{t}", "summary": "",
                     "pub_date": datetime(2026, 10, 20, tzinfo=UTC)} for t in titles]
        return fetch

    assert [a["title"] for a in store.articles("search:rates", page(["a", "b"]))] == ["a", "b"]
    assert len(store.articles("search:rates", page(["c"]))) == 2 and len(asked) == 1   # within the TTL
    later = store.articles("search:rates", page(["b", "c"]), ttl=timedelta(seconds=0))
    assert sorted(a["title"] for a in later) == ["a", "b", "c"] and len(asked) == 2    # nothing lost, no duplicates


def test_snapshots_are_kept_per_session_and_histories_reused(tmp_path):
    store, asked = MarketStore(tmp_path), []

    def fetch():
        asked.append(1)
        return {"mean_target": 100 + len(asked)}

    assert store.snapshot("AAA", "price_targets", fetch, session="2026-10-20") == {"mean_target": 101}
    assert store.snapshot("AAA", "price_targets", fetch, session="2026-10-20") == {"mean_target": 101}
    store.snapshot("AAA", "price_targets", fetch, session="2026-10-21")
    assert [s for s, _ in store.snapshots("AAA", "price_targets")] == ["2026-10-20", "2026-10-21"]
    store.snapshot("AAA", "upgrades", fetch, ttl=timedelta(days=7))
    store.snapshot("AAA", "upgrades", fetch, ttl=timedelta(days=7))
    assert len(asked) == 3
