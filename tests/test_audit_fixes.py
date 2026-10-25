"""Fixes from the 2026-10-25 audit validation: shadow idempotence, deadlines past the loaded prices."""

import pandas as pd
import pytest

from tradingagents.stocks.positions import POLICIES, Book

pytestmark = pytest.mark.unit


def test_a_lot_bought_on_the_latest_session_keeps_a_real_deadline():
    sessions = pd.bdate_range("2026-10-01", "2026-10-23")
    book = Book(sessions, cash=1000.0)
    lot = book.buy(sessions[-1], "AAA", 200.0, 100.0, POLICIES["barrier_20"], 0.02, 0.004)
    assert lot.deadline > sessions[-1] + pd.Timedelta(days=20)        # 20 sessions on the exchange calendar
    assert book.exits_due(sessions[-1], pd.Series({"AAA": 100.0})) == []
    mid = book.buy(sessions[5], "BBB", 200.0, 100.0, POLICIES["barrier_20"], 0.02, 0.004)
    assert mid.deadline > sessions[-1]                                  # not clamped to the last row


def test_the_legacy_scorer_compounds_baskets_and_annualizes_on_actual_spacing():
    import numpy as np

    from tradingagents.stocks.evaluate import portfolio

    sessions = pd.bdate_range("2020-01-01", periods=400)
    dates = sessions[::50]                      # spaced far wider than horizon + 1
    rows = []
    for d in dates:
        for i in range(30):
            label = np.log(1.10) if i % 2 else np.log(0.90)    # half +10%, half -10%: a flat basket
            rows.append({"date": d, "ticker": f"T{i}", "label": label, "net_score": i, "mom_12_1": i,
                         "current_trend": 1, "cost_rt": 0.0})
    frame = pd.DataFrame(rows).set_index(["date", "ticker"])
    out = portfolio(frame, "net_score", sessions, horizon=20)
    assert out["per_year"] == pytest.approx(252 / 50)               # not 252 / 21
    assert out["all"]["mean"] == pytest.approx(0.0, abs=1e-12)      # mean of simple returns, not of logs


def test_the_etf_backtest_trades_the_session_after_its_plan():
    import numpy as np

    from tradingagents.optimizer.backtest import _executed, _simulate

    days = pd.bdate_range("2026-01-05", periods=5)
    returns = pd.DataFrame({"X": [0.0, 0.5, 0.0, 0.0, 0.0]}, index=days)   # +50% on the second session
    cash = pd.Series(0.0, index=days)
    plan = [(days[0], {"X": 1.0})]
    same, _ = _simulate(returns, cash, plan)
    lagged, _ = _simulate(returns, cash, _executed(plan, days, 1))
    assert same.iloc[-1] == pytest.approx(1.5)      # the old assumption caught the jump
    assert lagged.iloc[-1] == pytest.approx(1.0)    # bought at the next close, after it
    assert _executed([(days[-1], {"X": 1.0})], days, 1) == []
    assert np.isfinite(lagged).all()


def test_a_past_cutoff_never_reuses_a_model_fitted_later_that_week(tmp_path):
    import pickle

    from tradingagents.stocks import daily
    from tradingagents.stocks import panel as P
    from tradingagents.stocks.targets import Thresholds

    policy = POLICIES["barrier_20"]
    friday, wednesday = pd.Timestamp("2026-10-23"), pd.Timestamp("2026-10-21")
    key = daily.bundle_key(friday, policy, P.features_for(), Thresholds())
    assert key == daily.bundle_key(wednesday, policy, P.features_for(), Thresholds())    # same ISO week
    (tmp_path / "decision_models").mkdir()
    (tmp_path / "decision_models" / f"{key}.pkl").write_bytes(pickle.dumps({"meta": {"cutoff": "2026-10-23"}}))
    bundle, art = daily.fit_bundle(None, None, friday, policy, tmp_path, train=False)
    assert art["reused"] and bundle["meta"]["cutoff"] == "2026-10-23"
    with pytest.raises(ValueError, match="not fitted"):
        daily.fit_bundle(None, None, wednesday, policy, tmp_path, train=False)


def test_as_of_data_drops_every_row_after_the_cutoff():
    from tradingagents.stocks.daily import as_of_data
    from tradingagents.stocks.data import StockData

    days = pd.bdate_range("2026-10-19", "2026-10-23")
    frame = pd.DataFrame({"AAA": range(5)}, index=days, dtype=float)
    data = StockData(None, {"AAA": {}}, ["AAA"], {}, frame, frame, frame, frame,
                     actions={"AAA": {"dividends": [["2026-10-22", 1.0]], "splits": []}})
    cut = as_of_data(data, days[1])
    assert all(f.index.max() == days[1] for f in (cut.close, cut.high, cut.low, cut.volume))
    assert cut.actions == data.actions          # kept whole: undoing the adjustment needs every dividend


def test_a_recorded_split_scales_the_rebuilt_lots(tmp_path):
    from tradingagents.ledger.store import Store
    from tradingagents.stocks.daily import lots_from_ledger

    store = Store(tmp_path / "book")
    d = pd.Timestamp("2026-09-01").date()
    store.ledger.contribute(1000.0, d, sleeve="trading", settle_date=d)
    store.ledger.fill("trading", "buy", "AAA", 2.0, 100.0, d, settle_date=d)
    store.ledger.fill("trading", "buy", "AAA", 1.0, 130.0, pd.Timestamp("2026-09-02").date())
    store.ledger.record("corporate_action", "trading", pd.Timestamp("2026-09-10").date(), ticker="AAA",
                        quantity=3.0, note="2-for-1 split")
    lots = lots_from_ledger(store, "trading", pd.Timestamp("2026-09-11").date())
    assert [(lot["shares"], lot["entry_px"]) for lot in lots] == [(4.0, 50.0), (2.0, 65.0)]
    assert sum(lot["shares"] for lot in lots) == store.ledger.state("trading", pd.Timestamp("2026-09-11").date()).positions["AAA"]


def test_a_shadow_lot_takes_a_later_split_once():
    from tradingagents.stocks.shadow import _apply_splits

    sessions = pd.bdate_range("2026-09-01", "2026-10-23")
    book = Book(sessions, cash=100.0)
    book.buy(sessions[0], "AAA", 100.0, 100.0, POLICIES["barrier_20"], 0.02, 0.0)
    state = {"split_basis": {"AAA|2026-09-01": ""}}
    actions = {"AAA": {"splits": [["2026-09-15", 2.0]], "dividends": []}}
    _apply_splits(state, book, actions, pd.Timestamp("2026-09-10"))          # before the split: nothing
    assert book.lots[0].shares == pytest.approx(1.0)
    _apply_splits(state, book, actions, pd.Timestamp("2026-09-16"))
    _apply_splits(state, book, actions, pd.Timestamp("2026-09-17"))          # a second update: not again
    assert (book.lots[0].shares, book.lots[0].entry_px) == (pytest.approx(2.0), pytest.approx(50.0))
