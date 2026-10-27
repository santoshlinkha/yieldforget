"""The quant-upgrade pieces do what they say, on hand-made data: the fixed-size loss label, residual momentum,
the two-step limit model and the 8 rotation drivers."""

import numpy as np
import pandas as pd
import pytest

from tests.test_stock_targets import DAYS, _frames, _run

pytestmark = pytest.mark.unit


def test_a_fixed_eight_percent_fall_after_entry_is_flagged_whatever_the_stock_volatility():
    dip = np.array([100, 97, 91.5, 95, 100] + [101] * 15, float)        # entry at 100, a close 8.5% below
    assert _run(*_frames(dip, before_noise=0.05))["pos_dd8"] == 1     # a wild stock and
    assert _run(*_frames(dip, before_noise=0.005))["pos_dd8"] == 1    # a calm one: the same label
    shallow = np.array([100, 97, 93, 95, 100] + [101] * 15, float)
    assert _run(*_frames(shallow))["pos_dd8"] == 0


def test_residual_momentum_sees_the_stock_own_run_not_the_market():
    from tradingagents.stocks.extra_features import residual_momentum

    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2023-01-02", periods=300)
    spy = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    cash = pd.Series(0.0001, index=idx)
    noise = rng.normal(0, 0.01, (300, 2))
    stock = pd.DataFrame({"A": 1.2 * spy + noise[:, 0], "B": 1.2 * spy + noise[:, 1]})
    stock.iloc[-20:, 0] += 0.01                                           # A's own run, 1% a day for 20 days
    stock.iloc[-20:, 1] += 2 * spy.iloc[-20:]                             # B only rode the market harder
    fund = spy + rng.normal(0, 0.005, 300)
    sector = pd.DataFrame({"A": fund, "B": fund})
    r20, r60 = residual_momentum(stock, spy, cash, sector)
    assert r20["A"].iloc[-1] > 2 and r20["A"].iloc[-1] > r20["B"].iloc[-1]
    assert r60.notna().iloc[-1].all()


def test_two_step_limit_model_gives_fill_times_outcome_probabilities():
    from tradingagents.stocks.outcome_models import OutcomeModel

    rng = np.random.default_rng(2)
    n = 8000
    x = rng.normal(size=(n, 2))
    fills = rng.random(n) < 1 / (1 + np.exp(-2 * x[:, 0]))                # x0 decides the fill
    hit = np.where(rng.random(n) < 1 / (1 + np.exp(-2 * x[:, 1])), "target", "stop")   # x1 the outcome
    outcome = np.where(fills, hit, "nofill")
    excess = np.where(outcome == "target", 0.03, np.where(outcome == "stop", -0.03, 0.0))
    idx = pd.MultiIndex.from_arrays([pd.bdate_range("2015-01-01", periods=n), np.zeros(n, int).astype(str)],
                                    names=["date", "ticker"])
    df = pd.DataFrame({"f0": x[:, 0], "f1": x[:, 1], "outcome": outcome, "excess": excess}, index=idx)
    m = OutcomeModel("lgbm_seq", "limit_dip").fit(df.iloc[:6000], df.iloc[6000:7500], ["f0", "f1"])
    assert m.fill is not None
    late = pd.DataFrame({"f0": [3.0, -3.0, 0.0, 0.0], "f1": [0.0, 0.0, 3.0, -3.0]},
                        index=pd.MultiIndex.from_arrays([[idx[-1][0]] * 4, list("abcd")], names=["date", "ticker"]))
    p = m.predict(late)
    assert np.allclose(p.filter(like="p_").sum(axis=1), 1)
    assert p["p_nofill"].iloc[0] < 0.2 < 0.8 < p["p_nofill"].iloc[1]
    assert p["p_target"].iloc[2] > p["p_target"].iloc[3]
    with pytest.raises(ValueError):
        OutcomeModel("tree", "limit_dip")


def test_rotation_drivers_add_the_hedge_basket_and_the_threats_index_a_day_late(monkeypatch):
    from tradingagents.research import rotation as R

    idx = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(3)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, len(R.TICKERS))), axis=0)),
                         index=idx, columns=R.TICKERS)
    d = R.driver_moves(close)
    assert "Geopolitical threats" not in d                                # needs the download
    lr = np.log(close).diff()
    assert d["Gold/defense/oil vs SPY"].iloc[5] == pytest.approx(lr[["GLD", "ITA", "USO"]].iloc[5].mean() - lr["SPY"].iloc[5])
    threats = pd.Series(np.arange(40, dtype=float), index=idx)
    threats.iloc[20] = 100.0                                             # a spike in the news on day 20
    monkeypatch.setattr("tradingagents.stocks.extra_features.gpr_daily",
                        lambda cache: pd.DataFrame({"GPRD_THREAT": threats}))
    g = R.driver_moves(close, cache_dir="x")["Geopolitical threats"]
    assert g.abs().idxmax() in (idx[21], idx[22]) and g.iloc[20] == pytest.approx(np.log1p(19) - np.log1p(18))
    assert len(R.DRIVERS) == 8
