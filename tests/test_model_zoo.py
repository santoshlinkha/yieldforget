"""The side-by-side models: competitors are same-sector and correlated, each stock gets its own regression, and the
Mamba model sees only the days up to each decision close."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import model_zoo as Z

pytestmark = pytest.mark.unit
DAYS = pd.bdate_range("2020-01-01", periods=400)


def _prices(seed=0):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(0, 0.01, 400), rng.normal(0, 0.01, 400)
    cols = {}
    for i in range(6):                       # A0-A2 move with factor 1 (tech), B0-B2 with factor 2 (energy)
        f = f1 if i < 3 else f2
        cols[("A" if i < 3 else "B") + str(i % 3)] = 100 * np.exp(np.cumsum(f + rng.normal(0, 0.003, 400)))
    cols["SPY"] = 100 * np.exp(np.cumsum((f1 + f2) / 2))
    return pd.DataFrame(cols, index=DAYS)


def test_competitors_are_the_most_correlated_stocks_of_the_same_sector():
    close = _prices()
    sectors = {**{f"A{i}": "Tech" for i in range(3)}, **{f"B{i}": "Energy" for i in range(3)}}
    d = DAYS[[300]]
    f = Z.peer_features(close, sectors, d, list(sectors))
    a0 = f.loc[(d[0], "A0")]
    peers20 = np.log(close[["A1", "A2"]]).diff(20).loc[d[0]].mean()
    assert a0["peer_ret_20"] == pytest.approx(peers20)                     # its tech peers, never the energy names
    assert a0["own_minus_peer_20"] == pytest.approx(np.log(close["A0"]).diff(20).loc[d[0]] - peers20)


def test_each_stock_gets_its_own_regression_and_short_histories_none():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=120, freq="5B")
    rows = []
    for t, slope in (("UP", 1.0), ("DOWN", -1.0)):
        x = rng.normal(size=len(dates))
        rows.append(pd.DataFrame({"x": x, "label": slope * x + rng.normal(0, 0.1, len(dates))},
                                 index=pd.MultiIndex.from_arrays([dates, [t] * len(dates)], names=["date", "ticker"])))
    train = pd.concat(rows).sort_index()
    test = pd.DataFrame({"x": [1.0, 1.0, 1.0]}, index=pd.MultiIndex.from_arrays(
        [[dates[-1]] * 3, ["UP", "DOWN", "NEW"]], names=["date", "ticker"]))
    got = Z.fit_ridge_per_stock(train, test, ["x"])
    assert got[0] > 0.5 and got[1] < -0.5 and np.isnan(got[2])            # opposite stocks, opposite forecasts


def test_mamba_sees_only_days_up_to_the_decision_and_learns_a_simple_pattern():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mambapy")
    from tradingagents.stocks import seq_model as SM

    close = _prices(2)
    data = SimpleNamespace(close=close, high=close * 1.01, low=close * 0.99, volume=close * 0 + 1e6,
                           info={c: {"sector_etf": None} for c in close})
    bank = SM.SequenceBank(data, start="2020-01-01")
    d = DAYS[[200]]
    w = bank.batch(d, ["A0"])
    assert w.shape == (1, SM.LOOKBACK, SM.CHANNELS)
    later = close.copy()
    later.iloc[201:] *= 3                                                  # a future jump must change nothing
    w2 = SM.SequenceBank(SimpleNamespace(**{**vars(data), "close": later, "high": later * 1.01, "low": later * 0.99}),
                         start="2020-01-01").batch(d, ["A0"])
    assert np.allclose(w, w2)
    assert bank.batch(DAYS[[10]], ["A0"]).sum() == 0                       # not enough history: zeros, not a crash
    torch.manual_seed(0)
    old = (SM.EPOCHS, SM.ROWS_PER_EPOCH, SM.BATCH)
    SM.EPOCHS, SM.ROWS_PER_EPOCH, SM.BATCH = 2, 2000, 128
    try:
        dates = DAYS[70:390:2]
        idx = pd.MultiIndex.from_product([dates, [c for c in close if c != "SPY"]], names=["date", "ticker"])
        rows = pd.DataFrame({"f": 0.0}, index=idx)
        rows["label"] = np.log(close.shift(-5) / close).stack().reindex(idx).fillna(0).to_numpy()
        rows["label_end"] = rows.index.get_level_values("date")
        m = SM.MambaRanker(bank, "cpu").fit(rows, ["f"])
        assert np.isfinite(m.predict(rows.iloc[:10])).all()
    finally:
        SM.EPOCHS, SM.ROWS_PER_EPOCH, SM.BATCH = old


def test_the_ensemble_weights_member_ranks_and_reads_its_own_map(tmp_path):
    import json

    from tradingagents.stocks.adapters import score_to_net

    pct = np.linspace(0.01, 1, 100)
    bundle = {"version": "ensemble-v1", "members": ["a", "b"], "weights": {"a": 0.75, "b": 0.25},
              "map_net": score_to_net(pct, pct / 100), "map_excess": score_to_net(pct, pct / 100 - 0.005)}
    (tmp_path / "decision_models").mkdir()
    (tmp_path / "decision_models" / "ensemble.json").write_text(json.dumps(bundle))
    member = np.array([True, True, True, False])
    scores = {"a": np.array([3.0, 1.0, 2.0, 9.0]), "b": np.array([1.0, 3.0, 2.0, 0.0])}
    e = Z.ensemble_scores(scores, member, tmp_path)
    assert list(np.argsort(-e["pct"][:3])) == [0, 2, 1]          # a's order wins: it carries three quarters
    assert e["expected_net"][0] > e["expected_net"][1] and e["weights"] == {"a": 0.75, "b": 0.25}
    assert Z.ensemble_scores({"a": scores["a"]}, member, tmp_path) is None   # a missing member: no ensemble


def test_the_decision_ranks_by_the_ensemble_only_for_its_own_close():
    from tradingagents.stocks import daily

    today = pd.DataFrame({"score_pct": [0.9, 0.1], "expected_net": [0.005, 0.004], "expected_excess": [0.0, 0.0],
                          "net_sd": [0.1, 0.1], "net_q10": [-0.1, -0.1], "p_win": [0.5, 0.5]}, index=["X", "Y"])
    zoo = {"info_cutoff": "2026-10-26", "ensemble": {"weights": {"a": 1.0}},
           "stocks": {"X": {"ensemble": {"pct": 0.2, "forecast": 0.001, "excess_vs_spy": -0.002, "net_sd": 0.05}},
                      "Y": {"ensemble": {"pct": 0.95, "forecast": 0.012, "excess_vs_spy": 0.004, "net_sd": 0.06}}}}
    bundle = {"quality": SimpleNamespace(predict=lambda rows: np.full(len(rows), 0.55))}
    rows = pd.DataFrame({"f": [0.0, 0.0]}, index=["X", "Y"])
    out, used = daily.apply_ensemble(today, zoo, bundle, rows, pd.Timestamp("2026-10-26"))
    assert used["name"] == "ensemble" and out.loc["Y", "score_pct"] == 0.95 and out.loc["Y", "expected_net"] == 0.012
    stale, used2 = daily.apply_ensemble(today, zoo, bundle, rows, pd.Timestamp("2026-10-27"))
    assert used2["name"] == "xgb_rank" and stale.loc["X", "score_pct"] == 0.9
