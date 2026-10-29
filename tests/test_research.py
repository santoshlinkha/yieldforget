"""Forecasting research on labelled synthetic data: no look-ahead, purged folds, coherent forecasts.

The synthetic market has a planted signal: next month's return leans on a hidden
per-asset drift that also shows up in past momentum. A working model should find
some of it; nothing here is real market data.
"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research import features as F
from tradingagents.research.forecast import Probabilistic
from tradingagents.research.models import XGB, ElasticNet, Momentum, TrainMean, VolPrior, Zero
from tradingagents.research.walkforward import walk_forward

pytestmark = pytest.mark.unit


def _market(n_assets=30, years=12, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2008-01-01", periods=252 * years)
    n = len(days)
    drift = rng.normal(0, 0.0006, n_assets)
    # Drifts wander slowly, so past returns say something about the next month.
    path = np.cumsum(rng.normal(0, 0.00002, (n, n_assets)), axis=0) + drift
    ret = path + rng.normal(0, 0.012, (n, n_assets))
    tickers = [f"A{i:02d}" for i in range(n_assets)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=days, columns=tickers)
    close["CASH"] = 100 * np.exp(np.arange(n) * 0.02 / 252)
    spy = pd.Series(100 * np.exp(np.cumsum(ret.mean(axis=1))), index=days)
    high, low = close * 1.01, close * 0.99
    volume = pd.DataFrame(rng.integers(1e5, 1e6, close.shape), index=days, columns=close.columns).astype(float)
    market = {"SPY": spy, "^VIX": pd.Series(20.0, index=days), "^VIX3M": pd.Series(21.0, index=days),
              "^TNX": pd.Series(3.0, index=days), "^IRX": pd.Series(2.0, index=days)}
    return close, high, low, volume, market


def _panel(close, high, low, volume, market, horizon=20):
    dates = F.decision_dates(close.index, start=close.index[260], every=min(horizon, 20))
    return F.build_panel(close, high, low, volume, market, "CASH", dates, horizon=horizon)


def test_the_label_is_the_next_horizon_log_return_above_cash():
    close, high, low, volume, market = _market(n_assets=6, years=3)
    panel = _panel(close, high, low, volume, market)
    (date, ticker), row = next(iter(panel.dropna(subset=["label"]).iterrows()))
    i = close.index.get_loc(date)
    want = np.log(close[ticker].iloc[i + 20] / close[ticker].iloc[i]) - np.log(close["CASH"].iloc[i + 20] / close["CASH"].iloc[i])
    assert row["label"] == pytest.approx(want) and row["label_end"] == close.index[i + 20]


def test_changing_the_future_leaves_earlier_features_and_forecasts_unchanged():
    close, high, low, volume, market = _market(n_assets=12, years=8)
    cut = close.index[1500]
    shocked = close.copy()
    shocked.loc[shocked.index > cut] *= 1.7
    a = _panel(close, high, low, volume, market)
    b = _panel(shocked, high, low, volume, market)
    feats = [c for c in a.columns if c not in ("label", "label_end")]
    early = a.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(a.loc[early, feats], b.loc[early, feats])
    # A model trained on labels that ended before the cut forecasts the cut date identically.
    train_a = a[(pd.to_datetime(a["label_end"]) < cut) & a["label"].notna()]
    train_b = b[(pd.to_datetime(b["label_end"]) < cut) & b["label"].notna()]
    rows_a = a.xs(a.index.get_level_values("date")[early].max(), level="date", drop_level=False)
    rows_b = b.loc[rows_a.index]
    fa = Probabilistic(ElasticNet).fit(train_a, F.PRICE_FEATURES, 20).predict(rows_a[feats])
    fb = Probabilistic(ElasticNet).fit(train_b, F.PRICE_FEATURES, 20).predict(rows_b[feats])
    pd.testing.assert_frame_equal(fa, fb)


def test_no_training_label_overlaps_its_test_block():
    close, high, low, volume, market = _market(n_assets=10, years=9)
    panel = _panel(close, high, low, volume, market, horizon=60)
    result = walk_forward(panel, [TrainMean], F.PRICE_FEATURES, 60, first_test=str(close.index[1300].date()))
    for entry in result.fit_log:
        assert pd.Timestamp(entry["train_last_label_end"]) < pd.Timestamp(entry["block"])


def test_forecasts_are_coherent_and_models_beat_nothing_only_when_they_should():
    close, high, low, volume, market = _market()
    panel = _panel(close, high, low, volume, market)
    factories = [Zero, TrainMean, VolPrior, Momentum, ElasticNet, lambda: XGB(rounds=300),
                 lambda: XGB("quantile", rounds=300)]
    result = walk_forward(panel, factories, F.PRICE_FEATURES, 20, first_test=str(close.index[1500].date()))
    p = result.predictions
    for m in ("elastic_net", "xgb", "xgb_quantile", "momentum"):
        assert (p[f"{m}:q05"] <= p[f"{m}:median"] + 1e-12).all() and (p[f"{m}:median"] <= p[f"{m}:q95"] + 1e-12).all()
        assert p[f"{m}:p_gain"].between(0, 1).all()
    s = result.scores
    assert np.isnan(s["zero"].mean_ic)                       # no ranking at all
    assert s["momentum"].mean_ic > 0.02 and s["elastic_net"].mean_ic > 0.02   # the planted signal is found
    assert 0.75 < s["elastic_net"].coverage_90 < 0.98        # intervals from unseen errors are near 90%


def test_xgboost_is_reproducible_with_a_seed():
    close, high, low, volume, market = _market(n_assets=15, years=6)
    panel = _panel(close, high, low, volume, market).dropna(subset=["label"])
    one = XGB(rounds=200, seed=1).fit(panel, F.PRICE_FEATURES, 20).predict(panel)
    two = XGB(rounds=200, seed=1).fit(panel, F.PRICE_FEATURES, 20).predict(panel)
    np.testing.assert_array_equal(one, two)


def _fake_bars(close, high, low, volume, market):
    def bars(tickers, cache_dir=None):
        frames = {"Close": close.copy(), "High": high.copy(), "Low": low.copy(), "Volume": volume.copy()}
        for name, series in market.items():
            for f in frames:
                frames[f][name] = series if f == "Close" else 0.0
        return {f: frames[f].reindex(columns=list(tickers)) for f in frames}
    return bars


def test_a_forecast_run_is_saved_immutably_and_its_model_reloads_identically(tmp_path, monkeypatch):
    import sqlite3

    from tradingagents.ledger.db import connect
    from tradingagents.research import engine

    close, high, low, volume, market = _market(n_assets=12, years=7)
    close = close.rename(columns={"CASH": "BIL"})
    high, low, volume = (x.rename(columns={"CASH": "BIL"}) for x in (high, low, volume))
    monkeypatch.setattr(engine, "bars", _fake_bars(close, high, low, volume, market))
    con = connect(tmp_path / "p.sqlite3")
    run = engine.run_forecast(con, tmp_path, [c for c in close.columns if c != "BIL"], horizon=20,
                              model="xgb_quantile", device_setting="cpu", bags=2, artifact_dir=tmp_path / "models")
    assert run["info_cutoff"] == str(close.index[-1].date()) and len(run["forecasts"]) == 12
    f = run["forecasts"][0]
    assert f["q05"] <= f["median"] <= f["q95"] and 0 <= f["p_gain"] <= 1 and f["estimation_sd"] >= 0
    assert "no walk-forward trial backs this model at this horizon" in f["flags"]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE forecasts SET mean = 1 WHERE run_id = ?", (run["run_id"],))
    saved = engine.load_run(con, run["run_id"])
    model = engine.load_model(saved)
    assert model.horizon == 20 and saved["backend"] == "cpu"
    # The reloaded model reproduces the saved forecasts exactly.
    last = close.index[-1]
    panel = F.build_panel(close, high, low, volume, market, "BIL", pd.DatetimeIndex([last]), horizon=20)
    again = model.predict(panel.xs(last, level="date", drop_level=False))
    stored = {f["ticker"]: f["mean"] for f in saved["forecasts"]}
    assert len(again) == len(stored)
    for (_, t), m in again["mean"].items():
        assert m == pytest.approx(stored[t], abs=1e-9)


def test_a_stock_run_never_stands_in_for_the_plans_etf_run(tmp_path, monkeypatch):
    """The plan holds funds and reads the latest run; a newer stock run must not become that run."""
    from tradingagents.ledger.db import connect
    from tradingagents.research import engine

    close, high, low, volume, market = _market(n_assets=6, years=4)
    close = close.rename(columns={"CASH": "BIL"})
    high, low, volume = (x.rename(columns={"CASH": "BIL"}) for x in (high, low, volume))
    monkeypatch.setattr(engine, "bars", _fake_bars(close, high, low, volume, market))
    con = connect(tmp_path / "p.sqlite3")
    tickers = [c for c in close.columns if c != "BIL"]
    etf = engine.run_forecast(con, tmp_path, tickers, horizon=20, model="elastic_net", device_setting="cpu",
                              bags=0, artifact_dir=tmp_path / "models")
    stock = engine.run_forecast(con, tmp_path, tickers, horizon=20, model="elastic_net", universe="stock",
                                device_setting="cpu", bags=0, artifact_dir=tmp_path / "models")
    assert engine.latest_run(con, horizon=20)["id"] == etf["run_id"]
    assert engine.latest_run(con, horizon=20, universe="stock")["id"] == stock["run_id"]


def test_optimizer_inputs_are_shrunk_clipped_and_conservative():
    from tradingagents.research.engine import expected_returns

    run = {"calibration": {"calibration": 0.5}, "forecasts": [
        {"ticker": "AAA", "mean": 0.02, "sigma": 0.05, "estimation_sd": 0.004},
        {"ticker": "BBB", "mean": 0.50, "sigma": 0.05, "estimation_sd": 0.0},
        {"ticker": "CCC", "mean": 0.01, "sigma": 0.05, "estimation_sd": 0.0}]}
    mu = expected_returns(run, ["AAA", "BBB"], conservative=True, mode="absolute")
    assert mu["AAA"] == pytest.approx(0.5 * 0.02 - 0.004)
    assert mu["BBB"] == pytest.approx(0.10)            # 0.5 x 0.50 clipped at two sigmas
    assert "CCC" not in mu
    assert expected_returns({"calibration": {}, "forecasts": run["forecasts"]}, ["AAA"], mode="absolute")["AAA"] <= 0
    # Relative: the training premium for all, plus the calibrated difference from the day's average.
    rel = {"calibration": {"calibration_xs": 1.0}, "config": {"train_mean_label": 0.006}, "forecasts": [
        {"ticker": "AAA", "mean": 0.03, "sigma": 0.05}, {"ticker": "BBB", "mean": 0.01, "sigma": 0.05}]}
    mu = expected_returns(rel, ["AAA", "BBB"])
    assert mu["AAA"] == pytest.approx(0.016) and mu["BBB"] == pytest.approx(-0.004)
    flat = expected_returns({**rel, "calibration": {}}, ["AAA", "BBB"])
    assert flat["AAA"] == flat["BBB"] == pytest.approx(0.006)   # an unbacked model moves nothing


def test_chronos_challenger_uses_its_own_quantiles_without_downloading(monkeypatch):
    torch = pytest.importorskip("torch")
    from tradingagents.research import foundation

    class FakePipeline:
        calls = 0

        def predict_quantiles(self, contexts, prediction_length, quantile_levels):
            FakePipeline.calls += 1
            out = []
            for c in contexts:
                last = float(c[-1])
                steps = torch.tensor([[last - 0.05, last + 0.01, last + 0.06]] * prediction_length)
                out.append(steps.unsqueeze(0))
            return out, None

    monkeypatch.setattr(foundation, "pipeline", lambda device: FakePipeline())
    close, high, low, volume, market = _market(n_assets=5, years=3)
    panel = _panel(close, high, low, volume, market).dropna(subset=["label"])
    rows = panel.iloc[:10]
    model = Probabilistic(lambda: foundation.Chronos2(close)).fit(panel, F.PRICE_FEATURES, 20)
    got = model.predict(rows)
    cash = rows["cash_h"].to_numpy()
    np.testing.assert_allclose(got["median"], 0.01 - cash, atol=1e-6)
    assert (got["q05"] < got["median"]).all() and (got["median"] < got["q95"]).all()
    calls = FakePipeline.calls
    model.predict(rows)
    assert FakePipeline.calls == calls   # forecasts are cached per row and horizon


def test_lstm_challenger_trains_reproducibly_on_cpu():
    pytest.importorskip("torch")
    from tradingagents.research.neural import LSTMForecaster, Sequences

    close, high, low, volume, market = _market(n_assets=8, years=5)
    panel = _panel(close, high, low, volume, market).dropna(subset=["label"])
    seqs = Sequences(close.drop(columns="CASH"), market["SPY"])
    a = LSTMForecaster(seqs, epochs=3).fit(panel, F.PRICE_FEATURES, 20).predict(panel.iloc[:50])
    b = LSTMForecaster(seqs, epochs=3).fit(panel, F.PRICE_FEATURES, 20).predict(panel.iloc[:50])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    assert np.isfinite(a).all()


def test_a_batch_comparison_lines_up_evidence_agreement_and_flat_runs():
    from tradingagents.research.engine import compare_runs

    def run(rid, model, horizon, means, ic=None):
        return {"run_id": rid, "model": model, "horizon": horizon, "backend": "cpu", "seconds": 1.0,
                "calibration": {"trial": 9, "mean_ic": ic, "ic_ci": [0.0, 0.1], "p_value_holm": 0.01} if ic else {},
                "forecasts": [{"ticker": t, "mean": m} for t, m in means.items()]}

    rows = compare_runs([
        run("a", "xgb_rank", 60, {"X": 0.03, "Y": 0.02, "Z": 0.01}, ic=0.066),
        run("b", "elastic_net", 60, {"X": 0.02, "Y": 0.01, "Z": 0.00}, ic=0.045),
        run("c", "xgb", 60, {"X": 0.01, "Y": 0.01, "Z": 0.01}),
        run("d", "elastic_net", 20, {"X": 0.0, "Y": 0.1, "Z": 0.2}),
    ], top=2)
    by = {r["run_id"]: r for r in rows}
    assert [r["horizon"] for r in rows] == [20, 60, 60, 60]
    assert by["a"]["agreement"] == pytest.approx(1.0)        # same order as b; c is flat and correlates with nothing
    assert by["d"]["agreement"] is None                      # nothing else at 20 sessions
    assert not by["c"]["ranks"] and by["c"]["top"] == []
    assert [t["ticker"] for t in by["a"]["top"]] == ["X", "Y"] and by["a"]["mean_ic"] == 0.066


def test_the_live_record_scores_a_run_only_once_its_horizon_has_passed(tmp_path, monkeypatch):
    from tradingagents.ledger.db import connect
    from tradingagents.research import engine

    close, high, low, volume, market = _market(n_assets=8, years=4)
    close = close.rename(columns={"CASH": "BIL"})
    high, low, volume = (x.rename(columns={"CASH": "BIL"}) for x in (high, low, volume))
    tickers = [c for c in close.columns if c != "BIL"]
    con = connect(tmp_path / "p.sqlite3")
    # A run made 30 sessions ago (its horizon of 20 has passed), and one made today (it has not).
    cut = close.index[-31]
    early = {n: s.loc[:cut] for n, s in market.items()}
    monkeypatch.setattr(engine, "bars", _fake_bars(*(x.loc[:cut] for x in (close, high, low, volume)), early))
    engine.run_forecast(con, tmp_path, tickers, horizon=20, model="elastic_net", device_setting="cpu", bags=0,
                        artifact_dir=tmp_path / "models")
    monkeypatch.setattr(engine, "bars", _fake_bars(close, high, low, volume, market))
    engine.run_forecast(con, tmp_path, tickers, horizon=20, model="elastic_net", device_setting="cpu", bags=0,
                        artifact_dir=tmp_path / "models")
    rec = engine.live_record(con, tmp_path, "elastic_net", 20)
    assert rec["made"] == 2 and rec["resolved"] == 1
    assert rec["dates"][0]["date"] == str(cut.date()) and rec["dates"][0]["ended"] == str(close.index[-11].date())
    assert -1 <= rec["mean_ic"] <= 1 and rec["dates"][0]["n"] == 8


def test_the_combined_forecast_weights_models_by_their_tested_skill():
    from tradingagents.research.engine import combine_runs

    def run(rid, model, means, ic=None, p=0.01):
        cal = {"trial": 1, "mean_ic": ic, "p_value_holm": p} if ic is not None else {}
        return {"id": rid, "model": model, "calibration": cal,
                "forecasts": [{"ticker": t, "mean": m, "median": m, "q05": m - 0.1, "q95": m + 0.1, "p_gain": 0.5,
                               "estimation_sd": 0.01, "flags_json": "[]"} for t, m in means.items()]}

    out = combine_runs([run("a", "xgb_rank", {"X": 0.04, "Y": 0.00}, ic=0.3),
                        run("b", "elastic_net", {"X": 0.01, "Y": 0.02}, ic=0.1),
                        run("c", "momentum", {"X": 0.50, "Y": 0.50}, ic=0.05, p=0.4),     # not significant
                        run("d", "chronos2", {"X": -0.5, "Y": 0.9}, ic=0.2)])              # contamination unknown
    assert out["basis"] == "evidence"
    x = next(r for r in out["rows"] if r["ticker"] == "X")
    assert x["combined"]["mean"] == pytest.approx(0.75 * 0.04 + 0.25 * 0.01)
    assert x["combined"]["q05"] == pytest.approx(0.75 * -0.06 + 0.25 * -0.09)
    assert x["combined"]["spread"] == pytest.approx((0.75 * 0.0075 ** 2 + 0.25 * 0.0225 ** 2) ** 0.5)
    weights = {m["model"]: m["weight"] for m in x["models"]}
    assert weights == pytest.approx({"xgb_rank": 0.75, "elastic_net": 0.25, "momentum": 0.0, "chronos2": 0.0})
    assert out["rows"][0]["ticker"] == "X"          # sorted by combined mean, highest first

    unbacked = combine_runs([run("a", "xgb", {"X": 0.02}), run("b", "elastic_net", {"X": 0.04})])
    assert unbacked["basis"] == "equal" and unbacked["rows"][0]["combined"]["mean"] == pytest.approx(0.03)


def test_a_forecast_from_the_same_inputs_is_not_trained_or_saved_again(tmp_path, monkeypatch):
    from tradingagents.ledger.db import connect
    from tradingagents.research import engine

    close, high, low, volume, market = _market(n_assets=8, years=7)
    close = close.rename(columns={"CASH": "BIL"})
    high, low, volume = (x.rename(columns={"CASH": "BIL"}) for x in (high, low, volume))
    monkeypatch.setattr(engine, "bars", _fake_bars(close, high, low, volume, market))
    con = connect(tmp_path / "p.sqlite3")
    tickers = [c for c in close.columns if c != "BIL"]
    kw = {"horizon": 20, "model": "elastic_net", "device_setting": "cpu", "bags": 0, "artifact_dir": tmp_path / "m"}
    first = engine.run_forecast(con, tmp_path, tickers, **kw)
    again = engine.run_forecast(con, tmp_path, tickers, **kw)
    assert again["reused"] and again["run_id"] == first["run_id"] and len(again["forecasts"]) == len(first["forecasts"])
    assert con.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0] == 1
    other = engine.run_forecast(con, tmp_path, tickers[:-1], **kw)            # other tickers: other inputs
    forced = engine.run_forecast(con, tmp_path, tickers, force=True, **kw)
    assert not other.get("reused") and not forced.get("reused")
    assert con.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0] == 3
