"""The training store: every set saved once, the same data never fitted twice, training dates never scored."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import training_store as TS

pytestmark = pytest.mark.unit


def _table(seed=0, n=50):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=n // 5), list("ABCDE")], names=["date", "ticker"])
    return pd.DataFrame({"f1": rng.normal(size=n), "label": rng.normal(size=n)}, index=idx)


def test_the_fingerprint_follows_every_value_and_the_design():
    a = _table()
    assert TS.fingerprint(a) == TS.fingerprint(a.copy())
    b = a.copy()
    b.iloc[3, 0] += 1e-9
    assert TS.fingerprint(b) != TS.fingerprint(a)
    assert TS.fingerprint(a, {"policy": "x"}) != TS.fingerprint(a, {"policy": "y"})


def test_a_set_is_saved_once_and_found_with_its_bundle(tmp_path):
    t = _table()
    fp = TS.fingerprint(t)
    bundle = tmp_path / "decision_models" / "k.pkl"
    bundle.parent.mkdir()
    bundle.write_bytes(b"x")
    assert TS.find(tmp_path, fp) is None
    TS.save(tmp_path, fp, t, {"cutoff": "2020-03-01", "trained_through": "2020-02-07"}, "k", bundle)
    TS.save(tmp_path, fp, t, {"cutoff": "2020-03-01"}, "k", bundle)                  # once
    got = TS.find(tmp_path, fp)
    assert got["n_rows"] == 50 and got["bundle_key"] == "k" and got["trained_through"] == "2020-02-07"
    assert len(TS.listing(tmp_path)) == 1
    import duckdb

    con = duckdb.connect(str(TS.path(tmp_path)), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM training_rows WHERE fingerprint = ?", [fp]).fetchone()[0] == 50
    con.close()
    bundle.unlink()
    assert TS.find(tmp_path, fp) is None                                              # a missing fit is refitted


def test_training_dates_are_never_scored():
    TS.guard("2020-02-07", pd.DatetimeIndex(["2020-02-10"]))
    with pytest.raises(TS.TrainedOnError):
        TS.guard("2020-02-07", pd.DatetimeIndex(["2020-02-07", "2020-02-10"]))


def test_outcome_models_refuse_their_own_training_dates():
    from tradingagents.stocks.outcome_models import OutcomeModel

    rng = np.random.default_rng(1)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=400), list("ABCDEFGHIJ")], names=["date", "ticker"])
    f = pd.DataFrame({"f1": rng.normal(size=len(idx)), "outcome": rng.choice(["target", "stop", "expiry"], len(idx)),
                      "excess": rng.normal(0, .05, len(idx))}, index=idx)
    d = f.index.get_level_values("date")
    m = OutcomeModel("linear", "next_close").fit(f[d < "2020-12-01"], f[(d >= "2020-12-01") & (d < "2021-03-01")], ["f1"])
    m.predict(f[d >= "2021-04-01"])
    with pytest.raises(TS.TrainedOnError):
        m.predict(f[d >= "2021-02-01"])
