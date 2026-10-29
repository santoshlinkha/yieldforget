"""The sector model: labels from the next close, purged folds, no scoring of training dates, winners by sector."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research import sector_model as S
from tradingagents.research.rotation import SECTORS

pytestmark = pytest.mark.unit


def _close(n=900, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    out = {t: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for t in ["SPY", *SECTORS, "USO", "UUP", "HYG", "IEF"]}
    out.update({"^TNX": 3 + np.cumsum(rng.normal(0, .03, n)), "^IRX": np.full(n, 2.0), "^VIX": 15 + rng.normal(0, 1, n)})
    return pd.DataFrame(out, index=idx)


def test_the_label_is_the_return_over_spy_from_the_next_close():
    c = _close()
    p = S.features(c)
    d = p.index.get_level_values("date").unique()[300]
    row = p.loc[(d, "XLK")]
    i = c.index.get_loc(d)
    rel = np.log(c["XLK"]) - np.log(c["SPY"])
    assert row["label"] == pytest.approx(rel.iloc[i + 1 + S.H] - rel.iloc[i + 1])
    assert row["label_end"] == c.index[i + 1 + S.H]
    assert p.xs(c.index[-1], level="date")["label"].isna().all()              # the future is unknown


def test_a_fitted_model_refuses_its_training_dates_and_ranks_today():
    from tradingagents.stocks.training_store import TrainedOnError

    p = S.features(_close())
    feats = S.feature_names(p)
    lab = p[p["label"].notna()]
    cut = lab.index.get_level_values("date").unique()[400]
    m = S.fit(lab[pd.to_datetime(lab["label_end"]) < cut], feats)
    later = p[p.index.get_level_values("date") > cut]
    out = S.predict(m, later)
    assert out.groupby(level="date")["rank_pct"].max().eq(1.0).all()
    with pytest.raises(TrainedOnError):
        S.predict(m, lab[lab.index.get_level_values("date") <= pd.Timestamp(m["trained_through"])])


def test_the_winner_is_the_top_ranked_stock_of_each_sector():
    members = {"AAA": "Alpha (Information Technology)", "BBB": "Beta (Information Technology)", "CCC": "Gamma (Utilities)"}
    decision = {"scores": {"AAA": {"rank": 9, "of": 500, "member": True}, "BBB": {"rank": 3, "of": 500, "member": True},
                           "CCC": {"rank": 40, "of": 500, "member": True}}}
    w = S.winners(decision, members)
    assert w["XLK"]["ticker"] == "BBB" and w["XLU"]["ticker"] == "CCC" and w["XLK"]["name"] == "Beta"


def test_the_same_prices_and_code_load_the_saved_forecast_instead_of_refitting(tmp_path, monkeypatch):
    import tradingagents.research.prices as prices

    close = _close(n=4700)
    close.index = pd.bdate_range("2008-01-01", periods=len(close))     # the study's folds start in 2010
    monkeypatch.setattr(prices, "bars", lambda tickers, cache_dir: {"Close": close})
    studies = []
    real = S.study
    monkeypatch.setattr(S, "study", lambda *a, **k: (studies.append(1), real(*a, **k))[1])
    first = S.run(tmp_path)
    again = S.run(tmp_path)
    assert len(studies) == 1 and again["reused"] and again["forecast"] == first["forecast"]
    S.run(tmp_path, force=True)                                       # asked: fitted again
    assert len(studies) == 2
    close.iloc[-1, 0] *= 1.01                                          # a changed close is new data
    assert not S.run(tmp_path).get("reused") and len(studies) == 3
