"""Splitting the trading budget: weights sum to one, respect the cap, favour the less risky and the more promising,
and fall back to equal weights without price history. Synthetic prices only."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import allocation as A

pytestmark = pytest.mark.unit


def _prices(vols, n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0, v, n))) for i, v in enumerate(vols)}, index=idx)


def test_weights_sum_to_one_respect_the_cap_and_lean_away_from_risk():
    close = _prices([0.01, 0.01, 0.01, 0.04])
    out = A.weights(list(close.columns), [0.006] * 4, close)
    w = out["weights"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3) and max(w.values()) <= A.CAP + 1e-6
    assert w["S3"] < min(w["S0"], w["S1"], w["S2"])                 # four times the volatility: the smallest share


def test_a_higher_expected_return_gets_more_but_only_partly():
    close = _prices([0.015] * 3)
    base = A.weights(list(close.columns), [0.005, 0.005, 0.005], close)["weights"]
    tilt = A.weights(list(close.columns), [0.03, 0.005, 0.005], close)["weights"]
    assert tilt["S0"] > base["S0"] and tilt["S0"] <= A.CAP + 1e-6


def test_one_candidate_takes_all_and_missing_prices_fall_back_to_equal():
    assert A.weights(["A"], [0.01], None)["weights"] == {"A": 1.0}
    out = A.weights(["A", "B"], [0.01, 0.02], None)
    assert out["weights"] == {"A": 0.5, "B": 0.5} and out["method"] == "equal weights"
