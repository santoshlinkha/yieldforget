"""What the Models tab and the Trade view say about coverage, fallbacks and each number's definition.

Synthetic saved runs and decisions; no model is fitted.
"""

import pytest

from tradingagents import model_outputs as MO
from tradingagents.stocks import model_zoo, trade

pytestmark = pytest.mark.unit


def _run(with_ensemble: bool):
    stocks = {f"S{i}": {"xgb_rank": {"pct": i / 9, "forecast": None},
                        "chronos2": {"pct": i / 9, "forecast": 0.01}} for i in range(10)}
    stocks["S0"].pop("chronos2")                                         # partial coverage
    if with_ensemble:
        for i, m in enumerate(stocks.values()):
            m["ensemble"] = {"pct": i / 9, "forecast": 0.004}
    models = {k: {"label": v[0], "kind": v[1]} for k, v in model_zoo.MODELS.items()}
    return {"stocks": stocks, "models": models, "notes": {} if with_ensemble else {"ensemble": "not available"}}


def test_a_run_without_ensemble_scores_says_the_ranker_fell_back():
    d = model_zoo.describe(_run(with_ensemble=False))
    assert d["ranking_source"] == "xgb_rank" and "fallback" in d["ranking_note"]
    assert d["models"]["ensemble"]["coverage"] == 0 and d["models"]["xgb_rank"]["lifecycle"] == "fallback_decision"
    assert d["models"]["chronos2"]["coverage"] == 9 and d["models"]["chronos2"]["of"] == 10   # partial, shown as such
    with_ens = model_zoo.describe(_run(with_ensemble=True))
    assert with_ens["ranking_source"] == "ensemble" and with_ens["models"]["ensemble"]["lifecycle"] == "active_decision"


def test_each_zoo_forecast_number_names_its_own_unit_and_clock():
    assert model_zoo.output_key("ensemble") == "zoo.ensemble_expected_net"
    assert model_zoo.output_key("chronos2") == model_zoo.output_key("chronos2_covariates") == "zoo.chronos_log_return"
    assert model_zoo.output_key("lgbm_regression") == "zoo.forecast_log_excess"
    assert model_zoo.output_key("xgb_rank") is None                      # a rank, no return forecast
    ens, reg, chron = (MO.OUTPUTS[k] for k in ("zoo.ensemble_expected_net", "zoo.forecast_log_excess",
                                                "zoo.chronos_log_return"))
    assert ens.unit == "arithmetic return" and reg.unit == "log return" and chron.entry != reg.entry


def test_the_trade_view_says_when_p_profit_is_a_transfer_from_the_base_ranker():
    decision = {"id": "d", "info_cutoff": "2026-10-26", "execution_date": "2026-10-27", "mode": "experimental",
                "entries": [{"ticker": "AAA", "action": "WAIT", "paper_action": "PAPER ENTER", "price": 10.0,
                             "sigma": 0.02}], "lots": [],
                "model": {"ranker": {"name": "ensemble", "quality_matched": False, "coverage": 400, "of": 500,
                                     "quality_note": "P(profit after costs) comes from the base ranker's model"}}}
    notes = " ".join(trade.plan(decision, 100.0)["notes"])
    assert "base ranker's model" in notes and "400 of 500" in notes
