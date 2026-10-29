"""The model output contract: every family and output is described, and the pages name each number by it.

Synthetic: no model is fitted; the pages are read as text.
"""

from pathlib import Path

import numpy as np
import pytest

from tradingagents import model_outputs as MO

pytestmark = pytest.mark.unit

STATIC = Path(__file__).resolve().parents[1] / "tradingagents" / "webapp" / "static"
REQUIRED_FAMILIES = {"forecast_engine", "stock_signals", "daily_decision", "model_zoo", "policy_outcomes",
                     "sector_model", "risk_research", "sequence_challengers", "llm_roles", "text_and_quality"}


def test_every_family_has_a_lifecycle_consumer_evidence_and_known_outputs():
    assert set(MO.FAMILIES) >= REQUIRED_FAMILIES
    for f in MO.FAMILIES.values():
        assert f.lifecycle in MO.LIFECYCLES and f.evidence in MO.EVIDENCE and f.consumer and f.source
        assert all(o in MO.OUTPUTS for o in f.outputs), f.key


def test_every_output_states_target_unit_benchmark_horizon_clock_and_costs():
    for o in MO.OUTPUTS.values():
        assert o.target and o.unit and o.entry and o.exit and o.costs and o.calibration and o.label, o.key
    log = {k for k, o in MO.OUTPUTS.items() if o.unit == "log return"}
    arith = {k for k, o in MO.OUTPUTS.items() if o.unit == "arithmetic return"}
    assert "forecast.mean" in log and "daily.expected_net" in arith and "zoo.ensemble_expected_net" in arith
    # The ensemble's mapped P&L and an ordinary zoo forecast are different units and clocks: never one column.
    assert MO.OUTPUTS["zoo.ensemble_expected_net"].unit != MO.OUTPUTS["zoo.forecast_log_excess"].unit
    assert MO.OUTPUTS["zoo.chronos_log_return"].entry != MO.OUTPUTS["zoo.forecast_log_excess"].entry


def test_the_forecast_probability_is_of_a_positive_total_not_of_beating_cash():
    # The audit's synthetic case: standardized errors [-2..2], zero predicted excess, positive cash, scale 0.1.
    z, mean, cash, sigma = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 0.0, 0.05, 0.1
    p_total = 1 - np.searchsorted(z, (-cash - mean) / sigma, side="right") / len(z)
    p_excess = 1 - np.searchsorted(z, (-mean) / sigma, side="right") / len(z)
    assert p_total == pytest.approx(0.6) and p_excess == pytest.approx(0.4)
    spec = MO.OUTPUTS["forecast.p_total_positive"]
    assert "total return" in spec.target and "beating cash" not in spec.label
    assert MO.measures("fund")["probability"]["key"] == "forecast.p_total_positive"
    assert MO.measures("stock")["probability"]["key"] == "daily.p_win"


@pytest.mark.parametrize("page", ["index.html", "ticker.html"])
def test_pages_do_not_call_the_total_return_probability_beating_cash(page):
    text = (STATIC / page).read_text(encoding="utf-8")
    assert "Chance of beating cash" not in text and ">Up</th>" not in text
    assert 'P(gain)"' not in text


def test_the_trade_page_leads_with_the_tested_entry_not_a_last_close_limit():
    text = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<b>limit ${usd(x.buy_limit)}</b>" not in text
    assert "Weight" in text and "The Trading Floor" in text and "not an order price" in text


def test_a_procedure_id_ignores_refits_and_dollars_but_not_gates():
    base = {"recipe": {"model": "xgb_rank", "params": {"depth": 3}}, "ranking": "ensemble",
            "ensemble": {"members": ["a"]}, "calibration": {"kind": "logistic"}, "gates": {"min_p_win": 0.5},
            "ordering": "score_pct", "sizing": {"cap": 0.25}, "costs": {"rt": "spread"},
            "execution": {"entry": "next_close"}}
    a = MO.procedure_id(**base)
    assert MO.procedure_id(**base) == a
    assert MO.procedure_id(**{**base, "gates": {"min_p_win": 0.55}}) != a
    assert MO.procedure_id(**{**base, "ordering": "expected_net"}) != a
