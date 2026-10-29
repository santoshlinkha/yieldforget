"""Outcome models, the decision layer and the study's purging: calibration boundaries, abstention, held positions."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import policy_study as S
from tradingagents.stocks.decision_layer import LayerConfig, decide
from tradingagents.stocks.outcome_models import OutcomeModel, brier

pytestmark = pytest.mark.unit
FEATS = ["f1", "f2"]


def _rows(n=6000, seed=0, policy="next_close"):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    p_t = 1 / (1 + np.exp(-(f1 - 0.5)))
    u = rng.random(n)
    outcome = np.where(u < p_t * 0.7, "target", np.where(u < 0.85, "stop", "expiry"))
    if policy == "limit_dip":
        outcome = np.where(rng.random(n) < 0.3, "nofill", outcome)
    excess = np.where(outcome == "target", 0.05, np.where(outcome == "stop", -0.05, 0.0)) + rng.normal(0, .01, n)
    excess = np.where(outcome == "nofill", 0.0, excess)
    return pd.DataFrame({"f1": f1, "f2": f2, "outcome": outcome, "excess": excess})


@pytest.mark.parametrize("kind", ["lgbm", "linear"])
@pytest.mark.parametrize("policy", ["next_close", "limit_dip"])
def test_models_learn_calibrated_probabilities_that_sum_to_one(kind, policy):
    tr, cal, te = _rows(8000, 1, policy), _rows(6000, 2, policy), _rows(4000, 3, policy)
    m = OutcomeModel(kind, policy).fit(tr, cal, FEATS)
    p = m.predict(te)
    cols = [c for c in p if c.startswith("p_")]
    assert np.allclose(p[cols].sum(axis=1), 1.0)
    assert m.cal_method == "isotonic"                                # 6,000 calibration rows
    y = te["outcome"].map({c: i for i, c in enumerate(m.classes)}).to_numpy()
    base = np.tile(m.base, (len(te), 1))
    assert brier(p[cols].to_numpy(), y) < brier(base, y)             # beats the base rate when signal exists
    assert (p["q10_excess"] <= p["mean_excess"]).all()


def test_a_small_calibration_slice_uses_sigmoid_and_a_tiny_one_none():
    tr = _rows(6000, 1)
    assert OutcomeModel("linear", "next_close").fit(tr, _rows(1000, 2), FEATS).cal_method == "sigmoid"
    assert OutcomeModel("linear", "next_close").fit(tr, _rows(100, 2), FEATS).cal_method.startswith("none")


def test_the_split_purges_rows_whose_labels_reach_the_test_period_and_calibrates_on_later_rows():
    d = pd.bdate_range("2010-01-01", "2016-12-31", freq="10B")
    idx = pd.MultiIndex.from_product([d, ["A"]], names=["date", "ticker"])
    rows = pd.DataFrame(index=idx)
    ends = d + pd.Timedelta(days=35)
    lab = pd.DataFrame({"label_end_date": ends, "outcome": "expiry"}, index=idx)
    start = pd.Timestamp("2016-01-01")
    fit, cal = S.split(rows, lab, start)
    assert not (fit & cal).any()
    assert (ends[fit | cal] < start).all()                             # nothing overlaps the test year
    assert (ends[fit] < start - pd.DateOffset(years=2)).all()          # the fit ends before calibration starts
    assert d[cal].min() >= start - pd.DateOffset(years=2)
    gap = d[(d < start) & ~(fit | cal)]
    assert len(gap) and (ends[(d < start) & ~(fit | cal)] >= start - pd.DateOffset(years=2)).all()


VALID = {"validated": True, "state": "validated"}
P = {"next_close": {"mean_excess": 0.02, "q10_excess": -0.06, "p_target": 0.4, "p_stop": 0.3},
     "limit_dip": {"mean_excess": 0.012, "q10_excess": -0.05, "p_nofill": 0.4}}


def test_the_layer_abstains_without_validation_or_with_stale_data_but_keeps_its_view():
    d = decide(P, held=False, cost_rt=0.002, status={"validated": False, "state": "not validated", "why": "no skill"},
               cfg=LayerConfig())
    assert d["action"] == "insufficient evidence" and d["model_choice"] == "next_close" and "no skill" in d["reasons"][0]
    assert decide(P, False, 0.002, VALID, LayerConfig(), stale_days=9)["action"] == "insufficient evidence"
    assert decide({}, False, 0.002, VALID, LayerConfig())["action"] == "insufficient evidence"


def test_no_trade_region_and_risk_budget_leave_cash():
    small = {k: {**v, "mean_excess": 0.003} for k, v in P.items()}
    assert decide(small, False, 0.002, VALID, LayerConfig(margin=0.005))["action"] == "cash"
    risky = {"next_close": {**P["next_close"], "q10_excess": -0.30}, "limit_dip": {**P["limit_dip"], "mean_excess": 0.0}}
    d = decide(risky, False, 0.002, VALID, LayerConfig())
    assert d["action"] == "cash" and any("risk budget" in b for o in d["options"] for b in o.get("blocked", []))
    assert decide(P, False, 0.002, VALID, LayerConfig())["action"] == "next_close"


def test_a_held_position_is_judged_from_now_not_from_its_purchase_price():
    weak = {"next_close": {"mean_excess": -0.02, "q10_excess": -0.1}}
    assert decide(weak, True, 0.002, VALID, LayerConfig())["action"] == "sell"
    meh = {"next_close": {"mean_excess": -0.003, "q10_excess": -0.1}}     # inside the no-trade region
    assert decide(meh, True, 0.002, VALID, LayerConfig())["action"] == "hold"


def test_the_vectorized_choice_matches_the_layer():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), t) for t in "ABCD"], names=["date", "ticker"])
    nc = pd.DataFrame({"mean_excess": [0.02, 0.001, 0.02, 0.01], "q10_excess": [-0.05, -0.05, -0.5, -0.05]}, index=idx)
    lim = pd.DataFrame({"mean_excess": [0.01, 0.002, 0.012, 0.03], "q10_excess": [-0.05] * 4}, index=idx)
    cfg = LayerConfig(margin=0.005)
    got = S.choose(nc, lim, 0.005, cfg)["pick"].tolist()
    want = [decide({"next_close": nc.iloc[i].to_dict(), "limit_dip": lim.iloc[i].to_dict()}, False, 0.0, VALID, cfg)["action"]
            for i in range(4)]
    assert got == want == ["next_close", "cash", "limit_dip", "limit_dip"]


def test_a_date_with_fewer_picks_than_slots_holds_cash_in_the_rest():
    d = pd.Timestamp("2020-01-01")
    idx = pd.MultiIndex.from_tuples([(d, "A"), (d, "B")], names=["date", "ticker"])
    picks = pd.DataFrame({"pick": ["next_close", "cash"], "score": [0.02, -np.inf]}, index=idx)
    s = S.per_date(picks, {"next_close": pd.Series([0.10, 0.5], index=idx)})
    assert s.loc[d] == pytest.approx(0.10 / 5)


def test_the_promotion_rule_refuses_without_skill_or_interval():
    rep = {"classification": {"lgbm:next_close:holdout": {
        "per_class": {"target": {"skill": 0.01}, "stop": {"skill": -0.002}}, "q10_coverage": 0.11}},
        "strategies": {"lgbm_layer:holdout": {"mean_excess": 0.002, "ci95": [-0.001, 0.005]},
                       "existing_rule:holdout": {"mean_excess": 0.001}}}
    st = S.status_from(rep)
    assert not st["validated"] and "stop" in st["why"] and "exclude zero" in st["why"]


def test_agents_get_model_evidence_only_when_it_is_validated_and_never_a_conclusion():
    from tradingagents.stocks.decision_layer import agent_evidence

    row = decide(P, False, 0.002, VALID, LayerConfig())
    dec = {"policy": {"available": True, "model_version": "v1", "status": {**VALID, "evaluated": "2026-10-27"},
                      "stocks": {"AAA": row}}}
    text = agent_evidence(dec, "aaa")
    assert "expected excess over cash" in text and "weigh them against your own analysis" in text
    assert not any(w in text.lower() for w in ("you should", "recommend", "must buy", "conclude"))
    unvalidated = {"policy": {**dec["policy"], "status": {"validated": False}}}
    assert agent_evidence(unvalidated, "AAA") == "" and agent_evidence(None, "AAA") == ""


@pytest.mark.parametrize("q10", [None, float("nan"), float("inf"), float("-inf")])
def test_a_buy_needs_a_finite_downside_estimate_unless_the_bypass_is_on_and_shown(q10):
    blind = {"next_close": {"mean_excess": 0.02, "q10_excess": q10}, "limit_dip": {"mean_excess": 0.01, "q10_excess": q10}}
    d = decide(blind, False, 0.002, VALID, LayerConfig())
    assert d["action"] == "cash" and not any(o.get("eligible") for o in d["options"] if o["action"] != "cash")
    assert all("no downside estimate" in " ".join(o["blocked"]) for o in d["options"] if o["action"] != "cash")
    bypass = decide(blind, False, 0.002, VALID, LayerConfig(allow_missing_downside=True))
    assert bypass["action"] == "next_close"
    assert next(o for o in bypass["options"] if o["action"] == "next_close")["risk_bypass"]
