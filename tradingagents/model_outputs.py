"""What every model output means: one registry the API and the pages read their labels from.

The app has several model families whose numbers look alike but are not: a forecast's mean is a
log return over cash from today's close, the daily decision's expected net is an arithmetic
barrier-policy P&L after costs from the next close, an ensemble percentile is a rank, and a
"probability" may be of a positive total return, of a profitable trade, or of a large loss.
Each ``Output`` states exactly what is measured, so a page can name it precisely instead of
"forecast", "Up" or "correct", and so two numbers are only put in one column when they measure
the same thing.

Each ``Family`` states its lifecycle (what it is used for today) and its evidence status. The
lifecycle words are the ones the code inventory uses:

- ``active_decision``: decides what the Trade page proposes;
- ``active_context``: shown beside decisions, decides nothing;
- ``research``: run on request or in experiments, not in the daily workflow;
- ``compatibility``: kept for old records or callers, not for new work;
- ``archive``: kept to read history only;
- ``uncalled``: no production caller found (a library or test surface).

Identities are separate on purpose: a *training recipe* (how a model is fitted), a *decision
procedure* (recipe plus ranking, ensemble, calibration, gates, ordering, sizing, costs and
execution: what evidence is about) and an individual *fitted artifact* (one refit's hash). A
scheduled refit under the same procedure keeps the procedure's evidence; a change to the
procedure needs new evidence. ``procedure_id`` hashes the procedure's parts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

LIFECYCLES = ("active_decision", "active_context", "research", "compatibility", "archive", "uncalled")
EVIDENCE = ("none", "development", "development_and_prospective", "validated")


@dataclass(frozen=True)
class Output:
    key: str                    # stable id, e.g. "forecast.mean"
    label: str                  # the words a page shows
    target: str                 # what is predicted
    unit: str                   # "log return", "arithmetic return", "probability", "percentile rank", ...
    benchmark: str | None       # what it is measured against, if anything
    horizon_sessions: int | None
    entry: str                  # when the measured holding starts
    exit: str                   # when it ends
    costs: str                  # how trading costs are treated
    calibration: str            # how (if at all) it is calibrated
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Family:
    key: str
    name: str
    lifecycle: str
    consumer: str               # where its outputs are actually used
    outputs: tuple[str, ...]    # Output keys
    evidence: str
    information_cutoff: str     # what the model may know
    source: str                 # where in the code
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


H = 20
CLOSE_T = "the close of the information cutoff session (t)"
NEXT_CLOSE = "the next session's close (t+1)"
BARRIER_EXIT = "the barrier policy's exit: a close through the target or stop, or the 20-session deadline"

OUTPUTS: dict[str, Output] = {o.key: o for o in [
    # --- forecast engine (Forecasts tab, ticker page, fund verdicts) ---
    Output("forecast.mean", "Expected log return over cash", "total log return minus the cash (T-bill) log return",
           "log return", "cash (3-month T-bill)", H, CLOSE_T, "the close h sessions later", "none",
           "none (point estimate)"),
    Output("forecast.interval", "5th-95th percentile range (log return over cash)",
           "quantiles of the log return over cash", "log return", "cash (3-month T-bill)", H, CLOSE_T,
           "the close h sessions later", "none",
           "residual quantiles from a held-back slice before the training end; coverage is checked in walk-forward"),
    Output("forecast.p_total_positive", "P(total return > 0)",
           "probability the total return, including the cash yield, is above zero", "probability", None, H, CLOSE_T,
           "the close h sessions later", "none", "empirical, from held-back standardized errors",
           "not the chance of beating cash: that event needs the excess return above zero"),
    # --- daily stock decision (Trade, Decisions, ticker) ---
    Output("daily.score_pct", "Model score percentile (100 = top)", "the ranker's percentile among today's S&P 500 "
           "members", "percentile rank", None, H, NEXT_CLOSE, BARRIER_EXIT, "not applicable", "none: a rank, not a "
           "probability"),
    Output("daily.expected_net", "Expected net P&L of the trade (after costs)",
           "arithmetic net return of the barrier-policy trade", "arithmetic return", None, H, NEXT_CLOSE, BARRIER_EXIT,
           "estimated round-trip cost deducted", "score-percentile bins mapped to out-of-fold trade outcomes"),
    Output("daily.p_win", "P(trade profitable after costs)", "probability the barrier-policy trade's net return is "
           "above zero", "probability", None, H, NEXT_CLOSE, BARRIER_EXIT, "estimated round-trip cost deducted",
           "logistic on the score percentile, calibrated out of fold"),
    Output("daily.p_risk", "P(large loss at the exit close)", "probability of the large-loss event at the exit "
           "close", "probability", None, H, NEXT_CLOSE, BARRIER_EXIT, "not applicable", "XGBoost classifier; flagged "
           "uninformative when it gives every member the same value"),
    Output("daily.expected_excess_vs_spy", "Expected excess over SPY (trade)", "net trade return minus SPY over the "
           "same dates", "arithmetic return", "SPY", H, NEXT_CLOSE, BARRIER_EXIT, "estimated round-trip cost deducted",
           "score-percentile bins"),
    # --- model zoo (Models tab, screener, ticker) ---
    Output("zoo.pct", "Percentile among today's stocks (100 = top)", "each model's rank of the stock", "percentile rank",
           None, H, NEXT_CLOSE, "the close 20 sessions after entry (t+21)", "not applicable", "none: a rank"),
    Output("zoo.forecast_log_excess", "Model's own forecast: log return over cash",
           "20-session log return over cash, entry at the next close", "log return", "cash (3-month T-bill)", H,
           NEXT_CLOSE, "the close at t+21", "none", "none (regression output)",
           "Chronos rows forecast from the current close (t) instead; see zoo.chronos_log_return"),
    Output("zoo.chronos_log_return", "Chronos forecast: log return from the current close",
           "median of the 20-step-ahead price path relative to the last close", "log return", None, H, CLOSE_T,
           "the close 20 sessions later (t+20)", "none", "none (pretrained, zero-shot)",
           "a different clock from the trade labels (t+1 to t+21); compare ranks, not returns"),
    Output("zoo.ensemble_expected_net", "Ensemble expected net P&L (barrier trade, after costs)",
           "the learned-weight ensemble's percentile mapped to barrier-policy net P&L", "arithmetic return", None, H,
           NEXT_CLOSE, BARRIER_EXIT, "estimated round-trip cost deducted", "percentile bins on out-of-fold outcomes"),
    Output("zoo.mean_pct", "Equal-weight mean percentile (N models)", "the plain average of the fitted models' "
           "percentiles that scored the stock", "percentile rank", None, H, NEXT_CLOSE, "the close at t+21",
           "not applicable", "none", "not the learned ensemble; N can differ by stock"),
    # --- policy outcome layer ---
    Output("policy.expected_excess", "Expected excess over cash of the policy (after costs)",
           "arithmetic return of the policy's trade minus cash", "arithmetic return", "cash (3-month T-bill)", H,
           "the policy's entry (next close, or a limit fill)", "the policy's exit", "estimated costs deducted",
           "LightGBM regression, out of fold"),
    Output("policy.q10_excess", "Bad case: 10th percentile of excess over cash", "10th percentile of the policy's "
           "excess return", "arithmetic return", "cash (3-month T-bill)", H, "the policy's entry", "the policy's exit",
           "estimated costs deducted", "quantile model; a buy needs a finite value"),
    Output("policy.p_stop", "P(stop hit first)", "probability the stop is reached before the target or deadline",
           "probability", None, H, "the policy's entry", "the policy's exit", "not applicable", "calibrated classifier",
           "not P(losing money)"),
    # --- sector model ---
    Output("sector.expected_log_excess_spy", "Expected log return over SPY (sector)", "sector ETF log return minus "
           "SPY's", "log return", "SPY", H, NEXT_CLOSE, "the close at t+21", "none", "LambdaRank and regression"),
    # --- track record ---
    Output("record.fixed_horizon_net", "Fixed-horizon diagnostic: net return at the deadline close",
           "close-to-close return from the entry close to the deadline close, minus the round-trip cost",
           "arithmetic return", None, H, NEXT_CLOSE, "the deadline close (no target or stop)",
           "estimated round-trip cost deducted", "not applicable",
           "a diagnostic, not the policy's result: the policy can exit earlier"),
    Output("record.fixed_horizon_raw", "Fixed-horizon diagnostic: raw return (before costs)",
           "close-to-close return from the entry close to the deadline close", "arithmetic return", None, H,
           NEXT_CLOSE, "the deadline close (no target or stop)", "none", "not applicable"),
]}

FAMILIES: dict[str, Family] = {f.key: f for f in [
    Family("forecast_engine", "Forecast engine (zero, training mean, volatility prior, momentum, Elastic Net, "
           "XGBoost, Chronos-2, '+chronos' variants)", "active_context",
           "Forecasts tab, ticker page forecasts, fund ranks in verdicts",
           ("forecast.mean", "forecast.interval", "forecast.p_total_positive"), "development",
           "closes and features through the run's cutoff", "tradingagents/research/engine.py, research/forecast.py"),
    Family("stock_signals", "Stock signal pipeline (trend, breakout, breakdown, loss, drawdown classifiers; "
           "return quantiles)", "active_context", "Signals, screener, entry plans, ticker",
           (), "development", "through the cutoff close", "tradingagents/stocks/pipeline.py",
           "event definitions differ per model; each probability names its own event"),
    Family("daily_decision", "Daily stock decision (XGBoost ranker or the ensemble, policy P&L map, trade quality, "
           "position risk)", "active_decision", "Trade, Decisions, ticker verdicts",
           ("daily.score_pct", "daily.expected_net", "daily.p_win", "daily.p_risk", "daily.expected_excess_vs_spy"),
           "development", "features through the cutoff close; entry at the next close", "tradingagents/stocks/daily.py"),
    Family("model_zoo", "Model zoo and learned ensemble (XGBoost x2, LightGBM x2, Ridge x2, Mamba, Chronos-2 x2, "
           "baselines)", "active_decision", "Models tab, screener, ticker; the ensemble can replace the daily ranker",
           ("zoo.pct", "zoo.forecast_log_excess", "zoo.chronos_log_return", "zoo.ensemble_expected_net", "zoo.mean_pct"),
           "development", "through the cutoff close", "tradingagents/stocks/model_zoo.py",
           "the ensemble decides only when the run for that close has it; otherwise the XGBoost ranker does"),
    Family("policy_outcomes", "Policy outcome layer (LightGBM plus linear/logistic baselines)", "active_context",
           "Trade 'policy model' panel, ticker", ("policy.expected_excess", "policy.q10_excess", "policy.p_stop"),
           "development", "through the cutoff close", "tradingagents/stocks/outcome_models.py, decision_layer.py",
           "abstains unless validated"),
    Family("sector_model", "Sector model (LightGBM LambdaRank and regression)", "active_context", "sector view",
           ("sector.expected_log_excess_spy",), "development", "through the cutoff close",
           "tradingagents/research/sector_model.py"),
    Family("risk_research", "Risk research (trailing/EWMA/GARCH volatility; sample/Ledoit-Wolf/factor covariance)",
           "research", "experiments/risk_forecast_study.py", (), "development", "through each study date",
           "tradingagents/research/risk_models.py", "does not configure production risk"),
    Family("sequence_challengers", "Sequence and pretrained challengers (TimesFM 2.5, Kronos-mini, PatchTST, LSTM)",
           "research", "experiments/challenger_*_study.py", (), "development", "through each study date",
           "experiments/challenger_stock_study.py", "pretrained weights may have seen the test period"),
    Family("llm_roles", "LLM agent team (4 analysts, bull/bear, research manager, trader, 3 risk roles, portfolio "
           "manager, reflector)", "active_context", "agent reports, ratings shown beside the quant decision",
           (), "none", "the evidence packet's cutoff", "tradingagents/llm_clients/roles.py",
           "13 roles often share one model; their opinions never decide the quant action"),
    Family("text_and_quality", "Sentiment (FinBERT or word list), financial quality checks, moat rating, screening",
           "active_context", "screener, ticker", (), "none", "statements known by the cutoff",
           "tradingagents/stocks/news.py, research/quality.py, moat.py",
           "classifiers and rules, not validated return signals"),
    Family("optimizer_tracker", "Retired optimizer tracker and auto-invest", "uncalled", "tests only", (), "none",
           "n/a", "tradingagents/optimizer/tracker.py, optimizer/auto_invest.py", "kept until archived deliberately"),
]}


def output(key: str) -> dict:
    return OUTPUTS[key].as_dict()


def measures(kind: str) -> dict:
    """The outputs behind a verdict's ``expected_net`` and ``p_win`` fields, by score kind ('stock' or 'fund')."""
    if kind == "fund":
        return {"expected": output("forecast.mean"), "probability": output("forecast.p_total_positive")}
    return {"expected": output("daily.expected_net"), "probability": output("daily.p_win")}


def catalog() -> dict:
    return {"outputs": {k: o.as_dict() for k, o in OUTPUTS.items()},
            "families": {k: f.as_dict() for k, f in FAMILIES.items()},
            "lifecycles": list(LIFECYCLES), "evidence_levels": list(EVIDENCE)}


def procedure_id(*, recipe: dict, ranking: str, ensemble: dict | None, calibration: dict | None, gates: dict,
                 ordering: str, sizing: dict, costs: dict, execution: dict, risk: dict | None = None,
                 features: list[str] | None = None) -> str:
    """A decision procedure's identity: everything that changes what evidence is about, and nothing that changes
    with a routine refit (no fitted weights, no dates, no dollar amounts)."""
    parts = {"recipe": recipe, "ranking": ranking, "ensemble": ensemble, "calibration": calibration, "gates": gates,
             "ordering": ordering, "sizing": sizing, "costs": costs, "execution": execution, "risk": risk,
             "features": sorted(features or [])}
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Identity:
    """The three identities of one decision: how models are trained, the procedure evidence is about, and the
    fitted artifacts this decision used."""
    recipe_id: str
    procedure_id: str
    artifacts: dict = field(default_factory=dict)     # name -> sha256 of each fitted model/map/weights used

    def as_dict(self) -> dict:
        return asdict(self)
