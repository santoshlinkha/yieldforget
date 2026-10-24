"""Mode A: does a forecaster make the optimizer's portfolio better, after costs?

Run after forecast_study.py etf, from the TradingAgents folder:
    PYTHONPATH=. .venv/Scripts/python.exe experiments/forecast_portfolio_study.py

At each monthly rebalance the optimizer gets the forecasts the walk-forward made with
data up to that day (its out-of-sample predictions for the latest prediction date
on or before the rebalance), clipped to two predictive sigmas and **not** shrunk by
any calibration measured later. Everything else is the usual backtest: the ETF
universe, the worst-case stress limit, monthly rebalancing at the close,
10 bp per dollar traded. Compared with the current rule (vol_prior) and with
holding VTI, SPY or SGOV. Risk levels 4 (10%), 7 (30%) and 10 (no stress cap).
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent / "output"
MODELS = ("vol_prior", "momentum", "elastic_net", "xgb", "xgb_rank")
# "rel:" variants use the forecast only *between* assets: everyone gets the average
# return above cash among labels already resolved at the rebalance, plus the forecast's
# difference from that date's average forecast (slope 1, nothing fitted on later data).
VARIANTS = (*MODELS, "rel:momentum", "rel:elastic_net", "rel:xgb", "rel:xgb_quantile")
LIMITS = (0.10, 0.30, None)
START = "2012-02-01"


def load():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.optimizer.data import fetch_closes
    from tradingagents.optimizer.universe import ETFS
    from tradingagents.portfolio import load_portfolio

    held = [p.ticker.upper() for p in load_portfolio("my_portfolio.json").positions]
    candidates = [t for t in dict.fromkeys([*ETFS, *held]) if t not in ("SGOV", "BIL")]
    closes = fetch_closes([*candidates, "QQQ", "VBR", "BIL", "SGOV", "VTI", "SPY", "TLT", "GLD", "DBC"],
                          required=["BIL"], cache_dir=Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer")
    return closes, [t for t in candidates if t in closes]


def run(args):
    variant, limit = args
    relative = variant.startswith("rel:")
    model = variant.removeprefix("rel:")
    from tradingagents.optimizer.backtest import rebalance_plan, run_backtest
    from tradingagents.optimizer.plan import OptimizerSettings

    closes, candidates = load()
    pred = pd.read_pickle(HERE / "pred_etf_20.pkl")
    settings = OptimizerSettings(loss_limit=limit, return_model="forecast" if model != "vol_prior" else "vol_prior",
                                 forecast_horizon=20)
    pdates = np.sort(pred.index.get_level_values("date").unique())

    def mu_for(day):
        known = pdates[pdates <= np.datetime64(day)]
        if not len(known):
            return {}
        rows = pred.xs(known[-1], level="date")
        mean, sigma = rows[f"{model}:mean"], (rows[f"{model}:q95"] - rows[f"{model}:q05"]) / 3.29
        if relative:
            # Labels resolved by now: prediction dates at least 31 calendar days back (20 sessions).
            done = pred[pred.index.get_level_values("date") <= pd.Timestamp(day) - pd.Timedelta(days=31)]
            premium = float(done["label"].mean()) if len(done) else 0.0
            mean = premium + (mean - mean.mean())
        return dict(np.clip(mean, -2 * sigma, 2 * sigma).items())

    days = list(closes.index[closes.index >= pd.Timestamp(START)])[::21]
    schedule, history = rebalance_plan(closes, candidates, settings, days,
                                       mu_for=mu_for if model != "vol_prior" else None)
    result = run_backtest(closes, candidates, settings, start=START, every=21, cost_bps=10,
                          schedule=schedule, history=history)
    m = result.metrics
    return {"model": variant, "limit": limit, "final": m["Optimizer"].final, "cagr": m["Optimizer"].cagr,
            "max_drawdown": m["Optimizer"].max_drawdown, "volatility": m["Optimizer"].volatility,
            "sharpe": m["Optimizer"].sharpe, "turnover": result.turnover,
            "invested": float(np.mean([h.invested for h in result.history])),
            "benchmarks": {k: {"final": v.final, "cagr": v.cagr, "max_drawdown": v.max_drawdown}
                           for k, v in m.items() if k != "Optimizer"}}


def main():
    load()   # fill the price store once, before the workers read it
    with Pool(6) as pool:
        rows = pool.map(run, [(m, lim) for lim in LIMITS for m in VARIANTS])
    (HERE / "forecast_portfolio.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['model']:17} limit {str(r['limit']):5} ${r['final']:.2f} per $1  {r['cagr']:.1%}/yr  "
              f"maxDD {r['max_drawdown']:.1%}  sharpe {r['sharpe']:.2f}  turnover {r['turnover']:.1f}/yr  "
              f"invested {r['invested']:.0%}")
    b = rows[0]["benchmarks"]
    print({k: f"${v['final']:.2f} {v['cagr']:.1%}/yr maxDD {v['max_drawdown']:.1%}" for k, v in b.items()})


if __name__ == "__main__":
    main()
