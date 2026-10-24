"""Walk-forward study: worst-case vs crash-scenario risk model, risk levels and growth targets.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/risk_model_study.py
Writes experiments/output/study.json; the 2026-10-23 results are in docs/BACKTEST_RISK_MODELS.md.
"""
import json
import pickle
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent / "output"
HERE.mkdir(exist_ok=True)
START, EVERY = "2008-06-02", 21
PERIODS = [("2008-06-02", "2012-12-31", "2008-12 (GFC, euro crisis)"),
           ("2013-01-01", "2019-12-31", "2013-19 (calm bull)"),
           ("2020-01-01", "2026-12-31", "2020-26 (COVID, 2022 rates)")]
STRATEGIES = ([(m, lim, None) for m in ("worst_case", "scenarios") for lim in (0.025, 0.10, 0.20, 0.30)]
              + [(m, 0.30, g) for m in ("worst_case", "scenarios") for g in (0.06, 0.08, 0.10)])


def load():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.optimizer.data import fetch_closes
    from tradingagents.optimizer.universe import ETFS
    from tradingagents.portfolio import load_portfolio
    held = [p.ticker.upper() for p in load_portfolio("my_portfolio.json").positions]
    candidates = list(dict.fromkeys([*ETFS, *held]))
    closes = fetch_closes([*candidates, "QQQ", "VBR", "BIL", "SGOV", "VTI", "BND", "SPY", "TLT", "GLD", "DBC"],
                          required=["BIL"], cache_dir=Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer")
    return closes, [t for t in candidates if t in closes], held


def work(args):
    model, limit, target = args
    from tradingagents.optimizer.backtest import rebalance_plan
    from tradingagents.optimizer.plan import OptimizerSettings
    closes, candidates = pickle.load(open(HERE / "closes.pkl", "rb"))
    settings = OptimizerSettings(loss_limit=limit, risk_model=model, target_growth=target)
    days = list(closes.index[closes.index >= pd.Timestamp(START)])[::EVERY]
    t = time.time()
    schedule, history = rebalance_plan(closes, [c for c in candidates if c not in ("BIL", "SGOV")], settings, days)
    print(f"done {args} in {time.time() - t:.0f}s", flush=True)
    return args, schedule, history


def main():
    from tradingagents.optimizer.backtest import _metrics, _simulate, run_backtest
    from tradingagents.optimizer.plan import OptimizerSettings
    from tradingagents.portfolio import load_portfolio

    closes, candidates, held = load()
    pickle.dump((closes, candidates), open(HERE / "closes.pkl", "wb"))
    book = load_portfolio("my_portfolio.json")
    prices = {t: float(closes[t].dropna().iloc[-1]) for t in held if t in closes}
    values = {p.ticker.upper(): p.quantity * prices.get(p.ticker.upper(), 0.0) for p in book.positions}
    total = sum(values.values())
    mix = {t: v / total for t, v in values.items() if t != "SGOV" and v > 0}

    with Pool(min(len(STRATEGIES), 11)) as pool:
        planned = pool.map(work, STRATEGIES, chunksize=1)

    returns = closes.pct_change(fill_method=None)
    cash = returns["BIL"].fillna(0.0)
    out = []
    for (model, limit, target), schedule, history in planned:
        settings = OptimizerSettings(loss_limit=limit, risk_model=model, target_growth=target)
        row = {"model": model, "limit": limit, "target": target}
        for bps in (0, 10):
            res = run_backtest(closes, candidates, settings, start=START, every=EVERY, current_mix=mix,
                               cost_bps=bps, schedule=schedule, history=[*history])
            row[f"bt{bps}"] = asdict(res)
            row[f"breaches{bps}"] = len(res.breaches)
        daily, _ = _simulate(returns[returns.index >= pd.Timestamp(START)], cash, schedule, 0.0)
        row["periods"] = {}
        for a, b, name in PERIODS:
            seg = daily.loc[a:b]
            if len(seg) > 60:
                row["periods"][name] = asdict(_metrics(seg, cash))
        out.append(row)
    # Benchmarks by period, from the last strategy's run (identical for all).
    bench = {}
    r = returns[returns.index >= pd.Timestamp(START)]
    days = list(r.index)[::EVERY]
    for name, m in {"VTI only": {"VTI": 1.0}, "60/40 (VTI/BND)": {"VTI": 0.6, "BND": 0.4}, "Your mix now": mix,
                    "Cash (BIL)": {}}.items():
        fixed = [(d, {t: w for t, w in m.items() if r[t].loc[:d].notna().any()}) for d in days]
        curve, _ = _simulate(r, cash, fixed)
        bench[name] = {"all": asdict(_metrics(curve, cash)),
                       **{n: asdict(_metrics(curve.loc[a:b], cash)) for a, b, n in PERIODS}}
    json.dump({"strategies": out, "benchmarks": bench, "start": START, "every": EVERY, "mix": mix},
              open(HERE / "study.json", "w"), default=str, indent=1)
    print("wrote", HERE / "study.json")


if __name__ == "__main__":
    main()
