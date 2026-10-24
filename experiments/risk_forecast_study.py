"""Volatility and covariance forecasters, scored out of sample on the ETF universe.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/risk_forecast_study.py
Monthly decision dates from 2010; 20-session horizon; close-to-close realized variance.
"""
import json
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent / "output"
HERE.mkdir(exist_ok=True)


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.optimizer.universe import ETFS
    from tradingagents.portfolio import load_portfolio
    from tradingagents.research import features as F
    from tradingagents.research.prices import bars
    from tradingagents.research.risk_models import covariance_study, score_vol, vol_forecasts

    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    held = [p.ticker.upper() for p in load_portfolio("my_portfolio.json").positions]
    tickers = [t for t in dict.fromkeys([*ETFS, *held]) if t not in ("SGOV", "BIL")]
    close = bars(tickers, cache)["Close"]
    close = close[close.index >= pd.Timestamp("2000-01-01")]
    dates = F.decision_dates(close.index, start="2010-01-01")
    out = {}
    for h in (5, 20, 60):
        t = time.time()
        frame = vol_forecasts(close, dates, h)
        out[f"vol_h{h}"] = score_vol(frame)
        print(f"vol h={h}: {time.time() - t:.0f}s", json.dumps(out[f"vol_h{h}"]), flush=True)
    t = time.time()
    out["covariance_h20"] = covariance_study(close, dates, 20)
    print(f"covariance: {time.time() - t:.0f}s", json.dumps(out["covariance_h20"]), flush=True)
    (HERE / "risk_forecasts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
