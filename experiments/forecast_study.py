"""Walk-forward comparison of return forecasters on two universes and three horizons.

Run from the TradingAgents folder:
    PYTHONPATH=. .venv/Scripts/python.exe experiments/forecast_study.py etf
    PYTHONPATH=. .venv/Scripts/python.exe experiments/forecast_study.py sp500

Settings are fixed here, before results are seen, and every run is written to
the trial registry (research_trials in portfolio.sqlite3), whatever it finds.

- etf: the 39 long-lived ETFs plus your holdings; price, volume and market features.
  One pooled model across ETFs. Tests from 2012.
- sp500: today's S&P 500 members, plus Yahoo's dated analyst actions and earnings
  surprises. One pooled stock model. **Survivorship-biased**: today's members are
  the companies that survived and grew, so absolute returns are flattered. Rank
  comparisons between models are less affected, but still optimistic. Tests from 2015,
  so the analyst history (from about 2012) has years to train on.

Horizons 5, 20 and 60 sessions; decisions every min(horizon, 20) sessions; refit yearly;
10 bp per side on top-fifth turnover.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent / "output"
HERE.mkdir(exist_ok=True)
HORIZONS = (5, 20, 60)
FIRST_TEST = {"etf": "2012-01-01", "sp500": "2015-01-01"}


def build(universe: str):
    from tradingagents.dataflows.vendors.index_members import get_sp500_members
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.optimizer.universe import ETFS
    from tradingagents.portfolio import load_portfolio
    from tradingagents.research import features as F
    from tradingagents.research.prices import MARKET, bars

    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    if universe == "etf":
        held = [p.ticker.upper() for p in load_portfolio("my_portfolio.json").positions]
        tickers = [t for t in dict.fromkeys([*ETFS, *held]) if t not in ("SGOV", "BIL")]
    else:
        tickers = sorted(set(get_sp500_members(cache)))
    b = bars([*tickers, "BIL", *MARKET], cache)
    close, high, low, volume = (b[f][[*tickers, "BIL"]] for f in ("Close", "High", "Low", "Volume"))
    market = {m: b["Close"][m].dropna() for m in MARKET if b["Close"][m].notna().any()}
    close = close[close.index >= pd.Timestamp("2004-01-01")]
    return close, high, low, volume, market


def run(args):
    universe, horizon = args
    from tradingagents.research import features as F
    from tradingagents.research.models import XGB, ElasticNet, Momentum, TrainMean, VolPrior, Zero
    from tradingagents.research.walkforward import walk_forward

    close, high, low, volume, market = build(universe)
    close, high, low, volume = (x.reindex(close.index) for x in (close, high, low, volume))
    dates = F.decision_dates(close.index, start="2006-01-01", every=min(horizon, 20))
    feats = list(F.PRICE_FEATURES) + list(F.MARKET_FEATURES)
    analyst = earnings = None
    if universe == "sp500":
        from tradingagents.research.yahoo_extra import earnings as get_earnings
        from tradingagents.research.yahoo_extra import upgrades as get_upgrades

        analyst, earnings = {}, {}
        for t in [c for c in close.columns if c != "BIL"]:
            try:
                analyst[t] = F.analyst_features(get_upgrades(t), close[t], dates)
                earnings[t] = F.earnings_features(get_earnings(t), close.index, dates)
            except Exception as exc:  # a ticker without Yahoo analyst data keeps its price features
                print(f"{t}: no analyst/earnings data ({str(exc)[:60]})", flush=True)
        feats += list(F.ANALYST_FEATURES) + list(F.EARNINGS_FEATURES)
    t0 = time.time()
    panel = F.build_panel(close, high, low, volume, market, "BIL", dates, horizon=horizon,
                          analyst=analyst, earnings=earnings)
    print(f"{universe} h={horizon}: panel {panel.shape} in {time.time() - t0:.0f}s", flush=True)
    factories = [Zero, TrainMean, VolPrior, Momentum, ElasticNet, lambda: XGB(), lambda: XGB("rank"),
                 lambda: XGB("quantile")]
    result = walk_forward(panel, factories, feats, horizon, FIRST_TEST[universe], universe=universe,
                          progress=lambda n, of, start: print(f"{universe} h={horizon} block {n}/{of} {start.date()}",
                                                                 flush=True))
    result.predictions.to_pickle(HERE / f"pred_{universe}_{horizon}.pkl")
    return universe, horizon, result, feats, panel.shape


def main():
    from dataclasses import asdict

    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.store import Store
    from tradingagents.research.walkforward import record_trial

    universe = sys.argv[1]
    build(universe)   # fill the price store once, before the workers read it
    with Pool(len(HORIZONS)) as pool:
        results = pool.map(run, [(universe, h) for h in HORIZONS])
    store = Store(Path("my_portfolio.json").resolve().parent)
    summary = {}
    for universe, horizon, result, feats, shape in results:
        config = {"features": feats, "first_test": FIRST_TEST[universe], "every": min(horizon, 20),
                  "refit": "yearly", "cost_bps": 10, "rows": shape[0]}
        trial = record_trial(store.con, result, f"forecast_study:{universe}:{horizon}", config, code_version())
        summary[f"{universe}:{horizon}"] = {"trial": trial, "first_test": result.first_test,
                                            "last_test": result.last_test, "rows": shape[0],
                                            "seconds": result.seconds,
                                            "scores": {m: asdict(s) for m, s in result.scores.items()},
                                            "by_period": result.by_period,
                                            "gain_last_block": next((e["fit"].get("gain") for e in reversed(result.fit_log)
                                                                     if e["model"] == "xgb"), None)}
    (HERE / f"forecast_{universe}.json").write_text(json.dumps(summary, indent=1, default=str))
    print("wrote", HERE / f"forecast_{universe}.json")


if __name__ == "__main__":
    main()
