"""Does Chronos-2's forecast help the other models as an extra input? (stacking)

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/stacking_study.py
Same ETF universe, decision dates (every 20 sessions from 2006), labels, yearly refits,
first test year (2012) and costs as forecast_study.py etf and challenger_study.py. Each
model runs twice on the same panel: with the price and market inputs only, and with
Chronos-2's median forecast and its 5-95% range added ("+chronos"). Both are recorded as
trials, whatever they show. Chronos-2 may have seen these prices in pretraining
(contamination risk unknown), which would flatter the "+chronos" rows; scores on test
dates after its release (October 2025) are reported separately for that reason.
"""
import json
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent / "output"


def named(model, name):
    model.name = name
    return model


def main():
    import torch

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.store import Store
    from tradingagents.optimizer.universe import ETFS
    from tradingagents.portfolio import load_portfolio
    from tradingagents.research import features as F
    from tradingagents.research.foundation import FEATURES, RELEASE, chronos_features
    from tradingagents.research.models import XGB, ElasticNet
    from tradingagents.research.prices import MARKET, bars
    from tradingagents.research.walkforward import record_trial, score, walk_forward

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    held = [p.ticker.upper() for p in load_portfolio("my_portfolio.json").positions]
    tickers = [t for t in dict.fromkeys([*ETFS, *held]) if t not in ("SGOV", "BIL")]
    b = bars([*tickers, "BIL", *MARKET], cache)
    close = b["Close"][[*tickers, "BIL"]]
    close = close[close.index >= pd.Timestamp("2004-01-01")]
    high, low, volume = (b[f][[*tickers, "BIL"]].reindex(close.index) for f in ("High", "Low", "Volume"))
    market = {m: b["Close"][m].dropna() for m in MARKET if b["Close"][m].notna().any()}
    base = list(F.PRICE_FEATURES) + list(F.MARKET_FEATURES)
    store = Store(Path("my_portfolio.json").resolve().parent)
    summary = {}
    for horizon in (20, 60):
        dates = F.decision_dates(close.index, start="2006-01-01", every=20)
        panel = F.build_panel(close, high, low, volume, market, "BIL", dates, horizon=horizon)
        t = time.time()
        panel = panel.join(chronos_features(close[tickers], panel, horizon, device))
        print(f"h={horizon}: Chronos-2 inputs for {len(panel)} rows in {time.time() - t:.0f}s", flush=True)
        results = {}
        for tag, feats in (("base", base), ("chronos", base + FEATURES)):
            suffix = "" if tag == "base" else "+chronos"
            factories = [lambda s=suffix: named(ElasticNet(), "elastic_net" + s),
                         lambda s=suffix: named(XGB(), "xgb" + s),
                         lambda s=suffix: named(XGB("rank"), "xgb_rank" + s),
                         lambda s=suffix: named(XGB("quantile"), "xgb_quantile" + s)]
            result = walk_forward(panel, factories, feats, horizon, "2012-01-01", universe="etf",
                                  progress=lambda n, of, s, tag=tag: print(f"h={horizon} {tag} block {n}/{of} {s.date()}",
                                                                           flush=True))
            config = {"features": feats, "first_test": "2012-01-01", "every": 20, "refit": "yearly", "cost_bps": 10,
                      "device": device, "chronos_release": RELEASE}
            trial = record_trial(store.con, result, f"stacking_study:etf:{horizon}:{tag}", config, code_version())
            pred = result.predictions
            after = pred[pred.index.get_level_values("date") >= pd.Timestamp(RELEASE)]
            results[tag] = {"trial": trial, "scores": {m: s.__dict__ for m, s in result.scores.items()},
                            "after_release": {m: score(after, m).__dict__ for m in result.scores}
                            if len(after) else {}}
        summary[str(horizon)] = results
        for tag in results:
            for m, s in results[tag]["scores"].items():
                print(f"h={horizon} {m:22} IC {s['mean_ic']:+.3f} ({s['ic_ci'][0]:+.3f},{s['ic_ci'][1]:+.3f}) "
                      f"net {s['top_minus_all_net']:+.4f} p {s['p_value']:.3f}", flush=True)
    (HERE / "stacking.json").write_text(json.dumps(summary, indent=1, default=str))
    print("wrote", HERE / "stacking.json")


if __name__ == "__main__":
    main()
