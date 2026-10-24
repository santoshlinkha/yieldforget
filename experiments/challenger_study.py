"""The neural (LSTM) and pretrained (Chronos-2) challengers, on the ETF study's folds and scoring.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/challenger_study.py
Same universe, decision dates, labels, yearly refits and costs as forecast_study.py etf,
so the scores compare directly with trials 4-6. Runs on the GPU when there is one
(one job at a time). Chronos-2 is also scored on test dates after its public release
only, since its training data may include these prices.
"""
import json
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent / "output"


def main():
    import torch

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.store import Store
    from tradingagents.optimizer.universe import ETFS
    from tradingagents.portfolio import load_portfolio
    from tradingagents.research import features as F
    from tradingagents.research.foundation import RELEASE, Chronos2
    from tradingagents.research.neural import LSTMForecaster, Sequences
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
    feats = list(F.PRICE_FEATURES) + list(F.MARKET_FEATURES)
    seqs = Sequences(close[tickers], market["SPY"])
    store = Store(Path("my_portfolio.json").resolve().parent)
    summary = {}
    for horizon in (20, 60):
        dates = F.decision_dates(close.index, start="2006-01-01", every=20)
        panel = F.build_panel(close, high, low, volume, market, "BIL", dates, horizon=horizon)
        chronos = Chronos2(close[tickers], device=device)
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        t = time.time()
        result = walk_forward(panel, [lambda: LSTMForecaster(seqs, device=device), lambda: chronos], feats, horizon,
                              "2012-01-01", universe="etf",
                              progress=lambda n, of, s: print(f"h={horizon} block {n}/{of} {s.date()}", flush=True))
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else None
        result.predictions.to_pickle(HERE / f"pred_etf_{horizon}_challengers.pkl")
        config = {"models": ["lstm", "chronos2"], "device": device, "peak_vram_mb": peak, "seconds": time.time() - t,
                  "torch": torch.__version__, "chronos_release": RELEASE}
        trial = record_trial(store.con, result, f"challenger_study:etf:{horizon}", config, code_version())
        pred = result.predictions
        after = pred[pred.index.get_level_values("date") >= pd.Timestamp(RELEASE)]
        post = score(after, "chronos2").__dict__ if after.index.get_level_values("date").nunique() >= 3 else None
        summary[horizon] = {"trial": trial, "config": config,
                            "scores": {m: s.__dict__ for m, s in result.scores.items()},
                            "chronos2_after_release": post}
        for m, s in result.scores.items():
            print(f"h={horizon} {m:9} IC {s.mean_ic:6.3f} ({s.ic_ci[0]:6.3f},{s.ic_ci[1]:6.3f}) "
                  f"xs-cal {s.calibration_xs:5.2f} timing {s.timing_corr:5.2f} cov {s.coverage_90:.2f} "
                  f"brier {s.brier:.4f}/{s.brier_null:.4f} net {s.top_minus_all_net * 100:6.2f}%", flush=True)
        if post:
            print(f"h={horizon} chronos2 after release: IC {post['mean_ic']:.3f} over {post['dates']} dates", flush=True)
        print(f"h={horizon}: {config['seconds']:.0f}s on {device}, peak VRAM {peak}", flush=True)
    (HERE / "challengers.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
