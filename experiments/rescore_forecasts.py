"""Rescore saved walk-forward predictions with the current scoring code, and record them.

The predictions are unchanged (same models, same folds); only the scores are recomputed.
Each record names the trial it rescores, so the registry does not count it as a new experiment.
Run: PYTHONPATH=. .venv/Scripts/python.exe experiments/rescore_forecasts.py etf
"""
import json
import sys
from pathlib import Path

import pandas as pd

from tradingagents.ledger.recommendations import code_version
from tradingagents.ledger.store import Store
from tradingagents.research.walkforward import WalkForwardResult, holm, record_trial, score

HERE = Path(__file__).parent / "output"
universe = sys.argv[1]
store = Store(Path("my_portfolio.json").resolve().parent)
summary = json.loads((HERE / f"forecast_{universe}.json").read_text())
for key, old in summary.items():
    horizon = int(key.split(":")[1])
    pred = pd.read_pickle(HERE / f"pred_{universe}_{horizon}.pkl")
    models = sorted({c.split(":")[0] for c in pred.columns if ":" in c})
    scores = {m: score(pred, m) for m in models}
    for m, p in holm({m: s.p_value for m, s in scores.items()}).items():
        scores[m].p_value = p
    dates = pred.index.get_level_values("date")
    result = WalkForwardResult(horizon, universe, str(dates.min().date()), str(dates.max().date()), pred, scores,
                               old["by_period"])
    trial = record_trial(store.con, result, f"forecast_study:{universe}:{horizon}:rescore",
                         {"rescores_trial": old["trial"]}, code_version())
    for m, s in scores.items():
        print(f"{universe} h={horizon} {m:13} IC {s.mean_ic:6.3f} ({s.ic_ci[0]:6.3f},{s.ic_ci[1]:6.3f}) "
              f"xs-cal {s.calibration_xs:5.2f} timing {s.timing_corr:5.2f} cov {s.coverage_90:.2f} "
              f"brier {s.brier:.4f}/{s.brier_null:.4f} net {s.top_minus_all_net * 100:6.2f}% pHolm {s.p_value:.3f}")
    print(f"recorded trial {trial} (rescores {old['trial']})")
