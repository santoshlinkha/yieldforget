"""CPU vs GPU for the stock models on a real panel: time and numerical agreement.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/stock_gpu_benchmark.py
"""
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates
    from tradingagents.stocks import data as D
    from tradingagents.stocks import macro as M
    from tradingagents.stocks import panel as P
    from tradingagents.stocks.models import XGBEvent

    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    data = D.load([], cache)
    out = {"gpu": cuda_usable()}
    for h in (5, 20):
        dates = decision_dates(data.close.index, start="2006-01-01", every=P.every_for(h))
        t = time.time()
        panel = P.build(data, h, dates=dates, macro=M.features(dates, cache))
        out[f"panel_{h}_seconds"] = round(time.time() - t, 1)
        train = panel[panel["in_index"] & panel["up_break"].notna()]
        feats = P.features_for()
        res = {}
        for device in ("cpu", "cuda") if out["gpu"] else ("cpu",):
            t = time.time()
            m = XGBEvent(device).fit(train, feats, "up_break")
            res[device] = {"seconds": round(time.time() - t, 1), "rounds": m.best_rounds,
                           "p": m.predict_proba(train.iloc[-20000:])}
        row = {"rows": len(train), "features": len(feats),
               **{d: {"seconds": r["seconds"], "rounds": r["rounds"]} for d, r in res.items()}}
        if "cuda" in res:
            row["max_abs_prob_diff"] = float(np.max(np.abs(res["cpu"]["p"] - res["cuda"]["p"])))
            row["mean_abs_prob_diff"] = float(np.mean(np.abs(res["cpu"]["p"] - res["cuda"]["p"])))
        out[f"h{h}_up_break"] = row
        print(json.dumps(out, indent=1), flush=True)
    (Path(__file__).parent / "output" / "stock_gpu_benchmark.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
