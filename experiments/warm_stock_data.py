"""Fill the stock module's caches once: SEC facts, splits and dividends, FRED vintages, prices.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/warm_stock_data.py
"""
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.stocks import data as D  # noqa: E402

if __name__ == "__main__":
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    t = time.time()
    got = D.load([], cache, progress=lambda stage, **k: print(f"{time.time() - t:6.0f}s {stage} {k}", flush=True))
    print("stocks", len(got.stocks), "errors", len(got.errors), "rejected", got.rejected)
    for k, v in list(got.errors.items())[:15]:
        print(" ", k, v)
    print("no facts:", sum(1 for f in got.facts.values() if not f))
