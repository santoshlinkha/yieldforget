"""Fill the local store with Yahoo's dated analyst actions and earnings for the S&P 500 and your holdings.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/fetch_yahoo_histories.py
Each ticker is fetched at most weekly; a rerun only fills what is missing or stale.
"""
import sys
import time
from pathlib import Path

from tradingagents.dataflows.vendors.index_members import get_sp500_members
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.research.yahoo_extra import earnings, upgrades

cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
symbols = sorted(set(get_sp500_members(cache)))
ok = failed = 0
start = time.time()
for n, s in enumerate(symbols, 1):
    try:
        upgrades(s, cache)
        earnings(s, cache)
        ok += 1
    except Exception as exc:
        failed += 1
        print(f"{s}: {str(exc)[:120]}", flush=True)
    if n % 50 == 0:
        print(f"{n}/{len(symbols)} done, {failed} failed, {time.time() - start:.0f}s", flush=True)
print(f"finished: {ok} ok, {failed} failed in {time.time() - start:.0f}s", flush=True)
