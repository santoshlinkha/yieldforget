"""Run the policy study (``tradingagents.stocks.policy_study``) and write its report.

    .venv/Scripts/python.exe -m experiments.policy_study [--device cuda]

Reads the local price store only (no paid API). Writes a ``research_trials`` row (``policy_study:h20``),
the live model bundle under the cache's ``decision_models`` and ``docs/POLICY_STUDY.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ledger.store import Store
    from tradingagents.stocks import policy_study as S
    from tradingagents.stocks.report_policy import write

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--book", default=".")
    args = ap.parse_args()
    t0 = time.time()
    store = Store(Path(args.book))
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    report = S.run(store, cache, args.device, lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True))
    path = write(report, Path("docs/POLICY_STUDY.md"))
    print(json.dumps({"status": report["status"], "trial": report["trial"], "report": str(path)}, indent=1))


if __name__ == "__main__":
    main()
