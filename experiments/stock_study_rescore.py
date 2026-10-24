"""Rescore the stock study from its saved predictions (no refitting), and record the result as new trials.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/stock_study_rescore.py
Used after a scoring fix: 2026-10-24, the Brier-skill interval was stored high-first.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, str(Path(__file__).parent))
import stock_study as S  # noqa: E402


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.store import Store
    from tradingagents.stocks import data as D

    data = D.load([], Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer")
    store = Store(Path("my_portfolio.json").resolve().parent)
    for h in (5, 20, 60):
        old = json.loads((S.HERE / f"stock_study_{h}.json").read_text())
        pred = pd.read_pickle(S.HERE / f"stock_pred_{h}.pkl")
        summary = S.summarize(h, pred, data, [])
        summary["volatility"], summary["seconds"] = old.get("volatility"), old.get("seconds")
        summary["fits"], summary["fit_seconds"] = old.get("fits"), old.get("fit_seconds")
        summary["rescored"] = "Brier-skill interval order fixed; ranked_dates added; predictions unchanged"
        with transaction(store.con):
            store.con.execute(
                "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
                " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now_utc(), f"stock_study:h{h}", "stock_pit", h, S.EDGES[0], str(pd.Timestamp.now().date()),
                 json.dumps({"rescore_of": f"stock_study:h{h}", "note": summary["rescored"]}), code_version(),
                 json.dumps(summary, default=str), "{}", summary["seconds"] or 0.0))
        (S.HERE / f"stock_study_{h}.json").write_text(json.dumps(summary, indent=1, default=str))
        print("rescored", h, flush=True)


if __name__ == "__main__":
    main()
