"""Were the suggestions right? Every saved pick, followed until its 20 sessions are over.

**The model's picks.** From each saved daily decision (the latest one for each data date: what the page
last showed for that day), the best buy and the runners-up: the candidates that passed the model's gates,
in its order. A pick is bought at the close of the session after the data date (the earliest it could
trade) and judged at the close ``horizon`` sessions later:

- **correct**: the return over those sessions, dividends included, beat the round-trip cost (the model's own
  claim: "profitable after costs"); **wrong** otherwise. Beating SPY over the same days is shown beside it.
- **open**: fewer than ``horizon`` sessions have closed; the return so far is shown, not judged.

**The AI agents' calls.** Each rating on record (the memory log), judged the same way from the session after
its analysis date: Buy and Overweight are correct when the stock rose, Sell and Underweight when it fell
(they say avoid or reduce); Hold is not judged.

Prices are the local store's adjusted closes (total return). Nothing here changes any record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UP, DOWN = ("Buy", "Overweight"), ("Sell", "Underweight")


def _picks(con, per_day: int = 5) -> list[dict]:
    rows = con.execute("SELECT id, created_at, info_cutoff, execution_date, mode, payload_json FROM stock_decisions"
                       " ORDER BY info_cutoff, created_at").fetchall()
    latest = {}
    for r in rows:   # the last decision saved for each data date
        latest[r["info_cutoff"]] = r
    import json

    out = []
    for cutoff, r in sorted(latest.items(), reverse=True):
        payload = json.loads(r["payload_json"])
        passing = [e for e in payload.get("entries", []) if e.get("paper_action") or e.get("action") == "ENTER"]
        for i, e in enumerate(passing[:per_day]):
            out.append({"decision_id": r["id"], "data_to": cutoff, "executes": r["execution_date"], "mode": r["mode"],
                        "ticker": e["ticker"], "place": i + 1, "role": "best buy" if i == 0 else "runner-up",
                        "expected_net": e.get("expected_net"), "p_win": e.get("p_win"), "cost_rt": e.get("cost_rt") or 0.0,
                        "horizon": int(payload.get("horizon") or 20)})
    return out


def _follow(close: pd.DataFrame, ticker: str, after: str, horizon: int) -> dict:
    """Bought at the close of the first session after ``after``; held ``horizon`` sessions (or to the latest close)."""
    if ticker not in close:
        return {"status": "no prices"}
    px = close[ticker].dropna()
    later = px.index[px.index > pd.Timestamp(after)]
    if not len(later):
        return {"status": "not bought yet", "sessions": 0}
    entry_day = later[0]
    path = px.loc[entry_day:]
    k = min(len(path) - 1, horizon)
    out = {"entry_date": str(entry_day.date()), "entry": float(path.iloc[0]), "sessions": int(k),
           "last_date": str(path.index[k].date()), "last": float(path.iloc[k]),
           "return": float(path.iloc[k] / path.iloc[0] - 1)}
    if "SPY" in close:
        spy = close["SPY"].dropna()
        s = spy.loc[entry_day:path.index[k]]
        out["spy"] = float(s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 else 0.0
    out["status"] = "done" if k >= horizon else "open"
    return out


def build(con, cache_dir: Path, results_dir: Path | None = None, config: dict | None = None) -> dict:
    from tradingagents.research.prices import bars

    picks = _picks(con)
    agents = []
    if config is not None:
        from tradingagents.memory import TradingMemoryLog

        for e in TradingMemoryLog(config).load_entries():
            if e.get("rating") and e.get("ticker") and e.get("date"):
                agents.append({"ticker": e["ticker"].upper(), "data_to": str(e["date"])[:10],
                               "rating": str(e["rating"]).title(), "horizon": 20})
    tickers = sorted({p["ticker"] for p in picks} | {a["ticker"] for a in agents} | {"SPY"})
    close = bars(tickers, cache_dir)["Close"] if len(tickers) > 1 else pd.DataFrame()
    for p in picks:
        p.update(_follow(close, p["ticker"], p["data_to"], p["horizon"]))
        if p["status"] == "done":
            p["verdict"] = "correct" if p["return"] - p["cost_rt"] > 0 else "wrong"
            p["beat_spy"] = p.get("spy") is not None and p["return"] > p["spy"]
    for a in agents:
        a.update(_follow(close, a["ticker"], a["data_to"], a["horizon"]))
        if a["status"] == "done" and (a["rating"] in UP or a["rating"] in DOWN):
            rose = a["return"] > 0
            a["verdict"] = "correct" if rose == (a["rating"] in UP) else "wrong"

    def summary(rows, key=None):
        done = [r for r in rows if r.get("verdict") and (key is None or key(r))]
        if not done:
            return {"judged": 0, "open": sum(1 for r in rows if r.get("status") == "open" and (key is None or key(r)))}
        rets = np.array([r["return"] for r in done])
        return {"judged": len(done), "correct": sum(r["verdict"] == "correct" for r in done),
                "hit_rate": float(np.mean([r["verdict"] == "correct" for r in done])),
                "avg_return": float(rets.mean()),
                "avg_vs_spy": float(np.mean([r["return"] - r["spy"] for r in done if r.get("spy") is not None] or [np.nan])),
                "open": sum(1 for r in rows if r.get("status") == "open" and (key is None or key(r)))}

    return {"picks": picks, "agents": sorted(agents, key=lambda a: a["data_to"], reverse=True),
            "summary": {"best_buy": summary(picks, lambda r: r["place"] == 1), "all_picks": summary(picks),
                        "agents": summary(agents)},
            "rules": {"correct_pick": "the return over 20 sessions from the next close beat the round-trip cost",
                      "correct_agent": "Buy/Overweight: the stock rose over 20 sessions; Sell/Underweight: it fell; "
                                       "Hold: not judged"}}
