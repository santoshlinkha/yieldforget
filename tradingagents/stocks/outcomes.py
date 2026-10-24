"""What happened after each saved stock forecast, recorded once its horizon has passed.

A forecast made at cutoff t for ``h`` sessions matures when the close of t+1+h is in
(the tradable return enters at t+1). Its outcome is computed with the same target
definitions and the thresholds version the run used, from today's price history,
and saved once in ``stock_outcomes`` next to the forecast, never instead of it.
Old forecasts are never regenerated: the saved forecast is what gets scored.

``scorecard`` summarizes the matured forecasts per horizon: the rank correlation of
expected and realized returns on each run date (IC), how often the realized return
fell inside the 5-95% range, and for each event the Brier score against the
training base rate saved with the run. A few run dates say little; the count is shown.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .targets import EVENTS, Thresholds, labels


def pending(con, last_close: date) -> list[dict]:
    from tradingagents.ledger.sessions import next_session

    rows = con.execute(
        "SELECT f.run_id, f.ticker, f.horizon, f.payload_json, r.info_cutoff, r.thresholds_json FROM stock_forecasts f"
        " JOIN stock_runs r ON r.id = f.run_id LEFT JOIN stock_outcomes o"
        " ON o.run_id = f.run_id AND o.ticker = f.ticker AND o.horizon = f.horizon WHERE o.run_id IS NULL").fetchall()
    out = []
    for r in rows:
        matures = next_session(date.fromisoformat(r["info_cutoff"]), r["horizon"] + 1)
        if matures <= last_close:
            out.append({**dict(r), "matures": matures})
    return out


def resolve(con, cache_dir: Path) -> int:
    """Record the outcome of every forecast that has matured and has none yet; returns how many."""
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.research.prices import bars

    from .data import CASH

    spy = bars(["SPY"], cache_dir)["Close"]["SPY"].dropna()
    todo = pending(con, spy.index[-1].date())
    if not todo:
        return 0
    tickers = sorted({r["ticker"] for r in todo})
    b = bars([*tickers, "SPY", CASH], cache_dir)
    written = []
    for (cutoff, thr_json), group in pd.DataFrame(todo).groupby(["info_cutoff", "thresholds_json"]):
        th = Thresholds(**json.loads(thr_json))
        when = pd.Timestamp(cutoff)
        if when not in b["Close"].index:
            continue
        for h, g in group.groupby("horizon"):
            names = [t for t in g["ticker"] if t in b["Close"].columns]
            if not names:
                continue
            lab = labels(b["Close"][names], b["High"][names], b["Low"][names], b["Close"][CASH], b["Close"]["SPY"],
                         pd.DatetimeIndex([when]), int(h), th)
            for _, r in g.iterrows():
                if (when, r["ticker"]) not in lab.index:
                    continue
                real = lab.loc[(when, r["ticker"])]
                fc = (json.loads(r["payload_json"]) or {}).get("forecast") or {}
                realized = {k: (None if pd.isna(real[k]) else float(real[k]))
                            for k in ("label", "label_rel", "ret_raw", "drawdown", "trend_state", "failed_up",
                                      "failed_down", *EVENTS)}
                if fc:
                    realized["inside_5_95"] = (None if realized["label"] is None or fc.get("q05") is None
                                               else bool(fc["q05"] <= realized["label"] <= fc["q95"]))
                written.append((r["run_id"], r["ticker"], int(h), str(r["matures"]), now_utc(), json.dumps(realized)))
    with transaction(con):
        con.executemany("INSERT OR IGNORE INTO stock_outcomes VALUES (?, ?, ?, ?, ?, ?)", written)
    return len(written)


def scorecard(con) -> dict:
    """The matured forecasts scored, per horizon."""
    rows = con.execute(
        "SELECT o.run_id, o.ticker, o.horizon, o.realized_json, f.payload_json, f.status, r.info_cutoff"
        " FROM stock_outcomes o JOIN stock_forecasts f USING (run_id, ticker, horizon)"
        " JOIN stock_runs r ON r.id = o.run_id").fetchall()
    out = {}
    frame = []
    for r in rows:
        real = json.loads(r["realized_json"])
        payload = json.loads(r["payload_json"]) or {}
        fc = payload.get("forecast") or {}
        frame.append({"run": r["run_id"], "date": r["info_cutoff"], "h": r["horizon"], "status": r["status"],
                      "forecast": fc.get("mean"), "realized": real.get("label"), "inside": real.get("inside_5_95"),
                      **{f"p_{e}": fc.get(f"p_{e}") for e in EVENTS}, **{e: real.get(e) for e in EVENTS},
                      **{f"base_{e}": (payload.get("base_rates") or {}).get(e) for e in EVENTS}})
    if not frame:
        return {"matured": 0}
    f = pd.DataFrame(frame)
    for h, g in f.groupby("h"):
        ok = g[(g["status"] == "ok") & g["forecast"].notna() & g["realized"].notna()]
        ics = [x["forecast"].rank().corr(x["realized"].rank()) for _, x in ok.groupby("run") if len(x) >= 5]
        ics = [v for v in ics if np.isfinite(v)]
        events = {}
        for e in EVENTS:
            sub = ok[ok[f"p_{e}"].notna() & ok[e].notna()]
            if len(sub):
                y = sub[e].astype(float)
                events[e] = {"n": len(sub), "brier": float(((sub[f"p_{e}"] - y) ** 2).mean()),
                             "brier_base": float(((sub[f"base_{e}"] - y) ** 2).mean()), "happened": float(y.mean())}
        out[int(h)] = {"matured": int(len(g)), "abstained": int((g["status"] != "ok").sum()), "run_dates": int(ok["run"].nunique()),
                       "ic_mean": float(np.mean(ics)) if ics else None, "ic_dates": len(ics),
                       "inside_5_95": float(ok["inside"].dropna().astype(float).mean()) if ok["inside"].notna().any() else None,
                       "events": events}
    return out
