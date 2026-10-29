"""Were the suggestions right? Every saved pick, followed until its 20 sessions are over.

**The model's picks.** From each saved daily decision (the latest one for each data date: what the page
last showed for that day), the best buy and the runners-up: the candidates that passed the model's gates,
in its order. A pick is bought at the close of the session after the data date (the earliest it could
trade) and judged at the close ``horizon`` sessions later:

- **correct**: the return over those sessions, dividends included, beat the round-trip cost (the model's own
  claim: "profitable after costs"); **wrong** otherwise. Beating SPY over the same days is shown beside it.
- **open**: fewer than ``horizon`` sessions have closed; the return so far is shown, not judged.

**The second model's suggestions** (the policy layer, ``decision_layer``). From the same decisions, the stocks
where its own choice was to buy, best expected excess over cash first (at most five a day). It abstains
("insufficient evidence") until validated, so these are what it *would* have done, recorded and judged anyway:

- ``next_close``: bought at the close after the data date, judged like the model's picks.
- ``limit_dip``: a buy limit at close x exp(-0.5 sigma sqrt(5)) for 5 sessions, filled on the first session whose
  low is 0.1% below it, at the limit; then held 20 sessions from the fill. Not filled: **no fill**, money stays in
  cash, not judged.

**The AI agents' calls.** Each rating on record (the memory log), judged the same way from the session after
its analysis date: Buy and Overweight are correct when the stock rose, Sell and Underweight when it fell
(they say avoid or reduce); Hold is not judged.

**Deleting** a row hides it from this view (``track_record_hidden``); the decision or rating it came from is
never changed, and restoring brings it back. Summaries count only the rows shown.

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
            out.append({"key": f"pick|{cutoff}|{e['ticker']}|{i + 1}",
                        "decision_id": r["id"], "data_to": cutoff, "executes": r["execution_date"], "mode": r["mode"],
                        "ticker": e["ticker"], "place": i + 1, "role": "1st pick" if i == 0 else "runner-up",
                        "expected_net": e.get("expected_net"), "p_win": e.get("p_win"), "cost_rt": e.get("cost_rt") or 0.0,
                        "horizon": int(payload.get("horizon") or 20)})
    return out


def _policy_picks(con, per_day: int = 5) -> list[dict]:
    """The second model's buy choices per data date (its view even while it abstains)."""
    import json

    rows = con.execute("SELECT id, info_cutoff, execution_date, payload_json FROM stock_decisions"
                       " ORDER BY info_cutoff, created_at").fetchall()
    latest = {}
    for r in rows:
        latest[r["info_cutoff"]] = r
    out = []
    for cutoff, r in sorted(latest.items(), reverse=True):
        payload = json.loads(r["payload_json"])
        pv = payload.get("policy") or {}
        if not pv.get("available"):
            continue
        held = {lot["ticker"] for lot in payload.get("lots", [])}
        scores = payload.get("scores") or {}
        spec = (pv.get("policies") or {}).get("limit_dip") or {}
        buys = []
        for t, d in (pv.get("stocks") or {}).items():
            choice = d.get("model_choice")
            if t in held or choice not in ("next_close", "limit_dip"):
                continue
            opt = next((o for o in d.get("options", []) if o.get("action") == choice), {})
            buys.append((opt.get("expected_excess") or 0.0, t, choice, opt, d))
        buys.sort(key=lambda b: (-b[0], b[1]))
        for i, (ex, t, choice, opt, d) in enumerate(buys[:per_day]):
            out.append({"key": f"policy|{cutoff}|{t}", "decision_id": r["id"], "data_to": cutoff,
                        "executes": r["execution_date"], "ticker": t, "place": i + 1, "choice": choice,
                        "expected_excess": ex, "q10_excess": opt.get("q10_excess"), "p_nofill": opt.get("p_nofill"),
                        "shown_as": d.get("action"), "validated": bool(d.get("validated")),
                        "cost_rt": (scores.get(t) or {}).get("cost_rt") or 0.0, "horizon": 20,
                        "limit_k": spec.get("limit_k", 0.5), "limit_days": spec.get("limit_days", 5),
                        "fill_buffer": spec.get("fill_buffer", 0.001)})
    return out


def _follow_limit(close: pd.DataFrame, low: pd.DataFrame, sigma: pd.DataFrame, p: dict) -> dict:
    """A limit below the data date's close, good for ``limit_days`` sessions; then held ``horizon`` from the fill."""
    t = p["ticker"]
    if t not in close or t not in low or t not in sigma:
        return {"status": "no prices"}
    c, lo = close[t].dropna(), low[t].reindex(close.index)
    at = c.index[c.index <= pd.Timestamp(p["data_to"])]
    s = sigma[t].reindex(at).dropna()
    if not len(at) or not len(s):
        return {"status": "no prices"}
    limit = float(c.loc[at[-1]] * np.exp(-p["limit_k"] * s.iloc[-1] * np.sqrt(p["limit_days"])))
    window = c.index[c.index > at[-1]][:p["limit_days"]]
    fill = next((d for d in window if np.isfinite(lo.get(d, np.nan)) and lo[d] <= limit * (1 - p["fill_buffer"])), None)
    out = {"limit": limit}
    if fill is None:
        out["status"] = "no fill" if len(window) == p["limit_days"] else "waiting for a fill"
        out["sessions"] = len(window)
        return out
    path = c.loc[fill:]
    k = min(len(path) - 1, p["horizon"])
    out.update({"entry_date": str(fill.date()), "entry": limit, "sessions": int(k), "last_date": str(path.index[k].date()),
                "last": float(path.iloc[k]), "return": float(path.iloc[k] / limit - 1)})
    if "SPY" in close:
        spy = close["SPY"].dropna().loc[fill:path.index[k]]
        out["spy"] = float(spy.iloc[-1] / spy.iloc[0] - 1) if len(spy) > 1 else 0.0
    out["status"] = "done" if k >= p["horizon"] else "open"
    return out


def hidden(con) -> set[str]:
    try:
        return {r[0] for r in con.execute("SELECT key FROM track_record_hidden")}
    except Exception:   # an older database without the table: nothing hidden
        return set()


def hide(con, key: str) -> None:
    from tradingagents.ledger.db import now_utc, transaction

    if not key:
        raise ValueError("no row to delete")
    with transaction(con):
        con.execute("INSERT OR IGNORE INTO track_record_hidden VALUES (?, ?)", (str(key), now_utc()))


def unhide(con, key: str) -> int:
    from tradingagents.ledger.db import transaction

    with transaction(con):
        return con.execute("DELETE FROM track_record_hidden WHERE key = ?", (str(key),)).rowcount


def restore_all(con) -> int:
    from tradingagents.ledger.db import transaction

    with transaction(con):
        return con.execute("DELETE FROM track_record_hidden").rowcount


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


def _day_weights(picks: list[dict], close: pd.DataFrame) -> dict[str, float]:
    """Each day's split of $1 across its 1st pick and runners-up, by the Trade page's method (``allocation``),
    from prices up to that day's data date only: what the page would have shown then."""
    from .allocation import weights

    out: dict[str, float] = {}
    by_day: dict[str, list[dict]] = {}
    for p in picks:
        by_day.setdefault(p["data_to"], []).append(p)
    for day, rows in by_day.items():
        rows = sorted(rows, key=lambda r: r["place"])
        hist = close.loc[:pd.Timestamp(day)] if not close.empty else None
        w = weights([r["ticker"] for r in rows], [r.get("expected_net") for r in rows], hist)["weights"]
        for r in rows:
            out[r["key"]] = float(w.get(r["ticker"], 0.0))
    return out


def build(con, cache_dir: Path, results_dir: Path | None = None, config: dict | None = None) -> dict:
    from tradingagents.research.prices import bars

    gone = hidden(con)
    picks = [p for p in _picks(con) if p["key"] not in gone]
    policy = [p for p in _policy_picks(con) if p["key"] not in gone]
    agents = []
    if config is not None:
        from tradingagents.memory import TradingMemoryLog

        for e in TradingMemoryLog(config).load_entries():
            if e.get("rating") and e.get("ticker") and e.get("date"):
                row = {"ticker": e["ticker"].upper(), "data_to": str(e["date"])[:10],
                       "rating": str(e["rating"]).title(), "horizon": 20}
                row["key"] = f"agent|{row['data_to']}|{row['ticker']}|{row['rating']}"
                if row["key"] not in gone:
                    agents.append(row)
    tickers = sorted({p["ticker"] for p in picks + policy} | {a["ticker"] for a in agents} | {"SPY"})
    frames = bars(tickers, cache_dir) if len(tickers) > 1 else {"Close": pd.DataFrame(), "Low": pd.DataFrame()}
    close, low = frames["Close"], frames.get("Low", pd.DataFrame())
    from .targets import ewma_vol

    sigma = ewma_vol(close) if not close.empty else pd.DataFrame()
    for p in policy:
        p.update(_follow(close, p["ticker"], p["data_to"], p["horizon"]) if p["choice"] == "next_close"
                 else _follow_limit(close, low, sigma, p))
        if p["status"] == "done":
            p["verdict"] = "correct" if p["return"] - p["cost_rt"] > 0 else "wrong"
            p["beat_spy"] = p.get("spy") is not None and p["return"] > p["spy"]
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

    def summary(rows, key=None, costs=True):
        """Net of the estimated round-trip cost first (the same cost the verdict uses), the raw return beside it,
        the number of distinct data dates (overlapping 20-session holds are not independent) and a standard
        error over those dates."""
        done = [r for r in rows if r.get("verdict") and (key is None or key(r))]
        still_open = sum(1 for r in rows if r.get("status") == "open" and (key is None or key(r)))
        if not done:
            return {"judged": 0, "open": still_open}
        raw = np.array([r["return"] for r in done])
        cost = np.array([float(r.get("cost_rt") or 0.0) for r in done]) if costs else np.zeros(len(done))
        net = raw - cost
        spy = [(n, r["spy"]) for n, r in zip(net, done, strict=True) if r.get("spy") is not None]
        by_date = pd.Series(net, index=[r["data_to"] for r in done]).groupby(level=0).mean()
        se = float(by_date.std(ddof=1) / np.sqrt(len(by_date))) if len(by_date) > 1 else None
        return {"judged": len(done), "correct": sum(r["verdict"] == "correct" for r in done),
                "hit_rate": float(np.mean([r["verdict"] == "correct" for r in done])),
                "avg_net_return": float(net.mean()) if costs else None,
                "avg_net_vs_spy": float(np.mean([n - s_ for n, s_ in spy])) if costs and spy else None,
                "avg_raw_return": float(raw.mean()),
                "avg_raw_vs_spy": float(np.mean([r["return"] - r["spy"] for r in done if r.get("spy") is not None]
                                                or [np.nan])),
                "net_se_by_date": se if costs else None, "dates": int(len(by_date)),
                "costs": "estimated round trip deducted" if costs else "none (a direction call, not a trade)",
                "open": still_open}

    def one_dollar(rows, key=None, costs=True, follows=lambda r: True, stake=lambda r: 1.0):
        """$1 on every recommendation followed exactly (bought at the next close, sold at the 20th close after, less
        the round-trip cost): what was put in, what it is worth, and the profit, for finished and open ones apart.
        Open ones are valued at the latest close. Day by day, the running total of the finished ones."""
        chosen = [r for r in rows if (key is None or key(r)) and follows(r) and r.get("return") is not None
                  and r.get("status") in ("done", "open")]
        net = lambda r: stake(r) * (r["return"] - (float(r.get("cost_rt") or 0.0) if costs else 0.0))  # noqa: E731
        held = lambda r: stake(r) * (r["return"] - (float(r.get("cost_rt") or 0.0) / 2 if costs else 0.0))  # noqa: E731
        done = [r for r in chosen if r["status"] == "done"]
        still = [r for r in chosen if r["status"] == "open"]
        by_day = {}
        for r in done:
            day = by_day.setdefault(r["data_to"], [0.0, 0.0])
            day[0] += stake(r)
            day[1] += net(r)
        running, n, total = [], 0.0, 0.0
        for day in sorted(by_day):
            n, total = n + by_day[day][0], total + by_day[day][1]
            running.append({"date": day, "invested": round(n, 4), "profit": round(total, 4)})
        put_done, put_open = sum(stake(r) for r in done), sum(stake(r) for r in still)
        return {"each_usd": 1.0, "finished": {"trades": len(done), "invested": round(put_done, 4),
                                              "profit": round(sum(net(r) for r in done), 4),
                                              "value": round(put_done + sum(net(r) for r in done), 4)},
                # Still held: valued at the latest close less the entry half of the cost (the exit half is not paid
                # yet), the same rule as the curve.
                "open": {"trades": len(still), "invested": round(put_open, 4),
                         "profit": round(sum(held(r) for r in still), 4),
                         "value": round(put_open + sum(held(r) for r in still), 4)},
                "costs": "estimated round trip deducted" if costs else "none", "running": running}

    def value_curve(rows, key=None, costs=True, follows=lambda r: True, stake=lambda r: 1.0) -> list[dict]:
        """Day by day: dollars put in ($1 on each recommendation from its entry close) and what they are worth. A
        position held is valued at that day's close less half the round-trip cost (the entry half); once sold at its
        20th close it is worth 1 + its return less the whole cost from then on."""
        chosen = [r for r in rows if (key is None or key(r)) and follows(r) and r.get("entry_date")
                  and r.get("status") in ("done", "open") and r["ticker"] in close]
        if not chosen or close.empty:
            return []
        days = close.index
        invested = pd.Series(0.0, index=days)
        value = pd.Series(0.0, index=days)
        for r in chosen:
            cost = float(r.get("cost_rt") or 0.0) if costs else 0.0
            k = stake(r)
            e, last = pd.Timestamp(r["entry_date"]), pd.Timestamp(r["last_date"])
            px = close[r["ticker"]].ffill()
            invested[days >= e] += k
            held = (days >= e) & (days <= last)
            value[held] += k * ((px[held] / r["entry"]).to_numpy() - cost / 2)
            if r["status"] == "done":
                value[days == last] += -k * cost / 2             # sold that close: the exit half of the cost
                value[days > last] += k * (1.0 + r["return"] - cost)
        on = invested > 0
        return [{"date": str(d.date()), "invested": round(float(i), 2), "value": round(float(v), 4)}
                for d, i, v in zip(days[on], invested[on], value[on], strict=True)]

    buys = lambda r: r.get("rating") in UP  # noqa: E731   (the agents' Buy/Overweight calls: $1 bought on each)
    weight = _day_weights(picks, close)
    split = lambda r: weight.get(r["key"], 0.0)  # noqa: E731   ($1 a day split by the Trade page's weights)
    dollars = {"best_buy": one_dollar(picks, lambda r: r["place"] == 1), "all_picks": one_dollar(picks),
               "weighted": one_dollar(picks, stake=split),
               "policy": one_dollar(policy), "agents_buys": one_dollar(agents, costs=False, follows=buys)}
    curves = {"best_buy": value_curve(picks, lambda r: r["place"] == 1), "all_picks": value_curve(picks),
              "weighted": value_curve(picks, stake=split),
              "policy": value_curve(policy), "agents_buys": value_curve(agents, costs=False, follows=buys)}
    for p in picks:
        p["weight"] = weight.get(p["key"])
    for k, c in curves.items():
        dollars[k]["curve"] = c

    return {"picks": picks, "policy": policy, "agents": sorted(agents, key=lambda a: a["data_to"], reverse=True),
            "one_dollar": dollars,
            "hidden": len(gone),
            "summary": {"best_buy": summary(picks, lambda r: r["place"] == 1), "all_picks": summary(picks),
                        "policy": summary(policy), "agents": summary(agents, costs=False)},
            "outcome": "fixed-horizon diagnostic: held to the 20th close after entry (the limit record from its fill), "
                       "not the barrier policy's result and not an account; see model_outputs record.fixed_horizon_*",
            "rules": {"correct_pick": "the return over 20 sessions from the next close beat the round-trip cost",
                      "correct_policy": "the same, from the next close or from the limit fill; a limit that never "
                                        "filled is not judged",
                      "correct_agent": "Buy/Overweight: the stock rose over 20 sessions; Sell/Underweight: it fell; "
                                       "Hold: not judged"}}
