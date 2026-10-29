"""The internal model's live record: every prediction saved, scored when it matures, and fed back.

**Saved.** Each daily decision (``daily.run``) saves every ranked stock's prediction in
``model_predictions``: rank, score, expected net P&L, P(profit after costs), P(large loss), the
as-traded close and the action, with the fitted model it came from. Before any outcome is known,
and never edited.

**Scored.** Once ``horizon`` sessions have closed after the next session (when the stock could
first be bought), ``resolve`` saves what happened in ``model_outcomes``, twice over:

- the **policy outcome** (``policy_net``, ``policy_exit_on``, ``policy_reason``): the trade the
  model was trained on, run through the same engine as its labels (``positions.trades``: entry at
  the next close, close-confirmed target and stop, the deadline, costs), from the volatility saved
  with the prediction. This is what calibration learns from;
- a **fixed-horizon diagnostic** (``realized_return``, ``won``): the close-to-close return held to
  the deadline, before costs, and whether it beat the round-trip cost. Shown, never calibrated on.

**Fed back.** ``calibration`` fits, for one decision procedure (``procedure_id``) and one knowledge
cutoff, how often a stock with a given raw P(profit) really made money under the policy (isotonic)
and how the policy's net results relate to the raw expected net (a straight line, shrunk to the
original when it is noisy). Only outcomes whose exit close is on or before the cutoff count, so a
replay of an earlier day never sees later results; only predictions of the same procedure count, so
another model's record never corrects this one; and one prediction per (data date, ticker) counts
(the latest run of that day), so running a day again does not weigh its outcomes twice. ``apply``
puts it on today's scores once enough has matured (``MIN_ROWS`` predictions over ``MIN_DATES``
decision days); raw numbers, the calibration version and the corrected numbers are all saved.
``scorecard`` reports the live rank correlation (IC), the top fifth against the rest, and the
calibration table. ``export`` writes all of it to CSV for retraining and research.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ROWS, MIN_DATES = 300, 10
FIELDS = ("rank", "of_n", "score_pct", "expected_net", "p_win", "p_risk", "cost_rt")


def record(con, decision: dict, model_key: str | None = None) -> int:
    """Save every scored stock of a decision (and its lots' actions); returns how many rows."""
    from tradingagents.ledger.db import transaction

    actions = {e["ticker"]: e.get("paper_action") or e["action"] for e in decision.get("entries", [])}
    actions |= {lot["ticker"]: lot.get("action") for lot in decision.get("lots", [])}
    closes = {e["ticker"]: e.get("price") for e in decision.get("entries", [])}
    model = decision.get("model") or {}
    procedure = model.get("procedure_id")
    cal_version = (model.get("live_calibration") or {}).get("version")
    rows = []
    for t, s in (decision.get("scores") or {}).items():
        rows.append((decision["id"], t, decision["info_cutoff"], int(decision.get("horizon") or 20), s.get("rank"),
                     s.get("of"), s.get("score_pct"), s.get("expected_net"), s.get("p_win"), s.get("p_risk"),
                     s.get("cost_rt"), closes.get(t), actions.get(t), model_key,
                     s.get("p_win_raw", s.get("p_win")), s.get("expected_net_raw", s.get("expected_net")),
                     cal_version, s.get("sigma"), procedure))
    with transaction(con):
        con.executemany(
            "INSERT OR IGNORE INTO model_predictions (decision_id, ticker, info_cutoff, horizon, rank, of_n, score_pct,"
            " expected_net, p_win, p_risk, cost_rt, close, action, model_key, p_win_raw, expected_net_raw,"
            " calibration_version, sigma, procedure_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows)
    return len(rows)


def pending(con, last_close: date) -> pd.DataFrame:
    from tradingagents.ledger.sessions import next_session

    rows = pd.read_sql_query(
        "SELECT p.decision_id, p.ticker, p.info_cutoff, p.horizon, p.cost_rt, p.sigma FROM model_predictions p"
        " LEFT JOIN model_outcomes o ON o.decision_id = p.decision_id AND o.ticker = p.ticker"
        " WHERE o.decision_id IS NULL", con)
    if rows.empty:
        return rows
    matures = [next_session(date.fromisoformat(c), int(h) + 1) for c, h in zip(rows["info_cutoff"], rows["horizon"])]
    rows["matures"] = matures
    return rows[rows["matures"] <= last_close]


def _policy_outcomes(todo: pd.DataFrame, engine: dict) -> dict:
    """(decision_id, ticker) -> (net, exit date, reason) of the policy trade, from ``positions.trades``."""
    from .positions import trades

    px, div, low, close, policy = (engine[k] for k in ("px", "div", "low", "close", "policy"))
    ok = (todo["ticker"].isin(px.columns) & pd.to_datetime(todo["info_cutoff"]).isin(px.index)
          & todo["sigma"].notna() & (todo["horizon"] == policy.horizon))
    rows = todo[ok]
    if rows.empty:
        return {}
    cols = sorted(set(rows["ticker"]))
    t = trades(px[cols], low.reindex(index=px.index, columns=cols), close.reindex(index=px.index, columns=cols),
               div[cols], pd.DataFrame({"date": pd.to_datetime(rows["info_cutoff"]).to_numpy(),
                                        "ticker": rows["ticker"].to_numpy(), "sigma": rows["sigma"].astype(float)}),
               policy, rows["cost_rt"].fillna(0.004).to_numpy(dtype=float))
    out = {}
    for key, r in zip(zip(rows["decision_id"], rows["ticker"]), t.itertuples()):
        if r.complete:
            out[key] = (float(r.net), str(pd.Timestamp(r.exit_date).date()), str(r.reason))
    return out


def resolve(con, cache_dir: Path, engine: dict | None = None) -> int:
    """Record what happened to every matured prediction without an outcome; returns how many.

    ``engine``: ``{"px", "div", "low", "close", "policy"}``, the executable prices, dividends, adjusted lows and
    closes and the exit policy the daily decision uses. With it each outcome gets the policy's own result; without
    it only the fixed-horizon diagnostic is saved, and such an outcome never feeds the calibration.
    """
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.research.prices import bars

    spy = bars(["SPY"], cache_dir)["Close"]["SPY"].dropna()
    last = spy.index[-1]
    if engine:   # only what the engine's prices reach: an outcome saved without its policy result stays so
        last = min(last, engine["px"].index[-1])
    todo = pending(con, last.date())
    if todo.empty:
        return 0
    close = bars(sorted({*todo["ticker"], "SPY"}), cache_dir)["Close"]
    sessions = close.index
    policy = _policy_outcomes(todo, engine) if engine else {}
    version = engine["policy"].version() if engine else None
    out = []
    for r in todo.itertuples():
        after = sessions[sessions > pd.Timestamp(r.info_cutoff)]
        if len(after) <= r.horizon or r.ticker not in close:
            continue
        entry_day, exit_day = after[0], after[r.horizon]
        a, b = close.at[entry_day, r.ticker], close.at[exit_day, r.ticker]
        if not (np.isfinite(a) and np.isfinite(b) and a > 0):
            continue
        ret = float(b / a - 1)
        s = float(close.at[exit_day, "SPY"] / close.at[entry_day, "SPY"] - 1)
        pol = policy.get((r.decision_id, r.ticker))
        out.append((r.decision_id, r.ticker, str(exit_day.date()), now_utc(), float(a), float(b), ret, s,
                    int(ret > float(r.cost_rt or 0.0)), version if pol else None, *(pol or (None, None, None))))
    with transaction(con):
        con.executemany(
            "INSERT OR IGNORE INTO model_outcomes (decision_id, ticker, matured_on, recorded_at, entry_close,"
            " exit_close, realized_return, spy_return, won, policy_version, policy_net, policy_exit_on,"
            " policy_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", out)
    return len(out)


def matured(con) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT p.*, o.matured_on, o.realized_return, o.spy_return, o.won FROM model_predictions p"
        " JOIN model_outcomes o ON o.decision_id = p.decision_id AND o.ticker = p.ticker", con)


def _isotonic(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
    grid = np.linspace(0, 1, 21)
    return {"x": grid.round(3).tolist(), "y": np.round(iso.predict(grid), 4).tolist()}


def observations(con, cutoff, procedure_id: str | None) -> pd.DataFrame:
    """The matured policy outcomes calibration may use at ``cutoff`` for ``procedure_id``: exits on or before the
    cutoff, predictions made before it, the same procedure, one per (data date, ticker): the latest run's."""
    m = pd.read_sql_query(
        "SELECT p.rowid AS _rid, p.*, o.policy_net, o.policy_exit_on, o.policy_reason, o.policy_version"
        " FROM model_predictions p JOIN model_outcomes o ON o.decision_id = p.decision_id AND o.ticker = p.ticker"
        " WHERE o.policy_net IS NOT NULL AND o.policy_exit_on <= ? AND p.info_cutoff < ? AND p.procedure_id IS ?",
        con, params=(str(cutoff)[:10], str(cutoff)[:10], procedure_id))
    if m.empty:
        return m
    m = m.sort_values("_rid").drop_duplicates(["info_cutoff", "ticker"], keep="last")
    m["p_win_x"] = m["p_win_raw"].fillna(m["p_win"])
    m["expected_net_x"] = m["expected_net_raw"].fillna(m["expected_net"])
    return m.dropna(subset=["p_win_x", "expected_net_x", "policy_net"])


def calibration(con, cutoff, procedure_id: str | None, min_rows: int = MIN_ROWS, min_dates: int = MIN_DATES) -> dict:
    """What the live record of ``procedure_id``, as known at ``cutoff``, says about its raw numbers; ``active`` only
    with enough matured predictions. Fitted on the policy's own net results, never on the fixed-horizon diagnostic."""
    m = observations(con, cutoff, procedure_id)
    dates = m["info_cutoff"].nunique() if len(m) else 0
    out = {"rows": int(len(m)), "dates": int(dates), "min_rows": min_rows, "min_dates": min_dates,
           "cutoff": str(cutoff)[:10], "procedure_id": procedure_id, "target": "policy net P&L (barrier trade, costs)",
           "active": bool(len(m) >= min_rows and dates >= min_dates)}
    out["version"] = f"live-{procedure_id}-{out['cutoff']}-{len(m)}" if out["active"] else "none"
    if not out["active"]:
        out["note"] = (f"{len(m)} matured predictions of this decision procedure over {dates} decision days known by "
                       f"{out['cutoff']}; the live calibration starts at {min_rows} over {min_dates} days. Until then "
                       "the model's own numbers are used.")
        return out
    won = (m["policy_net"] > 0).to_numpy(dtype=float)
    out["p_win_map"] = _isotonic(m["p_win_x"].to_numpy(), won)
    x, y = m["expected_net_x"].to_numpy(), m["policy_net"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1) if np.std(x) > 0 else (1.0, 0.0)
    # Shrink toward the model's own numbers (slope 1, intercept 0) by how noisy the fit is.
    resid = y - (slope * x + intercept)
    se = float(np.sqrt(resid.var() / max(len(x) * x.var(), 1e-12)))
    weight = float(1 / (1 + (se / max(abs(slope - 1), 1e-6)) ** 2))
    out["net_map"] = {"slope": float(1 + weight * (slope - 1)), "intercept": float(weight * intercept),
                      "raw_slope": float(slope), "raw_intercept": float(intercept), "weight": weight}
    return out


def apply(today: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Today's scores with the live calibration put on P(profit) and expected net; the raw numbers kept beside
    them (``p_win_model``, ``expected_net_model``), and saved as ``p_win_raw`` / ``expected_net_raw``."""
    out = today.copy()
    out["p_win_model"], out["expected_net_model"] = out["p_win"], out["expected_net"]
    if not cal.get("active"):
        return out
    pm = cal["p_win_map"]
    out["p_win"] = np.interp(out["p_win"].astype(float), pm["x"], pm["y"])
    nm = cal["net_map"]
    out["expected_net"] = nm["slope"] * out["expected_net"].astype(float) + nm["intercept"]
    return out


def scorecard(con) -> dict:
    """The live record: rank correlation per decision day, the top fifth against the rest, calibration bins."""
    m = matured(con)
    if m.empty:
        n = con.execute("SELECT COUNT(*), COUNT(DISTINCT info_cutoff) FROM model_predictions").fetchone()
        return {"matured": 0, "saved": int(n[0]), "saved_days": int(n[1]),
                "note": "No prediction has matured yet: each needs its horizon (20 sessions) after the next session."}
    ics, tops = [], []
    for day, g in m.groupby("info_cutoff"):
        g = g.dropna(subset=["score_pct", "realized_return"])
        if len(g) >= 20:
            ics.append({"date": day, "ic": float(g["score_pct"].rank().corr(g["realized_return"].rank())), "n": len(g)})
            top = g["rank"] <= max(1, int(len(g) * 0.2))
            tops.append(float(g.loc[top, "realized_return"].mean() - g.loc[~top, "realized_return"].mean()))
    bins = pd.cut(m["p_win"], [0, .4, .45, .5, .55, .6, .65, 1.0])
    table = m.groupby(bins, observed=True).agg(n=("won", "size"), predicted=("p_win", "mean"), realized=("won", "mean"))
    ic = [x["ic"] for x in ics]
    return {"matured": int(len(m)), "days": len(ics),
            "ic_mean": float(np.mean(ic)) if ic else None,
            "ic_t": float(np.mean(ic) / (np.std(ic, ddof=1) / np.sqrt(len(ic)))) if len(ic) > 2 and np.std(ic) > 0 else None,
            "top_fifth_edge": float(np.mean(tops)) if tops else None, "by_day": ics[-30:],
            "calibration": [{"bin": str(k), **{c: (None if pd.isna(v) else float(v)) for c, v in row.items()}}
                            for k, row in table.iterrows()],
            "calibration_state": calibration(con, date.today(), _latest_procedure(con)),
            "outcome_definitions": {"realized_return": "fixed-horizon diagnostic: close to close over the horizon, "
                                                       "before costs (not the policy's result)",
                                    "policy_net": "the barrier policy's trade, net of costs: what calibration uses"}}


def _latest_procedure(con) -> str | None:
    row = con.execute("SELECT procedure_id FROM model_predictions ORDER BY rowid DESC LIMIT 1").fetchone()
    return row[0] if row else None


def export(con, folder: Path) -> Path:
    """All predictions with their outcomes (blank until matured) as one CSV, for retraining and research."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model_predictions.csv"
    rows = con.execute(
        "SELECT p.*, o.matured_on, o.entry_close, o.exit_close, o.realized_return, o.spy_return, o.won"
        " FROM model_predictions p LEFT JOIN model_outcomes o ON o.decision_id = p.decision_id AND o.ticker = p.ticker"
        " ORDER BY p.info_cutoff, p.rank")
    cols = [d[0] for d in rows.description]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    tmp.replace(path)
    (folder / "README.txt").write_text(
        "model_predictions.csv: every prediction the internal stock model made in a daily decision, saved before the "
        "outcome was known, with the outcome once matured: policy_net is the barrier policy's trade net of costs (what "
        "calibration uses); realized_return is a fixed-horizon diagnostic (hold to the horizon, before costs, from the "
        "close after the cutoff). Use it to score models live, recalibrate them, or add the live period to training.\n"
        f"Calibration in force: {json.dumps(calibration(con, date.today(), _latest_procedure(con)))}\n",
        encoding="utf-8")
    return path
