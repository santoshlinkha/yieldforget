"""The internal model's live record: every prediction saved, scored when it matures, and fed back.

**Saved.** Each daily decision (``daily.run``) saves every ranked stock's prediction in
``model_predictions``: rank, score, expected net P&L, P(profit after costs), P(large loss), the
as-traded close and the action, with the fitted model it came from. Before any outcome is known,
and never edited.

**Scored.** Once ``horizon`` sessions have closed after the next session (when the stock could
first be bought), ``resolve`` saves what happened in ``model_outcomes``: the total return from
that close to ``horizon`` sessions later, SPY's over the same days, and whether it beat the
round-trip cost. A hold to the deadline, not the barrier policy's stop and target, so it is an
approximation of the trade the model sizes; it is the same for every prediction.

**Fed back.** ``calibration`` fits, from the matured predictions only, how often a stock with a
given P(profit) really made money (isotonic, by prediction) and how realized returns relate to
the expected ones (a straight line, shrunk to the original when it is noisy). ``apply`` puts
those on today's scores once enough has matured (``MIN_ROWS`` predictions over ``MIN_DATES``
decision days); the model's own numbers stay beside them. ``scorecard`` reports the live rank
correlation (IC), the top fifth against the rest, and the calibration table. ``export`` writes
all of it to CSV for retraining and research.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# About two years of matured decision days: a calibration fitted on a few weeks follows one market spell
# (it is what made P(loss) and the edge swing day to day); over two years it averages across regimes.
MIN_ROWS, MIN_DATES = 20000, 500
FIELDS = ("rank", "of_n", "score_pct", "expected_net", "p_win", "p_risk", "cost_rt")


def record(con, decision: dict, model_key: str | None = None) -> int:
    """Save every scored stock of a decision (and its lots' actions); returns how many rows."""
    from tradingagents.ledger.db import transaction

    actions = {e["ticker"]: e.get("paper_action") or e["action"] for e in decision.get("entries", [])}
    actions |= {lot["ticker"]: lot.get("action") for lot in decision.get("lots", [])}
    closes = {e["ticker"]: e.get("price") for e in decision.get("entries", [])}
    rows = []
    for t, s in (decision.get("scores") or {}).items():
        rows.append((decision["id"], t, decision["info_cutoff"], int(decision.get("horizon") or 20), s.get("rank"),
                     s.get("of"), s.get("score_pct"), s.get("expected_net"), s.get("p_win"), s.get("p_risk"),
                     s.get("cost_rt"), closes.get(t), actions.get(t), model_key))
    with transaction(con):
        con.executemany("INSERT OR IGNORE INTO model_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def pending(con, last_close: date) -> pd.DataFrame:
    from tradingagents.ledger.sessions import next_session

    rows = pd.read_sql_query(
        "SELECT p.decision_id, p.ticker, p.info_cutoff, p.horizon, p.cost_rt FROM model_predictions p"
        " LEFT JOIN model_outcomes o ON o.decision_id = p.decision_id AND o.ticker = p.ticker"
        " WHERE o.decision_id IS NULL", con)
    if rows.empty:
        return rows
    matures = [next_session(date.fromisoformat(c), int(h) + 1) for c, h in zip(rows["info_cutoff"], rows["horizon"])]
    rows["matures"] = matures
    return rows[rows["matures"] <= last_close]


def resolve(con, cache_dir: Path) -> int:
    """Record what happened to every matured prediction without an outcome; returns how many."""
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.research.prices import bars

    spy = bars(["SPY"], cache_dir)["Close"]["SPY"].dropna()
    todo = pending(con, spy.index[-1].date())
    if todo.empty:
        return 0
    close = bars(sorted({*todo["ticker"], "SPY"}), cache_dir)["Close"]
    sessions = close.index
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
        out.append((r.decision_id, r.ticker, str(exit_day.date()), now_utc(), float(a), float(b), ret, s,
                    int(ret > float(r.cost_rt or 0.0))))
    with transaction(con):
        con.executemany("INSERT OR IGNORE INTO model_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", out)
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


def calibration(con, min_rows: int = MIN_ROWS, min_dates: int = MIN_DATES) -> dict:
    """What the live record says about the model's numbers; ``active`` only with enough matured predictions."""
    m = matured(con).dropna(subset=["p_win", "expected_net", "realized_return"])
    dates = m["info_cutoff"].nunique() if len(m) else 0
    out = {"rows": int(len(m)), "dates": int(dates), "min_rows": min_rows, "min_dates": min_dates,
           "active": bool(len(m) >= min_rows and dates >= min_dates)}
    if not out["active"]:
        out["note"] = (f"{len(m)} matured predictions over {dates} decision days; the live calibration starts at "
                       f"{min_rows} over {min_dates} days. Until then the model's own numbers are used.")
        return out
    out["p_win_map"] = _isotonic(m["p_win"].to_numpy(), m["won"].to_numpy(dtype=float))
    x, y = m["expected_net"].to_numpy(), (m["realized_return"] - m["cost_rt"].fillna(0)).to_numpy()
    slope, intercept = np.polyfit(x, y, 1) if np.std(x) > 0 else (1.0, 0.0)
    # Shrink toward the model's own numbers (slope 1, intercept 0) by how noisy the fit is.
    resid = y - (slope * x + intercept)
    se = float(np.sqrt(resid.var() / max(len(x) * x.var(), 1e-12)))
    weight = float(1 / (1 + (se / max(abs(slope - 1), 1e-6)) ** 2))
    out["net_map"] = {"slope": float(1 + weight * (slope - 1)), "intercept": float(weight * intercept),
                      "raw_slope": float(slope), "raw_intercept": float(intercept), "weight": weight}
    return out


def apply(today: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Today's scores with the live calibration put on P(profit) and expected net; the originals kept."""
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
            "calibration_state": calibration(con)}


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
        "outcome was known, with the outcome once matured (hold to the horizon, total return, from the close after "
        "the cutoff). Use it to score models live, recalibrate them, or add the live period to training.\n"
        f"Calibration in force: {json.dumps(calibration(con))}\n", encoding="utf-8")
    return path
