"""Scores for the stock forecasts: events, trend states, drawdown severity and a portfolio test.

Uncertainty is computed over **dates**, never over stock-rows: five hundred stocks
on one date share one market, so per-date averages are bootstrapped in blocks
of 12 dates (``research.walkforward.block_bootstrap``).

Events (breakouts, large loss, deep drawdown):
- Brier score and log loss against the null of the training base rate, the Brier
  skill (1 - Brier / null) with its 95% interval, and the ROC AUC;
- a reliability table (forecast probability in tenths against how often it happened);
- **alerts**: on each date the top tenth of stocks by probability. Precision (how
  many alerts came true), recall (how many events were alerted; the rest are
  missed), lift over the base rate, false alarms; for breakouts, how many alerted
  breakouts that happened then failed; for deep drawdowns, the lead time (sessions
  from the forecast to the first breach).

Trend states: log loss against the training mix, accuracy against always guessing
the most common state, and recall per state.

Severity: the share of outcomes below the forecast 5th-percentile drawdown (target 5%)
and its pinball loss.

Portfolio: on non-overlapping decision dates, the top fifth by expected return net of
the estimated round-trip cost, equal weighted and long only, against all stocks equal
weighted, a momentum rule (top fifth by 12-1 month momentum) and a trend rule (stocks
already in an uptrend). Returns are the tradable label: next close to ``h`` sessions
later, above cash; costs are charged on the turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.walkforward import block_bootstrap

from .models import log_loss


def _dates(frame: pd.DataFrame):
    return frame.index.get_level_values("date")


def event_scores(frame: pd.DataFrame, target: str, prob: str, base: str, extra: dict | None = None) -> dict:
    """``frame``: rows with the outcome in ``target``, the forecast in ``prob`` and the training
    base rate in ``base``."""
    from sklearn.metrics import roc_auc_score

    f = frame[frame[target].notna() & frame[prob].notna()]
    if f.empty:
        return {}
    y, p, q = f[target].to_numpy(float), f[prob].to_numpy(float), f[base].to_numpy(float)
    brier, null = float(np.mean((p - y) ** 2)), float(np.mean((q - y) ** 2))
    per_date = pd.DataFrame({"d": (q - y) ** 2 - (p - y) ** 2}, index=f.index).groupby(level="date")["d"].mean()
    lo, hi = block_bootstrap(per_date.to_numpy())
    out = {"rows": int(len(f)), "dates": int(per_date.size), "base_rate": float(y.mean()),
           "brier": brier, "brier_null": null, "skill": 1 - brier / null if null else float("nan"),
           # per_date is the null's error minus the model's, so its interval scaled by the null is the skill's
           # (low first). The pooled skill and this per-date interval can differ slightly.
           "skill_ci": [lo / null, hi / null] if null and np.isfinite(lo) else [None, None],
           "log_loss": log_loss(y, p), "log_loss_null": log_loss(y, q),
           "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}
    bins = np.clip((p * 10).astype(int), 0, 9)
    out["reliability"] = [[float(p[bins == b].mean()), float(y[bins == b].mean()), int((bins == b).sum())]
                          for b in range(10) if (bins == b).any()]
    rank = f[prob].groupby(level="date").rank(pct=True, method="first")
    alert = (rank > 0.9).to_numpy()
    hits = alert & (y == 1)
    out["alerts"] = {"share": float(alert.mean()), "precision": float(hits.sum() / max(alert.sum(), 1)),
                     "recall": float(hits.sum() / max((y == 1).sum(), 1)),
                     "missed": float(1 - hits.sum() / max((y == 1).sum(), 1)),
                     "false_alarms": float((alert & (y == 0)).sum() / max(alert.sum(), 1)),
                     "lift": float(hits.sum() / max(alert.sum(), 1) / max(y.mean(), 1e-9))}
    for name, col in (extra or {}).items():
        v = f[col].to_numpy(float)
        sel = hits & np.isfinite(v)
        out["alerts"][name] = float(v[sel].mean()) if sel.any() else None
    return out


def trend_scores(frame: pd.DataFrame, probs: list[str], prior: np.ndarray) -> dict:
    f = frame[frame["trend_state"].notna()]
    y = f["trend_state"].astype(int).to_numpy()
    p = np.clip(f[probs].to_numpy(float), 1e-6, 1)
    p = p / p.sum(axis=1, keepdims=True)
    ll = float(-np.mean(np.log(p[np.arange(len(y)), y])))
    ll_null = float(-np.mean(np.log(np.clip(prior[y], 1e-6, 1))))
    guess = p.argmax(axis=1)
    return {"rows": int(len(y)), "log_loss": ll, "log_loss_null": ll_null, "skill": 1 - ll / ll_null,
            "accuracy": float((guess == y).mean()), "accuracy_majority": float((y == np.bincount(y).argmax()).mean()),
            "mix": (np.bincount(y, minlength=4) / len(y)).tolist(),
            "recall": [float((guess[y == k] == k).mean()) if (y == k).any() else None for k in range(4)]}


def severity_scores(frame: pd.DataFrame, q05: str) -> dict:
    f = frame[frame["drawdown"].notna() & frame[q05].notna()]
    y, q = f["drawdown"].to_numpy(float), f[q05].to_numpy(float)
    d = y - q
    return {"rows": int(len(f)), "below_q05": float((y < q).mean()),
            "pinball_q05": float(np.mean(np.maximum(0.05 * d, (0.05 - 1) * d)))}


def spaced_dates(dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Decision dates at least ``horizon`` + 1 sessions apart, so holding periods do not overlap."""
    keep, last = [], None
    pos = {d: i for i, d in enumerate(sessions)}
    for d in sorted(dates):
        if last is None or pos[d] - pos[last] >= horizon + 1:
            keep.append(d)
            last = d
    return pd.DatetimeIndex(keep)


def portfolio(frame: pd.DataFrame, score: str, sessions: pd.DatetimeIndex, horizon: int) -> dict:
    """Long-only top fifth by ``score`` (expected return net of cost), with three benchmarks.

    Each period is an equal-weight basket: its return is the mean of the names' simple returns, less cost,
    taken back to a log return so periods compound. Annualized figures use the periods' actual spacing
    (dates skip when a day has too few names), not the minimum spacing."""
    f = frame[frame["label"].notna()]
    dates = spaced_dates(pd.DatetimeIndex(_dates(f).unique()), sessions, horizon)
    rules = {"model": score, "momentum": "mom_12_1"}
    held = {k: set() for k in (*rules, "trend")}
    rows = []
    for d in dates:
        day = f.xs(d, level="date")
        if len(day) < 25:
            continue
        rec = {"date": d, "all": float(np.log1p(np.expm1(day["label"]).mean()))}
        picks = {k: set(day[c].nlargest(max(1, len(day) // 5)).index) for k, c in rules.items() if c in day}
        picks["trend"] = set(day.index[day["current_trend"] == 1]) if "current_trend" in day else set()
        for k, names in picks.items():
            if not names:
                rec[k] = rec["all"]
                continue
            turnover = 1 - len(names & held[k]) / len(names) if held[k] else 1.0
            cost = float(day.loc[list(names), "cost_rt"].mean()) * turnover
            rec[k] = float(np.log1p(np.expm1(day.loc[list(names), "label"]).mean() - cost))
            rec[f"{k}_turnover"] = turnover
            held[k] = names
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("date")
    pos = {d: i for i, d in enumerate(sessions)}
    gaps = np.diff([pos[d] for d in t.index])
    out = {"periods": int(len(t)), "per_year": 252 / float(gaps.mean()) if len(gaps) else 252 / (horizon + 1)}
    for k in ("model", "momentum", "trend", "all"):
        if k not in t:
            continue
        r = t[k].dropna()
        wealth = np.exp(r.cumsum())
        out[k] = {"mean": float(r.mean()), "ann": float(r.mean() * out["per_year"]),
                  "total": float(wealth.iloc[-1] - 1) if len(wealth) else None,
                  "max_drawdown": float((wealth / wealth.cummax() - 1).min()) if len(wealth) else None,
                  "turnover": float(t.get(f"{k}_turnover", pd.Series(dtype=float)).mean()) if f"{k}_turnover" in t else None,
                  "by_period": {name: float(r[(r.index >= a) & (r.index <= b)].mean())
                                for a, b, name in (("2000", "2019-12-31", "to 2019"), ("2020", "2100", "2020 on"))
                                if ((r.index >= a) & (r.index <= b)).any()}}
    excess = (t["model"] - t["all"]).dropna().to_numpy()
    out["model_minus_all"] = {"mean": float(excess.mean()) if len(excess) else None,
                              "ci": list(block_bootstrap(excess, block=6)) if len(excess) >= 12 else [None, None]}
    out["series"] = {k: [[str(i.date()), float(v)] for i, v in t[k].dropna().items()] for k in ("model", "all")
                     if k in t}
    return out
