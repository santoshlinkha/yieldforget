"""The decision study: cross-fitted entry models, development-only rule selection, exit policies compared on
identical entries, and a capital-constrained historical simulation. Pure functions; the runner is
``experiments/decision_study.py``.

Design (fixed before results):
- Decision dates every 5 sessions from 2006 (entries weekly; exits are checked every session).
- Shared walk-forward folds: one year each from 2012. Training rows for a fold are members on
  their date whose trade **exited** and whose label **ended** before the fold began.
- Primary models produce out-of-fold (OOF) scores. The trade-quality model, the score-to-P&L map
  and the risk model for fold k learn only from OOF rows of earlier folds (cross-fitting), so no
  input to them was fitted on the rows it scores.
- Development: 2012 to June 2024. Thresholds and the policy are chosen there only. July 2024 on
  is **exploratory**: it was inspected in an earlier study, so it is not an untouched test.
- Promotion needs the complete policy (entry, sizing, exit) to beat cash-flow-matched SPY and
  cash after costs on development with acceptable drawdown and calibration; IC or AUC alone is not
  enough. Even then it stays experimental until the prospective record confirms it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .adapters import (
    Ranker,
    TradeQuality,
    WithinDateLinear,
    apply_map,
    score_to_net,
    within_date_pct,
)
from .positions import Book, Policy, benchmark, trades

DEV_END = pd.Timestamp("2024-06-30")
FIRST_TEST = pd.Timestamp("2012-01-01")
EVERY = 5


def fold_edges(dates: pd.DatetimeIndex, start=FIRST_TEST) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    years = range(start.year, dates.max().year + 1)
    edges = [pd.Timestamp(f"{y}-01-01") for y in years] + [dates.max() + pd.Timedelta(days=1)]
    return list(zip(edges, edges[1:], strict=False))


def attach_trades(panel: pd.DataFrame, px, low, close, div, policy: Policy) -> pd.DataFrame:
    """The policy's trade outcome for every member row: net arithmetic P&L, exit date, reason."""
    rows = panel[panel["in_index"]].reset_index()
    sigma = rows["ewma_vol_daily"].fillna(rows["raw_vol_3m"] / math.sqrt(252))
    t = trades(px, low, close, div, pd.DataFrame({"date": rows["date"], "ticker": rows["ticker"], "sigma": sigma}),
               policy, rows["cost_rt"].fillna(0.004).to_numpy())
    t.index = pd.MultiIndex.from_frame(rows[["date", "ticker"]])
    return t


def crossfit_primary(panel, features, edges, adapters) -> pd.DataFrame:
    """Out-of-fold scores for each adapter (and within-date percentiles), purged by label end."""
    out = []
    member = panel[panel["in_index"]]
    d = member.index.get_level_values("date")
    for start, end in edges:
        test = member[(d >= start) & (d < end)]
        train = member[(pd.to_datetime(member["label_end"]) < start) & member["label"].notna()]
        if test.empty or train.index.get_level_values("date").nunique() < 100:
            continue
        frame = pd.DataFrame(index=test.index)
        for make in adapters:
            a = make().fit(train, features)
            s = a.predict(test)
            frame[a.name] = s
            frame[f"{a.name}_pct"] = within_date_pct(test, s)
        out.append(frame)
    return pd.concat(out)


def crossfit_second(panel, oof, tr, edges, primary: str, features: list[str], device="cpu") -> pd.DataFrame:
    """Fold k's trade-quality probabilities, mapped expected net P&L and position-risk probability,
    each fitted only on OOF rows of earlier folds whose trades had exited before fold k."""
    from .models import XGBEvent

    rows = oof.join(tr[["net", "exit_date", "complete"]]).join(
        panel[[*features, "label_end", "pos_large_loss"]], how="left")
    rows["win"] = np.where(rows["complete"], (rows["net"] > 0).astype(float), np.nan)
    rows["score_pct"] = rows[f"{primary}_pct"]
    q_feats = [*features, "score_pct"]
    out = []
    d = rows.index.get_level_values("date")
    for start, end in edges:
        test = rows[(d >= start) & (d < end)]
        past = rows[(pd.to_datetime(rows["exit_date"]) < start) & (pd.to_datetime(rows["label_end"]) < start)
                    & rows["complete"]]
        if test.empty or past.index.get_level_values("date").nunique() < 100:
            continue
        frame = pd.DataFrame(index=test.index)
        mapping = score_to_net(past["score_pct"].to_numpy(), past["net"].to_numpy())
        frame["expected_net"] = apply_map(mapping, test["score_pct"].to_numpy())
        for model, cal in (("logit", "none"), ("logit", "sigmoid"), ("xgb", "none"), ("xgb", "sigmoid")):
            q = TradeQuality(model, cal, device).fit(past, q_feats)
            frame[q.name] = q.predict(test)
        risk_rows = past[past["pos_large_loss"].notna()]
        frame["p_risk"] = XGBEvent(device).fit(risk_rows, features, "pos_large_loss").predict_proba(test)
        out.append(frame)
    return pd.concat(out)


def calibration_table(p: np.ndarray, y: np.ndarray, bins=(0, .3, .4, .45, .5, .55, .6, .7, 1.0)) -> list[dict]:
    """Reliability around the action thresholds: forecast vs observed in each band."""
    out = []
    for a, b in zip(bins, bins[1:], strict=False):
        sel = (p >= a) & (p < b) & np.isfinite(y)
        if sel.any():
            out.append({"band": f"{a:.2f}-{b:.2f}", "n": int(sel.sum()), "forecast": float(p[sel].mean()),
                        "observed": float(y[sel].mean())})
    return out


def block_length(per_period: np.ndarray, horizon: int, cadence: int) -> int:
    """Bootstrap block from dependence and cadence: at least the holding overlap (horizon / cadence
    periods), longer when the series' first-lag autocorrelation says so."""
    overlap = max(1, math.ceil(horizon / cadence))
    x = per_period[np.isfinite(per_period)]
    if len(x) < 10 or np.std(x) == 0:
        return overlap
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    extra = math.ceil(3 / (1 - rho)) if 0 < rho < 0.95 else 1
    return max(overlap, extra)


def paired_ci(diffs: np.ndarray, block: int, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    from tradingagents.research.walkforward import block_bootstrap

    return block_bootstrap(diffs[np.isfinite(diffs)], block=block, reps=reps, seed=seed)


# --- the capital-constrained simulation ---

def simulate(signals: pd.DataFrame, px: pd.DataFrame, div: pd.DataFrame, policy: Policy, capital: float,
             start, end, max_names=5, max_share=0.25, sizing="equal", rule=None, label="model") -> dict:
    """Daily book from ``start`` to ``end``: on decision dates, candidates passing ``rule`` (a function of
    the signal row) are bought at the next close within settled cash, the name limit and the per-name
    cap (existing holdings included); exits are close-confirmed every session; dividends are cash."""
    sessions = px.index[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
    book = Book(px.index, cash=capital)
    marks = px.ffill()          # lots are marked at their last known price; trades need a price that day
    pending, log = [], []
    by_date = {d: g.droplevel("date") for d, g in signals.groupby(level="date")}
    equity, exposure = [], []
    for day in sessions:
        book.settle(day)
        prices = px.loc[day]
        if div is not None:
            book.dividends(day, div.loc[day])
        # Orders decided at the previous close execute now.
        for order in pending:
            t, dollars, sigma, cost = order
            p = float(prices.get(t, np.nan))
            if np.isfinite(p) and dollars <= book.cash + 1e-9:
                book.buy(day, t, dollars, p, policy, sigma, cost, source=label)
        pending = []
        for lot, reason in [x for x in getattr(book, "_due", [])]:
            p = float(prices.get(lot.ticker, np.nan))
            if lot in book.lots and np.isfinite(p) and day > lot.entry_date:
                book.sell(day, lot, p, reason)
        book._due = book.exits_due(day, prices)
        value = book.value(marks.loc[day])
        equity.append((day, value))
        held = sum(l.shares * float(marks.loc[day].get(l.ticker, 0) or 0) for l in book.lots)
        exposure.append(held / value if value else 0.0)
        if day in by_date:
            today = by_date[day]
            if rule is not None:
                today = today[today.apply(rule, axis=1)]
            held_names = {l.ticker for l in book.lots}
            slots = max(0, max_names - len(held_names))
            picks = [t for t in today.sort_values("rank_key", ascending=False).index if t not in held_names][:slots]
            if picks:
                per_cap = max_share * value
                if sizing == "vol":
                    inv = 1 / today.loc[picks, "sigma"].clip(lower=1e-4)
                    weights = inv / inv.sum()
                else:
                    weights = pd.Series(1 / len(picks), index=picks)
                budget = book.cash
                for t in picks:
                    dollars = min(per_cap, budget * float(weights[t]))
                    if dollars >= 1.0:
                        pending.append((t, dollars, float(today.loc[t, "sigma"]), float(today.loc[t, "cost_rt"])))
    eq = pd.Series(dict(equity))
    rets = eq.pct_change().dropna()
    trades_log = [x for x in book.log if x["action"] in ("sell", "reduce")]
    return {"label": label, "equity": eq, "final": float(eq.iloc[-1]) if len(eq) else capital,
            "total": float(eq.iloc[-1] / capital - 1) if len(eq) else 0.0,
            "ann": float((eq.iloc[-1] / capital) ** (252 / max(len(eq), 1)) - 1) if len(eq) else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0,
            "exposure": float(np.mean(exposure)) if exposure else 0.0, "costs": round(book.costs, 2),
            "trades": len(trades_log), "daily": rets,
            "exit_reasons": pd.Series([x.get("reason") for x in trades_log]).value_counts().to_dict()}


def buy_and_hold(close_adj: pd.Series, capital: float, start, end) -> dict:
    s = close_adj[(close_adj.index >= pd.Timestamp(start)) & (close_adj.index <= pd.Timestamp(end))].dropna()
    eq = capital * s / s.iloc[0]
    return {"equity": eq, "total": float(eq.iloc[-1] / capital - 1),
            "ann": float((eq.iloc[-1] / capital) ** (252 / len(eq)) - 1),
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "daily": eq.pct_change().dropna()}


def summary(sim: dict, bench: dict, cash: dict, trials: int = 1) -> dict:
    """The simulation against SPY and cash. ``trials``: how many rule and policy variants were tried to pick
    this one, so the deflated Sharpe ratio of the daily excess over SPY charges for the search."""
    from tradingagents.research.walkforward import deflated_sharpe

    ex = (sim["daily"] - bench["daily"].reindex(sim["daily"].index).fillna(0)).to_numpy()
    lo, hi = paired_ci(ex, block=20)
    keep = {k: v for k, v in sim.items() if k not in ("equity", "daily")}
    return {**keep, "vs_spy_daily_mean": float(np.nanmean(ex)), "vs_spy_ci": [lo, hi],
            "vs_spy_deflated_sharpe": deflated_sharpe(ex, trials),
            "spy_total": bench["total"], "cash_total": cash["total"], "spy_max_drawdown": bench["max_drawdown"]}


def continuation_states(tr: pd.DataFrame, px: pd.DataFrame, div: pd.DataFrame, close_adj: pd.DataFrame,
                        ewma: pd.DataFrame, cash_adj: pd.Series, checkpoints=(5, 10, 15), horizon=20) -> pd.DataFrame:
    """Position states of the fixed-horizon trades at each checkpoint (sessions after entry), with the
    continuation benefit: holding from the next close to the deadline exit, against selling at the
    next close and holding the cash fund instead. Every entry contributes, survivors or not."""
    sessions = px.index
    rows = []
    t_ok = tr[tr["complete"]]
    pos = {d: i for i, d in enumerate(sessions)}
    P, D, C = px.to_numpy(float), div.to_numpy(float), cash_adj.reindex(sessions).to_numpy(float)
    E = ewma.reindex(index=sessions, columns=px.columns).to_numpy(float)
    col = {t: i for i, t in enumerate(px.columns)}
    for (date, ticker), r in t_ok.iterrows():
        e = pos[pd.Timestamp(r["entry_date"])]
        x = pos[pd.Timestamp(r["exit_date"])]
        j = col[ticker]
        s0 = E[e - 1, j] if e > 0 else np.nan
        for k in checkpoints:
            if e + k + 1 >= x:
                break
            now, nxt = P[e + k, j], P[e + k + 1, j]
            hold = (P[x, j] + np.nansum(D[e + k + 2: x + 1, j])) / nxt - 1
            cash = C[x] / C[e + k + 1] - 1 if np.isfinite(C[x]) and np.isfinite(C[e + k + 1]) else 0.0
            rows.append({"entry_decision": date, "ticker": ticker, "check": k, "age": k / horizon,
                         "unrealized": now / r["entry_px"] - 1,
                         "unrealized_sigma": (now / r["entry_px"] - 1) / (s0 * math.sqrt(horizon)) if s0 else np.nan,
                         "vol_ratio": E[e + k, j] / s0 if s0 else np.nan,
                         "mom_5": now / P[e + k - 5, j] - 1 if k >= 5 else np.nan,
                         "benefit": hold - cash, "check_date": sessions[e + k], "exit_date": r["exit_date"]})
    return pd.DataFrame(rows)
