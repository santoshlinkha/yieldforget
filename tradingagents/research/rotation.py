"""Sector rotation: which sectors money is moving into or out of, caught early, and the likely cause.

**Where each sector stands.** Each SPDR sector fund is compared with SPY the way a relative
rotation graph does: the *ratio* is the fund's price relative to SPY over its own 63-session
average (above 100: beating the market on trend), the *momentum* is that ratio against itself 10
sessions ago (above 100: the ratio is rising). Together they give four stages, which sectors tend
to pass through in order: Improving -> Leading -> Weakening -> Lagging -> Improving.

**Alerts, ahead of the turn.** The early stages are the alerts: *rotating in* when a sector still
behind the market has turned up (Improving), *rotating out* when a leader has started to slip
(Weakening). A sector that has just crossed into Leading or Lagging is reported as the rotation
being confirmed. The stage says nothing certain about what comes next: sectors also turn back.

**The cause.** For each sector, the daily return over SPY is regressed (ridge, standardized, over
the last 252 sessions) on the daily moves of six market drivers: the 10-year yield, the yield
curve (10-year less 13-week), oil, the dollar, credit (high-yield bonds against Treasuries) and the
VIX. The sector's move over the last 21 sessions is split into each driver's part (its sensitivity
times the driver's move); the largest parts, in the direction of the move, are given as the cause,
with how much of the sector's daily moves the drivers explain at all (R squared). What is left is
the sector's own news, which this cannot name.

The internal ETF forecast for the fund (the latest saved run, ``research.engine``) is attached
when there is one, as a second opinion on the next 20 sessions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = {"XLK": "Technology", "XLC": "Communication", "XLY": "Consumer discretionary", "XLF": "Financials",
           "XLV": "Health care", "XLI": "Industrials", "XLE": "Energy", "XLB": "Materials", "XLP": "Consumer staples",
           "XLU": "Utilities", "XLRE": "Real estate"}
# name -> (how its daily move is measured, units, what a rise means, what a fall means)
DRIVERS = {
    "10-year yield": ("diff:^TNX", "pts", "rates rose", "rates fell"),
    "Yield curve (10y - 3m)": ("curve", "pts", "the yield curve steepened", "the yield curve flattened"),
    "Oil": ("ret:USO", "%", "oil rose", "oil fell"),
    "Dollar": ("ret:UUP", "%", "the dollar strengthened", "the dollar weakened"),
    "Junk bonds vs Treasuries": ("spread:HYG:IEF", "%", "credit was calm (junk bonds beat Treasuries)",
                                 "credit stress rose (junk bonds trailed Treasuries)"),
    "VIX": ("diff:^VIX", "pts", "fear rose (VIX up)", "fear eased (VIX down)"),
}
TICKERS = ["SPY", *SECTORS, "^TNX", "^IRX", "USO", "UUP", "HYG", "IEF", "^VIX"]
TREND, MOMENTUM, WINDOW, FIT = 63, 10, 21, 252


def stages(close: pd.DataFrame) -> pd.DataFrame:
    """Ratio and momentum per sector per session (relative-rotation style)."""
    rs = close[list(SECTORS)].div(close["SPY"], axis=0)
    ratio = 100 * rs / rs.rolling(TREND).mean()
    mom = 100 * ratio / ratio.shift(MOMENTUM)
    return pd.concat({"ratio": ratio, "momentum": mom}, axis=1)


def stage_of(ratio: float, mom: float) -> str:
    if ratio >= 100:
        return "Leading" if mom >= 100 else "Weakening"
    return "Improving" if mom >= 100 else "Lagging"


def driver_moves(close: pd.DataFrame) -> pd.DataFrame:
    """Each driver's daily move: yields and VIX in points, the rest in log returns."""
    out = {}
    for name, (how, *_) in DRIVERS.items():
        kind, *args = how.split(":")
        if kind == "diff":
            out[name] = close[args[0]].diff()
        elif kind == "curve":
            out[name] = (close["^TNX"] - close["^IRX"]).diff()
        elif kind == "ret":
            out[name] = np.log(close[args[0]]).diff()
        elif kind == "spread":
            out[name] = np.log(close[args[0]]).diff() - np.log(close[args[1]]).diff()
    return pd.DataFrame(out)


def causes(excess: pd.Series, drivers: pd.DataFrame, window: int = WINDOW, alpha: float = 1.0) -> dict:
    """Split the sector's move over ``window`` sessions into each driver's part (ridge on standardized moves)."""
    data = pd.concat([excess.rename("y"), drivers], axis=1).dropna().iloc[-FIT:]
    if len(data) < 60:
        return {"r2": None, "parts": [], "move": None}
    X, y = data[list(drivers)], data["y"]
    mu, sd = X.mean(), X.std().replace(0, np.nan)
    Z = ((X - mu) / sd).fillna(0.0).to_numpy()
    yc = y.to_numpy() - y.mean()
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ yc)
    fitted = Z @ beta
    r2 = float(1 - ((yc - fitted) ** 2).sum() / max((yc ** 2).sum(), 1e-12))
    recent = data.iloc[-window:]
    zr = ((recent[list(drivers)] - mu) / sd).fillna(0.0)
    raw_beta = pd.Series(beta, index=list(drivers)) / sd           # per unit of the driver's own move
    parts = []
    for name in drivers:
        moved = float(recent[name].sum())
        part = float(beta[list(drivers).index(name)] * zr[name].sum())
        parts.append({"driver": name, "driver_move": moved, "units": DRIVERS[name][1], "part": part,
                      "sensitivity": float(raw_beta[name]) if np.isfinite(raw_beta[name]) else None})
    return {"r2": max(r2, 0.0), "parts": parts, "move": float(recent["y"].sum())}


def _move_text(p: dict) -> str:
    m = p["driver_move"]
    size = f"{abs(m):.2f} pts" if p["units"] == "pts" else f"{abs(np.expm1(m)):.1%}"
    return f"{p['driver']} {'+' if m >= 0 else '−'}{size}"


def explain(name: str, c: dict, top: int = 2) -> dict:
    """The drivers that moved the sector most, in the direction it moved, as text."""
    if c["move"] is None:
        return {"text": "not enough history to tell", "drivers": []}
    move = c["move"]
    same = sorted([p for p in c["parts"] if p["part"] * move > 0], key=lambda p: -abs(p["part"]))[:top]
    explained = sum(p["part"] for p in c["parts"])
    bits = []
    for p in same:
        _, _, rise, fall = DRIVERS[p["driver"]]
        happened = rise if p["driver_move"] >= 0 else fall
        helps = (p["sensitivity"] or 0) * p["driver_move"] > 0
        bits.append(f"{happened} ({_move_text(p)}), and {name} usually {'beats' if helps else 'trails'} the market "
                    f"when that happens: {p['part']:+.1%}")
    if not bits:
        text = "none of the market drivers explains it: likely the sector's own news (earnings, regulation, deals)"
    else:
        text = "; ".join(bits)
        own = move - explained
        if abs(own) > abs(move) * 0.5:
            text += f". The rest ({own:+.1%}) is the sector's own news, which this cannot name"
    return {"text": text, "drivers": [{k: p[k] for k in ("driver", "driver_move", "units", "part")} for p in same],
            "explained": explained, "r2": c["r2"]}


def _forecasts(con) -> dict[str, dict]:
    if con is None:
        return {}
    row = con.execute("SELECT r.id, r.model, r.horizon, r.info_cutoff FROM forecast_runs r WHERE r.universe = 'etf'"
                      " ORDER BY r.created_at DESC LIMIT 1").fetchone()
    if not row:
        return {}
    run = dict(row)
    out = {}
    for f in con.execute("SELECT ticker, mean, p_gain FROM forecasts WHERE run_id = ?", (run["id"],)):
        if f["ticker"] in SECTORS:
            out[f["ticker"]] = {"mean": f["mean"], "p_gain": f["p_gain"], "model": run["model"],
                                "horizon": run["horizon"], "data_to": run["info_cutoff"]}
    if out:
        ranked = sorted(out, key=lambda t: -(out[t]["mean"] or 0))
        for i, t in enumerate(ranked):
            out[t]["rank"], out[t]["of"] = i + 1, len(ranked)
    return out


def rotation(cache_dir: str | Path, con=None, close: pd.DataFrame | None = None) -> dict:
    """Every sector's stage, and the rotation alerts with their likely causes."""
    if close is None:
        from tradingagents.research.prices import bars

        close = bars(TICKERS, cache_dir)["Close"]
    close = close.ffill()
    missing = [t for t in ["SPY", *SECTORS] if t not in close or close[t].dropna().empty]
    if missing:
        raise ValueError(f"no prices for {', '.join(missing)}")
    st = stages(close).dropna(how="all")
    drivers = driver_moves(close)
    spy = np.log(close["SPY"]).diff()
    fc = _forecasts(con)
    rows, alerts = [], []
    last = st.index[-1]
    for t, name in SECTORS.items():
        ratio, mom = st[("ratio", t)], st[("momentum", t)]
        if not np.isfinite(ratio.iloc[-1]) or not np.isfinite(mom.iloc[-1]):
            continue
        now = stage_of(ratio.iloc[-1], mom.iloc[-1])
        hist = [stage_of(r, m) for r, m in zip(ratio.iloc[-MOMENTUM - 1:], mom.iloc[-MOMENTUM - 1:], strict=True)
                if np.isfinite(r) and np.isfinite(m)]
        since = next((i for i, s in enumerate(reversed(hist)) if s != now), len(hist))   # sessions in this stage
        rel = float(close[t].iloc[-1] / close[t].iloc[-WINDOW - 1] / (close["SPY"].iloc[-1] / close["SPY"].iloc[-WINDOW - 1]) - 1)
        excess = np.log(close[t]).diff() - spy
        why = explain(name, causes(excess, drivers))
        row = {"ticker": t, "name": name, "stage": now, "sessions_in_stage": since, "ratio": round(float(ratio.iloc[-1]), 2),
               "momentum": round(float(mom.iloc[-1]), 2), "vs_spy_1m": round(rel, 4), "cause": why["text"],
               "drivers": why["drivers"], "r2": why["r2"], "forecast": fc.get(t)}
        rows.append(row)
        alert = None
        if now == "Improving":
            alert = ("in", "early", f"{name} is still behind the market but has turned up")
        elif now == "Weakening":
            alert = ("out", "early", f"{name} still leads the market but has started to slip")
        elif now == "Leading" and since <= 5:
            alert = ("in", "confirmed", f"{name} has just moved into the lead")
        elif now == "Lagging" and since <= 5:
            alert = ("out", "confirmed", f"{name} has just fallen behind")
        if alert:
            direction, stage, text = alert
            f = fc.get(t)
            agree = None
            if f and f.get("mean") is not None:
                agree = (f["mean"] > 0) == (direction == "in")
            alerts.append({"ticker": t, "name": name, "direction": direction, "stage": stage, "text": text,
                           "strength": round(abs(float(mom.iloc[-1]) - 100), 2), "vs_spy_1m": row["vs_spy_1m"],
                           "cause": why["text"], "drivers": why["drivers"], "r2": why["r2"], "forecast": f,
                           "forecast_agrees": agree})
    # Early alerts first (they are the ones ahead of the move), the strongest turns first.
    alerts.sort(key=lambda a: (a["stage"] != "early", -a["strength"]))
    recent = drivers.iloc[-WINDOW:].sum()
    return {"as_of": str(last.date()), "window": WINDOW, "sectors": sorted(rows, key=lambda r: -r["vs_spy_1m"]),
            "alerts": alerts,
            "drivers": {k: {"move": round(float(recent[k]), 4), "units": DRIVERS[k][1]} for k in DRIVERS},
            "method": "Stage: relative strength vs SPY over 63 sessions and its 10-session momentum. Cause: the "
                      "sector's daily return over SPY regressed on six market drivers over a year; the parts of the "
                      "last 21 sessions' move. A stage is not a forecast; sectors also turn back."}
