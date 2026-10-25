"""The quality and Wall Street screen: the ticker page's checklist and analyst data for many stocks, ranked.

For each stock (today's S&P 500 plus the watchlist):

- **Quality**: the moat checks (``research.moat``) and the checklist's financial strength and
  profitability items (``research.quality``). Quality score = the share of those tests passed, 0-100
  (an item marked "check" counts half, "flag" zero; items without data are left out).
- **Valuation**, kept apart from quality so cheap and good stay distinguishable: forward P/E against
  its own history, EV/EBITDA, FCF yield, and the growth today's price implies (reverse DCF), with the
  checklist's pass / check / flag counts.
- **Wall Street**: this month's analyst counts (strong buy to strong sell) as a consensus score from 1
  (strong buy) to 5 (strong sell), the number of analysts, and the upside to the mean price target.

**Combined rank**: the average of three percentile ranks among the stocks screened (quality score,
analyst consensus with stronger buys ranked higher, price-target upside); a stock missing one of them is
ranked on the others and says so. Ties break by quality score, then ticker. It is a screen of facts and
opinions as of today, not a forecast: none of it was tested for predicting returns (see the decision
study for what was), and Wall Street data is Yahoo's version as of today.

Built only when asked (``build``), four stocks at a time with every fetch cached (statements and
analyst histories weekly, profiles and analyst consensus daily), and saved; reading it fetches nothing.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
WEIGHT = {"ok": 1.0, "warn": 0.5, "flag": 0.0}


def _consensus(analysts: dict | None) -> tuple[float | None, int, str | None]:
    """(mean score 1 = strong buy .. 5 = strong sell, analysts, label) from this month's counts."""
    rec = next((r for r in (analysts or {}).get("recommendations") or [] if r.get("period") == "0m"), None)
    if not rec:
        return None, 0, None
    counts = [rec.get(k, 0) or 0 for k in ("strongBuy", "buy", "hold", "sell", "strongSell")]
    n = sum(counts)
    if not n:
        return None, 0, None
    mean = sum((i + 1) * c for i, c in enumerate(counts)) / n
    label = "Strong buy" if mean <= 1.5 else "Buy" if mean <= 2.5 else "Hold" if mean <= 3.5 else "Sell" \
        if mean <= 4.5 else "Strong sell"
    return mean, n, label


def score_row(ticker: str, profile: dict, financials: dict, closes, dividends: list, analysts: dict | None) -> dict:
    """One stock's screen row from data already fetched (pure: no downloads)."""
    from tradingagents.research.moat import moat
    from tradingagents.research.quality import checklist

    m = moat(financials, profile.get("sector"), profile.get("industry"))
    financial = str(m.get("basis", "")).startswith("bank")
    q = checklist(financials, profile, closes, dividends, analysts, financial)
    sections = {sec["key"]: sec for sec in q["sections"]}
    points, counted = float(m["passed"]), float(m["scored"])
    for key in ("strength", "profitability"):
        for item in sections[key]["items"]:
            if item["status"] in WEIGHT:
                points += WEIGHT[item["status"]]
                counted += 1
    val = sections["valuation"]
    items = {i["name"]: i for i in val["items"]}
    implied = (val.get("implied") or {}).get("growth")
    mean, n, label = _consensus(analysts)
    targets = (analysts or {}).get("price_targets") or {}
    price = float(closes.dropna().iloc[-1]) if closes is not None and closes.notna().any() else None
    upside = targets.get("mean") / price - 1 if targets.get("mean") and price else None
    return {"ticker": ticker, "name": profile.get("longName") or profile.get("shortName") or ticker,
            "sector": profile.get("sector"), "price": price,
            "moat": m["verdict"], "moat_passed": m["passed"], "moat_scored": m["scored"],
            "strength": sections["strength"]["counts"], "profitability": sections["profitability"]["counts"],
            "downturn": sections["strength"].get("answer"),
            "quality_score": round(100 * points / counted, 1) if counted else None, "quality_tests": int(counted),
            "valuation": val["counts"], "implied_growth": implied,
            "forward_pe": profile.get("forwardPE"), "ev_ebitda": items.get("EV/EBITDA", {}).get("value"),
            "fcf_yield": items.get("FCF yield", {}).get("value"),
            "analyst_mean": round(mean, 2) if mean is not None else None, "analysts": n, "analyst_label": label,
            "target_mean": targets.get("mean"), "target_upside": upside}


def rank(rows: list[dict]) -> list[dict]:
    """Percentile ranks and the combined rank; ties by quality score, then ticker."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows
    parts = {"quality_pct": frame["quality_score"].rank(pct=True),
             # A lower consensus score is a stronger buy: rank it higher.
             "analyst_pct": (-frame["analyst_mean"]).rank(pct=True),
             "upside_pct": frame["target_upside"].rank(pct=True)}
    for k, v in parts.items():
        frame[k] = v
    frame["combined"] = frame[list(parts)].mean(axis=1, skipna=True)
    frame["missing"] = frame[list(parts)].isna().sum(axis=1)
    frame = frame.sort_values(["combined", "quality_score", "ticker"], ascending=[False, False, True], na_position="last")
    frame["rank"] = np.arange(1, len(frame) + 1)
    out = frame.replace({np.nan: None}).to_dict("records")
    for r in out:
        if r["missing"]:
            r["note"] = f"ranked on {3 - int(r['missing'])} of 3 measures"
    return out


def build(cache_dir: Path, tickers: list[str], results_dir: Path, workers: int = 4, progress=None) -> dict:
    """Fetch (cached), score, rank and save the screen; returns it."""
    from tradingagents.dataflows.vendors.yahoo import history as yh
    from tradingagents.research.prices import bars

    say = progress or (lambda *_: None)
    closes = bars(tickers, cache_dir)["Close"]
    rows, errors, done = [], {}, 0

    def one(t):
        try:
            profile = yh.profile(t, cache_dir) or {}
            if profile.get("quoteType") not in (None, "EQUITY"):
                return t, None, "not a stock"
            fin = yh.financials(t, cache_dir)
            acts = yh.actions(t, cache_dir)
            analysts = yh.current_analysts(t, cache_dir)
            series = closes[t] if t in closes else None
            return t, score_row(t, profile, fin, series, acts.get("dividends") or [], analysts), None
        except Exception as exc:  # one stock's failure stays with that stock
            return t, None, str(exc)[:160]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, row, err in pool.map(one, tickers):
            done += 1
            if row:
                rows.append(row)
            else:
                errors[t] = err
            if done % 25 == 0:
                say(f"scored {done} of {len(tickers)}")
    out = {"built_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "as_of": str(closes.index[-1].date()) if len(closes) else None, "stocks": len(rows),
           "errors": errors, "rows": rank(rows)}
    folder = Path(results_dir) / "screen"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"quality_{out['as_of']}.json").write_text(json.dumps(out, default=str), encoding="utf-8")
    return out


def latest(results_dir: Path) -> dict | None:
    folder = Path(results_dir) / "screen"
    files = sorted(folder.glob("quality_*.json")) if folder.is_dir() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None
