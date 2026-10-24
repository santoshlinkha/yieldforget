"""Dividends, splits and share-count changes, from ``history.corporate_actions`` output.

- **Dividends**: every payment (split-adjusted, per share), the totals by calendar year,
  the last 12 months and the yield on today's price, how many years in a row the
  dividend was raised, and its growth rate over the last five full years. Raises and
  cuts compare each year's typical (median) payment, not its total: a total moves with
  how many payments fall in the calendar year. A typical payment down more than 5% is
  a cut; payments that stopped are a suspension.
- **Splits**: each one as "4-for-1" (more shares) or "1-for-10" (a reverse split).
- **Share count**: Yahoo reports shares outstanding as they were, so a 4-for-1 split
  looks like the count quadrupling. Each count is restated in today's shares (multiplied
  by every later split) before the year-end counts are compared: a rise of 2% or more in
  a year is **dilution** (new shares issued: stock pay, acquisitions, raising money), a
  fall of 2% or more is a **buyback**. A year's count is the median of its last 90
  days of reports, since Yahoo's history has one-day spikes. Funds are left out: an
  ETF's count moves with investors coming and going, not with dilution.

Every event lands on one timeline, newest first.
"""

from __future__ import annotations

import statistics
from datetime import date, timedelta

CUT = 0.05        # a full year's dividends down more than this is a cut
SHARE_MOVE = 0.02  # a year's share-count change at least this is dilution or a buyback


def _split_label(ratio: float) -> str:
    if ratio >= 1:
        return f"{ratio:g}-for-1 split"
    return f"1-for-{1 / ratio:g} reverse split"


def _dividends(payments: list, as_of: date, price: float | None) -> tuple[dict | None, list[dict]]:
    if not payments:
        return None, []
    by_year: dict[int, float] = {}
    paid: dict[int, list[float]] = {}
    for d, amount in payments:
        by_year[int(d[:4])] = by_year.get(int(d[:4]), 0.0) + amount
        paid.setdefault(int(d[:4]), []).append(amount)
    typical = {y: statistics.median(v) for y, v in paid.items()}
    full = sorted(y for y in by_year if y < as_of.year)
    year_ago = (as_of - timedelta(days=365)).isoformat()
    ttm = sum(a for d, a in payments if d > year_ago)
    events = []
    first = payments[0][0]
    events.append({"date": first, "kind": "dividend", "text": f"first dividend on record, ${payments[0][1]:.4g} a share"})
    for prev, year in zip(full, full[1:], strict=False):
        if year == prev + 1 and typical[year] < typical[prev] * (1 - CUT):
            events.append({"date": f"{year}-12-31", "kind": "cut",
                           "text": f"dividend cut {1 - typical[year] / typical[prev]:.0%} in {year} "
                                   f"(typical payment ${typical[prev]:.4g} to ${typical[year]:.4g} a share)"})
    if ttm == 0:
        events.append({"date": payments[-1][0], "kind": "suspended",
                       "text": "dividend suspended: the last one paid, none in the 12 months since"})
    streak = 0
    for year in reversed(full):
        if year - 1 in typical and typical[year] > typical[year - 1]:
            streak += 1
        else:
            break
    growth = None
    if full and full[-1] - 5 in by_year and by_year[full[-1] - 5] > 0:
        growth = (by_year[full[-1]] / by_year[full[-1] - 5]) ** (1 / 5) - 1
    summary = {"ttm": ttm, "yield": ttm / price if price and ttm else None, "since": first[:4],
               "raised_years": streak, "growth_5y": growth, "count": len(payments),
               "by_year": [[y, by_year[y], y >= as_of.year] for y in sorted(by_year)],
               "payments": [{"date": d, "amount": a} for d, a in reversed(payments)]}
    return summary, events


def _shares(points: list, splits: list, as_of: date) -> tuple[list[dict], list[dict]]:
    """Year-end share counts in today's shares, with each year's change; and the events among them."""
    if not points:
        return [], []

    def restated(d, count):
        factor = 1.0
        for when, ratio in splits:
            if when > d and ratio > 0:
                factor *= ratio
        return count * factor

    by_year: dict[int, list[tuple[str, float]]] = {}
    for d, count in points:
        if count > 0:
            by_year.setdefault(int(d[:4]), []).append((d, restated(d, count)))
    year_end = {}
    for y, pts in by_year.items():
        last = date.fromisoformat(pts[-1][0])    # points are oldest first
        recent = [c for d, c in pts if date.fromisoformat(d) >= last - timedelta(days=90)]
        year_end[y] = (pts[-1][0], statistics.median(recent))
    rows, events = [], []
    years = sorted(year_end)
    for i, y in enumerate(years):
        d, count = year_end[y]
        prev = year_end.get(y - 1) if i else None
        change = count / prev[1] - 1 if prev else None
        rows.append({"year": y, "as_of": d, "shares": count, "change": change})
        if change is not None and abs(change) >= SHARE_MOVE:
            kind = "dilution" if change > 0 else "buyback"
            events.append({"date": d, "kind": kind,
                           "text": f"share count {'up' if change > 0 else 'down'} {abs(change):.1%} "
                                   + (f"so far in {y}" if y == as_of.year else f"in {y}")
                                   + (" (new shares issued)" if change > 0 else " (shares bought back)")})
    return rows, events


def summarize(actions: dict, as_of: date, price: float | None = None, fund: bool = False) -> dict:
    """The dividend summary, the splits, year-end share counts and one event timeline, newest first."""
    splits = [(d, r) for d, r in actions.get("splits") or [] if r and r != 1]
    dividends, events = _dividends(actions.get("dividends") or [], as_of, price)
    if fund:   # a fund passes on what its holdings pay: a smaller year is not a decision to cut
        events = [e for e in events if e["kind"] != "cut"]
    shares, share_events = ([], []) if fund else _shares(actions.get("shares") or [], splits, as_of)
    events += share_events
    events += [{"date": d, "kind": "split", "text": _split_label(r)} for d, r in splits]
    net = None
    if len(shares) >= 2:
        net = {"from": shares[0]["year"], "to": shares[-1]["year"],
               "change": shares[-1]["shares"] / shares[0]["shares"] - 1}
    return {"fund": fund, "dividends": dividends, "splits": [{"date": d, "ratio": r, "label": _split_label(r)} for d, r in splits],
            "shares": list(reversed(shares)), "shares_net": net,
            "events": sorted(events, key=lambda e: e["date"], reverse=True)}
