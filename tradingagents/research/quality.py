"""A company checklist beyond the moat: financial strength, profitability and cash flow, valuation.

Every figure comes from Yahoo's statements (annual, and the last four quarters for
"trailing twelve months" flows), its quote profile, and the app's saved prices.
Each item has a status: ``ok``, ``warn`` (worth a look), ``flag`` (fails the rule),
or ``info`` (context, no rule).

**Financial strength.** Debt to equity (flag above 2); interest coverage, operating
income over interest expense (flag under 3); current ratio, current assets over
current liabilities (flag under 1.0); and a high-rate downturn test: operating
income 30% lower while all debt is refinanced 2 points dearer, coverage flagged
under 1.5 and thin under 3.

**Profitability and cash flow.** Free cash flow (positive every year, its margin,
how much of net income it turns into cash); ROIC against a 15% target (the moat's
own calculation); and whether gross, operating and net margins are expanding,
stable (within a point) or shrinking from the first year to the last.

**Valuation.** Forward P/E against the company's own P/E at each fiscal year end
(year-end price over that year's diluted EPS; prices are restored from the app's
dividend-adjusted closes using the dividend history first); EV/EBITDA; free-cash-flow
yield; and a reverse DCF: the yearly growth in free cash flow over ten years that
today's enterprise value implies, at a 9% discount rate and 2.5% growth after,
set against the company's revenue record and analysts' estimates for next year.
These rates are assumptions, stated on the page; the implied growth moves with them.

Banks and insurers: debt is their raw material and they report no current assets,
so strength and EV-based measures are left out and said so.
"""

from __future__ import annotations

import statistics

import pandas as pd

from .moat import roic_by_year

DISCOUNT = 0.09
TERMINAL = 0.025
YEARS = 10


def _item(name: str, status: str | None, value: str, rule: str = "") -> dict:
    return {"name": name, "status": status, "value": value, "rule": rule}


def _annual(statements: dict, kind: str, line: str) -> dict[str, float]:
    table = (statements.get(kind) or {}).get("annual") or {}
    values = (table.get("rows") or {}).get(line) or []
    return {p: v for p, v in zip(table.get("periods") or [], values, strict=False) if v is not None}


def _latest(series: dict[str, float]):
    return series[max(series)] if series else None


def _ttm(statements: dict, kind: str, line: str):
    """The last four quarters summed when Yahoo has all four, else the latest year; and which."""
    table = (statements.get(kind) or {}).get("quarterly") or {}
    values = [v for v in ((table.get("rows") or {}).get(line) or [])[:4] if v is not None]
    if len(values) == 4:
        return sum(values), "last 4 quarters"
    return _latest(_annual(statements, kind, line)), "latest year"


def _pct(x):
    return f"{x:.1%}" if abs(x) < 0.1 else f"{x:.0%}"


def raw_closes(adjusted: pd.Series, dividends: list) -> pd.Series:
    """Closes as they traded, from dividend-adjusted ones: each dividend scaled earlier prices
    down by 1 − dividend ÷ the close before it went ex; undo that, newest dividend first."""
    adjusted = adjusted.dropna()
    factor = pd.Series(1.0, index=adjusted.index)
    running = 1.0
    for d, amount in sorted(dividends, reverse=True):
        before = adjusted.loc[:pd.Timestamp(d) - pd.Timedelta(days=1)]
        if before.empty or amount <= 0:
            continue
        # The close before the ex-date carries this dividend's own adjustment: adjusted = (raw − dividend) × later ones.
        raw_prev = before.iloc[-1] / running + amount
        running *= 1 - amount / raw_prev
        factor.loc[before.index] = running
    return adjusted / factor


def implied_growth(value: float, fcf: float, discount: float = DISCOUNT, terminal: float = TERMINAL,
                   years: int = YEARS) -> float | None:
    """The yearly free-cash-flow growth over ``years`` that makes the discounted cash flows,
    plus a terminal value growing at ``terminal``, equal ``value``. None when FCF is not positive."""
    if not fcf or fcf <= 0 or value <= 0:
        return None

    def worth(g):
        total, f = 0.0, fcf
        for t in range(1, years + 1):
            f *= 1 + g
            total += f / (1 + discount) ** t
        return total + f * (1 + terminal) / (discount - terminal) / (1 + discount) ** years

    lo, hi = -0.5, 1.5
    if worth(hi) < value:
        return hi
    if worth(lo) > value:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if worth(mid) < value else (lo, mid)
    return (lo + hi) / 2


def _strength(s: dict, financial: bool) -> list[dict]:
    if financial:
        equity, assets = _latest(_annual(s, "balance", "Stockholders Equity")), _latest(_annual(s, "balance", "Total Assets"))
        out = [_item("Not the usual tests", "info", "a bank or insurer borrows as its business and reports no current "
                     "assets, so debt to equity, interest coverage and the current ratio say little; its capital "
                     "ratios (CET1) are not in Yahoo's data")]
        if equity and assets:
            out.append(_item("Equity to assets", "info", f"{equity / assets:.1%} of assets funded by shareholders"))
        return out
    debt = _latest(_annual(s, "balance", "Total Debt")) or 0.0
    equity = _latest(_annual(s, "balance", "Stockholders Equity"))
    cash = _latest(_annual(s, "balance", "Cash And Cash Equivalents")) or 0.0
    operating = _latest(_annual(s, "income", "Operating Income"))
    interest = _latest(_annual(s, "income", "Interest Expense"))
    current_a, current_l = _latest(_annual(s, "balance", "Current Assets")), _latest(_annual(s, "balance", "Current Liabilities"))
    out = []
    if equity is None:
        out.append(_item("Debt to equity", None, "not reported", "flag above 2"))
    elif equity <= 0:
        out.append(_item("Debt to equity", "warn", f"shareholders' equity is negative (${equity / 1e9:.1f}B), "
                         "usually from buybacks; the ratio is not meaningful", "flag above 2"))
    else:
        de = debt / equity
        out.append(_item("Debt to equity", "flag" if de > 2 else "ok",
                         f"{de:.2f} (debt ${debt / 1e9:.1f}B, cash ${cash / 1e9:.1f}B)", "flag above 2"))
    if operating is None:
        out.append(_item("Interest coverage", None, "not reported", "flag under 3"))
    elif not interest:
        out.append(_item("Interest coverage", "ok", "no interest expense reported", "flag under 3"))
    else:
        cover = operating / abs(interest)
        out.append(_item("Interest coverage", "flag" if cover < 3 else "ok",
                         f"{cover:.1f}× (operating income over interest)", "flag under 3"))
    if current_a and current_l:
        cr = current_a / current_l
        out.append(_item("Current ratio", "flag" if cr < 1 else "ok", f"{cr:.2f}", "flag under 1.0"))
    else:
        out.append(_item("Current ratio", None, "not reported", "flag under 1.0"))
    if operating is not None:
        rule = "operating income −30%, all debt refinanced 2 points dearer: flag under 1.5×, thin under 3×"
        if cash >= debt:
            out.append(_item("High-rate downturn", "ok", "more cash than debt: rates barely touch it", rule))
        else:
            stressed_interest = abs(interest or 0.0) + 0.02 * debt
            cover = 0.7 * operating / stressed_interest if stressed_interest else float("inf")
            out.append(_item("High-rate downturn", "flag" if cover < 1.5 else "warn" if cover < 3 else "ok",
                             f"coverage would be {cover:.1f}×", rule))
    return out


def _profitability(s: dict, financial: bool) -> list[dict]:
    revenue = _annual(s, "income", "Total Revenue")
    income = _annual(s, "income", "Net Income")
    fcf = _annual(s, "cashflow", "Free Cash Flow")
    out = []
    if fcf and not financial:
        ttm_fcf, basis = _ttm(s, "cashflow", "Free Cash Flow")
        ttm_rev, _ = _ttm(s, "income", "Total Revenue")
        years = sorted(fcf)
        conv = [fcf[p] / income[p] for p in years if income.get(p, 0) > 0]
        margin = f", {ttm_fcf / ttm_rev:.0%} of revenue" if ttm_rev else ""
        out.append(_item("Free cash flow", "ok" if all(fcf[p] > 0 for p in years) else "flag",
                         f"${ttm_fcf / 1e9:.1f}B ({basis}){margin}; positive in {sum(fcf[p] > 0 for p in years)} of "
                         f"{len(years)} years" + (f"; turns {statistics.mean(conv):.0%} of net income into cash" if conv else ""),
                         "positive every year"))
    elif financial:
        out.append(_item("Free cash flow", "info", "not meaningful for a bank or insurer: lending and deposits run "
                         "through its cash flow"))
    if not financial:
        roic = roic_by_year(s)
        if roic:
            latest = roic[max(roic)][0]
            avg = statistics.mean(r for r, _ in roic.values())
            out.append(_item("ROIC", "ok" if latest >= 0.15 else "flag",
                             f"{latest:.0%} latest, {avg:.0%} on average over {len(roic)} years", "target above 15%"))
    for name, line in (("Gross margin", "Gross Profit"), ("Operating margin", "Operating Income"), ("Net margin", "Net Income")):
        num = _annual(s, "income", line)
        m = {p: num[p] / revenue[p] for p in num if revenue.get(p)}
        if len(m) < 2:
            continue
        first, last = m[min(m)], m[max(m)]
        change = last - first
        trend = "expanding" if change > 0.01 else "shrinking" if change < -0.01 else "stable"
        out.append(_item(f"{name} trend", "warn" if trend == "shrinking" else "ok",
                         f"{trend}: {_pct(first)} in {min(m)[:4]} to {_pct(last)} in {max(m)[:4]} "
                         f"({'+' if change >= 0 else '−'}{abs(change) * 100:.1f} points)", "expanding or stable"))
    return out


def _valuation(s: dict, profile: dict, closes: pd.Series | None, dividends: list, analysts: dict | None,
               financial: bool) -> tuple[list[dict], dict | None]:
    out = []
    price = float(closes.dropna().iloc[-1]) if closes is not None and closes.notna().any() else None
    shares = profile.get("sharesOutstanding") or _latest(_annual(s, "balance", "Ordinary Shares Number"))
    cap = profile.get("marketCap") or (price * shares if price and shares else None)

    forward = profile.get("forwardPE") or (price / profile["forwardEps"] if price and profile.get("forwardEps", 0) > 0 else None)
    eps = _annual(s, "income", "Diluted EPS")
    history = []
    if closes is not None and eps:
        raw = raw_closes(closes, dividends)
        for p, e in sorted(eps.items()):
            at = raw.loc[:pd.Timestamp(p)]
            if e > 0 and len(at):
                history.append((p[:4], at.iloc[-1] / e))
    if forward and history:
        avg = statistics.mean(pe for _, pe in history)
        lo, hi = min(pe for _, pe in history), max(pe for _, pe in history)
        where = "above" if forward > avg * 1.15 else "below" if forward < avg * 0.85 else "in line with"
        out.append(_item("Forward P/E vs its history", "warn" if where == "above" else "info",
                         f"forward {forward:.1f} vs {avg:.1f} on average at its last {len(history)} year ends "
                         f"({lo:.0f} to {hi:.0f}): {where} its own history", "context"))
    elif forward:
        out.append(_item("Forward P/E", "info", f"{forward:.1f} (no positive past earnings to compare)", "context"))

    implied = None
    if not financial and cap:
        debt = _latest(_annual(s, "balance", "Total Debt")) or 0.0
        cash = _latest(_annual(s, "balance", "Cash And Cash Equivalents")) or 0.0
        ev = cap + debt - cash
        ebitda, basis = _ttm(s, "income", "EBITDA")
        if ebitda and ebitda > 0:
            m = ev / ebitda
            out.append(_item("EV/EBITDA", "info", f"{m:.1f}× ({basis}): " + ("low" if m < 10 else "moderate" if m < 20 else "high")
                             + f"; enterprise value ${ev / 1e9:,.0f}B", "under 10 low, 10-20 moderate, over 20 high"))
        fcf, basis = _ttm(s, "cashflow", "Free Cash Flow")
        if fcf is not None:
            y = fcf / cap
            out.append(_item("FCF yield", "info", f"{y:.1%} ({basis}): " + ("high" if y > 0.05 else "moderate" if y > 0.02 else "low"),
                             "over 5% high, 2-5% moderate, under 2% low"))
        g = implied_growth(ev, fcf or 0.0)
        if g is not None:
            revenue = _annual(s, "income", "Total Revenue")
            years = sorted(revenue)
            record = ((revenue[years[-1]] / revenue[years[0]]) ** (1 / (len(years) - 1)) - 1
                      if len(years) >= 2 and revenue[years[0]] > 0 else None)
            ahead = None
            for row in (analysts or {}).get("revenue_estimate") or []:
                if row.get("period") == "+1y" and row.get("growth") is not None:
                    ahead = row["growth"]
            bench = max(x for x in (record, ahead, -1.0) if x is not None)
            if record is None and ahead is None:
                status, verdict = "info", "no growth record to compare"
            elif g <= bench:
                status, verdict = "ok", "no faster than its record or analysts' estimate: a modest assumption"
            elif g <= bench + 0.05:
                status, verdict = "warn", "a little above its record and analysts' estimate: achievable but not cautious"
            else:
                status, verdict = "flag", "well above its record and analysts' estimate: a demanding assumption"
            if g > 0.20:
                verdict += "; few companies sustain 20%+ for a decade"
            implied = {"growth": g, "record": record, "analysts_next_year": ahead, "verdict": verdict}
            parts = [f"revenue grew {record:.1%} a year {years[0][:4]}-{years[-1][:4]}" if record is not None else None,
                     f"analysts expect {ahead:.1%} next year" if ahead is not None else None]
            out.append(_item("Implied growth (reverse DCF)", status,
                             f"today's price assumes free cash flow grows about {g:.1%} a year for {YEARS} years; "
                             + "; ".join(p for p in parts if p) + f". {verdict[0].upper() + verdict[1:]}.",
                             f"{DISCOUNT:.0%} discount rate, {TERMINAL:.1%} growth after year {YEARS}"))
        elif fcf is not None and fcf <= 0:
            out.append(_item("Implied growth (reverse DCF)", None, "free cash flow is not positive, so no growth rate "
                             "can be read from the price", ""))
    elif financial:
        equity = _latest(_annual(s, "balance", "Stockholders Equity"))
        if cap and equity and equity > 0:
            out.append(_item("Price to book", "info", f"{cap / equity:.2f}×: how banks are usually valued", "context"))
    return out, implied


def _summary(items: list[dict]) -> dict:
    counted = [i for i in items if i["status"] in ("ok", "warn", "flag")]
    return {"ok": sum(i["status"] == "ok" for i in counted), "warn": sum(i["status"] == "warn" for i in counted),
            "flag": sum(i["status"] == "flag" for i in counted)}


def checklist(statements: dict, profile: dict | None = None, closes: pd.Series | None = None,
              dividends: list | None = None, analysts: dict | None = None, financial: bool = False) -> dict:
    """Financial strength, profitability and cash flow, and valuation, each with its items and counts."""
    profile = profile or {}
    strength = _strength(statements, financial)
    profit = _profitability(statements, financial)
    value, implied = _valuation(statements, profile, closes, dividends or [], analysts, financial)
    flags = [i["name"] for i in strength if i["status"] == "flag"]
    downturn = next((i for i in strength if i["name"] == "High-rate downturn"), None)
    if financial:
        survive = "Not assessed: a bank's resilience rests on capital ratios Yahoo does not report."
    elif not flags and downturn and downturn["status"] == "ok":
        survive = "Likely: no strength test fails, and coverage holds in the downturn test."
    elif downturn and downturn["status"] == "flag":
        survive = "At risk: in the downturn test its operating income would barely cover interest."
    else:
        survive = ("Probably, with caveats: " + ", ".join(flags).lower() + " flagged." if flags
                   else "Probably: coverage would be thin in the downturn test.")
    return {"sections": [
        {"key": "strength", "title": "Financial strength", "items": strength, "counts": _summary(strength),
         "answer": survive},
        {"key": "profitability", "title": "Profitability and cash flow", "items": profit, "counts": _summary(profit)},
        {"key": "valuation", "title": "Valuation", "items": value, "counts": _summary(value), "implied": implied},
    ], "assumptions": {"discount": DISCOUNT, "terminal": TERMINAL, "years": YEARS}}
