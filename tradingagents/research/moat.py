"""Quantitative moat indicators: signs of a durable competitive advantage in a company's own statements.

Not Morningstar's moat rating, which is a proprietary analyst judgment. These are
five checks, over the years of annual statements Yahoo gives (usually four), on
what a moat should leave in the numbers:

- **Pricing power**: gross margin of 40% or more on average, never below 30%,
  and stable (a year-to-year spread under 5 points).
- **Returns on capital**: return on invested capital of 15% or more in every year.
  ROIC = operating income × (1 − 21% tax) ÷ (equity + total debt − cash).
- **Profitability**: operating margin of 15% or more on average.
- **Cash generation**: positive free cash flow in every year.
- **Steady growth**: revenue grew in at least all but one of the year-on-year steps.

The verdict counts the checks passed: 4 or 5 "strong indicators", 2 or 3 "some",
0 or 1 "few". A check without the data it needs is "not enough data" and is not
counted either way. Banks, insurers and REITs report differently (no gross
margin, debt is their raw material), so the checks say less about them; funds
have no statements at all.
"""

from __future__ import annotations

import statistics

TAX = 0.21


def _series(statements: dict, kind: str, line: str) -> list[float | None]:
    table = (statements.get(kind) or {}).get("annual") or {}
    return list((table.get("rows") or {}).get(line) or [])


def _ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def _check(name: str, passed: bool | None, value: str, rule: str) -> dict:
    return {"name": name, "passed": passed, "value": value, "rule": rule}


def moat(statements: dict) -> dict:
    """The five checks and a verdict, from ``history.financials`` output."""
    periods = ((statements.get("income") or {}).get("annual") or {}).get("periods") or []
    revenue = _series(statements, "income", "Total Revenue")
    gross = _series(statements, "income", "Gross Profit")
    operating = _series(statements, "income", "Operating Income")
    equity = _series(statements, "balance", "Stockholders Equity")
    debt = _series(statements, "balance", "Total Debt")
    cash = _series(statements, "balance", "Cash And Cash Equivalents")
    fcf = _series(statements, "cashflow", "Free Cash Flow")
    n = len(periods)
    checks = []

    gm = [m for m in (_ratio(g, r) for g, r in zip(gross, revenue, strict=False)) if m is not None]
    if len(gm) >= 2:
        avg, spread = statistics.mean(gm), max(gm) - min(gm)
        checks.append(_check("Pricing power", avg >= 0.40 and min(gm) >= 0.30 and spread < 0.05,
                             f"gross margin {avg:.0%} on average ({min(gm):.0%} to {max(gm):.0%})",
                             "40%+ on average, never under 30%, within 5 points"))
    else:
        checks.append(_check("Pricing power", None, "no gross margin reported", "40%+ on average, stable"))

    roic = []
    for i in range(n):
        o = operating[i] if i < len(operating) else None
        capital = sum(x[i] or 0.0 for x in (equity, debt) if i < len(x)) - (cash[i] or 0.0 if i < len(cash) else 0.0)
        if o is not None and capital > 0:
            roic.append(o * (1 - TAX) / capital)
    if len(roic) >= 2:
        checks.append(_check("Returns on capital", min(roic) >= 0.15,
                             f"ROIC {statistics.mean(roic):.0%} on average, lowest {min(roic):.0%}",
                             "15%+ every year"))
    else:
        checks.append(_check("Returns on capital", None, "not enough data", "15%+ every year"))

    om = [m for m in (_ratio(o, r) for o, r in zip(operating, revenue, strict=False)) if m is not None]
    checks.append(_check("Profitability", statistics.mean(om) >= 0.15 if om else None,
                         f"operating margin {statistics.mean(om):.0%} on average" if om else "not enough data",
                         "15%+ on average"))

    f = [x for x in fcf if x is not None]
    checks.append(_check("Cash generation", all(x > 0 for x in f) if len(f) >= 2 else None,
                         f"free cash flow positive in {sum(x > 0 for x in f)} of {len(f)} years" if f else "not enough data",
                         "positive every year"))

    # Statements list the newest year first: growth is each year against the one before it.
    r = [x for x in revenue if x is not None]
    steps = [new > old for new, old in zip(r, r[1:], strict=False)]
    checks.append(_check("Steady growth", sum(steps) >= len(steps) - 1 if len(steps) >= 2 else None,
                         f"revenue grew in {sum(steps)} of {len(steps)} years" if steps else "not enough data",
                         "growth in all but at most one year"))

    scored = [c for c in checks if c["passed"] is not None]
    passed = sum(1 for c in scored if c["passed"])
    if len(scored) < 3:
        verdict = "not enough data"
    elif passed >= 4:
        verdict = "strong indicators"
    elif passed >= 2:
        verdict = "some indicators"
    else:
        verdict = "few indicators"
    return {"verdict": verdict, "passed": passed, "scored": len(scored), "checks": checks,
            "years": f"{periods[-1][:4]} to {periods[0][:4]}" if periods else None}
