"""Quantitative moat indicators: signs of a durable competitive advantage in a company's own statements.

Not Morningstar's moat rating, which is a proprietary analyst judgment. These are
five checks, over the years of annual statements Yahoo gives (usually four), on
what a moat should leave in the numbers.

For most companies:

- **Pricing power**: a gross margin above the sector's bar on average, never far
  below it, and stable (a year-to-year spread under 5 points).
- **Returns on capital**: return on invested capital above the bar in every year.
  ROIC = operating income × (1 − the year's own tax rate) ÷ invested capital,
  averaged over the year's start and end; invested capital is equity + total debt
  − the cash beyond 2% of revenue a business needs to run.
- **Profitability**: an operating margin above the sector's bar on average.
- **Cash generation**: positive free cash flow in every year.
- **Steady growth**: revenue grew in at least all but one of the year-on-year steps.

Banks and insurers report no gross margin or operating income, and debt is their
raw material, so they get their own five: return on equity, return on assets, net
margin, growth in book value per share, and revenue growth.

The bars are the app's rough judgment of what is high for a sector (software
earns wider margins than a grocer), not measured sector medians. The verdict
counts the checks passed: 4 or 5 "strong indicators", 2 or 3 "some", 0 or 1
"few". A check without the data it needs is "not enough data" and is not counted
either way; fewer than three counted checks give no verdict. Funds have no
statements at all.
"""

from __future__ import annotations

import statistics

TAX = 0.21  # when a year's own rate can't be worked out
OPERATING_CASH = 0.02  # of revenue: cash a business needs to run, the rest is excess

# Gross margin on average, its floor, operating margin on average, ROIC every year.
DEFAULT_BARS = (0.40, 0.30, 0.15, 0.15)
SECTOR_BARS = {
    "Technology": (0.50, 0.40, 0.20, 0.15),
    "Communication Services": (0.45, 0.35, 0.18, 0.15),
    "Healthcare": (0.50, 0.40, 0.15, 0.15),
    "Consumer Defensive": (0.30, 0.20, 0.10, 0.15),
    "Consumer Cyclical": (0.35, 0.25, 0.10, 0.15),
    "Industrials": (0.30, 0.20, 0.12, 0.15),
    "Basic Materials": (0.25, 0.15, 0.12, 0.15),
    "Energy": (0.25, 0.15, 0.12, 0.15),
    # Regulated and asset-heavy: capital earns less, and that is the business.
    "Utilities": (0.35, 0.25, 0.15, 0.08),
    "Real Estate": (0.50, 0.40, 0.25, 0.08),
}
# Hardware makes things, software copies them: a hardware maker's margins are lower at any moat.
INDUSTRY_BARS = {
    "Consumer Electronics": (0.35, 0.25, 0.12, 0.15),
    "Computer Hardware": (0.35, 0.25, 0.12, 0.15),
    "Electronic Components": (0.30, 0.20, 0.12, 0.15),
    "Communication Equipment": (0.40, 0.30, 0.12, 0.15),
}
# Banks and insurers: return on equity every year, return on assets and net margin on average.
FINANCIAL_BARS = (0.12, 0.01, 0.20)


def _series(statements: dict, kind: str, line: str) -> dict[str, float]:
    """A line's values by period date; years a statement doesn't cover are simply absent."""
    table = (statements.get(kind) or {}).get("annual") or {}
    values = (table.get("rows") or {}).get(line) or []
    return {p: v for p, v in zip(table.get("periods") or [], values, strict=False) if v is not None}


def _ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def _average(series: dict, periods: list[str], i: int):
    """The value over year ``i``: the mean of its start (the year before's end) and its end."""
    end = series.get(periods[i])
    start = series.get(periods[i + 1]) if i + 1 < len(periods) else None
    if end is None:
        return None
    return (end + start) / 2 if start is not None else end


def _check(name: str, passed: bool | None, value: str, rule: str) -> dict:
    return {"name": name, "passed": passed, "value": value, "rule": rule}


def _growth(name: str, values: list[float], what: str) -> dict:
    # Statements list the newest year first: growth is each year against the one before it.
    steps = [new > old for new, old in zip(values, values[1:], strict=False)]
    return _check(name, sum(steps) >= len(steps) - 1 if len(steps) >= 2 else None,
                  f"{what} grew in {sum(steps)} of {len(steps)} years" if steps else "not enough data",
                  "growth in all but at most one year")


def _is_financial(sector: str | None, line) -> bool:
    if line("income", "Operating Income") or line("income", "Gross Profit"):
        return False
    return sector == "Financial Services" or bool(line("income", "Net Income") and line("balance", "Total Assets"))


def _roic(line, periods: list[str]) -> dict[str, tuple[float, float]]:
    """``{period: (ROIC, tax rate)}``, newest first, for the years with the data.

    ROIC = operating income × (1 − the year's own tax rate) ÷ invested capital averaged
    over the year's start and end; invested capital is equity + total debt − the cash
    beyond 2% of revenue. A year with no positive capital is left out.
    """
    revenue, operating = line("income", "Total Revenue"), line("income", "Operating Income")
    pretax, tax = line("income", "Pretax Income"), line("income", "Tax Provision")
    equity, debt, cash = line("balance", "Stockholders Equity"), line("balance", "Total Debt"), line("balance", "Cash And Cash Equivalents")

    def invested(p):
        if p not in equity:
            return None
        excess = max(0.0, cash.get(p, 0.0) - OPERATING_CASH * revenue[p]) if p in revenue else cash.get(p, 0.0)
        return equity[p] + debt.get(p, 0.0) - excess

    capital = {p: c for p in periods if (c := invested(p)) is not None}
    out = {}
    for i, p in enumerate(periods):
        rate = _ratio(tax.get(p), pretax.get(p)) if pretax.get(p, 0) > 0 else None
        rate = min(max(rate, 0.0), 0.5) if rate is not None else TAX
        c = _average(capital, periods, i)
        if p in operating and c is not None and c > 0:
            out[p] = (operating[p] * (1 - rate) / c, rate)
    return out


def roic_by_year(statements: dict) -> dict[str, tuple[float, float]]:
    """``{period: (ROIC, tax rate)}`` from ``history.financials`` output; see ``_roic``."""
    periods = ((statements.get("income") or {}).get("annual") or {}).get("periods") or []
    return _roic(lambda kind, name: _series(statements, kind, name), periods)


def _company_checks(line, periods, sector, industry) -> tuple[list[dict], str]:
    if industry in INDUSTRY_BARS:
        bars, basis = INDUSTRY_BARS[industry], f"bars for {industry} companies"
    elif sector in SECTOR_BARS:
        bars, basis = SECTOR_BARS[sector], f"bars for {sector} companies"
    else:
        bars, basis = DEFAULT_BARS, "general bars" + (f" ({sector} has none of its own)" if sector else "")
    gm_bar, gm_floor, om_bar, roic_bar = bars
    revenue, gross, operating = line("income", "Total Revenue"), line("income", "Gross Profit"), line("income", "Operating Income")
    checks = []

    gm = [m for m in (_ratio(gross.get(p), revenue.get(p)) for p in periods) if m is not None]
    rule = f"{gm_bar:.0%}+ on average, never under {gm_floor:.0%}, within 5 points"
    if len(gm) >= 2:
        avg, spread = statistics.mean(gm), max(gm) - min(gm)
        checks.append(_check("Pricing power", avg >= gm_bar and min(gm) >= gm_floor and spread < 0.05,
                             f"gross margin {avg:.0%} on average ({min(gm):.0%} to {max(gm):.0%})", rule))
    else:
        checks.append(_check("Pricing power", None, "no gross margin reported", rule))

    by_year = _roic(line, periods)
    roic, rates = [r for r, _ in by_year.values()], [t for _, t in by_year.values()]
    rule = f"{roic_bar:.0%}+ every year"
    if len(roic) >= 2:
        checks.append(_check("Returns on capital", min(roic) >= roic_bar,
                             f"ROIC {statistics.mean(roic):.0%} on average, lowest {min(roic):.0%} "
                             f"(tax rate {statistics.mean(rates):.0%})", rule))
    else:
        checks.append(_check("Returns on capital", None, "not enough data", rule))

    om = [m for m in (_ratio(operating.get(p), revenue.get(p)) for p in periods) if m is not None]
    checks.append(_check("Profitability", statistics.mean(om) >= om_bar if om else None,
                         f"operating margin {statistics.mean(om):.0%} on average" if om else "not enough data",
                         f"{om_bar:.0%}+ on average"))

    f = [fcf for p in periods if (fcf := line("cashflow", "Free Cash Flow").get(p)) is not None]
    checks.append(_check("Cash generation", all(x > 0 for x in f) if len(f) >= 2 else None,
                         f"free cash flow positive in {sum(x > 0 for x in f)} of {len(f)} years" if f else "not enough data",
                         "positive every year"))

    checks.append(_growth("Steady growth", [revenue[p] for p in periods if p in revenue], "revenue"))
    return checks, basis


def _financial_checks(line, periods) -> tuple[list[dict], str]:
    roe_bar, roa_bar, nm_bar = FINANCIAL_BARS
    revenue, income = line("income", "Total Revenue"), line("income", "Net Income")
    equity, assets, shares = line("balance", "Stockholders Equity"), line("balance", "Total Assets"), line("balance", "Ordinary Shares Number")
    roe, roa = [], []
    for i, p in enumerate(periods):
        e, a = _average(equity, periods, i), _average(assets, periods, i)
        if p in income and e and e > 0:
            roe.append(income[p] / e)
        if p in income and a and a > 0:
            roa.append(income[p] / a)
    checks = [
        _check("Returns on equity", min(roe) >= roe_bar if len(roe) >= 2 else None,
               f"ROE {statistics.mean(roe):.0%} on average, lowest {min(roe):.0%}" if roe else "not enough data",
               f"{roe_bar:.0%}+ every year"),
        _check("Returns on assets", statistics.mean(roa) >= roa_bar if len(roa) >= 2 else None,
               f"ROA {statistics.mean(roa):.2%} on average" if roa else "not enough data",
               f"{roa_bar:.0%}+ on average"),
    ]
    nm = [m for m in (_ratio(income.get(p), revenue.get(p)) for p in periods) if m is not None]
    checks.append(_check("Profitability", statistics.mean(nm) >= nm_bar if nm else None,
                         f"net margin {statistics.mean(nm):.0%} on average" if nm else "not enough data",
                         f"{nm_bar:.0%}+ on average"))
    book = [b for p in periods if (b := _ratio(equity.get(p), shares.get(p))) is not None]
    checks.append(_growth("Book value growth", book, "book value per share"))
    checks.append(_growth("Steady growth", [revenue[p] for p in periods if p in revenue], "revenue"))
    return checks, "bank and insurer checks: debt is their raw material, so no gross margin or ROIC"


def moat(statements: dict, sector: str | None = None, industry: str | None = None) -> dict:
    """The five checks and a verdict, from ``history.financials`` output and Yahoo's sector and industry."""
    periods = ((statements.get("income") or {}).get("annual") or {}).get("periods") or []

    def line(kind, name):
        return _series(statements, kind, name)

    if _is_financial(sector, line):
        checks, basis = _financial_checks(line, periods)
    else:
        checks, basis = _company_checks(line, periods, sector, industry)

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
    return {"verdict": verdict, "passed": passed, "scored": len(scored), "checks": checks, "basis": basis,
            "years": f"{periods[-1][:4]} to {periods[0][:4]}" if periods else None}
