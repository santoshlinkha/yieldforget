"""The money amounts the owner sets for the trading account: the trading budget (and, from earlier versions, the
daily SGOV buy, the match rate and the dry-yield line, which the page no longer shows).

- **trading_budget_usd** (default $0): the money set aside for buying stocks. What is left of it is the budget
  less what went into stocks since it was last set, plus what came back from sales (``ledger.income.budget``).

One canonical interface (``load``, ``update``, ``snapshot``) over the ``capital`` settings
scope, validated, with every change kept in ``settings_history``.

What they mean:

- **daily_sgov_usd** (default $29.00): the deposit bought into SGOV each market session. SGOV is
  the base of the account and is never traded; it earns interest every month.
- **match_rate** (default 3%): the broker's match on contributions (Robinhood's IRA match is 3%
  with Gold, 1% without; the match is clawed back on early withdrawal, see the broker's terms).
- **dry_yield** (default 2%): the yearly SGOV yield under which its interest has "dried up", and
  the app suggests where else the daily deposit could go (``ledger.income.health``).

The trading budget is not a setting: it is the SGOV interest and the match earned since the
strategy started, less what has gone into stocks, plus what came back (``ledger.income.budget``).
The ledger stays the financial truth: changing a setting never changes a balance.

Every recommendation, paper decision and experiment can attach ``snapshot()``: the values,
the settings-history version and a hash, so a later change never alters a past record.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

SCOPE = "capital"
DEFAULTS = {"trading_budget_usd": 0.0, "daily_sgov_usd": 29.00, "match_rate": 0.03, "dry_yield": 0.02}
LIMITS = {"trading_budget_usd": (0.0, 10_000_000.0), "daily_sgov_usd": (0.0, 100_000.0), "match_rate": (0.0, 0.10),
          "dry_yield": (0.0, 0.10)}
UNITS = {"trading_budget_usd": "dollars", "daily_sgov_usd": "dollars", "match_rate": "a fraction (0.03 = 3%)",
         "dry_yield": "a fraction (0.02 = 2%)"}
# Keys of the retired Investing/Trading split, removed when this scope is next written.
RETIRED = ("trading_capital_usd", "investing_capital_usd", "daily_investment_usd")


@dataclass(frozen=True)
class Capital:
    trading_budget_usd: float
    daily_sgov_usd: float
    match_rate: float
    dry_yield: float


def validate(key: str, value) -> float:
    if key not in DEFAULTS:
        raise ValueError(f"unknown capital setting {key}; one of {', '.join(DEFAULTS)}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be {UNITS[key]}, not {value!r}") from None
    lo, hi = LIMITS[key]
    if not math.isfinite(v) or v < lo or v > hi:
        raise ValueError(f"{key} must be between {lo:g} and {hi:g} ({UNITS[key]})")
    return round(v, 2) if key.endswith("_usd") else round(v, 4)


def ensure(store) -> None:
    """Create any capital setting that is missing, at its default."""
    for key, value in DEFAULTS.items():
        if store.con.execute("SELECT 1 FROM settings WHERE scope = ? AND key = ?", (SCOPE, key)).fetchone() is None:
            store.set(SCOPE, key, value, source="migration")


def load(store) -> Capital:
    ensure(store)
    return Capital(**{k: float(store.get(SCOPE, k)) for k in DEFAULTS})


def version(store) -> int:
    """The id of the latest change to any capital setting: a monotonic settings version."""
    ensure(store)
    row = store.con.execute("SELECT MAX(id) FROM settings_history WHERE scope = ?", (SCOPE,)).fetchone()
    return int(row[0] or 0)


def snapshot(store) -> dict:
    """The values, their settings version and a hash: what a record attaches."""
    values = asdict(load(store))
    text = json.dumps(values, sort_keys=True)
    return {**values, "version": version(store), "hash": hashlib.sha256(text.encode()).hexdigest()[:12]}


def update(store, values: dict, source: str = "user") -> dict:
    """Validate and save; returns the new snapshot. Balances are untouched."""
    ensure(store)
    clean = {k: validate(k, v) for k, v in values.items()}
    for key, value in clean.items():
        if float(store.get(SCOPE, key)) != value:
            store.set(SCOPE, key, value, source=source)
    return snapshot(store)


def history(store, limit: int = 50) -> list[dict]:
    return [dict(r) for r in store.con.execute(
        "SELECT id, key, value_json, changed_at, source FROM settings_history WHERE scope = ? ORDER BY id DESC LIMIT ?",
        (SCOPE, limit))]
