"""The three money amounts the owner sets: Trading capital, Investing capital, daily investment.

One canonical interface (``load``, ``update``, ``snapshot``) over the ``capital`` settings
scope, validated, with every change kept in ``settings_history``.

What they mean:

- **trading_capital_usd** (default $1,000): the capital assigned to the Trading sleeve, a
  target for how much has been *put into* it, net. **Not a daily trading budget.** Trading's
  actual equity then moves with its own P&L, fees, corporate actions and explicit transfers.
- **investing_capital_usd** (default $4,800): the same target for the Investing sleeve.
- **daily_investment_usd** (default $27.61): the Investing purchase planned for each market
  session, bounded by settled cash on hand. A plan to buy, not proof that money arrived:
  deposits are recorded separately (``ledger.contribute``).

The ledger stays the financial truth. Changing a target never changes a balance: it yields a
**reallocation proposal** (``reallocation``) comparing each target with what the sleeve has
been funded with so far (contributions and transfers in, net, not its current value, so P&L is
neither erased nor topped up), and money moves only when a proposed transfer is recorded.

Earlier saved values are kept: on first use, ``trading.initial_allocation`` and
``investing.daily_contribution`` (the old keys) seed the new ones when they were set. The old
keys remain readable and writable as aliases of the new ones.

Every recommendation, paper decision and experiment can attach ``snapshot()``: the values,
the settings-history version and a hash, so a later change never alters a past record.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date

SCOPE = "capital"
DEFAULTS = {"trading_capital_usd": 1000.00, "investing_capital_usd": 4800.00, "daily_investment_usd": 27.61}
LIMITS = {"trading_capital_usd": (0.0, 10_000_000.0), "investing_capital_usd": (0.0, 10_000_000.0),
          "daily_investment_usd": (0.0, 100_000.0)}
# Old per-sleeve keys that now read and write the canonical ones.
ALIASES = {("trading", "initial_allocation"): "trading_capital_usd",
           ("investing", "daily_contribution"): "daily_investment_usd"}


@dataclass(frozen=True)
class Capital:
    trading_capital_usd: float
    investing_capital_usd: float
    daily_investment_usd: float


def validate(key: str, value) -> float:
    if key not in DEFAULTS:
        raise ValueError(f"unknown capital setting {key}; one of {', '.join(DEFAULTS)}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number of dollars, not {value!r}") from None
    lo, hi = LIMITS[key]
    if not math.isfinite(v) or v < lo or v > hi:
        raise ValueError(f"{key} must be between ${lo:,.2f} and ${hi:,.2f}")
    return round(v, 2)


def _stored(con, scope: str, key: str):
    row = con.execute("SELECT value_json FROM settings WHERE scope = ? AND key = ?", (scope, key)).fetchone()
    return json.loads(row[0]) if row else None


def ensure(store) -> None:
    """Create the capital settings once, from earlier saved values when there are any."""
    con = store.con
    if con.execute("SELECT 1 FROM settings WHERE scope = ? LIMIT 1", (SCOPE,)).fetchone():
        return
    seeds = dict(DEFAULTS)
    for (scope, key), canonical in ALIASES.items():
        old = _stored(con, scope, key)
        if old is not None:
            seeds[canonical] = validate(canonical, old)
    for key, value in seeds.items():
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
    """Validate and save; returns the new snapshot. Balances are untouched (see ``reallocation``)."""
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


def funded(store, sleeve: str, as_of) -> float:
    """Net money put into a sleeve so far: contributions and opening balances plus transfers in,
    minus withdrawals and transfers out. Not its value: P&L is not funding."""
    return store.ledger.flows(sleeve, date(1900, 1, 1), as_of)


def reallocation(store, as_of, prices: dict[str, float] | None = None) -> dict:
    """A reviewable proposal to bring each sleeve's net funding to its capital target.

    Only settled, unreserved cash can move, and only what the source sleeve really has: the
    proposal never creates money. It changes nothing until a transfer is recorded.
    """
    cap = load(store)
    out = {"as_of": str(as_of), "targets": asdict(cap), "sleeves": {}, "transfers": [], "notes": []}
    for sleeve, target in (("trading", cap.trading_capital_usd), ("investing", cap.investing_capital_usd)):
        state = store.ledger.state(sleeve, as_of, prices)
        out["sleeves"][sleeve] = {"target": target, "funded": round(funded(store, sleeve, as_of), 2),
                                  "equity": round(state.equity, 2) if prices is not None else None,
                                  "settled_cash": round(state.cash.available, 2),
                                  "unsettled": round(state.cash.unsettled, 2), "reserved": round(state.cash.reserved, 2)}
    gap = out["sleeves"]["trading"]["target"] - out["sleeves"]["trading"]["funded"]
    if abs(gap) >= 0.01:
        source, dest, amount = ("investing", "trading", gap) if gap > 0 else ("trading", "investing", -gap)
        movable = out["sleeves"][source]["settled_cash"]
        move = round(min(amount, movable), 2)
        if move >= 0.01:
            out["transfers"].append({"from": source, "to": dest, "cash": move})
        if move < round(amount, 2):
            out["notes"].append(f"{source} has ${movable:,.2f} of settled cash; ${amount - move:,.2f} more would need "
                                "sales there first (proposed separately), or a smaller target")
    total_target = cap.trading_capital_usd + cap.investing_capital_usd
    total_funded = out["sleeves"]["trading"]["funded"] + out["sleeves"]["investing"]["funded"]
    if total_target > total_funded + 0.01:
        out["notes"].append(f"targets add up to ${total_target:,.2f} but ${total_funded:,.2f} has been put in: the "
                            "difference needs a deposit, recorded when it arrives; no transfer can create it")
    out["notes"].append("Nothing moves until a transfer is recorded. Targets are net funding, not value: profits and "
                        "losses stay with the sleeve that earned them.")
    return out


def daily_plan(store, as_of, reserved_today: float = 0.0) -> dict:
    """The planned Investing purchase for the next market session, bounded by settled cash."""
    from tradingagents.ledger.sessions import is_session, next_session, session_on_or_after

    cap = load(store)
    day = session_on_or_after(as_of)
    state = store.ledger.state("investing", as_of)
    available = max(0.0, state.cash.available - reserved_today)
    buy = round(min(cap.daily_investment_usd, available), 2)
    notes = []
    if not is_session(as_of):
        notes.append(f"{as_of} is not a market session; the plan is for {day}")
    if buy < cap.daily_investment_usd:
        notes.append(f"settled cash ${available:,.2f} covers ${buy:,.2f} of the ${cap.daily_investment_usd:,.2f} "
                     "planned; the rest is skipped this session (it does not carry over)")
    notes.append("A purchase plan, not a deposit: record deposits when they arrive.")
    return {"session": str(day), "next_session": str(next_session(day)), "planned": cap.daily_investment_usd,
            "settled_available": round(available, 2), "buy": buy, "shortfall": round(cap.daily_investment_usd - buy, 2),
            "notes": notes}
