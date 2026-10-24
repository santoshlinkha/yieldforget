"""Sleeve accounting inside one IRA: Investing, Trading, and Combined (their sum).

Sleeves are bookkeeping, not separate brokerage accounts: the broker sees one
account, and its rules (settlement, restrictions, any forced liquidation) apply
to the whole of it. A sleeve owns the shares and cash its events give it:

- holdings = the sum of each ticker's signed quantities;
- settled cash = the cash of events whose settle date has arrived. A buy is
  debited at once, since a cash account can only spend settled money;
- unsettled cash = sale proceeds still waiting for their settle date (T+1);
- reserved cash = settled cash set aside for open orders or option collateral;
- available to buy = settled minus reserved.

Holdings change only through recorded events: fills, transfers, contributions,
dividends, corporate actions or audited adjustments. Accepting a recommendation
changes nothing.

An internal transfer is two events with one ``transfer_id``: the same shares or
cash leave one sleeve and enter the other at the same price, so the combined
account is unchanged, and for each sleeve the transfer is a capital flow, not
profit. Contributions and withdrawals are the account's only external flows.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field

from .db import now_utc, transaction
from .sessions import as_date, next_session

SLEEVES = ("investing", "trading")
DEFAULT_ACCOUNT = "rh-roth-ira"
EPS = 1e-9


class LedgerError(ValueError):
    """A recorded event would break the books (sell more than held, spend unsettled cash...)."""


@dataclass
class Cash:
    settled: float = 0.0
    unsettled: float = 0.0
    reserved: float = 0.0

    @property
    def available(self) -> float:
        return self.settled - self.reserved

    @property
    def total(self) -> float:
        return self.settled + self.unsettled


@dataclass
class SleeveState:
    sleeve: str                     # 'investing', 'trading' or 'combined'
    as_of: str
    positions: dict[str, float]
    cash: Cash
    values: dict[str, float] = field(default_factory=dict)   # by ticker, at the prices given

    @property
    def equity(self) -> float:
        return sum(self.values.values()) + self.cash.total


def _uid(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:32]


class Ledger:
    def __init__(self, con, account_id: str = DEFAULT_ACCOUNT):
        self.con, self.account_id = con, account_id

    # --- setup ---

    def ensure_account(self, name="Robinhood Roth IRA", broker="robinhood", account_type="roth_ira",
                       cash_account=True):
        with transaction(self.con):
            self.con.execute("INSERT OR IGNORE INTO accounts VALUES (?, ?, ?, ?, ?, ?)",
                             (self.account_id, name, broker, account_type, int(cash_account), now_utc()))
            for sleeve in SLEEVES:
                self.con.execute("INSERT OR IGNORE INTO sleeves VALUES (?, ?, ?)", (self.account_id, sleeve, now_utc()))

    # --- writing ---

    def record(self, kind: str, sleeve: str, trade_date, *, settle_date=None, ticker=None, quantity=0.0,
               price=None, cash=0.0, transfer_id=None, order_id=None, source="manual", source_ref=None,
               note=None, supersedes=None, uid=None) -> tuple[int, bool]:
        """Append one event; ``(id, True)``, or ``(existing id, False)`` when its uid is already recorded.

        The uid is the dedupe key: a broker's reference when there is one, otherwise
        a hash of the event's content, so importing the same file twice adds nothing.
        """
        if sleeve not in SLEEVES:
            raise LedgerError(f"unknown sleeve {sleeve}")
        trade = as_date(trade_date).isoformat()
        settle = as_date(settle_date).isoformat() if settle_date else trade
        uid = uid or _uid(self.account_id, kind, sleeve, trade, settle, ticker, round(quantity, 9),
                          price, round(cash, 6), transfer_id, order_id, source, source_ref)
        with transaction(self.con):
            row = self.con.execute("SELECT id FROM events WHERE event_uid = ?", (uid,)).fetchone()
            if row:
                return row["id"], False
            cur = self.con.execute(
                "INSERT INTO events (event_uid, account_id, sleeve, kind, trade_date, settle_date, ticker, quantity,"
                " price, cash, transfer_id, order_id, source, source_ref, note, supersedes, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, self.account_id, sleeve, kind, trade, settle, ticker, float(quantity), price, float(cash),
                 transfer_id, order_id, source, source_ref, note, supersedes, now_utc()))
            return cur.lastrowid, True

    def void(self, event_id: int, reason: str, replaced_by: int | None = None):
        """Take an event out of the books; the event itself stays, marked void."""
        with transaction(self.con):
            self.con.execute("INSERT INTO event_voids VALUES (?, ?, ?, ?)", (event_id, now_utc(), reason, replaced_by))

    def contribute(self, amount: float, trade_date, sleeve="investing", settle_date=None, source="manual",
                   source_ref=None, note=None) -> int:
        """New money from outside the IRA: an external flow, never profit."""
        if amount <= 0:
            raise LedgerError("a contribution must be positive")
        return self.record("contribution", sleeve, trade_date, settle_date=settle_date, cash=amount,
                           source=source, source_ref=source_ref, note=note)[0]

    def withdraw(self, amount: float, trade_date, sleeve="investing", source="manual", note=None) -> int:
        if amount <= 0:
            raise LedgerError("a withdrawal must be positive")
        if self.state(sleeve, trade_date).cash.available < amount - EPS:
            raise LedgerError(f"{sleeve} has less than ${amount:,.2f} of settled, unreserved cash")
        return self.record("withdrawal", sleeve, trade_date, cash=-amount, source=source, note=note)[0]

    def fill(self, sleeve: str, side: str, ticker: str, quantity: float, price: float, trade_date, *,
             fee: float = 0.0, settle_date=None, order_id=None, source="manual", source_ref=None,
             strict: bool = True, note=None, uid=None) -> tuple[int, list[str]]:
        """A buy or sell that happened. Buys spend settled cash at once; sale proceeds
        settle the next session. ``strict`` refuses what the books cannot pay for;
        a broker's own record (``strict=False``) is kept with a warning instead."""
        if side not in ("buy", "sell") or quantity <= 0 or price <= 0:
            raise LedgerError("a fill needs side buy or sell, a positive quantity and a positive price")
        ticker = ticker.upper()
        state = self.state(sleeve, trade_date)
        warnings = []
        if side == "buy":
            cost = quantity * price + fee
            if cost > state.cash.available + 0.005:
                warnings.append(f"buy of ${cost:,.2f} {ticker} is more than {sleeve}'s ${state.cash.available:,.2f} "
                                f"of settled, unreserved cash")
            cash, settle, qty = -cost, trade_date, quantity
        else:
            if quantity > state.positions.get(ticker, 0.0) + 1e-6:
                warnings.append(f"sells {quantity:g} {ticker} but {sleeve} holds {state.positions.get(ticker, 0.0):g}")
            cash, qty = quantity * price - fee, -quantity
            settle = settle_date or next_session(trade_date)
        if warnings and strict:
            raise LedgerError("; ".join(warnings))
        event_id, _ = self.record("fill", sleeve, trade_date, settle_date=settle, ticker=ticker, quantity=qty,
                                  price=price, cash=cash, order_id=order_id, source=source,
                                  source_ref=source_ref, note=note, uid=uid)
        return event_id, warnings

    def transfer(self, from_sleeve: str, to_sleeve: str, trade_date, *, ticker: str | None = None,
                 quantity: float = 0.0, cash: float = 0.0, price: float | None = None, note=None) -> str:
        """Move shares (at ``price``, for the flow's value) or settled cash between sleeves.

        The combined account does not change. Only what the source sleeve really
        holds can move: shares it owns, or settled cash not reserved, so no
        sleeve can be funded with money the account does not have.
        """
        if from_sleeve == to_sleeve:
            raise LedgerError("a transfer needs two different sleeves")
        if bool(ticker) == bool(cash):
            raise LedgerError("transfer either shares (ticker, quantity, price) or cash, not both")
        state = self.state(from_sleeve, trade_date)
        transfer_id = uuid.uuid4().hex
        with transaction(self.con):
            if ticker:
                ticker = ticker.upper()
                if quantity <= 0 or price is None or price <= 0:
                    raise LedgerError("a share transfer needs a positive quantity and the price it moves at")
                if quantity > state.positions.get(ticker, 0.0) + 1e-9:
                    raise LedgerError(f"{from_sleeve} holds {state.positions.get(ticker, 0.0):g} {ticker}, "
                                      f"not {quantity:g}")
                self.record("transfer", from_sleeve, trade_date, ticker=ticker, quantity=-quantity, price=price,
                            transfer_id=transfer_id, note=note)
                self.record("transfer", to_sleeve, trade_date, ticker=ticker, quantity=quantity, price=price,
                            transfer_id=transfer_id, note=note)
            else:
                if cash <= 0:
                    raise LedgerError("a cash transfer must be positive")
                if cash > state.cash.available + 0.005:
                    raise LedgerError(f"{from_sleeve} has ${state.cash.available:,.2f} of settled, unreserved cash, "
                                      f"not ${cash:,.2f}; unsettled proceeds cannot move until they settle")
                self.record("transfer", from_sleeve, trade_date, cash=-cash, transfer_id=transfer_id, note=note)
                self.record("transfer", to_sleeve, trade_date, cash=cash, transfer_id=transfer_id, note=note)
        return transfer_id

    def reserve(self, sleeve: str, amount: float, kind: str = "order", ref: str | None = None, as_of=None) -> str:
        state = self.state(sleeve, as_of or now_utc()[:10])
        if amount > state.cash.available + 0.005:
            raise LedgerError(f"cannot reserve ${amount:,.2f}: {sleeve} has ${state.cash.available:,.2f} available")
        rid = uuid.uuid4().hex
        with transaction(self.con):
            self.con.execute("INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                             (rid, self.account_id, sleeve, kind, amount, ref, now_utc()))
        return rid

    def release(self, reservation_id: str):
        with transaction(self.con):
            self.con.execute("UPDATE reservations SET released_at = ? WHERE id = ? AND released_at IS NULL",
                             (now_utc(), reservation_id))

    # --- reading ---

    def events(self, sleeve: str | None = None, as_of=None, include_void=False) -> list[dict]:
        sql = ("SELECT e.*, v.event_id IS NOT NULL AS void FROM events e LEFT JOIN event_voids v ON v.event_id = e.id"
               " WHERE e.account_id = ?")
        args: list = [self.account_id]
        if sleeve:
            sql += " AND e.sleeve = ?"
            args.append(sleeve)
        if as_of:
            sql += " AND e.trade_date <= ?"
            args.append(as_date(as_of).isoformat())
        rows = [dict(r) for r in self.con.execute(sql + " ORDER BY e.trade_date, e.id", args)]
        return rows if include_void else [r for r in rows if not r["void"]]

    def state(self, sleeve: str | None, as_of, prices: dict[str, float] | None = None) -> SleeveState:
        """Holdings and cash of one sleeve, or of the whole account (``None`` or 'combined'), at
        the end of ``as_of``."""
        sleeve = None if sleeve in (None, "combined") else sleeve
        day = as_date(as_of).isoformat()
        positions: dict[str, float] = {}
        cash = Cash()
        for e in self.events(sleeve, day):
            if e["ticker"] and e["quantity"]:
                positions[e["ticker"]] = positions.get(e["ticker"], 0.0) + e["quantity"]
            if e["cash"]:
                if e["settle_date"] <= day:
                    cash.settled += e["cash"]
                else:
                    cash.unsettled += e["cash"]
        sql = "SELECT COALESCE(SUM(amount), 0) FROM reservations WHERE account_id = ? AND released_at IS NULL"
        args = [self.account_id]
        if sleeve:
            sql += " AND sleeve = ?"
            args.append(sleeve)
        cash.reserved = float(self.con.execute(sql, args).fetchone()[0])
        positions = {t: q for t, q in positions.items() if abs(q) > EPS}
        values = {t: q * prices[t] for t, q in positions.items() if prices and t in prices}
        return SleeveState(sleeve or "combined", day, positions, cash, values)

    def flows(self, sleeve: str | None, start, end) -> float:
        """Net capital put into a sleeve (or, for None, the account) after ``start`` up to ``end``.

        External contributions minus withdrawals, plus, for one sleeve, transfers in
        minus transfers out, shares valued at the price they moved at. For the whole
        account the two legs of every internal transfer cancel.
        """
        total = 0.0
        a, b = as_date(start).isoformat(), as_date(end).isoformat()
        for e in self.events(None if sleeve in (None, "combined") else sleeve, b):
            if not a < e["trade_date"] <= b:
                continue
            if e["kind"] in ("contribution", "withdrawal", "opening_cash"):
                total += e["cash"]
            elif e["kind"] == "opening_position":
                total += e["quantity"] * (e["price"] or 0.0)
            elif e["kind"] == "transfer" and sleeve not in (None, "combined"):
                total += e["cash"] + e["quantity"] * (e["price"] or 0.0)
        return total

    def reconcile(self, broker_positions: dict[str, float], broker_cash: float | None, as_of,
                  tolerance: float = 1e-4) -> list[str]:
        """Differences between the books (all sleeves together) and what the broker shows."""
        state = self.state(None, as_of)
        problems = []
        for t in sorted(set(state.positions) | set(broker_positions)):
            mine, theirs = state.positions.get(t, 0.0), broker_positions.get(t, 0.0)
            if abs(mine - theirs) > tolerance:
                problems.append(f"{t}: books {mine:g}, broker {theirs:g}")
        if broker_cash is not None and abs(state.cash.total - broker_cash) > 0.01:
            problems.append(f"cash: books ${state.cash.total:,.2f}, broker ${broker_cash:,.2f}")
        return problems
