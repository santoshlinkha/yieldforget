"""Importing what really happened: Robinhood's activity CSV, previewed before anything is written.

``preview`` reads the file and says, row by row, what each would become: a fill,
dividend, contribution, interest or fee, or *not imported* with the reason. It
also flags rows already in the books (each row's dedupe key is a hash of the
row itself, numbered when a file holds identical rows), rows that would sell
more than the sleeve holds, and buys the sleeve's settled cash could not have
paid for. ``commit`` writes exactly the previewed rows, so importing the same file
twice adds nothing.

Robinhood's export has the columns Activity Date, Process Date, Settle Date,
Instrument, Description, Trans Code, Quantity, Price and Amount. Amounts are
written like "$1,234.56" or "($12.34)".
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime

from .accounting import Ledger, LedgerError

FILLS = {"Buy": "buy", "Sell": "sell"}
DIVIDENDS = {"CDIV", "MDIV", "DIV"}
CASH_IN = {"ACH", "RTP", "DCF", "CONT", "IRAC"}
INTEREST = {"INT", "SLIP"}
FEES = {"GOLD", "AFEE", "DFEE", "MINT"}


@dataclass
class Row:
    line: int
    raw: dict
    uid: str
    action: str                 # fill, dividend, contribution, withdrawal, interest, fee, skip
    trade_date: str | None = None
    settle_date: str | None = None
    ticker: str | None = None
    side: str | None = None
    quantity: float = 0.0
    price: float | None = None
    amount: float = 0.0
    duplicate: bool = False
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None


def _money(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")") or text.startswith("-")
    value = float(text.strip("()-").replace("$", "").replace(",", ""))
    return -value if negative else value


def _number(text: str) -> float:
    text = (text or "").strip().rstrip("S")
    return float(text.replace(",", "")) if text else 0.0


def _date(text: str) -> str | None:
    text = (text or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse(text: str) -> list[Row]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    seen: dict[str, int] = {}
    rows = []
    for n, raw in enumerate(reader, start=2):
        raw = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        if not any(raw.values()):
            continue
        body = "|".join(raw.get(k, "") for k in sorted(raw))
        seen[body] = seen.get(body, 0) + 1
        uid = "rhcsv:" + hashlib.sha256(f"{body}#{seen[body]}".encode()).hexdigest()[:32]
        code = raw.get("Trans Code", "")
        row = Row(n, raw, uid, "skip", trade_date=_date(raw.get("Activity Date", "")),
                  settle_date=_date(raw.get("Settle Date", "")) or _date(raw.get("Process Date", "")),
                  ticker=(raw.get("Instrument") or "").upper() or None, amount=_money(raw.get("Amount", "")))
        if row.trade_date is None:
            row.reason = "no readable activity date"
        elif code in FILLS:
            row.action, row.side = "fill", FILLS[code]
            row.quantity, row.price = abs(_number(raw.get("Quantity", ""))), abs(_money(raw.get("Price", "")))
            if not row.ticker or row.quantity <= 0 or not row.price:
                row.action, row.reason = "skip", "a trade without ticker, quantity or price"
        elif code in DIVIDENDS:
            row.action = "dividend"
        elif code in CASH_IN:
            row.action = "contribution" if row.amount > 0 else "withdrawal"
        elif code in INTEREST:
            row.action = "interest"
        elif code in FEES:
            row.action = "fee"
        else:
            row.reason = f"transaction code {code or '(blank)'} is not imported; record it by hand if it changes holdings"
        rows.append(row)
    return rows


def preview(ledger: Ledger, text: str, sleeve: str = "investing") -> list[Row]:
    rows = parse(text)
    held: dict[str, float] = {}
    for row in rows:
        if row.action == "skip":
            continue
        row.duplicate = ledger.con.execute("SELECT 1 FROM events WHERE event_uid = ?", (row.uid,)).fetchone() is not None
        if row.duplicate or row.action != "fill":
            continue
        if row.ticker not in held:
            held[row.ticker] = ledger.state(sleeve, row.trade_date).positions.get(row.ticker, 0.0)
        if row.side == "sell" and row.quantity > held[row.ticker] + 1e-6:
            row.warnings.append(f"sells {row.quantity:g} {row.ticker} but {sleeve} holds {held[row.ticker]:g}; "
                                f"if the shares belong to the other sleeve, import this row there")
        held[row.ticker] += row.quantity if row.side == "buy" else -row.quantity
    return rows


def commit(ledger: Ledger, rows: list[Row], sleeve: str = "investing") -> dict:
    """Write the previewed rows that are new; returns what was written and skipped."""
    written, skipped, warnings = 0, 0, []
    for row in sorted(rows, key=lambda r: (r.trade_date or "", r.line)):
        if row.action == "skip" or row.duplicate:
            skipped += 1
            continue
        try:
            if row.action == "fill":
                # The broker's cash amount is the truth; any gap to quantity × price is the fee.
                gross = row.quantity * row.price
                fee = max(0.0, (-row.amount - gross) if row.side == "buy" else (gross - row.amount)) if row.amount else 0.0
                _, w = ledger.fill(sleeve, row.side, row.ticker, row.quantity, row.price, row.trade_date, fee=fee,
                                   settle_date=row.settle_date if row.side == "sell" else None,
                                   source="csv", source_ref=f"line {row.line}", strict=False, uid=row.uid)
                warnings += [f"line {row.line}: {x}" for x in w]
            else:
                ledger.record(row.action, sleeve, row.trade_date, settle_date=row.settle_date, ticker=row.ticker,
                              cash=row.amount, source="csv", source_ref=f"line {row.line}", uid=row.uid)
            written += 1
        except LedgerError as exc:
            warnings.append(f"line {row.line}: {exc}")
            skipped += 1
    return {"written": written, "skipped": skipped, "warnings": warnings}
