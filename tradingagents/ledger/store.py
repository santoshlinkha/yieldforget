"""The application's records in one place: open the database, import the old files once,
and read or change settings per sleeve.

First open: the files that came before the database (holdings, buys, risk level,
scan list, paper tracker) are copied into ``legacy_imports`` byte for byte. Your
holdings become opening positions in the Investing sleeve, dated the day the
holdings file was last saved, and the risk level becomes Investing's setting.
The files are left where they are; the database is the record from then on.
The Trading sleeve starts empty: funding it is a transfer you make yourself
(see ``trading_assignment_preview``).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .accounting import DEFAULT_ACCOUNT, Ledger
from .db import DEFAULT_NAME, connect, now_utc, transaction

# Settings each sleeve has, with their defaults. Values are JSON.
SLEEVE_DEFAULTS = {
    "investing": {"risk_level": 2, "risk_model": "worst_case", "target_growth": None, "daily_contribution": 27.61,
                  "return_model": "vol_prior", "forecast_horizon": 20, "forecast_model": "xgb_quantile"},
    "trading": {"risk_level": 2, "risk_model": "worst_case", "target_growth": None, "daily_contribution": 0.0,
                "initial_allocation": 1000.0, "return_model": "vol_prior", "forecast_horizon": 20,
                "forecast_model": "xgb_quantile"},
}
GLOBAL_DEFAULTS = {
    "ai_budget": {"daily_usd": 1.0, "monthly_usd": 10.0, "daily_requests": 200, "monthly_requests": 2000},
    "ai_prices": {},          # "provider/model": {"input": $/M, "output": $/M, "cached_input": $/M}
    "ai_analysts": ["market"],
    "ai_shortlist": 10,
    "ai_fallbacks": [],       # other providers a failed paid call may move to; none by default
    "ai_estimates": {"rating": 0.30, "market_brief": 0.02},   # dollars reserved per job before it runs
    "compute_backend": "auto",
}
LEGACY_FILES = ("my_portfolio.json", "my_buys.json", "my_risk.json", "my_universe.json", "paper_tracker.json")


class Store:
    def __init__(self, folder: str | Path, account_id: str = DEFAULT_ACCOUNT, db_name: str = DEFAULT_NAME):
        self.folder = Path(folder)
        self.path = self.folder / db_name
        self.con = connect(self.path, backup_dir=self.folder / "backups")
        self.ledger = Ledger(self.con, account_id)
        self.ledger.ensure_account()
        self._import_legacy()

    # --- settings ---

    def get(self, scope: str, key: str):
        row = self.con.execute("SELECT value_json FROM settings WHERE scope = ? AND key = ?", (scope, key)).fetchone()
        if row:
            return json.loads(row[0])
        defaults = GLOBAL_DEFAULTS if scope == "global" else SLEEVE_DEFAULTS.get(scope, {})
        if key not in defaults:
            raise KeyError(f"no setting {scope}.{key}")
        return defaults[key]

    def settings(self, scope: str) -> dict:
        defaults = GLOBAL_DEFAULTS if scope == "global" else SLEEVE_DEFAULTS[scope]
        return {k: self.get(scope, k) for k in defaults}

    def set(self, scope: str, key: str, value, source: str = "user"):
        self.get(scope, key)   # refuses unknown keys
        text = json.dumps(value)
        with transaction(self.con):
            self.con.execute("INSERT OR REPLACE INTO settings VALUES (?, ?, ?, ?)", (scope, key, text, now_utc()))
            self.con.execute("INSERT INTO settings_history (scope, key, value_json, changed_at, source)"
                             " VALUES (?, ?, ?, ?, ?)", (scope, key, text, now_utc(), source))

    # --- work claims: one process at a time for a named piece of background work ---

    def claim(self, key: str, owner: str, ttl: timedelta = timedelta(minutes=30)) -> bool:
        now = datetime.now(UTC)
        stamp = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with transaction(self.con):
            row = self.con.execute("SELECT owner, expires_at FROM work_claims WHERE key = ?", (key,)).fetchone()
            if row and row["expires_at"] > stamp and row["owner"] != owner:
                return False
            self.con.execute("INSERT OR REPLACE INTO work_claims VALUES (?, ?, ?, ?)",
                             (key, owner, stamp, (now + ttl).strftime("%Y-%m-%dT%H:%M:%S.%fZ")))
            return True

    def release(self, key: str, owner: str):
        with transaction(self.con):
            self.con.execute("DELETE FROM work_claims WHERE key = ? AND owner = ?", (key, owner))

    # --- legacy files ---

    def _import_legacy(self):
        if self.con.execute("SELECT COUNT(*) FROM legacy_imports").fetchone()[0]:
            return
        found = [self.folder / name for name in LEGACY_FILES if (self.folder / name).exists()]
        if not found:
            return
        with transaction(self.con):
            for path in found:
                data = path.read_bytes()
                self.con.execute("INSERT OR IGNORE INTO legacy_imports (source_path, sha256, imported_at, payload)"
                                 " VALUES (?, ?, ?, ?)", (str(path), hashlib.sha256(data).hexdigest(), now_utc(), data))
            holdings = self.folder / "my_portfolio.json"
            if holdings.exists():
                book = json.loads(holdings.read_text(encoding="utf-8"))
                day = datetime.fromtimestamp(holdings.stat().st_mtime).date().isoformat()
                for p in book.get("positions", []):
                    if p.get("quantity"):
                        self.ledger.record("opening_position", "investing", day, ticker=p["ticker"].upper(),
                                           quantity=float(p["quantity"]), source="migration",
                                           source_ref="my_portfolio.json",
                                           note="holdings at migration; price and cost basis not recorded")
                if book.get("cash"):
                    self.ledger.record("opening_cash", "investing", day, cash=float(book["cash"]), source="migration",
                                       source_ref="my_portfolio.json", note="settled cash at migration")
            risk = self.folder / "my_risk.json"
            if risk.exists():
                saved = json.loads(risk.read_text(encoding="utf-8"))
                for key, name in (("level", "risk_level"), ("model", "risk_model"), ("target_growth", "target_growth")):
                    if key in saved:
                        self.set("investing", name, saved[key], source="migration")
            buys = self.folder / "my_buys.json"
            if buys.exists():
                data = json.loads(buys.read_text(encoding="utf-8"))
                per_day = sum((data.get("daily") or {}).values()) + sum((data.get("weekly") or {}).values()) / 5
                self.set("investing", "daily_contribution", round(per_day, 2), source="migration")

    def legacy(self, name: str) -> bytes | None:
        row = self.con.execute("SELECT payload FROM legacy_imports WHERE source_path LIKE ? ORDER BY id LIMIT 1",
                               (f"%{name}",)).fetchone()
        return row[0] if row else None

    # --- the Trading sleeve ---

    def trading_assignment_preview(self, amount: float, prices: dict[str, float], as_of,
                                   prefer: tuple[str, ...] = ("SGOV",)) -> dict:
        """What could move from Investing to fund Trading with about ``amount``: settled cash
        first, then the cash-like fund, then nothing else unless you pick it. A preview;
        nothing moves until you confirm with ``ledger.transfer``."""
        state = self.ledger.state("investing", as_of, prices)
        steps, left = [], amount
        if state.cash.available > 0.005:
            take = min(left, state.cash.available)
            steps.append({"cash": round(take, 2)})
            left -= take
        for t in prefer:
            if left <= 0.005 or t not in state.positions or t not in prices:
                continue
            qty = min(state.positions[t], left / prices[t])
            steps.append({"ticker": t, "quantity": round(qty, 6), "price": prices[t], "value": round(qty * prices[t], 2)})
            left -= qty * prices[t]
        return {"amount": amount, "steps": steps, "short_by": round(max(left, 0.0), 2),
                "unsettled_not_movable": round(state.cash.unsettled, 2),
                "note": ("Moves only what Investing really holds. Unsettled proceeds cannot move until they settle. "
                         "To use other holdings, pick them yourself.")}


    # --- views for the optimizer and the old files ---

    def portfolio(self, sleeve: str, as_of):
        """The optimizer's view of a sleeve, and its cash split: ``(PortfolioContext, Cash)``.

        The context's cash is settled plus unsettled (both are worth something);
        orders may only spend ``Cash.available``.
        """
        from tradingagents.portfolio import PortfolioContext, Position

        state = self.ledger.state(sleeve, as_of)
        positions = [Position(ticker=t, quantity=q) for t, q in sorted(state.positions.items())]
        return PortfolioContext(positions=positions, cash=state.cash.total, currency="USD"), state.cash

    def export_holdings(self, path: Path, as_of):
        """Write the whole account's holdings in the old file format, for code that still reads it."""
        context, cash = self.portfolio("combined", as_of)
        data = {"currency": "USD", "positions": [p.model_dump(exclude_none=True) for p in context.positions],
                "cash": round(cash.settled, 2) or None,
                "_note": "exported from portfolio.sqlite3; edit holdings in the app, not here"}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def adjust_holdings(self, wanted: dict[str, float], as_of, sleeve: str = "investing", note: str | None = None) -> list[int]:
        """Bring the account's share counts to ``wanted`` with audited adjustment events in one
        sleeve: for corrections the broker's records cannot supply. Refuses to take a sleeve
        below zero."""
        combined = self.ledger.state(None, as_of).positions
        own = self.ledger.state(sleeve, as_of).positions
        ids = []
        with transaction(self.con):
            for t in sorted(set(combined) | set(wanted)):
                delta = float(wanted.get(t, 0.0)) - combined.get(t, 0.0)
                if abs(delta) < 1e-9:
                    continue
                if own.get(t, 0.0) + delta < -1e-9:
                    raise ValueError(f"{t}: the change would leave {sleeve} with {own.get(t, 0.0) + delta:g} shares; "
                                     f"some are in the other sleeve")
                # Each edit is its own event: two identical edits on one day are both real.
                ids.append(self.ledger.record("adjustment", sleeve, as_of, ticker=t, quantity=delta, source="manual",
                                              note=note or "holdings edited in the app", uid=uuid.uuid4().hex)[0])
        return ids


def open_store(folder: str | Path) -> Store:
    return Store(folder)
