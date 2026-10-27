"""The application's records in one place: open the database, import the old files once,
and read or change settings.

There is one account, kept in the ``trading`` sleeve: SGOV as its base and the stocks bought
with SGOV's interest and the broker's match (``ledger.income``). The Investing sleeve of
earlier versions is retired by ``retire_investing``, once.

First open: the files that came before the database (holdings, buys, risk level,
scan list, paper tracker) are copied into ``legacy_imports`` byte for byte. Your
holdings become opening positions, dated the day the holdings file was last saved,
and the strategy starts that day. The files are left where they are; the database
is the record from then on.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .accounting import DEFAULT_ACCOUNT, Ledger
from .db import DEFAULT_NAME, connect, now_utc, transaction

# Settings each sleeve has, with their defaults. Values are JSON.
SLEEVE_DEFAULTS = {
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
    # Free OpenRouter models tried in order when the providers before them fail (openrouter in ai_fallbacks).
    # (Inkling's free tier refuses this app: "only available on agentic harnesses".)
    "ai_openrouter_models": ["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free"],
    "ai_estimates": {"rating": 0.30, "market_brief": 0.02},   # dollars reserved per job before it runs
    # Agent roles on their own provider and model (llm_clients.roles); empty: all on the run's provider.
    "ai_agent_models": {},
    # The agents' default provider and models ({"provider", "quick", "deep"}); empty: the app's configured ones.
    "ai_primary": {},
    "compute_backend": "auto",
    # The day the SGOV-interest trading strategy started: interest and match count from here (ledger.income).
    "income_start": None,
    # Stock research: the saved watchlist, and the limits stock recommendations respect.
    "stock_watchlist": [],
    "stock_rules": {"sleeve": "trading", "max_position_share": 0.25, "max_names": 5, "min_order": 1.0,
                    "max_large_loss_prob": 0.15, "llm_shortlist": 5},
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
        from . import capital

        row = self.con.execute("SELECT value_json FROM settings WHERE scope = ? AND key = ?", (scope, key)).fetchone()
        if row:
            return json.loads(row[0])
        defaults = GLOBAL_DEFAULTS if scope == "global" else capital.DEFAULTS if scope == capital.SCOPE             else SLEEVE_DEFAULTS.get(scope, {})
        if key not in defaults:
            raise KeyError(f"no setting {scope}.{key}")
        return defaults[key]

    def settings(self, scope: str) -> dict:
        from . import capital

        defaults = GLOBAL_DEFAULTS if scope == "global" else capital.DEFAULTS if scope == capital.SCOPE             else SLEEVE_DEFAULTS[scope]
        return {k: self.get(scope, k) for k in defaults}

    def set(self, scope: str, key: str, value, source: str = "user"):
        from . import capital

        if scope == capital.SCOPE:
            value = capital.validate(key, value)   # the backend checks every amount, whoever sends it
        if (scope, key) == ("global", "ai_primary") and value:
            from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

            value = {k: str(value.get(k) or "").strip() for k in ("provider", "quick", "deep")}
            value["provider"] = value["provider"].lower()
            if value["provider"] not in PROVIDER_API_KEY_ENV or not value["quick"] or not value["deep"]:
                raise ValueError("the default needs a known provider, a quick model and a deep model")
        if (scope, key) == ("global", "ai_agent_models"):
            from tradingagents.llm_clients.roles import validate

            value = validate(value)
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
                        self.ledger.record("opening_position", "trading", day, ticker=p["ticker"].upper(),
                                           quantity=float(p["quantity"]), source="migration",
                                           source_ref="my_portfolio.json",
                                           note="holdings at migration; price and cost basis not recorded")
                if book.get("cash"):
                    self.ledger.record("opening_cash", "trading", day, cash=float(book["cash"]), source="migration",
                                       source_ref="my_portfolio.json", note="settled cash at migration")
                self.set("global", "income_start", day, source="migration")
            buys = self.folder / "my_buys.json"
            if buys.exists():
                data = json.loads(buys.read_text(encoding="utf-8"))
                per_day = sum((data.get("daily") or {}).values()) + sum((data.get("weekly") or {}).values()) / 5
                if per_day:
                    from . import capital

                    self.set(capital.SCOPE, "daily_sgov_usd", round(per_day, 2), source="migration")

    def legacy(self, name: str) -> bytes | None:
        row = self.con.execute("SELECT payload FROM legacy_imports WHERE source_path LIKE ? ORDER BY id LIMIT 1",
                               (f"%{name}",)).fetchone()
        return row[0] if row else None

    # --- one account ---

    def retire_investing(self, as_of, backup_dir: Path | None = None) -> dict | None:
        """Fold the retired Investing sleeve into the one trading account, once; ``None`` when there is none.

        The database is copied to ``backup_dir`` first. Then every Investing record (its events, the
        transfers between the sleeves, its recommendations, orders, reservations, ideas and settings)
        is deleted, and what the whole account held on ``as_of`` is restated in Trading as opening
        positions and cash, so the account's holdings do not change. The strategy starts that day.
        """
        con = self.con
        if not con.execute("SELECT 1 FROM sleeves WHERE name = 'investing'").fetchone():
            return None
        before = self.ledger.state(None, as_of)
        folder = Path(backup_dir or self.folder / "backups")
        folder.mkdir(parents=True, exist_ok=True)
        backup = folder / f"{self.path.stem}_before_retiring_investing_{as_of}.sqlite3"
        dest = sqlite3.connect(backup)
        con.backup(dest)
        dest.close()
        transfers = [r[0] for r in con.execute("SELECT DISTINCT transfer_id FROM events WHERE sleeve = 'investing'"
                                               " AND transfer_id IS NOT NULL")]
        ids = [r[0] for r in con.execute(
            f"SELECT id FROM events WHERE sleeve = 'investing' OR transfer_id IN ({','.join('?' * len(transfers))})",
            transfers)]
        marks = ",".join("?" * len(ids))
        recs = [r[0] for r in con.execute("SELECT id FROM recommendations WHERE sleeve = 'investing'")]
        rmarks = ",".join("?" * len(recs))
        # The ledger is append-only by trigger; the guards are lifted for this one deletion, in its transaction.
        from .schema import _guards

        guarded = ("event_voids", "events", "order_status", "orders", "decisions", "recommendation_status",
                   "recommendations", "stock_ideas")
        with transaction(con):
            for table in guarded:
                con.execute(f"DROP TRIGGER IF EXISTS {table}_no_update")
                con.execute(f"DROP TRIGGER IF EXISTS {table}_no_delete")
            con.execute(f"DELETE FROM event_voids WHERE event_id IN ({marks}) OR replaced_by IN ({marks})", ids + ids)
            con.execute(f"UPDATE events SET supersedes = NULL WHERE supersedes IN ({marks})", ids)
            con.execute(f"DELETE FROM events WHERE id IN ({marks})", ids)
            con.execute(f"DELETE FROM order_status WHERE order_id IN (SELECT id FROM orders WHERE recommendation_id"
                        f" IN ({rmarks}) OR sleeve = 'investing')", recs)
            con.execute(f"DELETE FROM orders WHERE recommendation_id IN ({rmarks}) OR sleeve = 'investing'", recs)
            for table in ("decisions", "recommendation_status"):
                con.execute(f"DELETE FROM {table} WHERE recommendation_id IN ({rmarks})", recs)
            con.execute(f"DELETE FROM recommendations WHERE id IN ({rmarks})", recs)
            for table in ("reservations", "stock_ideas"):
                con.execute(f"DELETE FROM {table} WHERE sleeve = 'investing'")
            con.execute("DELETE FROM settings WHERE scope = 'investing'")
            con.execute("DELETE FROM settings WHERE scope = 'capital' AND key IN ('trading_capital_usd',"
                        " 'investing_capital_usd', 'daily_investment_usd')")
            con.execute("DELETE FROM sleeves WHERE name = 'investing'")
            for table in guarded:
                for statement in _guards(table).strip().splitlines():
                    con.execute(statement)
        now = self.ledger.state("trading", as_of)
        restated = {}
        for t in sorted(set(before.positions) | set(now.positions)):
            delta = before.positions.get(t, 0.0) - now.positions.get(t, 0.0)
            if abs(delta) > 1e-9:
                self.ledger.record("opening_position", "trading", as_of, ticker=t, quantity=delta, source="migration",
                                   note="the retired Investing sleeve, folded into the one account",
                                   uid=uuid.uuid4().hex)
                restated[t] = round(delta, 6)
        cash = before.cash.total - now.cash.total
        if abs(cash) > 0.005:
            self.ledger.record("opening_cash", "trading", as_of, cash=cash, source="migration",
                               note="the retired Investing sleeve's cash", uid=uuid.uuid4().hex)
        if not self.get("global", "income_start"):
            self.set("global", "income_start", str(as_of), source="migration")
        return {"backup": str(backup), "deleted_events": len(ids), "deleted_recommendations": len(recs),
                "restated": restated, "cash": round(cash, 2)}

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

    def adjust_holdings(self, wanted: dict[str, float], as_of, sleeve: str = "trading", note: str | None = None) -> list[int]:
        """Bring the account's share counts to ``wanted`` with audited adjustment events: for
        corrections the broker's records cannot supply. Refuses to take a holding below zero."""
        combined = self.ledger.state(None, as_of).positions
        own = self.ledger.state(sleeve, as_of).positions
        ids = []
        with transaction(self.con):
            for t in sorted(set(combined) | set(wanted)):
                delta = float(wanted.get(t, 0.0)) - combined.get(t, 0.0)
                if abs(delta) < 1e-9:
                    continue
                if own.get(t, 0.0) + delta < -1e-9:
                    raise ValueError(f"{t}: the change would leave {own.get(t, 0.0) + delta:g} shares")
                # Each edit is its own event: two identical edits on one day are both real.
                ids.append(self.ledger.record("adjustment", sleeve, as_of, ticker=t, quantity=delta, source="manual",
                                              note=note or "holdings edited in the app", uid=uuid.uuid4().hex)[0])
        return ids


def open_store(folder: str | Path) -> Store:
    return Store(folder)
