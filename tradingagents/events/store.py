"""The append-only event store.

Two tables, both append-only (triggers refuse UPDATE and DELETE):

- ``events``: one row per *version* of a scheduled or published event (a release date, a meeting, a rule). A
  reschedule, amendment or withdrawal is a new row with the same ``source_event_id`` and a different ``source_hash``;
  ``status`` says which (scheduled, rescheduled, released, cancelled). Times are UTC ISO strings; ``original_timezone``
  keeps the zone the source stated.
- ``observations``: one row per (series, observation date, vintage): an index value as one download stated it. A
  revised value arrives as a new vintage; nothing is overwritten.

``first_seen_at`` is when this app first received the row (system availability). ``source_available_at`` is when the
source made it public, when the source says so (else NULL: unknown). Strict replay uses ``first_seen_at``; research
replay may use documented lag assumptions instead (``features``).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "events-v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,                -- this version: source/source_event_id/hash
    source_event_id TEXT NOT NULL,         -- stable across versions (the source's own id, else a derived one)
    event_type TEXT NOT NULL,              -- cpi, employment, pce, gdp, ppi, fomc_statement, ...
    source TEXT NOT NULL,                  -- bls, bea, fomc, ...
    source_url TEXT,
    evidence_excerpt TEXT,                 -- the source's own words for it (title / summary), as data
    published_at TEXT,                     -- when the event's content was published (a release: its release time)
    source_available_at TEXT,              -- when the source made this version public, if stated
    first_seen_at TEXT NOT NULL,           -- when this app first received this version
    ingested_at TEXT NOT NULL,
    scheduled_at TEXT,                     -- UTC
    effective_at TEXT,
    reference_period TEXT,
    original_timezone TEXT,
    status TEXT NOT NULL,                  -- scheduled | rescheduled | released | cancelled | amended | withdrawn
    affected_entities TEXT, countries TEXT, sectors TEXT, products TEXT,
    raw_value REAL, units TEXT,
    vintage_id TEXT, revision_of TEXT,
    source_hash TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    reliability TEXT,
    missing_reason TEXT,
    UNIQUE (source, source_event_id, source_hash)
);
CREATE INDEX IF NOT EXISTS events_type ON events (event_type, scheduled_at);
CREATE TABLE IF NOT EXISTS observations (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    series TEXT NOT NULL,                  -- gpr_daily, gpr_act, gpr_threat, epu_monthly, ...
    obs_date TEXT NOT NULL,                -- the period the value describes (a day, or a month's first day)
    value REAL,
    units TEXT,
    source TEXT NOT NULL, source_url TEXT,
    vintage_id TEXT NOT NULL,              -- the download this value came from
    first_seen_at TEXT NOT NULL,           -- when this app received that download
    source_hash TEXT NOT NULL,             -- hash of the whole download
    extraction_version TEXT NOT NULL,
    UNIQUE (series, obs_date, vintage_id)
);
CREATE INDEX IF NOT EXISTS observations_series ON observations (series, obs_date);
CREATE TABLE IF NOT EXISTS fetches (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL, fetched_at TEXT NOT NULL, status TEXT NOT NULL,   -- ok | unavailable | error
    detail TEXT, rows_added INTEGER, source_hash TEXT
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_update BEFORE UPDATE ON observations
    BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_delete BEFORE DELETE ON observations
    BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
"""

EVENT_FIELDS = ("event_type", "source_url", "evidence_excerpt", "published_at", "source_available_at", "scheduled_at",
                "effective_at", "reference_period", "original_timezone", "status", "affected_entities", "countries",
                "sectors", "products", "raw_value", "units", "vintage_id", "revision_of", "reliability", "missing_reason")


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(payload) -> str:
    data = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:20]


def path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "events.sqlite3"


def connect(cache_dir: Path) -> sqlite3.Connection:
    p = path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def add_events(con, source: str, events: list[dict], extraction_version: str, seen_at: str | None = None) -> int:
    """Append each event version not stored yet; a changed schedule (same ``source_event_id``, new content) is a new
    row marked ``rescheduled`` with ``revision_of`` the previous version. Returns the rows added."""
    seen_at = seen_at or now_utc()
    added = 0
    with con:
        for e in events:
            content = {k: e.get(k) for k in EVENT_FIELDS if k not in ("status", "revision_of")}
            h = digest(content)
            sid = e["source_event_id"]
            if con.execute("SELECT 1 FROM events WHERE source = ? AND source_event_id = ? AND source_hash = ?",
                           (source, sid, h)).fetchone():
                continue
            prev = con.execute("SELECT event_id, scheduled_at FROM events WHERE source = ? AND source_event_id = ?"
                               " ORDER BY row_id DESC LIMIT 1", (source, sid)).fetchone()
            row = {k: e.get(k) for k in EVENT_FIELDS}
            if prev:
                row["revision_of"] = prev["event_id"]
                if prev["scheduled_at"] != e.get("scheduled_at") and row.get("status") == "scheduled":
                    row["status"] = "rescheduled"
            row.update(event_id=f"{source}/{sid}/{h}", source_event_id=sid, source=source, first_seen_at=seen_at,
                       ingested_at=now_utc(), source_hash=h, extraction_version=extraction_version)
            row["status"] = row.get("status") or "scheduled"
            cols = ", ".join(row)
            con.execute(f"INSERT INTO events ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
            added += 1
    return added


def add_observations(con, source: str, url: str, rows: list[tuple[str, str, float, str]], source_hash: str,
                     extraction_version: str, seen_at: str | None = None) -> int:
    """Append one download's ``(series, obs_date, value, units)`` rows as a vintage, keeping only values that are new
    or changed against the latest stored vintage (a revision is kept; an unchanged repeat is not)."""
    seen_at = seen_at or now_utc()
    vintage = f"{source}@{seen_at}"
    latest = {}
    for r in con.execute("SELECT series, obs_date, value FROM observations o WHERE source = ? AND row_id = ("
                         "SELECT MAX(row_id) FROM observations p WHERE p.series = o.series AND p.obs_date = o.obs_date)",
                         (source,)):
        latest[(r["series"], r["obs_date"])] = r["value"]
    added = 0
    with con:
        for series, obs_date, value, units in rows:
            old = latest.get((series, obs_date), "missing")
            if old != "missing" and (old == value or (old is not None and value is not None and abs(old - value) < 1e-9)):
                continue
            con.execute("INSERT OR IGNORE INTO observations (series, obs_date, value, units, source, source_url,"
                        " vintage_id, first_seen_at, source_hash, extraction_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (series, obs_date, value, units, source, url, vintage, seen_at, source_hash, extraction_version))
            added += 1
    return added


def log_fetch(con, source: str, status: str, detail: str | None = None, rows: int = 0, source_hash: str | None = None):
    with con:
        con.execute("INSERT INTO fetches (source, fetched_at, status, detail, rows_added, source_hash) VALUES (?,?,?,?,?,?)",
                    (source, now_utc(), status, (detail or "")[:500], rows, source_hash))


def status(con) -> list[dict]:
    """Each source's last fetch, and what it holds."""
    out = []
    for r in con.execute("SELECT source, MAX(row_id) AS last FROM fetches GROUP BY source ORDER BY source"):
        f = con.execute("SELECT * FROM fetches WHERE row_id = ?", (r["last"],)).fetchone()
        n_ev = con.execute("SELECT COUNT(*) FROM events WHERE source = ?", (r["source"],)).fetchone()[0]
        n_obs = con.execute("SELECT COUNT(*) FROM observations WHERE source = ?", (r["source"],)).fetchone()[0]
        out.append({"source": r["source"], "status": f["status"], "fetched_at": f["fetched_at"], "detail": f["detail"],
                    "events": n_ev, "observations": n_obs})
    return out
