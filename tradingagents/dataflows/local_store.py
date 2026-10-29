"""Downloaded market data kept on disk, so nothing is downloaded twice.

**Daily bars** (adjusted close, high, low, volume) live in one file per ticker
under ``<cache>/bars/``. The first request downloads a ticker's whole history.
After that, only the sessions since its last stored one are downloaded, with
ten days of overlap. When the overlapping closes differ, Yahoo has re-adjusted
the past (a dividend or split), and that ticker alone is downloaded in full
again. Only **completed** sessions are stored: a session counts once its close is
final, after 4:30 pm New York time. A plan made during the day therefore uses
the last final close, not a price still moving. A ticker Yahoo has nothing for is
asked again at most once a day.

**News** articles, from ticker news and searches, are kept in
``market_data.sqlite3`` with the searches that returned them. A search asked
again within ``NEWS_TTL`` is answered from the store; after that it is asked
anew and the new articles are added. A search therefore remembers everything it
ever returned, not just Yahoo's latest page. Articles are deduplicated by link,
or by title and publisher when there is no link.

**Snapshots** of Yahoo data with no history (analyst targets, EPS trends,
option-chain summaries) are kept once per ticker, kind and session. Data that
carries its own history (dated upgrades and downgrades, earnings dates) is
reused for ``HISTORY_TTL`` before being fetched again.

This is market data, so it is kept apart from the account records in
``portfolio.sqlite3``. ``TRADINGAGENTS_LOCAL_STORE=off`` turns the store off
(the test suite does), and every call then goes straight to the source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

NEW_YORK = pytz.timezone("America/New_York")
FIELDS = ("Close", "High", "Low", "Volume")
OVERLAP_DAYS = 10
REL_TOL = 1e-6
NEWS_TTL = timedelta(hours=3)
HISTORY_TTL = timedelta(days=7)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_meta (
    ticker TEXT PRIMARY KEY,
    first_date TEXT,
    last_date TEXT,              -- last stored (final) session; NULL when Yahoo had nothing
    checked_at TEXT NOT NULL,    -- last time the source was asked
    full_at TEXT                 -- last full download
);
CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL UNIQUE,
    title TEXT, publisher TEXT, link TEXT, pub_date TEXT, summary TEXT,
    first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_keys (
    key TEXT NOT NULL,
    article_id INTEGER NOT NULL REFERENCES news_articles(id),
    PRIMARY KEY (key, article_id)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    key TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    ticker TEXT NOT NULL,
    kind TEXT NOT NULL,
    session TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (ticker, kind, session)
);
"""


def enabled() -> bool:
    return os.environ.get("TRADINGAGENTS_LOCAL_STORE", "on").lower() not in ("off", "0", "false")


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def last_final_session(now: datetime | None = None) -> date:
    """The latest session whose close is final: today from 30 minutes after its close (4:30 pm New
    York, 1:30 pm on a half day), else the one before."""
    from tradingagents.ledger.sessions import is_session, previous_session, session_close

    local = (now or _now()).astimezone(NEW_YORK)
    today = local.date()
    if is_session(today):
        final_at = datetime.combine(today, session_close(today)) + timedelta(minutes=30)
        if local.replace(tzinfo=None) >= final_at:
            return today
    return previous_session(today)


def _safe(ticker: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else f"%{ord(c):02X}" for c in ticker)


class MarketStore:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.bars_dir = self.folder / "bars"
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "market_data.sqlite3"
        self._local = threading.local()
        self._lock = threading.Lock()
        self.con.executescript(_SCHEMA)
        self._memo: dict[str, tuple[float, pd.DataFrame]] = {}

    @property
    def con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
            self._local.con = con
        return con

    # --- bars ---

    def _file(self, ticker: str) -> Path:
        return self.bars_dir / f"{_safe(ticker)}.pkl"

    def _read(self, ticker: str) -> pd.DataFrame | None:
        path = self._file(ticker)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        hit = self._memo.get(ticker)
        if hit and hit[0] == mtime:
            return hit[1]
        frame = pd.read_pickle(path)
        self._memo[ticker] = (mtime, frame)
        return frame

    def _write(self, ticker: str, frame: pd.DataFrame, full: bool, now: datetime) -> bool:
        """Replace the ticker's file, then record its dates; ``False`` (metadata untouched) when
        the file could not be replaced, so the next call sees the old dates and asks again."""
        path = self._file(ticker)
        # Several processes may update one ticker at once: each writes its own file, and the
        # rename is retried while Windows has the target open; the last writer wins, and all
        # of them wrote the same sessions.
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        frame.to_pickle(tmp)
        for attempt in range(20):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        else:
            tmp.unlink(missing_ok=True)
            logger.warning("could not replace the cached bars of %s (the file stayed locked); "
                           "its dates are left as they were", ticker)
            return False
        now = _stamp(now)
        first = frame.index[0].date().isoformat() if len(frame) else None
        last = frame.index[-1].date().isoformat() if len(frame) else None
        self.con.execute(
            "INSERT INTO bars_meta (ticker, first_date, last_date, checked_at, full_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(ticker) DO UPDATE SET first_date = excluded.first_date, last_date = excluded.last_date,"
            " checked_at = excluded.checked_at, full_at = COALESCE(excluded.full_at, bars_meta.full_at)",
            (ticker, first, last, now, now if full else None))
        return True

    def _checked_empty(self, ticker: str, now: datetime):
        self.con.execute("INSERT INTO bars_meta (ticker, checked_at) VALUES (?, ?) ON CONFLICT(ticker)"
                         " DO UPDATE SET checked_at = excluded.checked_at", (ticker, _stamp(now)))

    def bars(self, tickers, download: Callable[[list[str], str | None, str], dict[str, pd.DataFrame]],
             now: datetime | None = None) -> dict[str, pd.DataFrame]:
        """Bars for ``tickers`` through the last final session, downloading only what is missing.

        ``download(tickers, start, end)`` returns ``{field: frame}`` with one column per
        ticker; ``start`` None means the whole history.
        """
        tickers = list(dict.fromkeys(tickers))
        now = now or _now()
        final = last_final_session(now)
        end = (final + timedelta(days=1)).isoformat()
        today = now.astimezone(NEW_YORK).date().isoformat()
        meta = {r["ticker"]: dict(r) for r in self.con.execute(
            f"SELECT * FROM bars_meta WHERE ticker IN ({','.join('?' * len(tickers))})", tickers)} if tickers else {}
        full, incremental = [], {}
        for t in tickers:
            m = meta.get(t)
            if m is None or (m["last_date"] and self._read(t) is None):
                full.append(t)
            elif m["last_date"] is None:
                if m["checked_at"][:10] < today:   # nothing last time; ask again once a day
                    full.append(t)
            elif m["last_date"] < final.isoformat() and m["checked_at"] < _stamp(now - timedelta(hours=1)):
                # Behind, and not asked in the last hour (a halted fund would otherwise be asked every call).
                incremental.setdefault(m["last_date"], []).append(t)
        with self._lock:
            for last, group in incremental.items():
                start = (date.fromisoformat(last) - timedelta(days=OVERLAP_DAYS)).isoformat()
                fresh = download(group, start, end)
                for t in group:
                    new = self._frame(fresh, t, final)
                    old = self._read(t)
                    if new is None or new.empty:
                        self._checked_empty(t, now)   # no new session yet (a holiday, a halt)
                        continue
                    shared = old.index.intersection(new.index)
                    a, b = old.loc[shared, "Close"].to_numpy(), new.loc[shared, "Close"].to_numpy()
                    ok = np.isfinite(a) & np.isfinite(b)
                    if len(shared) and np.allclose(a[ok], b[ok], rtol=REL_TOL, atol=0.0):
                        merged = pd.concat([old[old.index < new.index[0]], new]).sort_index()
                        self._write(t, merged[~merged.index.duplicated(keep="last")], full=False, now=now)
                    else:
                        full.append(t)   # Yahoo re-adjusted the past: start this one over
            if full:
                fresh = download(full, None, end)
                for t in full:
                    new = self._frame(fresh, t, final)
                    if new is None or new.empty:
                        self._checked_empty(t, now)
                    else:
                        self._write(t, new, full=True, now=now)
        out = {}
        frames = {t: self._read(t) for t in tickers}
        for f in FIELDS:
            cols = {t: fr[f] for t, fr in frames.items() if fr is not None and f in fr}
            out[f] = pd.DataFrame(cols).reindex(columns=tickers).sort_index()
        return out

    @staticmethod
    def _frame(fresh: dict[str, pd.DataFrame], ticker: str, final: date) -> pd.DataFrame | None:
        cols = {f: fresh[f][ticker] for f in FIELDS if f in fresh and ticker in fresh[f]}
        if "Close" not in cols or cols["Close"].dropna().empty:
            return None
        frame = pd.DataFrame(cols)
        frame.index = pd.DatetimeIndex(frame.index)
        frame = frame[frame.index <= pd.Timestamp(final)]
        return frame.dropna(subset=["Close"])

    def last_close(self, tickers) -> dict[str, float]:
        """The last stored close of each ticker; no download."""
        out = {}
        for t in tickers:
            frame = self._read(t)
            if frame is not None and len(frame):
                out[t] = float(frame["Close"].iloc[-1])
        return out

    def bars_checked_at(self) -> str | None:
        row = self.con.execute("SELECT MAX(checked_at) FROM bars_meta").fetchone()
        return row[0] if row else None

    # --- news ---

    def articles(self, key: str, fetch: Callable[[], list[dict]], ttl: timedelta = NEWS_TTL) -> list[dict]:
        """Every article ``key`` has returned, newest first, asking the source only when
        the last ask is older than ``ttl``. Articles are ``{title, publisher, link,
        pub_date (datetime or None), summary}``."""
        row = self.con.execute("SELECT fetched_at FROM fetch_log WHERE key = ?", (key,)).fetchone()
        if row is None or row[0] < _stamp(_now() - ttl):
            fresh = fetch()
            now = _stamp(_now())
            self.con.execute("BEGIN IMMEDIATE")
            try:
                for a in fresh:
                    uid = hashlib.sha256((a.get("link") or f"{a.get('title')}|{a.get('publisher')}").encode()).hexdigest()
                    pub = a.get("pub_date")
                    pub = pub.astimezone(UTC).isoformat() if isinstance(pub, datetime) else pub
                    self.con.execute("INSERT OR IGNORE INTO news_articles (uid, title, publisher, link, pub_date,"
                                     " summary, first_seen) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                     (uid, a.get("title"), a.get("publisher"), a.get("link"), pub, a.get("summary"), now))
                    self.con.execute("INSERT OR IGNORE INTO news_keys SELECT ?, id FROM news_articles WHERE uid = ?",
                                     (key, uid))
                if fresh:   # an empty answer is not remembered: the source is asked again next time
                    self.con.execute("INSERT OR REPLACE INTO fetch_log VALUES (?, ?)", (key, now))
                self.con.execute("COMMIT")
            except BaseException:
                self.con.execute("ROLLBACK")
                raise
        rows = self.con.execute("SELECT a.* FROM news_articles a JOIN news_keys k ON k.article_id = a.id"
                                " WHERE k.key = ? ORDER BY a.pub_date DESC", (key,)).fetchall()
        return [{"title": r["title"], "publisher": r["publisher"], "link": r["link"], "summary": r["summary"] or "",
                 "pub_date": datetime.fromisoformat(r["pub_date"]) if r["pub_date"] else None} for r in rows]

    # --- snapshots ---

    def snapshot(self, ticker: str, kind: str, fetch: Callable[[], object], session: str | None = None,
                 ttl: timedelta | None = None):
        """A JSON-able value of ``kind`` for ``ticker``: today's saved one, or with ``ttl``
        the newest saved within it; otherwise fetched and saved for ``session``."""
        session = session or last_final_session().isoformat()
        if ttl is None:
            row = self.con.execute("SELECT payload FROM snapshots WHERE ticker = ? AND kind = ? AND session = ?",
                                   (ticker, kind, session)).fetchone()
        else:
            row = self.con.execute("SELECT payload FROM snapshots WHERE ticker = ? AND kind = ? AND fetched_at >= ?"
                                   " ORDER BY fetched_at DESC LIMIT 1", (ticker, kind, _stamp(_now() - ttl))).fetchone()
        if row:
            return json.loads(row[0])
        value = fetch()
        self.con.execute("INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?)",
                         (ticker, kind, session, _stamp(_now()), json.dumps(value, default=str)))
        return value

    def snapshots(self, ticker: str, kind: str) -> list[tuple[str, object]]:
        """Every saved snapshot of a kind, oldest first: the history being built."""
        return [(r["session"], json.loads(r["payload"])) for r in self.con.execute(
            "SELECT session, payload FROM snapshots WHERE ticker = ? AND kind = ? ORDER BY session", (ticker, kind))]


_stores: dict[str, MarketStore] = {}
_stores_lock = threading.Lock()


def store(folder: str | Path | None = None) -> MarketStore | None:
    """The shared store for ``folder`` (the configured cache directory by default); None when off."""
    if not enabled():
        return None
    if folder is None:
        # One store for everything: the optimizer's cache folder, where prices are kept too.
        from tradingagents.dataflows.config import get_config
        folder = Path(get_config()["data_cache_dir"]) / "optimizer"
    key = str(Path(folder).resolve())
    with _stores_lock:
        if key not in _stores:
            _stores[key] = MarketStore(folder)
        return _stores[key]


def cached_call(kind: str, ttl: timedelta | None = None):
    """Keep a data tool's result, so the same request is not downloaded again.

    The result is saved under ``kind`` and the call's exact arguments: once per
    session by default, or reused for ``ttl`` when given (for sources that publish
    during the day, such as FRED). A call that fails is not saved. With the store
    off, the tool runs as it always did.
    """
    import functools

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            market_store = store()
            if market_store is None:
                return fn(*args, **kwargs)
            key = json.dumps([fn.__name__, args, kwargs], sort_keys=True, default=str)
            return market_store.snapshot(key, kind, lambda: fn(*args, **kwargs), ttl=ttl)
        return inner
    return wrap
