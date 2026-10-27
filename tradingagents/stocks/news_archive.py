"""A local archive of company news from SEC 8-K filings, back to 2006, with the minute each became public.

**Source.** SEC EDGAR, free and official: every 8-K (and 8-K/A) a company filed, from its submissions history
(``data.sec.gov/submissions``, including the older pages), with the SEC's acceptance time, the event items
(2.02 results, 5.02 executive changes, 1.01 material agreements, 2.01 acquisitions, 7.01 / 8.01 other
announcements ...) and, when fetched, the text of the press release attached as exhibit 99.

**Timing.** ``accepted_at`` is SEC's acceptance time in UTC; it is stored with its New York date and whether it came
before the 16:00 close. A filing accepted after the close is known only from the next session: ``known_session``
is the first exchange session whose close is after the acceptance, and every feature uses that, never the filing
date alone.

**Storage.** ``<data cache>/optimizer/sec_news.sqlite3``: ``filings`` (one row per 8-K), ``texts`` (exhibit text, fetched
separately because it is many more requests), ``companies`` (when each was last synced). Re-running only adds
what is new. SEC asks for at most 10 requests a second; this keeps well under it.

    python -m tradingagents.stocks.news_archive sync            # the filings index for S&P 500 members
    python -m tradingagents.stocks.news_archive texts --limit N # press-release text, newest first, resumable
"""

from __future__ import annotations

import html as _html
import re
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

START = "2006-01-01"
FORMS = ("8-K", "8-K/A")
PAUSE = 0.15                       # seconds between SEC requests (about 7 a second)
TEXT_CHARS = 60_000
_SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
ITEM_GROUPS = {"results": ("2.02",), "exec_change": ("5.02",), "deal": ("1.01", "2.01"),
               "announcement": ("7.01", "8.01"), "guidance_or_other": ("7.01",)}

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession TEXT PRIMARY KEY, cik TEXT NOT NULL, ticker TEXT, form TEXT NOT NULL, filed TEXT NOT NULL,
    accepted_at TEXT,                      -- UTC, as SEC states it
    ny_date TEXT,                          -- the New York calendar date of acceptance
    after_close INTEGER,                   -- 1 when accepted at or after 16:00 New York time
    items TEXT, primary_doc TEXT, report_date TEXT
);
CREATE INDEX IF NOT EXISTS filings_ticker ON filings (ticker, ny_date);
CREATE TABLE IF NOT EXISTS texts (
    accession TEXT PRIMARY KEY, document TEXT, text TEXT, fetched_at TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS companies (cik TEXT PRIMARY KEY, ticker TEXT, synced_at TEXT NOT NULL, filings INTEGER);
"""


def path(cache_dir) -> Path:
    """``cache_dir`` is the optimizer cache (``<data cache>/optimizer``), as everywhere in ``stocks``."""
    return Path(cache_dir) / "sec_news.sqlite3"


def connect(cache_dir) -> sqlite3.Connection:
    p = path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def _get(url: str):
    import requests

    from tradingagents.dataflows.vendors.sec_edgar import _user_agent

    for attempt in range(4):
        time.sleep(PAUSE)
        r = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:      # slow down and retry
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def _ny(accepted: str) -> tuple[str | None, int | None]:
    """(New York date, after the close) from SEC's UTC acceptance time."""
    if not accepted:
        return None, None
    t = pd.Timestamp(accepted)
    t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("America/New_York")
    return t.date().isoformat(), int(t.hour >= 16)


def _rows(block: dict, cik: str, ticker: str) -> list[tuple]:
    out = []
    n = len(block.get("accessionNumber", []))
    for i in range(n):
        form, filed = block["form"][i], block["filingDate"][i]
        if form not in FORMS or filed < START:
            continue
        accepted = block.get("acceptanceDateTime", [None] * n)[i]
        ny_date, after = _ny(accepted)
        out.append((block["accessionNumber"][i], cik, ticker, form, filed, accepted, ny_date, after,
                    block.get("items", [""] * n)[i], block.get("primaryDocument", [""] * n)[i],
                    block.get("reportDate", [""] * n)[i]))
    return out


def sync_company(con, cik: str, ticker: str) -> int:
    """Every 8-K since ``START`` for one company (recent filings and the older pages); returns how many rows."""
    data = _get(_SUBMISSIONS.format(name=f"CIK{cik}.json")).json()
    rows = _rows(data["filings"]["recent"], cik, ticker)
    for page in data["filings"].get("files", []):
        if page.get("filingTo", "9999") < START:
            continue
        rows += _rows(_get(_SUBMISSIONS.format(name=page["name"])).json(), cik, ticker)
    with con:
        con.executemany("INSERT OR IGNORE INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.execute("INSERT OR REPLACE INTO companies VALUES (?, ?, ?, ?)",
                    (cik, ticker, datetime.now(UTC).isoformat(timespec="seconds"), len(rows)))
    return len(rows)


def companies(cache_dir) -> dict[str, str]:
    """Ticker -> CIK for every S&P 500 member with a CIK (today's list; removed members have no reliable CIK)."""
    from .data import load as load_data

    data = load_data([], Path(cache_dir))
    return {t: info["cik"] for t, info in data.info.items() if info.get("cik")}


def sync(cache_dir, tickers: dict[str, str] | None = None, max_age_hours: float = 20, progress=print) -> dict:
    """Bring the filings index up to date; companies synced within ``max_age_hours`` are skipped."""
    con = connect(cache_dir)
    tickers = tickers or companies(cache_dir)
    fresh = {r["cik"] for r in con.execute("SELECT cik, synced_at FROM companies")
             if (datetime.now(UTC) - datetime.fromisoformat(r["synced_at"])).total_seconds() < max_age_hours * 3600}
    todo = [(t, c) for t, c in sorted(tickers.items()) if c not in fresh]
    added, failed = 0, []
    for k, (t, cik) in enumerate(todo, 1):
        try:
            added += sync_company(con, cik, t)
        except Exception as exc:  # one company's failure stays with it
            failed.append(f"{t}: {exc}")
        if k % 25 == 0 or k == len(todo):
            progress(f"SEC 8-K index: {k} of {len(todo)} companies")
    total = con.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    return {"companies": len(todo), "rows_seen": added, "filings": total, "failed": failed}


def _text_of(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</h\d>", "\n", raw)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()[:TEXT_CHARS]


def fetch_texts(cache_dir, limit: int = 500, items: tuple[str, ...] | None = ("2.02", "7.01", "8.01"),
                progress=print) -> dict:
    """Press-release text (exhibit 99, else the 8-K itself) for filings that have none yet, newest first."""
    con = connect(cache_dir)
    q = ("SELECT f.accession, f.cik, f.items, f.primary_doc FROM filings f LEFT JOIN texts t USING (accession)"
         " WHERE t.accession IS NULL ORDER BY f.accepted_at DESC")
    done = failed = 0
    for r in con.execute(q).fetchall():
        if done + failed >= limit:
            break
        if items and not any(i in (r["items"] or "") for i in items):
            continue
        acc = r["accession"].replace("-", "")
        try:
            listing = _get(_ARCHIVE.format(cik=int(r["cik"]), acc=acc, doc="index.json")).json()
            names = [x["name"] for x in listing["directory"]["item"]]
            ex = sorted(n for n in names if re.search(r"ex-?99", n, re.I) and n.lower().endswith((".htm", ".html", ".txt")))
            doc = ex[0] if ex else r["primary_doc"]
            text = _text_of(_get(_ARCHIVE.format(cik=int(r["cik"]), acc=acc, doc=doc)).text)
            row, done = (r["accession"], doc, text, "ok"), done + 1
        except Exception as exc:  # recorded so it is not retried endlessly
            row, failed = (r["accession"], None, str(exc)[:300], "failed"), failed + 1
        with con:
            con.execute("INSERT OR REPLACE INTO texts VALUES (?, ?, ?, ?, ?)",
                        (row[0], row[1], row[2], datetime.now(UTC).isoformat(timespec="seconds"), row[3]))
        if (done + failed) % 50 == 0:
            progress(f"SEC press releases: {done} saved, {failed} failed")
    return {"saved": done, "failed": failed,
            "texts": con.execute("SELECT COUNT(*) FROM texts WHERE status = 'ok'").fetchone()[0]}


def known_session(ny_date: pd.Series, after_close: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """The first session whose close comes after the filing: its own date if filed before 16:00 on a session day,
    else the next session."""
    d = pd.to_datetime(ny_date)
    pos = sessions.searchsorted(d, side="left")                  # first session on or after the date
    on_session = (pos < len(sessions)) & (sessions[pos.clip(max=len(sessions) - 1)] == d)
    pos = pos + (on_session & (after_close.fillna(1).astype(int) == 1)).astype(int)
    out = pd.Series(pd.NaT, index=ny_date.index, dtype="datetime64[ns]")
    ok = pos < len(sessions)
    out[ok] = sessions[pos[ok]]
    return out


def features(cache_dir, dates: pd.DatetimeIndex, tickers: list[str], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (date, ticker), from filings known by that date's close: 8-Ks in the last 5 and 20 sessions, sessions
    since the last results release (Item 2.02), and executive changes, deals and announcements in the last 60."""
    con = connect(cache_dir)
    f = pd.read_sql_query("SELECT ticker, ny_date, after_close, items FROM filings WHERE ny_date IS NOT NULL", con)
    f = f[f["ticker"].isin(tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    cols = ["n8k_5", "n8k_20", "since_results", "exec_change_60", "deal_60", "announce_60"]
    if f.empty:
        return pd.DataFrame(index=idx, columns=cols, dtype=float)
    f["known"] = known_session(f["ny_date"], f["after_close"], sessions)
    f = f.dropna(subset=["known"])
    f["k"] = sessions.get_indexer(f["known"])
    pos = pd.Series(sessions.get_indexer(dates), index=dates)
    out = []
    for t, g in f.groupby("ticker"):
        k = g["k"].to_numpy()
        res = g["items"].fillna("").str.contains("2.02", regex=False).to_numpy()
        flags = {c: g["items"].fillna("").str.contains("|".join(re.escape(i) for i in ITEM_GROUPS[c])).to_numpy()
                 for c in ("exec_change", "deal", "announcement")}
        for d, p in pos.items():
            if p < 0:
                continue
            seen = k <= p
            age = p - k
            last_res = age[seen & res]
            out.append((d, t, int((seen & (age < 5)).sum()), int((seen & (age < 20)).sum()),
                        float(last_res.min()) if len(last_res) else float("nan"),
                        int((seen & (age < 60) & flags["exec_change"]).sum()),
                        int((seen & (age < 60) & flags["deal"]).sum()),
                        int((seen & (age < 60) & flags["announcement"]).sum())))
    frame = pd.DataFrame(out, columns=["date", "ticker", *cols]).set_index(["date", "ticker"])
    return frame.reindex(idx)


if __name__ == "__main__":
    from tradingagents.default_config import DEFAULT_CONFIG

    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sync"
    if cmd == "sync":
        print(sync(cache))
    elif cmd == "texts":
        limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 500
        print(fetch_texts(cache, limit))
