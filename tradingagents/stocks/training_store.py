"""Every training set the decision models were fitted on, saved in DuckDB, and refits skipped on the same data.

``training.duckdb`` (next to the price store) holds, per fitted bundle:

- ``training_sets``: one row per distinct training set: its **fingerprint** (a hash of every value, row
  label and column of the training table), the cutoff, the dates it covers, the row count, the model
  bundle fitted on it and when;
- ``training_rows``: the training table itself (decision date, ticker, the features, the label, the
  label's end date), tagged with its set's fingerprint.

``daily.fit_bundle`` builds the training table, fingerprints it, and when that exact set was fitted
before, loads that bundle instead of training again, whatever the week; otherwise it fits, then saves the
set and its bundle here. Identical data is never trained twice, and every model can be traced to the
exact rows it learned from.

DuckDB lets one process write a file at a time, so connections are opened per call and retried briefly
when another process holds the lock. The price store stays in SQLite (``dataflows.local_store``):
the web app, decision runs and study scripts write it at the same time, which a single-writer file
would serialize.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

FILE = "training.duckdb"


def path(cache_dir) -> Path:
    return Path(cache_dir) / FILE


def _connect(cache_dir, read_only: bool = False, tries: int = 20):
    import duckdb

    p = path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if read_only and not p.exists():
        return None
    for i in range(tries):
        try:
            con = duckdb.connect(str(p), read_only=read_only)
            if not read_only:
                con.execute("CREATE TABLE IF NOT EXISTS training_sets (fingerprint VARCHAR PRIMARY KEY, created_at VARCHAR,"
                            " cutoff VARCHAR, first_date VARCHAR, last_date VARCHAR, trained_through VARCHAR,"
                            " n_rows BIGINT, n_dates INTEGER, bundle_key VARCHAR, bundle_path VARCHAR, meta_json VARCHAR)")
            return con
        except Exception as exc:   # another process holds the write lock: wait a moment
            if "lock" not in str(exc).lower() or i == tries - 1:
                raise
            time.sleep(0.5)


def fingerprint(frame: pd.DataFrame, extra: dict | None = None) -> str:
    """A hash of every value, the row labels and the columns, plus ``extra`` (the design: features, policy, code)."""
    h = hashlib.sha256()
    h.update(json.dumps([list(map(str, frame.columns)), extra or {}], sort_keys=True, default=str).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()[:32]


def find(cache_dir, fp: str) -> dict | None:
    """The saved set with this fingerprint and the bundle fitted on it, when its file still exists."""
    con = _connect(cache_dir, read_only=True)
    if con is None:
        return None
    try:
        row = con.execute("SELECT * FROM training_sets WHERE fingerprint = ?", [fp]).fetchone()
        cols = [d[0] for d in con.description] if row else []
    except Exception:   # an empty file has no table yet
        return None
    finally:
        con.close()
    if not row:
        return None
    out = dict(zip(cols, row, strict=True))
    return out if out.get("bundle_path") and Path(out["bundle_path"]).exists() else None


def save(cache_dir, fp: str, table: pd.DataFrame, meta: dict, bundle_key: str, bundle_path: Path) -> None:
    """Keep the training table and what was fitted on it (once per fingerprint)."""
    dates = pd.to_datetime(table.index.get_level_values("date"))
    frame = table.reset_index()
    frame.insert(0, "fingerprint", fp)
    con = _connect(cache_dir)
    try:
        if con.execute("SELECT 1 FROM training_sets WHERE fingerprint = ?", [fp]).fetchone():
            return
        con.register("incoming", frame)
        exists = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'training_rows'").fetchone()[0]
        if exists:
            # Columns follow the first set; a later design with other columns gets its own table.
            have = [r[0] for r in con.execute("DESCRIBE training_rows").fetchall()]
            if have == list(frame.columns):
                con.execute("INSERT INTO training_rows SELECT * FROM incoming")
            else:
                con.execute(f"CREATE TABLE training_rows_{fp[:12]} AS SELECT * FROM incoming")
        else:
            con.execute("CREATE TABLE training_rows AS SELECT * FROM incoming")
        con.execute("INSERT INTO training_sets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [fp, datetime.now().isoformat(timespec="seconds"), meta.get("cutoff"), str(dates.min().date()),
                     str(dates.max().date()), meta.get("trained_through"), len(table), int(dates.nunique()), bundle_key,
                     str(bundle_path), json.dumps(meta, default=str)])
    finally:
        con.close()


def listing(cache_dir, limit: int = 20) -> list[dict]:
    con = _connect(cache_dir, read_only=True)
    if con is None:
        return []
    try:
        rows = con.execute("SELECT fingerprint, created_at, cutoff, first_date, last_date, trained_through, n_rows, n_dates,"
                           " bundle_key FROM training_sets ORDER BY created_at DESC LIMIT ?", [limit]).fetchall()
        cols = [d[0] for d in con.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]
    except Exception:
        return []
    finally:
        con.close()


class TrainedOnError(ValueError):
    """Scoring rows the model was trained on: a result there says nothing about new data."""


def guard(trained_through, dates) -> None:
    """Refuse to score any decision date on or before the last date the model learned from."""
    if trained_through is None or dates is None or not len(dates):
        return
    first = pd.to_datetime(pd.Index(dates)).min()
    if first <= pd.Timestamp(trained_through):
        raise TrainedOnError(f"refusing to score {first.date()}: the model was trained on decision dates through "
                             f"{pd.Timestamp(trained_through).date()}; scoring them would test it on its own training data")
