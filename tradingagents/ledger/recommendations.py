"""Official recommendations: saved once, never changed, and kept apart from previews.

A preview is any plan computed on the page or the command line; nothing about it
is stored. Saving makes a recommendation **official**: it is written whole (the
plan, its inputs' identities, the code and settings that made it) and can only
be superseded by a later one, never edited. Prices it used are saved in the
price store, so its inputs can be rebuilt.

Which one counts. For a sleeve and a trading session S, the official
recommendation is the **last one saved for S before S's order cutoff**
(15:45 New York time, so orders can still be placed for the close). Its orders
cannot fill before ``earliest_execution``: the first session after its
information cutoff (the last close its prices include), and never earlier than
the session in which it was published. A recommendation saved after the
cutoff is for the next session.

Accepting a recommendation is a decision, not a trade; holdings change only
through fills.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
import zlib
from dataclasses import asdict, is_dataclass
from datetime import datetime, time
from pathlib import Path

import pytz

from . import price_store
from .db import now_utc, transaction
from .sessions import as_date, is_session, next_session, session_on_or_after

NEW_YORK = pytz.timezone("America/New_York")
ORDER_CUTOFF = time(15, 45)
STRATEGY_VERSION = "optimizer-v1"


def code_version(repo: Path | None = None) -> str:
    """The git commit, with '+dirty:<hash of the diff>' when the working tree has changes."""
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not head:
        return "unknown"
    return head + (f"+dirty:{hashlib.sha256(diff).hexdigest()[:12]}" if diff else "")


def session_for(published: datetime) -> str:
    """The session a recommendation published at ``published`` is for."""
    local = published.astimezone(NEW_YORK)
    day = local.date()
    if is_session(day) and local.time() < ORDER_CUTOFF:
        return day.isoformat()
    return next_session(day).isoformat()


def earliest_execution(info_cutoff, published: datetime) -> str:
    """First session an order may fill: after the information cutoff, and not before publication."""
    after_data = next_session(info_cutoff)
    return max(after_data, as_date(session_for(published))).isoformat()


def _json(x) -> str:
    return json.dumps(asdict(x) if is_dataclass(x) else x, default=str, sort_keys=True)


def save(con, *, account_id: str, sleeve: str, plan, suggestion, settings, closes, universe: list[str],
         info_cutoff: str, market_data_at: str, ai: dict | None = None, published: datetime | None = None,
         price_tickers: list[str] | None = None) -> dict:
    """Store a plan as the official recommendation for its session; returns its id and timing."""
    published = published or datetime.now(pytz.utc)
    session = session_for(published)
    tickers = sorted(set(price_tickers if price_tickers is not None else
                         [r.ticker for r in plan.rows] + [plan.cash.ticker]) & set(closes.columns))
    snapshot = price_store.save(con, closes, tickers)
    plan_json, suggestion_json = json.loads(_json(plan)), json.loads(_json(suggestion))
    artifact = {"plan": plan_json, "suggestion": suggestion_json, "settings": json.loads(_json(settings)),
                "price_snapshot_at": snapshot, "price_tickers": tickers}
    blob = json.dumps(artifact, sort_keys=True, default=str).encode()
    targets = {r["ticker"]: r["target"] for r in plan_json["rows"] if r["target"] > 1e-6}
    targets[plan_json["cash"]["ticker"]] = plan_json["cash"]["target"]
    orders = {"today": suggestion_json["orders"], "next_session": suggestion_json["tomorrow"]}
    metrics = {k: plan_json[k] for k in ("total", "current_stress", "target_stress", "expected_current",
                                         "expected_target", "stress_worst_case", "stress_scenarios",
                                         "worst_scenario", "current_history_dd", "target_history_dd", "goal")}
    reasons = {r["ticker"]: r["reason"] for r in plan_json["rows"] if r["target"] > 1e-6 or r["held"]}
    state = {"positions": {r["ticker"]: r["value"] for r in plan_json["rows"] if r["value"]},
             "cash_fund": plan_json["cash"]["value"], "settled_cash": suggestion_json["settled_cash"]}
    rec_id = uuid.uuid4().hex
    with transaction(con):
        prior = official(con, account_id, sleeve, session)
        con.execute(
            "INSERT INTO recommendations VALUES (" + ", ".join("?" * 23) + ")",
            (rec_id, account_id, sleeve, session, now_utc(), published.astimezone(pytz.utc).isoformat(),
             as_date(info_cutoff).isoformat(), market_data_at, earliest_execution(info_cutoff, published),
             prior["id"] if prior else None, code_version(), STRATEGY_VERSION, _json(settings), json.dumps(state),
             json.dumps(targets, sort_keys=True), json.dumps(orders, default=str), json.dumps(metrics, default=str),
             json.dumps(reasons), json.dumps(universe), json.dumps(ai or {}), json.dumps(tickers),
             hashlib.sha256(blob).hexdigest(), zlib.compress(blob, 6)))
        con.execute("INSERT INTO recommendation_status (recommendation_id, status, at) VALUES (?, 'official', ?)",
                    (rec_id, now_utc()))
        if prior:
            con.execute("INSERT INTO recommendation_status (recommendation_id, status, at, note) VALUES (?, 'superseded', ?, ?)",
                        (prior["id"], now_utc(), f"superseded by {rec_id}"))
    return {"id": rec_id, "session": session, "earliest_execution": earliest_execution(info_cutoff, published),
            "supersedes": prior["id"] if prior else None, "price_snapshot_at": snapshot}


def status(con, rec_id: str) -> str | None:
    row = con.execute("SELECT status FROM recommendation_status WHERE recommendation_id = ? ORDER BY id DESC LIMIT 1",
                      (rec_id,)).fetchone()
    return row["status"] if row else None


def official(con, account_id: str, sleeve: str, session: str) -> dict | None:
    """The current official recommendation for a sleeve and session, if any."""
    for row in con.execute("SELECT * FROM recommendations WHERE account_id = ? AND sleeve = ? AND session = ?"
                           " ORDER BY published_at DESC", (account_id, sleeve, session)):
        if status(con, row["id"]) == "official":
            return dict(row)
    return None


def load(con, rec_id: str) -> dict:
    row = con.execute("SELECT * FROM recommendations WHERE id = ?", (rec_id,)).fetchone()
    if row is None:
        raise KeyError(rec_id)
    out = dict(row)
    blob = zlib.decompress(out.pop("artifact"))
    if hashlib.sha256(blob).hexdigest() != out["artifact_sha256"]:
        raise ValueError(f"recommendation {rec_id} artifact does not match its hash")
    out["artifact"] = json.loads(blob)
    out["status"] = status(con, rec_id)
    return out


def history(con, account_id: str, sleeve: str | None = None, limit: int = 200) -> list[dict]:
    sql = ("SELECT id, sleeve, session, published_at, info_cutoff, earliest_execution, supersedes, code_version,"
           " strategy_version, metrics_json, targets_json FROM recommendations WHERE account_id = ?")
    args: list = [account_id]
    if sleeve:
        sql += " AND sleeve = ?"
        args.append(sleeve)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY published_at DESC LIMIT ?", [*args, limit])]
    for r in rows:
        r["status"] = status(con, r["id"])
        r["metrics"] = json.loads(r.pop("metrics_json"))
        r["targets"] = json.loads(r.pop("targets_json"))
    return rows


def decide(con, rec_id: str, decision: str, note: str | None = None) -> int:
    """Record what you decided about a recommendation. Nothing is bought or sold by this."""
    with transaction(con):
        cur = con.execute("INSERT INTO decisions (recommendation_id, decision, decided_at, note) VALUES (?, ?, ?, ?)",
                          (rec_id, decision, now_utc(), note))
        return cur.lastrowid


def decisions(con, rec_id: str) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM decisions WHERE recommendation_id = ? ORDER BY id", (rec_id,))]


def next_execution_session(rec: dict) -> str:
    return session_on_or_after(rec["earliest_execution"]).isoformat()
