"""A local web app for the daily optimization: one page and a small JSON API.

It runs on this machine because it needs the price feed and the LLM keys in
.env, and it serves only localhost: the page edits your records and can spend
LLM credit, so nothing else on the network should reach it.

Records live in ``portfolio.sqlite3`` next to the holdings file (see
``tradingagents.ledger``): one trading account (SGOV as its base, stocks bought
with SGOV's interest and the broker's match), every recorded fill, the daily
stock decisions and AI usage. The overview is the trade view (``trade``): the
budget, the best stock to buy and at what price, what to sell, whether SGOV's
interest is drying up, and the sector rotation alerts. Paid AI runs only on an
explicit POST, through the budgeted gateway.

Prices are downloaded once per day into the cache directory and reused. AI
ratings and backtests run in background threads, since they take minutes, and
the page polls for their progress.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tradingagents.dataflows import local_store
from tradingagents.dataflows.vendors.index_members import get_sp500_members
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.ledger import capital, imports as csv_import, recommendations
from tradingagents.ledger.accounting import SLEEVES
from tradingagents.ledger.ai_gateway import Budget, Gateway, Policy, Price
from tradingagents.ledger.db import transaction
from tradingagents.ledger.store import Store
from tradingagents.model_outputs import measures
from tradingagents.optimizer.ai_views import collect_ratings
from tradingagents.optimizer.data import fetch_closes, latest_prices
from tradingagents.optimizer.market import market_brief, market_overview, market_tickers, save_json
from tradingagents.optimizer.plan import OptimizerSettings
from tradingagents.optimizer.universe import (
    etf_list,
    GROUPS,
    THEME_GROUPS,
    describe,
    read_config,
    tickers_for,
)

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


class Job:
    """A background task the page polls: its progress, result or error."""

    STUCK_AFTER = 600   # seconds without progress before the page may stop waiting

    def __init__(self):
        self.running, self._progress, self.result, self.error, self.log = False, "", None, None, []
        self.attempts, self.events, self.gen, self.fraction = [], [], 0, None
        self.started_at = self.touched_at = 0.0
        self.switch = threading.Lock()   # held while a run starts or is let go, and while a run writes

    @property
    def progress(self) -> str:
        return self._progress

    @progress.setter
    def progress(self, text: str):
        """Every progress line also goes to the short live list the page shows."""
        self._progress, self.touched_at = text, time.time()
        self.events = [*self.events[-7:], {"at": time.strftime("%H:%M:%S"), "text": str(text)[:140]}]

    def start(self, work) -> bool:
        with self.switch:
            if self.running:
                return False
            self.gen += 1
            gen = self.gen
            self.running, self.result, self.error, self.log, self.attempts, self.events = True, None, None, [], [], []
            self.fraction = None
            self.started_at = time.time()
            self.progress = "starting"

        def run():
            result, error = None, None
            try:
                result = work(_Run(self, gen))
            except Exception as exc:  # shown on the page rather than lost in a thread
                logger.exception("background job failed")
                error = str(exc)
            finally:
                with self.switch:
                    if gen == self.gen:   # a run the page stopped waiting for never overwrites a newer one
                        self.result, self.error, self.running = result, error, False

        threading.Thread(target=run, daemon=True).start()
        return True

    def dismiss(self) -> bool:
        """Clear the last error of a job that is not running (the page's Dismiss)."""
        with self.switch:
            if self.running or not self.error:
                return False
            self.error, self.attempts = None, []
            return True

    def stop_waiting(self) -> bool:
        """Give up on a run that has made no progress for ``STUCK_AFTER`` seconds. Python cannot kill the thread:
        it may still finish in the background, but its result is ignored and a new run can start."""
        with self.switch:
            if not self.running or time.time() - self.touched_at < self.STUCK_AFTER:
                return False
            self.gen += 1
            self.running = False
        self.error = (f"stopped waiting after {(time.time() - self.started_at) / 60:.0f} min: no progress since "
                      f"'{self._progress}' for {(time.time() - self.touched_at) / 60:.0f} min (a download or a provider "
                      "never answered). Run it again; if it keeps happening, restart the app.")
        return True

    def say(self, text: str):
        self.progress = text

    def state(self) -> dict:
        now = time.time()
        return {"running": self.running, "progress": self._progress, "result": self.result,
                "error": self.error, "log": self.log, "attempts": self.attempts, "events": self.events,
                "fraction": self.fraction,
                "elapsed": round(now - self.started_at) if self.started_at else None,
                "idle": round(now - self.touched_at) if self.running else None,
                "stuck": bool(self.running and now - self.touched_at >= self.STUCK_AFTER)}


class _Run:
    """What one run's worker sees of its Job. While the run is the Job's current one, reads and
    writes go to the Job; once the page has stopped waiting for it (and maybe started another),
    they go to this handle's own copies, so a late worker never overwrites the new run's
    progress, log or attempts."""

    def __init__(self, job: Job, gen: int):
        object.__setattr__(self, "_job", job)
        object.__setattr__(self, "_gen", gen)
        object.__setattr__(self, "_own", {})

    @property
    def current(self) -> bool:
        return self._job.gen == self._gen

    def __getattr__(self, name):
        if self.current:
            return getattr(self._job, name)
        if name not in self._own:
            value = getattr(self._job, name)
            self._own[name] = [] if isinstance(value, list) else None if not callable(value) else value
        return self._own[name]

    def __setattr__(self, name, value):
        with self._job.switch:
            if self.current:
                setattr(self._job, name, value)
                return
        self._own[name] = value

    def say(self, text: str):
        self.progress = text


def _finite(value):
    """NaN and infinity as null: they are not JSON, and a browser refuses the whole response."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


# Free on a Gemini free-tier key (a Google project without billing), tried in order when the provider
# before fails; each model has its own free quota.
GEMINI_FALLBACKS = ("gemini-3.8-flash", "gemini-3.5-flash-lite")


JOB_NAMES = ("ai", "explain", "forecast", "daily", "stocks", "decisions", "screen", "analysis", "sector", "zoo",
             "morning")


def _short_error(exc: Exception) -> str:
    """A provider failure in a few words for the status line."""
    text = str(exc)
    for needle, short in (("free-models-per-day", "OpenRouter free daily limit reached"),
                          ("prepayment credits are depleted", "out of credit"), ("402", "out of credit"),
                          ("agentic harnesses", "not offered to this app"), ("needs about", "held by the AI budget"),
                          ("429", "rate limited"), ("rate", "rate limited"), ("quota", "out of quota"),
                          ("401", "bad key"), ("403", "not allowed"), ("404", "model not available"),
                          ("insufficient", "out of credit"), ("timed out", "timed out"), ("tool", "no tool support")):
        if needle in text.lower():
            return short
    return text[:80]


RATE_LIMIT_HOLD = timedelta(minutes=30)


def _next_midnight(now: datetime, tz: str) -> datetime:
    from datetime import UTC
    from zoneinfo import ZoneInfo

    local = now.astimezone(ZoneInfo(tz))
    return (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def _cooldown(provider: str, model: str, exc_text: str, now: datetime) -> tuple[str, datetime, str] | None:
    """How long a failure will keep repeating, so nothing more is sent there until then: ``(key, until, why)``, or
    None for a failure worth trying again at once. Out of credit and a free daily limit hold for the whole key
    (every model on it); a model's daily quota until the quota resets (midnight Pacific for Google, else 00:00 UTC);
    a per-minute rate limit for ``RATE_LIMIT_HOLD``, or longer when the provider asks for longer."""
    text = exc_text.lower()
    if "free-models-per-day" in text or "free daily limit" in text:
        # Only the key's free models are limited; its paid models still answer.
        return f"{provider}/*:free", _next_midnight(now, "UTC"), "free daily limit reached (resets 00:00 UTC)"
    if "prepayment credits are depleted" in text or "402" in text or "insufficient" in text and "credit" in text:
        return provider, now + timedelta(hours=6), "out of credit (top up, or the key is skipped for 6 hours)"
    limited = "429" in text or "rate limit" in text or "resource_exhausted" in text or "quota" in text
    if limited and re.search(r"per ?day|perday|per_day|daily", text):
        tz, at = ("America/Los_Angeles", "midnight Pacific") if provider == "google" else ("UTC", "00:00 UTC")
        return f"{provider}/{model}", _next_midnight(now, tz), f"daily quota used up (resets {at})"
    if limited:
        asked = re.search(r"retry(?:[ _-]?(?:in|after|delay))?\W{0,4}(\d+(?:\.\d+)?)\s*s", text)
        hold = max(RATE_LIMIT_HOLD, timedelta(seconds=float(asked[1]))) if asked else RATE_LIMIT_HOLD
        return f"{provider}/{model}", now + hold, "rate limited"
    return None


# What the morning run makes (Settings > Every morning); each is saved, and loaded again on the same inputs.
MORNING_STEPS = {"events": True, "funds": True, "models": True, "decision": True, "signals": True, "sector": True}


def _today() -> str:
    return date.today().isoformat()


_COUNT = re.compile(r"(\d+)\s*(?:of|/)\s*(\d+)")


def _counted(job, text: str, base: float = 0.0, span: float = 1.0):
    """Show ``text`` and move the job's bar from an "N of M" in it (never backwards)."""
    m = _COUNT.search(str(text))
    if m and int(m[2]):
        job.fraction = max(job.fraction or 0.0, min(base + span * int(m[1]) / int(m[2]), 1.0))
    job.progress = text


_MISSING = object()


def _number(body: dict, key: str, default=_MISSING) -> float:
    """A finite number from a request body; ``default`` when absent or blank, an error otherwise."""
    value = body.get(key)
    if value is None or value == "":
        if default is _MISSING:
            raise ValueError(f"{key} is required")
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number")
    return number


_OP_ID = re.compile(r"[A-Za-z0-9_.:-]{8,100}")


def _op_id(body: dict) -> str:
    """The page's id for this user action (kept across retries), or a fresh one."""
    given = body.get("op_id")
    if given is None:
        return uuid.uuid4().hex
    if not isinstance(given, str) or not _OP_ID.fullmatch(given):
        raise ValueError("op_id must be 8-100 letters, digits, '-', '_', '.' or ':'")
    return given


MAX_BODY = 8 * 1024 * 1024        # a broker CSV export fits easily
_LOOPBACK = {"127.0.0.1", "localhost", "[::1]"}
_HOST = re.compile(r"(\[[0-9a-f:]+\]|[^:]+)(?::(\d+))?")


def write_refusal(headers, csrf_token: str, port: int) -> str | None:
    """Why a state-changing request must be refused, or None.

    The server binds to loopback only, but a page on another site can still make the
    browser send a request to it. A write must therefore name this server as its Host
    (no DNS rebinding), come from this page's own origin when the browser says where it
    comes from, send JSON (which a cross-site form cannot do without a CORS preflight
    this server never grants), and carry the token printed into the page it was served.
    """
    host = (headers.get("Host") or "").strip().lower()
    parts = _HOST.fullmatch(host)
    if not parts or parts[1] not in _LOOPBACK or (parts[2] and parts[2] != str(port)):
        return "the request is not addressed to this local server"
    origin = headers.get("Origin")
    if origin is not None and origin.lower() != f"http://{host}":
        return "the request comes from another site"
    if (headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
        return "the request comes from another site"
    if (headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
        return "a write must send JSON (Content-Type: application/json)"
    if not csrf_token or not secrets.compare_digest(headers.get("X-CSRF-Token") or "", csrf_token):
        return "the page's session token is missing or stale; reload the page"
    return None


class App:
    """What the handlers share: the records, the files, the cache and the background jobs."""

    def __init__(self, portfolio_path: Path, schedule_path: Path, universe_path: Path, config: dict):
        self.portfolio_path = portfolio_path
        self.schedule_path = schedule_path
        self.universe_path = universe_path
        self.config = config
        self.cache_dir = Path(config["data_cache_dir"]) / "optimizer"
        self.tracker_path = portfolio_path.parent / "paper_tracker.json"
        self.store = Store(portfolio_path.parent)
        self.owner = f"web-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.lock = threading.Lock()   # one writer at a time on the shared connection
        self.csrf_token = secrets.token_urlsafe(32)   # printed into the pages; every write must return it
        self.morning = Job()
        self.ai, self.backtest, self.tracker, self.explain, self.forecast = Job(), Job(), Job(), Job(), Job()
        self.moat_jobs: dict[str, Job] = {}
        self.daily = Job()   # the daily Chronos-2 record, apart from the runs started on the page
        self.stocks = Job()  # a stock research batch
        self.decisions = Job()  # the daily stock decision (and, when asked, this week's model fit)
        self.screen = Job()     # the quality and Wall Street screen
        self.analysis = Job()   # one ticker's agent run for a chosen day, on request
        self.sector = Job()     # the sector model: study and next-20-session forecast
        self.zoo = Job()        # every stock forecasting model, fitted and scored side by side
        self.ratings: dict[str, str] = {}
        self.ratings_date: str | None = None
        self.last_targets: list[str] = []
        self.last_rows: dict[str, dict] = {}
        self._memo: dict[str, tuple[str, object]] = {}   # what was computed, and from which inputs

    def read_json(self, path: Path, default):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def universe(self) -> list[str]:
        return tickers_for(read_config(self.universe_path), self.cache_dir)

    def gateway(self) -> Gateway:
        g = self.store.settings("global")
        prices = {k: Price(**v) for k, v in g["ai_prices"].items()}
        # OpenRouter's ":free" models cost nothing; the budget still counts their requests.
        prices.update({f"openrouter/{m}": Price(0.0, 0.0) for m in g["ai_openrouter_models"]
                       if m.endswith(":free") and f"openrouter/{m}" not in prices})
        return Gateway(self.store.con, Policy(Budget(**g["ai_budget"]), prices, list(g["ai_fallbacks"])), self.owner)

    # --- planning ---

    def _inputs(self, *parts) -> str:
        """A fingerprint of everything a computed page section depends on: the stored prices,
        the books, the settings and the files it reads. Same fingerprint, same answer."""
        market_store = local_store.store(self.cache_dir)
        last_event = self.store.con.execute("SELECT MAX(id) FROM events").fetchone()[0]
        voids = self.store.con.execute("SELECT COUNT(*) FROM event_voids").fetchone()[0]
        files = [f.stat().st_mtime if f.exists() else 0 for f in (self.schedule_path, self.universe_path)]
        settings = [self.store.settings(sl) for sl in SLEEVES] + [capital.snapshot(self.store)]
        return json.dumps([market_store.bars_checked_at() if market_store else None, last_event, voids, files,
                           settings, self.ratings_date, _today(), *parts], sort_keys=True, default=str)

    def _remember(self, name: str, inputs: str, make):
        hit = self._memo.get(name)
        if hit and hit[0] == inputs:
            return hit[1]
        value = make()
        self._memo[name] = (inputs, value)
        return value

    # --- the trade view ---

    def income(self) -> dict:
        """The trading budget and whether SGOV's interest is drying up (alternatives when it is)."""
        from tradingagents.ledger import income

        today = date.today()
        books = self.books()["sleeves"]["combined"]
        sgov = (books["positions"].get(income.RESERVE) or {}).get("value") or 0.0
        return {"budget": income.budget(self.store, self.cache_dir, today), "sgov_value": sgov}

    def rotation(self) -> dict:
        from tradingagents.research.rotation import rotation

        return self._remember("rotation", self._inputs("rotation"), lambda: rotation(self.cache_dir, self.store.con))

    def trade(self) -> dict:
        """The overview: what to buy with the budget, at what price, and what to sell; from the internal model."""
        from tradingagents.stocks import daily, signals, trade

        inc = self.income()
        decision = daily.load(self.store.con)
        patient = {}
        if decision:
            sig = signals.latest(self.store.con, 20)
            for e in decision.get("entries", [])[:8]:
                s = sig.get(e["ticker"])
                if s and s.get("forecast"):
                    plan = self._entry_plan(e["ticker"], s)
                    dip = next((r for r in (plan or {}).get("rows", []) if r["name"] == "Median dip"), None)
                    if dip:
                        patient[e["ticker"]] = {"price": round(dip["price"], 2), "chance": dip["chance"],
                                                "stop": round(dip["stop"], 2), "target": round(dip["profit"], 2)}
        out = trade.plan(decision, inc["budget"]["available"], patient)
        asked = self.store.get("global", "ai_pick_state") or {}
        done = asked.get("date") == _today() and all(self._agents_ran_today(t) for t in asked.get("tickers") or [])
        out["ai_pick"] = {**asked, "today": done}      # asked today and every stock in it decided
        gg = self.store.settings("global")
        out["scans"] = {"used": len(self._scanned_today()), "limit": int(gg.get("ai_daily_scans") or 0),
                        "my_picks": gg.get("ai_my_picks") or []}
        scr = self.store.get("global", "ai_screen_state") or {}
        if scr.get("date") == _today():
            last = latest_prices(self.cache_dir, [p["ticker"] for p in scr.get("picks") or []])
            scr = {**scr, "picks": [{**p, "ltp": last.get(p["ticker"])} for p in scr.get("picks") or []]}
            out["ai_screen"] = scr
        else:
            out["ai_screen"] = None
        picks = ([out["best"]] if out.get("best") else []) + list(out.get("runners_up") or [])
        if picks:
            out["allocation"] = self._allocation(picks, inc["budget"]["available"])
        pv = (decision or {}).get("policy") or {}
        if pv.get("available"):
            st = pv.get("stocks") or {}
            ranked = sorted(((t, d) for t, d in st.items() if d["model_choice"] in ("next_close", "limit_dip")),
                            key=lambda x: -max(o["expected_excess"] for o in x[1]["options"]
                                               if o["action"] == x[1]["model_choice"]))
            out["policy"] = {k: pv.get(k) for k in ("model_version", "trained_to", "status", "margin")}
            out["policy"]["top"] = [{"ticker": t, **d} for t, d in ranked[:5]]
            out["policy"]["held"] = [{"ticker": t, **d} for t, d in st.items() if d["model_choice"] in ("hold", "sell")]
        else:
            out["policy"] = {"available": False, "note": pv.get("note") or "no decision with the policy model yet"}
        books = self.books()
        market_store = local_store.store(self.cache_dir)
        out.update({"income": inc, "account": books["sleeves"]["combined"],
                    "job": self._job_or_morning(self.decisions, "Daily decision"),
                    "freshness": {"prices_checked": market_store.bars_checked_at() if market_store else None,
                                  "last_event": books["events"][0]["recorded_at"] if books["events"] else None}})
        return out

    def portfolio_view(self) -> dict:
        """The stocks you recorded buying (not SGOV): units, average cost (first in, first out), value and profit."""
        from tradingagents.ledger.income import RESERVE
        from tradingagents.stocks.daily import lots_from_ledger

        today = _today()
        lots = [lot for lot in lots_from_ledger(self.store, "trading", today) if lot["ticker"] != RESERVE]
        fills = [e for e in self.store.ledger.events("trading", today) if e["kind"] == "fill" and e["ticker"] != RESERVE]
        tickers = sorted({lot["ticker"] for lot in lots} | {e["ticker"] for e in fills})
        prices = latest_prices(self.cache_dir, tickers) if tickers else {}
        rows = {}
        for lot in lots:
            r = rows.setdefault(lot["ticker"], {"ticker": lot["ticker"], "units": 0.0, "cost": 0.0, "since": lot["entry_date"]})
            r["units"] += lot["shares"]
            r["cost"] += lot["shares"] * (lot.get("entry_px") or 0.0)
            r["since"] = min(r["since"], lot["entry_date"])
        for r in rows.values():
            px = prices.get(r["ticker"])
            r["avg_cost"] = r["cost"] / r["units"] if r["units"] else None
            r["price"] = px
            r["value"] = r["units"] * px if px else None
            r["pnl"] = r["value"] - r["cost"] if r["value"] is not None else None
            r["pnl_pct"] = r["pnl"] / r["cost"] if r["pnl"] is not None and r["cost"] else None
        bought = -sum(e["cash"] for e in fills if e["cash"] < 0)
        sold = sum(e["cash"] for e in fills if e["cash"] > 0)
        open_cost = sum(r["cost"] for r in rows.values())
        value = sum(r["value"] or 0.0 for r in rows.values())
        # The whole account beside the stocks: SGOV, cash by state, and what is set aside, so "value" is never read
        # as the account total.
        whole = self.store.ledger.state(None, today)
        reserve_units = whole.positions.get(RESERVE, 0.0)
        reserve_px = latest_prices(self.cache_dir, [RESERVE]).get(RESERVE) if reserve_units else None
        missing = [r["ticker"] for r in rows.values() if r["value"] is None]
        if reserve_units and not reserve_px:
            missing.append(RESERVE)
        reserve_value = reserve_units * reserve_px if reserve_px else 0.0
        account = {"stocks": round(value, 2), "reserve_fund": RESERVE, "reserve_units": reserve_units,
                   "reserve_value": round(reserve_value, 2), "cash_settled": round(whole.cash.settled, 2),
                   "cash_unsettled": round(whole.cash.unsettled, 2), "cash_reserved": round(whole.cash.reserved, 2),
                   "cash_available": round(whole.cash.available, 2),
                   "total": round(value + reserve_value + whole.cash.total, 2), "missing_prices": missing}
        return {"as_of": today, "holdings": sorted(rows.values(), key=lambda r: -(r["value"] or 0)),
                "account": account,
                "value": value, "cost": open_cost, "unrealized": value - open_cost,
                "realized": sold - (bought - open_cost), "bought": bought, "sold": sold,
                "trades": [{k: e[k] for k in ("id", "trade_date", "ticker", "quantity", "price", "cash", "note")}
                           for e in sorted(fills, key=lambda e: (e["trade_date"], e["id"]), reverse=True)]}

    def sector_forecast(self) -> dict:
        """The saved sector forecast, its validation, and each sector's top stock by the internal model."""
        from tradingagents.research import sector_model
        from tradingagents.stocks import daily

        rep = sector_model.latest(self.cache_dir)
        out = {"job": self.sector.state(), "report": rep}
        if rep:
            out["winners"] = sector_model.winners(daily.load(self.store.con), get_sp500_members(self.cache_dir))
        return out

    def start_sector(self) -> bool:
        from tradingagents.research import sector_model

        def work(job):
            rep = sector_model.run(self.cache_dir, self.store, lambda text: job.say(text))
            return {"status": rep["status"], "as_of": rep["forecast"]["as_of"]}

        return self.sector.start(work)

    def _allocation(self, picks: list[dict], budget: float) -> dict:
        """The trading budget split across the 1st pick and the runners-up (``stocks.allocation``)."""
        from tradingagents.research.prices import bars
        from tradingagents.stocks import allocation

        tickers = [p["ticker"] for p in picks]
        try:
            close = bars(tickers, self.cache_dir)["Close"]
        except Exception:   # no price history: the split falls back to equal weights and says so
            close = None
        out = allocation.weights(tickers, [p.get("expected_net") for p in picks], close)
        out["budget"] = round(float(budget or 0.0), 2)
        out["dollars"] = {t: round(w * out["budget"], 2) for t, w in out["weights"].items()}
        return out

    def model_record(self) -> dict:
        from tradingagents.stocks import feedback

        return feedback.scorecard(self.store.con)

    # --- the books ---

    def books(self) -> dict:
        """Every sleeve and the combined account at the latest cached prices."""
        tickers = {e["ticker"] for e in self.store.ledger.events() if e["ticker"]}
        prices = latest_prices(self.cache_dir, tickers)
        out = {}
        for sleeve in (*SLEEVES, "combined"):
            s = self.store.ledger.state(sleeve, _today(), prices)
            out[sleeve] = {"positions": {t: {"quantity": q, "value": s.values.get(t), "name": describe(t)}
                                         for t, q in sorted(s.positions.items())},
                           "cash": {**asdict(s.cash), "available": s.cash.available}, "equity": s.equity,
                           "unpriced": sorted(t for t in s.positions if t not in prices)}
        events = self.store.ledger.events()[-50:]
        return {"as_of": _today(), "sleeves": out, "events": events[::-1],
                "settings": {s: self.store.settings(s) for s in SLEEVES}}

    def after_ledger_change(self):
        """Keep the old holdings file in step for code that still reads it."""
        self.store.export_holdings(self.portfolio_path, _today())

    # --- AI, on explicit request only ---

    def start_ai(self, analysts=None) -> bool:
        g = self.store.settings("global")
        analysts = analysts or g["ai_analysts"]
        context, _ = self.store.portfolio("combined", _today())
        cash = OptimizerSettings().cash_ticker
        # Holdings and what the last plan wanted to buy, capped at the shortlist size:
        # rating the whole scan list would cost one full analysis per fund.
        tickers = [t for t in dict.fromkeys([*(p.ticker.upper() for p in context.positions),
                                             *self.last_targets]) if t != cash][:int(g["ai_shortlist"])]

        config = self.agent_config(g)

        def work(job):
            ratings, failures = collect_ratings(tickers, config, analysts, gateway=self.gateway(),
                                                allow_paid=True, est_usd=g["ai_estimates"]["rating"],
                                                progress=lambda t, s: job.log.append(f"{t}: {s}"))
            self.ratings, self.ratings_date = ratings, _today()
            job.log += [f"{t} failed: {why}" for t, why in failures]
            return {"ratings": ratings, "date": self.ratings_date, "failures": len(failures)}

        return self.ai.start(work)

    def agent_config(self, g: dict | None = None) -> dict:
        """The config agent runs use: the default provider and models chosen under Settings (else the
        app's own), and the agent roles moved to other providers."""
        g = g or self.store.settings("global")
        primary = g.get("ai_primary") or {}
        config = {**self.config, "agent_models": g["ai_agent_models"]}
        # Free tiers answer "slow down" (429) under load; more retries ride that out instead of failing the run.
        config["llm_max_retries"] = max(int(config.get("llm_max_retries") or 0), 6)
        if primary.get("provider"):
            same = primary["provider"] == self.config["llm_provider"]
            config.update(llm_provider=primary["provider"], quick_think_llm=primary["quick"],
                          deep_think_llm=primary["deep"], backend_url=self.config.get("backend_url") if same else None)
        return config

    def agent_attempts(self, g: dict) -> list[dict]:
        """The configs an agent run tries in order: the configured one (Gemini Flash), then the free OpenRouter
        models, then the other free Gemini models, each only when the spending policy lists that provider as
        a fallback and its key is set. OpenRouter comes before the other Gemini models because those share the
        Google key's quota and outages: when Flash fails, they usually fail with it."""
        config = self.agent_config(g)
        attempts = [config]
        if "openrouter" in g["ai_fallbacks"] and os.environ.get("OPENROUTER_API_KEY"):
            for model in g["ai_openrouter_models"]:
                if (config["llm_provider"], config["deep_think_llm"]) != ("openrouter", model):
                    attempts.append({**config, "llm_provider": "openrouter", "quick_think_llm": model,
                                     "deep_think_llm": model, "backend_url": None, "agent_models": {}})
        # Roles moved to another provider (e.g. the debates on Groq) keep their seats when Gemini takes
        # over; if that attempt fails too, the next one runs every role on Gemini.
        elsewhere = {r: v for r, v in (config.get("agent_models") or {}).items() if v.get("provider") != "google"}
        if "google" in g["ai_fallbacks"] and os.environ.get("GOOGLE_API_KEY"):
            for model in GEMINI_FALLBACKS:
                if (config["llm_provider"], config["deep_think_llm"]) == ("google", model):
                    continue
                for moves in ([elsewhere, {}] if elsewhere else [{}]):
                    attempts.append({**config, "llm_provider": "google", "quick_think_llm": model,
                                     "deep_think_llm": model, "backend_url": None, "agent_models": moves})
        return attempts

    def ai_usage(self) -> dict:
        rows = [dict(r) for r in self.store.con.execute("SELECT * FROM ai_usage ORDER BY id DESC LIMIT 50")]
        from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
        from tradingagents.llm_clients.roles import ROLES, first_model, resolve, suggest

        g = self.store.settings("global")
        config = self.agent_config(g)
        # Which providers have a key in the environment (never the key itself).
        keys = sorted(p for p, env in PROVIDER_API_KEY_ENV.items() if env and os.environ.get(env))
        return {"remaining": self.gateway().remaining(), "recent": rows, "settings": g,
                "roles": [{"role": r, "label": label, "stage": stage, "tier": tier,
                           "provider": resolve(config)[r][0], "model": resolve(config)[r][1],
                           "moved": r in g["ai_agent_models"]} for r, (label, stage, tier) in ROLES.items()],
                "default": {"provider": config["llm_provider"], "quick": config["quick_think_llm"],
                            "deep": config["deep_think_llm"]},
                "app_default": {"provider": self.config["llm_provider"], "quick": self.config["quick_think_llm"],
                                "deep": self.config["deep_think_llm"]},
                "fallbacks": [f"{c['llm_provider']}/{c['deep_think_llm']}" for c in self.agent_attempts(g)[1:]
                              if not c.get("agent_models")],
                "providers": sorted(PROVIDER_API_KEY_ENV), "providers_with_keys": keys,
                "catalog": {p: {"quick": first_model(p, "quick"), "deep": first_model(p, "deep")} for p in keys},
                "suggested": suggest(config["llm_provider"], keys)}

    def context_config(self) -> dict:
        """The config for the app's non-agent AI work (market brief, ticker news, moat): Gemini on its own key
        (``GEMINI_CONTEXT_API_KEY``) when that key is set, else the usual config. Agent runs never use it."""
        from tradingagents.webapp.keys import CONTEXT_KEY

        key = os.environ.get(CONTEXT_KEY)
        if not key:
            return self.config
        model = GEMINI_FALLBACKS[0]
        return {**self.config, "llm_provider": "google", "quick_think_llm": model, "deep_think_llm": model,
                "backend_url": None, "context_google_api_key": key}

    def start_explain(self) -> bool:
        g = self.store.settings("global")

        def work(job):
            job.progress, job.fraction = "reading prices", 0.05
            overview = self.market()["overview"]
            job.progress, job.fraction = "reading headlines and economic data, then asking the AI", 0.35
            cfg = self.context_config()
            brief = market_brief(overview, cfg, gateway=self.gateway(), allow_paid=True,
                                 fallbacks=() if cfg is not self.config else tuple(g["ai_fallbacks"]))
            job.progress, job.fraction = "saving the brief", 0.95
            save_json(Path(self.config["results_dir"]) / "market" / f"brief_{overview['as_of']}.json", brief)
            return {"as_of": overview["as_of"]}

        return self.explain.start(work)

    # --- forecasts (local numerical work; no paid model is called) ---

    def start_forecast(self, body: dict) -> bool:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tradingagents.research.engine import UNIVERSES, compare_runs, factory, run_forecast

        horizons = [int(h) for h in body.get("horizons") or [body.get("horizon") or 20]]
        if not horizons or any(h not in (5, 20, 60) for h in horizons):
            raise ValueError("the forecast horizon is 5, 20 or 60 sessions")
        models = list(dict.fromkeys(body.get("models") or [body.get("model") or "elastic_net"]))
        for m in models:
            factory(m)   # an unknown name fails here, before anything starts
        from tradingagents.stocks.model_zoo import timesfm_allowed

        if "timesfm3" in models and not timesfm_allowed(self.store):
            raise ValueError("TimesFM 3.0 is research only: accept its non-commercial license in Settings first")
        universe = body.get("universe") or "etf"
        if universe == "etf":
            context, _ = self.store.portfolio("combined", _today())
            held = [p.ticker.upper() for p in context.positions]
            tickers = [t for t in dict.fromkeys([*etf_list(read_config(self.universe_path)), *held])
                       if t not in ("SGOV", "BIL")]
        elif universe == "stock":
            tickers = sorted(get_sp500_members(self.cache_dir))
            if not tickers:
                raise ValueError("the S&P 500 member list could not be downloaded, and no copy is saved")
        else:
            raise ValueError("the forecast universe is etf or stock")
        device = self.store.get("global", "compute_backend")

        combos = [(m, h) for h in horizons for m in models]

        def work(job):
            # Several at once: each run is its own thread with its own database connections.
            # Three at a time, since every boosted-tree fit already uses all the cores.
            done, runs, failed = 0, [], []
            job.progress = f"training {len(combos)} model{'s' * (len(combos) > 1)} on {UNIVERSES[universe]}"
            job.fraction = 0.02
            with ThreadPoolExecutor(max_workers=min(3, len(combos))) as pool:
                futures = {pool.submit(run_forecast, self.store.con, self.cache_dir, tickers, horizon=h, model=m,
                                       universe=universe, device_setting=device): (m, h) for m, h in combos}
                for fut in as_completed(futures):
                    m, h = futures[fut]
                    done += 1
                    job.fraction = max(job.fraction or 0.0, done / len(combos))
                    try:
                        runs.append(fut.result())
                    except Exception as exc:  # one failing model leaves the others' results
                        logger.exception("forecast %s at %s sessions failed", m, h)
                        failed.append({"model": m, "horizon": h, "error": str(exc)})
                    job.progress = f"{done} of {len(combos)} done on {UNIVERSES[universe]} (last: {m}, {h} sessions)"
            return {"universe": universe, "batch": compare_runs(runs), "failed": failed}

        return self.forecast.start(work)

    # --- every morning: the forecasts, every model's scores and the daily decision for the latest close ---

    def _job_or_morning(self, job: Job, prefix: str) -> dict:
        """A tab's own job, or the morning run while it is doing that tab's work (its progress starts with ``prefix``)."""
        own = job.state()
        if own["running"] or not self.morning.running or not str(self.morning.progress).startswith(prefix):
            return own
        return {**self.morning.state(), "via": "morning"}

    def schedule_state(self) -> dict:
        from tradingagents.ledger.sessions import next_session

        g = self.store.settings("global")
        sch, last = g.get("schedule") or {}, g.get("schedule_state") or {}
        cutoff = local_store.last_final_session()
        ens = self._ensemble_age()
        return {"settings": sch, "timesfm3": g.get("timesfm3") or {}, "last": last, "latest_close": cutoff.isoformat(),
                "done_for_latest_close": last.get("cutoff") == cutoff.isoformat() and not last.get("error"),
                "next_session": next_session(cutoff).isoformat(), "ensemble": ens, "job": self.morning.state()}

    def _ensemble_age(self) -> dict:
        from tradingagents.ledger.sessions import sessions_between
        from tradingagents.stocks import model_zoo

        b = model_zoo.ensemble_bundle(self.cache_dir)
        if not b:
            return {"trained": False}
        through = b.get("trained_through")
        age = sessions_between(through, local_store.last_final_session()) if through else None
        return {"trained": True, "trained_through": through, "created": b.get("created"), "age_sessions": age}

    def morning_due(self, now: datetime | None = None) -> bool:
        """After the set time, once per new close, with nothing else fitting models."""
        g = self.store.settings("global")
        sch = g.get("schedule") or {}
        if not sch.get("enabled"):
            return False
        now = now or datetime.now()
        hh, mm = (int(x) for x in str(sch.get("morning_at") or "07:30").split(":"))
        if (now.hour, now.minute) < (hh, mm):
            return False
        cutoff = local_store.last_final_session().isoformat()
        last = g.get("schedule_state") or {}
        if last.get("cutoff") == cutoff and (not last.get("error") or last.get("attempts", 0) >= 3):
            return False
        return not any(getattr(self, n).running for n in ("morning", "zoo", "decisions", "forecast", "daily"))

    def start_scheduler(self, every_seconds: int = 300) -> threading.Thread:
        """A background check every few minutes; the morning run starts when due (only while the app runs)."""
        def loop():
            while True:
                try:
                    if self.morning_due():
                        self.start_morning()
                except Exception:  # a failed check never stops the next one
                    logger.exception("morning schedule check failed")
                time.sleep(every_seconds)

        thread = threading.Thread(target=loop, daemon=True, name="morning-schedule")
        thread.start()
        return thread

    def start_morning(self, body: dict | None = None) -> bool:
        """Forecasts, every model's scores and the daily decision for the latest close. Models are refitted only when
        their last fit is ``retrain_every_sessions`` old, and the ensemble's weights then too; the same inputs as an
        earlier run are reused, so a second run the same morning costs nothing."""
        import subprocess
        import sys

        from tradingagents.research.engine import CHRONOS, run_forecast
        from tradingagents.stocks import daily, model_zoo

        g = self.store.settings("global")
        every = int((g.get("schedule") or {}).get("retrain_every_sessions") or 20)
        on = {**MORNING_STEPS, **((g.get("schedule") or {}).get("steps") or {})}   # Settings: what the morning makes
        refit_ensemble = bool((body or {}).get("refit_ensemble"))
        cutoff = local_store.last_final_session().isoformat()

        def work(job):
            steps, errors = [], []

            def step(name, frac, fn):
                job.progress, job.fraction = name, max(job.fraction or 0.0, frac)
                try:
                    out = fn()
                    steps.append({"step": name, "ok": True, **({"reused": True} if isinstance(out, dict)
                                                              and out.get("reused") else {})})
                except Exception as exc:  # one step failing leaves the others
                    logger.exception("morning step failed: %s", name)
                    steps.append({"step": name, "ok": False, "error": _short_error(exc)})
                    errors.append(f"{name}: {_short_error(exc)}")

            device = self.store.get("global", "compute_backend")
            context, _ = self.store.portfolio("combined", _today())
            held = [p.ticker.upper() for p in context.positions]
            etfs = [t for t in dict.fromkeys([*etf_list(read_config(self.universe_path)), *held])
                    if t not in ("SGOV", "BIL")]
            model = (self.store.settings("trading") or {}).get("forecast_model") or "xgb_quantile"
            if on["events"]:
                def events():
                    from tradingagents.events.sources import sync

                    report = sync(self.cache_dir)
                    bad = [f"{r['source']} {r['status']}" for r in report if r["status"] != "ok"]
                    if bad:
                        raise RuntimeError("; ".join(bad))
                step("Release calendars and policy / geopolitical indices", 0.01, events)
            if on["funds"]:
                for i, h in enumerate((20, 60)):
                    step(f"Fund forecasts, {h} sessions (Chronos-2)", 0.02 + 0.04 * i,
                         lambda h=h: run_forecast(self.store.con, self.cache_dir, etfs, horizon=h, model=CHRONOS,
                                                  universe="etf", device_setting=device))
                step(f"Fund forecasts, 20 sessions ({model})", 0.10,
                     lambda: run_forecast(self.store.con, self.cache_dir, etfs, horizon=20, model=model, universe="etf",
                                          device_setting=device))
                from tradingagents.stocks.model_zoo import timesfm_allowed

                if timesfm_allowed(self.store):
                    for h in (20, 60):
                        step(f"Fund forecasts, {h} sessions (TimesFM 3.0, research only)", 0.12,
                             lambda h=h: run_forecast(self.store.con, self.cache_dir, etfs, horizon=h, model="timesfm3",
                                                      universe="etf", device_setting=device))
            ens = self._ensemble_age()
            if on["models"] and (refit_ensemble or (ens.get("trained") and (ens.get("age_sessions") or 0) >= every)):
                root = Path(__file__).resolve().parents[2]
                if (root / "experiments" / "stacking_study.py").is_file():
                    def stack():
                        proc = subprocess.Popen([sys.executable, "-m", "experiments.stacking_study"], cwd=root,
                                                env={**os.environ, "STACKING_SKIP_CHRONOS": "1"},
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        tail = []
                        for line in proc.stdout:          # "HH:MM:SS step ..." lines: shown as the run's progress
                            line = line.strip()
                            if line[:2].isdigit() and line[2:3] == ":":
                                job.progress = f"Ensemble weights: {line[9:][:120]}"
                                job.fraction = min((job.fraction or 0.15) + 0.01, 0.39)
                            tail = (tail + [line])[-5:]
                        if proc.wait():
                            raise RuntimeError((tail[-1] if tail else "the ensemble refresh failed")[:300])
                    step(f"Refitting the ensemble's weights (every {every} sessions)", 0.15, stack)

            def within(prefix, lo, hi):
                """A step's own progress line and fraction, placed in its share of the morning run's bar."""
                def say(text, frac=None):
                    if frac is not None:
                        job.fraction = max(job.fraction or 0.0, lo + (hi - lo) * min(max(float(frac), 0.0), 1.0))
                    job.progress = f"{prefix}: {text}"
                return say

            def zoo():
                return model_zoo.run(self.store, self.cache_dir, within("Every model", 0.40, 0.65))
            if on["models"]:
                step("Every model's scores (refitted when due)", 0.40, zoo)

            def decide():
                return daily.run(self.store, self.cache_dir, date.today(), "barrier_20", True,
                                 within("Daily decision", 0.65, 0.8))
            if on["decision"]:
                step("The daily decision (models refitted when due)", 0.65, decide)

            def signals():
                from tradingagents.stocks.data import gather
                from tradingagents.stocks.pipeline import BatchConfig, Progress, run_batch

                names, sources = gather(self.store, "", "", True, True)
                for t in get_sp500_members(self.cache_dir):
                    sources.setdefault(t, "sp500")
                say = within("Screener signals", 0.8, 0.93)
                return run_batch(self.store.con, self.cache_dir, sorted(sources), sources,
                                 BatchConfig(horizons=(5, 20, 60), news=False, device=device), Progress(say))
            if on["signals"]:
                step("Screener signals, 5/20/60 sessions (models refitted when due)", 0.8, signals)

            def sector():
                from tradingagents.research import sector_model

                return sector_model.run(self.cache_dir, self.store, within("Sector forecast", 0.93, 1.0))
            if on["sector"]:
                step("Sector forecast", 0.93, sector)
            job.fraction = 1.0
            last = self.store.get("global", "schedule_state") or {}
            state = {"cutoff": cutoff, "finished_at": datetime.now().isoformat(timespec="minutes"), "steps": steps,
                     "error": "; ".join(errors) or None,
                     "attempts": (last.get("attempts", 0) + 1) if last.get("cutoff") == cutoff else 1}
            with self.lock:
                self.store.set("global", "schedule_state", state, source="schedule")
            if errors:
                raise RuntimeError("; ".join(errors))
            return state

        return self.morning.start(work)

    def start_daily_record(self) -> bool:
        """Chronos-2's forecasts for the latest close, at 20 and 60 sessions, once per trading day.

        Chronos-2 may have seen past prices in pretraining, so its walk-forward scores could be
        flattered; forecasts saved from now on are scored only on prices that came after them.
        Skipped quietly when Chronos-2 is not installed.
        """
        from tradingagents.research.engine import CHRONOS, run_forecast

        try:
            import chronos  # noqa: F401
        except ImportError:
            return False
        cutoff = local_store.last_final_session().isoformat()
        con = self.store.con
        due = [h for h in (20, 60) if not con.execute(
            "SELECT 1 FROM forecast_runs WHERE model = ? AND horizon = ? AND universe = 'etf' AND info_cutoff >= ?",
            (CHRONOS, h, cutoff)).fetchone()]
        if not due:
            return False
        context, _ = self.store.portfolio("combined", _today())
        held = [p.ticker.upper() for p in context.positions]
        tickers = [t for t in dict.fromkeys([*etf_list(read_config(self.universe_path)), *held])
                   if t not in ("SGOV", "BIL")]

        def work(job):
            done = []
            for i, h in enumerate(due):
                job.progress = f"recording Chronos-2's {h}-session forecasts"
                job.fraction = i / len(due)
                done.append(run_forecast(con, self.cache_dir, tickers, horizon=h, model=CHRONOS, universe="etf",
                                         device_setting=self.store.get("global", "compute_backend"))["run_id"])
            return {"runs": done}

        return self.daily.start(work)

    def combined_forecasts(self, horizon: int, universe: str = "etf") -> dict:
        """Each model's latest forecasts at ``horizon`` side by side, with their combination."""
        from tradingagents.research.engine import combine_runs, runs_to_combine

        runs = runs_to_combine(self.store.con, horizon, universe)
        out = combine_runs(runs) if runs else {"basis": None, "rows": [], "weights": []}
        for r in out["rows"]:
            r["name"] = describe(r["ticker"])
        return {**out, "horizon": horizon, "universe": universe,
                "info_cutoff": runs[0]["info_cutoff"] if runs else None}

    # --- stock research (API only: no page yet) ---

    def start_stocks(self, body: dict) -> bool:
        """A stock batch in the background: tickers pasted, from a CSV's text, the watchlist, holdings."""
        from tradingagents.stocks.data import gather
        from tradingagents.stocks.pipeline import BatchConfig, Progress, run_batch

        names, sources = gather(self.store, body.get("tickers", ""), body.get("csv", ""),
                                bool(body.get("watchlist")), bool(body.get("holdings")))
        if body.get("sp500"):   # every current member, as the screener's "compute signals for all" asks
            for t in get_sp500_members(self.cache_dir):
                sources.setdefault(t, "sp500")
            names = sorted(sources)
        if not names:
            raise ValueError("no tickers: paste some, send a CSV, or use the watchlist or holdings")
        horizons = tuple(int(h) for h in body.get("horizons") or (5, 20, 60))
        config = BatchConfig(horizons=horizons, news=body.get("news", True),
                             device=self.store.get("global", "compute_backend"))

        def work(job):
            stages = {"economic": 0.05, "building": 0.15, "fitting": 0.30, "collecting news": 0.50}

            def stage(text):
                at = next((v for k, v in stages.items() if str(text).startswith(k)), None)
                if at is not None:
                    job.fraction = max(job.fraction or 0.0, at)
                _counted(job, text, 0.0, 0.0)

            class Counted(Progress):
                def ticker(self, t, status, note=None):
                    super().ticker(t, status, note)
                    scored = sum(1 for x in self.tickers.values() if x["status"] in ("done", "abstained"))
                    job.fraction = max(job.fraction or 0.0, 0.55 + 0.45 * scored / max(len(names), 1))

            progress = Counted(stage)
            job.log = progress.tickers       # per-stock status, read while it runs
            out = run_batch(self.store.con, self.cache_dir, names, sources, config, progress, force=bool(body.get("force")))
            job.fraction = 1.0
            return out

        return self.stocks.start(work)

    def stock_run(self, run_id: str | None) -> dict:
        from tradingagents.stocks.pipeline import load_run

        runs = [dict(r) for r in self.store.con.execute(
            "SELECT id, created_at, info_cutoff, requested_json, seconds FROM stock_runs ORDER BY created_at DESC LIMIT 30")]
        for r in runs:
            r["requested"] = json.loads(r.pop("requested_json"))["tickers"]
        return {"run": load_run(self.store.con, run_id), "runs": runs, "job": self.stocks.state()}

    def stock_outcomes(self) -> dict:
        from tradingagents.stocks.ideas import prospective
        from tradingagents.stocks.outcomes import resolve, scorecard

        recorded = resolve(self.store.con, self.cache_dir)
        return {"recorded_now": recorded, "scorecard": scorecard(self.store.con),
                "prospective": prospective(self.store.con, self.cache_dir)}

    def stock_news_ai(self, body: dict) -> dict:
        """The paid AI reading of a shortlist's news (asked for explicitly; budgeted; cached)."""
        from datetime import UTC, datetime

        from tradingagents.stocks.llm import read_news
        from tradingagents.stocks.news import summary
        from tradingagents.stocks.pipeline import load_run

        run = load_run(self.store.con, body.get("run"))
        if run is None:
            raise ValueError("no stock run yet")
        limit = int(self.store.get("global", "stock_rules")["llm_shortlist"])
        ranked = sorted({f["ticker"]: abs((f.get("forecast") or {}).get("net") or 0) for f in run["forecasts"]
                         if f["horizon"] == 20}.items(), key=lambda kv: -kv[1])
        wanted = [t.upper() for t in body.get("tickers") or []] or [t for t, _ in ranked]
        g = self.store.settings("global")
        out = {}
        for t in wanted[:limit]:
            f = next((x for x in run["forecasts"] if x["ticker"] == t and x["horizon"] == 20), None)
            if not f or not f.get("forecast"):
                continue
            fc = f["forecast"]
            news = summary(self.store.con, t, datetime.now(UTC), days=14)
            numbers = (f"20-session expected return above cash {fc['mean']:+.3%} (net of cost {fc['net']:+.3%}), "
                       f"5-95% {fc['q05']:+.2%} to {fc['q95']:+.2%}, P(large loss) {fc['p_large_loss']:.0%}")
            out[t] = read_news(t, f.get("name") or t, run["info_cutoff"], news["items"], numbers, self.context_config(),
                               gateway=self.gateway(), allow_paid=True, fallbacks=tuple(g["ai_fallbacks"]))
        return {"shortlist": wanted[:limit], "readings": out}

    # --- capital: the three amounts, against what the books actually hold ---

    def signals(self, ticker: str | None = None, decision_id: str | None = None) -> dict:
        """Saved trend, breakout and crash signals with the daily decision's actions, and each stock's
        verdict (BUY / WATCH / HOLD / SELL / AVOID): reads only."""
        from tradingagents.memory import TradingMemoryLog
        from tradingagents.stocks import daily, screen, signals
        from tradingagents.stocks.verdict import verdict

        decision = daily.load(self.store.con, decision_id)
        scores = (decision or {}).get("scores") or {}
        horizon = (decision or {}).get("horizon") or 20
        out = signals.board(self.store.con, decision, 20, ticker)
        held = {t for t, q in self.store.ledger.state("combined", _today()).positions.items() if q}
        latest = screen.latest(Path(self.config["results_dir"])) or {}
        rows = {r["ticker"]: r for r in latest.get("rows", [])}
        from tradingagents.stocks.verdict import rating_on_record

        ratings = {}
        for e in TradingMemoryLog(self.config).load_entries():   # oldest first: the newest wins
            if e.get("rating"):
                ratings[e["ticker"].upper()] = rating_on_record(e)
        funds = self._fund_scores(horizon)
        if ticker and not out["rows"]:
            out["rows"] = [{"ticker": ticker.upper(), "signal": None, "decision": None}]
        if not ticker:   # held funds and stocks get a row too, so every holding has a verdict
            present = {r["ticker"] for r in out["rows"]}
            out["rows"] += [{"ticker": t, "signal": None, "decision": None}
                            for t in sorted(held - present) if t not in daily.CASH_FUNDS]
        # The 5-session trend forecast beside the 20-session one, for every row (the signals batch fits both).
        five = signals.latest(self.store.con, 5, ticker=ticker)
        for r in out["rows"]:
            t = r["ticker"]
            s5 = five.get(t)
            r["trend5"] = ({"horizon": 5, "trend": s5.get("trend"), "trend_p": s5.get("trend_p"), "data_to": s5.get("data_to"),
                            **{k: (s5.get("forecast") or {}).get(f"trend_{k}") for k in ("up", "down", "sideways", "unclear")}}
                           if s5 and s5.get("status") == "ok" else None)
            row = rows.get(t) or (self._screen_row(t) if ticker else None)
            r["screen"] = row and {k: row.get(k) for k in ("quality_score", "quality_rank", "analyst_mean",
                                                            "analyst_label", "target_upside", "rank")}
            r["rating"] = ratings.get(t)
            # The stock model's score when it has one, else the fund models' rank among funds; ``measures`` says
            # what its expected_net and p_win are, which differ between the two (model_outputs.measures).
            score = scores.get(t) or funds.get(t)
            if score:
                score = {**score, "measures": measures(score.get("kind") or "stock")}
            # Nothing on record (a cash fund, or a ticker no model covers): no verdict rather than a guess.
            known = (row or r["signal"] or r["decision"] or t in held or score) and t not in daily.CASH_FUNDS
            r["verdict"] = verdict(t in held, r["decision"], r["signal"], row, ratings.get(t), score=score,
                                   horizon=horizon, validated=(decision or {}).get("mode") == "validated")                 if known else None
            if ticker and r["signal"] and r["signal"].get("status") == "ok":
                r["entry"] = self._entry_plan(t, r["signal"])
            if ticker:
                r["policy"] = self._policy_row(decision, t)

        return out

    @staticmethod
    def _policy_row(decision: dict | None, ticker: str) -> dict:
        """The policy model's view of one stock from the saved decision: options, action, status; or why none."""
        pv = (decision or {}).get("policy") or {}
        if not pv.get("available"):
            return {"available": False, "note": pv.get("note") or "no daily decision with the policy model yet"}
        row = (pv.get("stocks") or {}).get(ticker)
        head = {k: pv.get(k) for k in ("model_version", "trained_to", "status", "margin", "policies")}
        head.update({"decided_at": decision.get("created_at"), "data_to": decision.get("info_cutoff"),
                     "executes": decision.get("execution_date"), "horizon": decision.get("horizon")})
        return {"available": row is not None, **head, **(row or {"note": "not scored: not an index member or no data"})}

    def _fund_scores(self, horizon: int) -> dict:
        """Every fund the fund models forecast, ranked among the funds by the combined forecast at
        ``horizon`` sessions (return above cash, chance of a gain): read from saved runs, kept a few minutes."""
        import time

        key = (horizon, time.time() // 300)
        if getattr(self, "_funds_cache", (None,))[0] == key:
            return self._funds_cache[1]
        rows = [r for r in self.combined_forecasts(horizon, "etf")["rows"] if r.get("combined")]
        rows.sort(key=lambda r: (-r["combined"]["mean"], -r["combined"]["p_gain"], r["ticker"]))
        n = len(rows)
        out = {r["ticker"]: {"kind": "fund", "rank": i + 1, "of": n, "rank_pct": round(1 - i / max(n - 1, 1), 4),
                             # expected_net/p_win keep the verdict's field names; for a fund they are the forecast's
                             # log return over cash and P(total return > 0): see model_outputs.measures("fund").
                             "expected_net": round(r["combined"]["mean"], 5), "p_win": round(r["combined"]["p_gain"], 4),
                             "expected_log_excess_vs_cash": round(r["combined"]["mean"], 5),
                             "p_total_positive": round(r["combined"]["p_gain"], 4),
                             "p_risk": None, "cost_rt": None, "models": r["combined"].get("models")}
               for i, r in enumerate(rows)}
        self._funds_cache = (key, out)
        return out

    def _entry_plan(self, symbol: str, sig: dict) -> dict | None:
        """The entry plan from the saved forecast and the close it was made at (cached prices; no fetch)."""
        import pandas as pd

        from tradingagents.research.prices import bars
        from tradingagents.stocks.signals import entry_plan

        try:
            close = bars([symbol], self.cache_dir)["Close"][symbol].dropna()
            close = close[close.index <= pd.Timestamp(sig["data_to"])]
            return entry_plan(sig["forecast"], float(close.iloc[-1]), sig.get("horizon", 20)) if len(close) else None
        except Exception as exc:  # the card then shows no plan rather than a wrong one
            logger.info("no entry plan for %s: %s", symbol, exc)
            return None

    def _screen_row(self, symbol: str) -> dict | None:
        """The screen's row for a stock outside the saved screen, from the cached data its page already read."""
        from tradingagents.dataflows.vendors.yahoo import history as yh
        from tradingagents.research.prices import bars
        from tradingagents.stocks import screen

        try:
            profile = yh.profile(symbol, self.cache_dir) or {}
            if profile.get("quoteType") not in (None, "EQUITY"):
                return None
            closes = bars([symbol], self.cache_dir)["Close"]
            return screen.score_row(symbol, profile, yh.financials(symbol, self.cache_dir),
                                    closes[symbol] if symbol in closes else None,
                                    yh.actions(symbol, self.cache_dir).get("dividends") or [],
                                    yh.current_analysts(symbol, self.cache_dir))
        except Exception as exc:  # the verdict then says what it lacked
            logger.info("no screen row for %s: %s", symbol, exc)
            return None

    def analysis_state(self) -> dict:
        return self.analysis.state()

    def _check_day(self, day: str) -> str:
        try:
            when = date.fromisoformat(day)
        except ValueError:
            raise ValueError("the date must look like 2026-10-23") from None
        if when > date.today():
            raise ValueError("the agents can only analyze a day that has happened")
        if when.weekday() >= 5:
            raise ValueError(f"{day} is a weekend: pick a trading day")
        return day

    def _analysts(self, body: dict, g: dict) -> list[str]:
        from tradingagents.optimizer.ai_views import ANALYST_ROLES

        analysts = [a for a in (body.get("analysts") or g["ai_analysts"]) if a in ANALYST_ROLES]
        if not analysts:
            raise ValueError("pick at least one analyst")
        return analysts

    def _run_agents(self, job, ticker: str, day: str, analysts: list[str], g: dict, force: bool = False) -> dict:
        """The agents on one ticker as of ``day``, trying each configured provider in turn (paid: only from a click).
        A run with the same inputs that already succeeded is not repeated unless ``force``."""
        from tradingagents.dataflows.symbols import safe_ticker_component
        from tradingagents.ledger.ai_gateway import AIBlocked, AIInProgress
        from tradingagents.ledger.db import now_utc

        folder = Path(self.config["results_dir"]) / safe_ticker_component(ticker) / day / "reports"
        # Reserved before the run (the actual spend is recorded per request): the first analyst costs the
        # most, each further one adds its report and a longer debate.
        est = g["ai_estimates"]["rating"] * (1 + 0.5 * (len(analysts) - 1))
        errors = []
        from tradingagents.stocks import daily
        from tradingagents.stocks.decision_layer import agent_evidence

        evidence = agent_evidence(daily.load(self.store.con), ticker)   # empty unless the model is validated
        attempts = self.agent_attempts(g)
        job.attempts = []                  # what the page shows: each provider tried, and how it went
        record = {"ticker": ticker, "as_of": day, "analysts": analysts, "evidence": evidence or {},
                  "attempts": attempts, "started_at": now_utc()}
        if not force:
            today_run = self._agents_ran_today(ticker)
            if today_run:   # at most one successful run a day per stock: the saved one is the answer
                job.progress = (f"{ticker}: the agents already analyzed it today "
                                f"({today_run['finished_at'][11:16]} UTC, data to {today_run['as_of']}); not run again")
                return {"ticker": ticker, "rating": today_run["rating"], "how": "reused", "reused": True,
                        "provider": today_run["actual_provider"], "model": today_run["actual_model"],
                        "final": {"action": today_run["final_action"]}, "fell_back": bool(today_run["fell_back"]),
                        "once_a_day": True}
            done = self._same_agent_run(record)
            if done and (folder / "complete_report.md").is_file():
                job.progress = (f"{ticker} as of {day}: already analyzed with the same data, analysts, evidence and "
                                f"models ({done['finished_at'][:16].replace('T', ' ')} UTC); nothing was re-run")
                return {"ticker": ticker, "rating": done["rating"], "how": "reused", "reused": True,
                        "provider": done["actual_provider"], "model": done["actual_model"],
                        "final": {"action": done["final_action"]}, "fell_back": bool(done["fell_back"])}
        limit = int(g.get("ai_daily_scans") or 0)
        if limit and ticker.upper() not in self._scanned_today() and len(self._scanned_today()) >= limit:
            from tradingagents.ledger.ai_gateway import AIBlocked

            raise AIBlocked(f"the daily limit of {limit} stocks analysed is reached (Settings: ai_daily_scans); "
                            "it starts again tomorrow")
        try:
            out = self._try_agents(job, ticker, day, analysts, folder, est, evidence, attempts, errors)
        except AIInProgress:
            raise                          # another run of the same request owns it and records itself
        except AIBlocked as exc:
            self._record_agent_run(record, job, None, "blocked", str(exc))
            raise
        except Exception as exc:
            self._record_agent_run(record, job, None, "failed", str(exc))
            raise
        self._record_agent_run(record, job, out, "ok", None)
        return out

    @staticmethod
    def _agent_chain(attempts: list[dict]) -> list:
        return [(c["llm_provider"], c["deep_think_llm"], sorted((c.get("agent_models") or {}).items())) for c in attempts]

    def _scanned_today(self) -> set[str]:
        """The stocks the agents were started on today (local calendar day), whatever the outcome."""
        start = datetime.combine(date.today(), datetime.min.time()).astimezone().astimezone(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            return {r[0].upper() for r in self.store.con.execute(
                "SELECT DISTINCT ticker FROM agent_runs WHERE started_at >= ?", (start,))}
        except Exception:
            return set()

    def _agents_ran_today(self, ticker: str):
        """Today's successful agent run on ``ticker`` (local calendar day), if any: the agents run once a day."""
        start = datetime.combine(date.today(), datetime.min.time()).astimezone().astimezone(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            return self.store.con.execute(
                "SELECT * FROM agent_runs WHERE ticker = ? AND status = 'ok' AND finished_at >= ?"
                " ORDER BY finished_at DESC LIMIT 1", (ticker.upper(), start)).fetchone()
        except Exception:   # an older database without the table
            return None

    def _same_agent_run(self, record: dict):
        """The newest successful run with exactly these inputs: ticker, data date, analysts, model evidence and the
        provider/model chain. Same inputs give the same analysis, so it is not paid for or saved twice."""
        from tradingagents.ledger.ai_gateway import stable_hash

        try:
            return self.store.con.execute(
                "SELECT * FROM agent_runs WHERE ticker = ? AND as_of = ? AND status = 'ok' AND analysts_json = ?"
                " AND evidence_hash = ? AND route_version = ? ORDER BY finished_at DESC LIMIT 1",
                (record["ticker"], record["as_of"], json.dumps(list(record["analysts"])),
                 stable_hash(record["evidence"]), stable_hash(self._agent_chain(record["attempts"])))).fetchone()
        except Exception:   # an older database without the table
            return None

    def dismiss_agent_failures(self, ticker: str) -> dict:
        """Remove a ticker's failed-run messages from view: the analysis job's last error (when it was this ticker's
        and nothing is running) and its failed runs in ``agent_runs`` since its last good one. The records stay."""
        from tradingagents.ledger.db import now_utc, transaction

        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("no ticker")
        job = self.analysis
        cleared = False
        # The page shows the analysis job's last error whichever stock it was for (its progress line starts with that
        # stock), so Dismiss clears it whenever nothing is running, and hides that stock's failed runs as well.
        tickers = {ticker}
        with job.switch:
            if not job.running and job.error:
                first = str(job.progress).split(" ", 1)[0].strip().upper()
                if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", first):
                    tickers.add(first)
                job.error, job.attempts = None, []
                cleared = True
        rows = []
        for t in sorted(tickers):
            rows += self.store.con.execute(
                "SELECT run_id FROM agent_runs WHERE ticker = ? AND status != 'ok' AND finished_at >"
                " COALESCE((SELECT MAX(finished_at) FROM agent_runs WHERE ticker = ? AND status = 'ok'), '')",
                (t, t)).fetchall()
        with transaction(self.store.con):
            for r in rows:
                self.store.con.execute("INSERT OR IGNORE INTO hidden_outputs VALUES ('agent_run', ?, ?)",
                                       (r["run_id"], now_utc()))
        return {"ticker": ticker, "job_error_cleared": cleared, "runs_hidden": len(rows)}

    def delete_agent_report(self, ticker: str, day: str, run: str | None) -> dict:
        """Remove one saved analysis from the pages by moving its folder to ``<results>/_deleted`` (recoverable by
        moving it back). ``run`` None is the day's latest run; ``archive/<stamp>`` an earlier one."""
        import shutil

        from tradingagents.dataflows.symbols import safe_ticker_component

        ticker = safe_ticker_component(ticker.strip().upper())
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day or ""):
            raise ValueError("date must be YYYY-MM-DD")
        if run and not re.fullmatch(r"archive/\d{8}T\d{6}(_\d+)?", run):
            raise ValueError("unknown run")
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        moved = []
        for root in self.report_roots():
            src = (root / ticker / day / (run or "reports")).resolve()
            if not src.is_dir() or root not in src.parents:
                continue
            dest = root / "_deleted" / ticker / f"{day}__{(run or 'latest').replace('/', '_')}__{stamp}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved.append(str(dest))
        if not moved:
            raise ValueError(f"no saved analysis of {ticker} for {day}" + (f" ({run})" if run else ""))
        return {"ticker": ticker, "date": day, "run": run, "moved_to": moved}

    def _record_agent_run(self, record: dict, job, out: dict | None, status: str, error: str | None) -> None:
        """One row in ``agent_runs`` per routed run (append-only): requested and actual provider, every attempt and
        why it moved on, the stages' models, the evidence hash and the result or the error."""
        from tradingagents.ledger.ai_gateway import stable_hash
        from tradingagents.ledger.db import now_utc, transaction
        from tradingagents.ledger.recommendations import code_version
        from tradingagents.llm_clients import roles

        chain = self._agent_chain(record["attempts"])
        first = record["attempts"][0] if record["attempts"] else {"llm_provider": "", "deep_think_llm": ""}
        ran = next((c for c in record["attempts"] if out and c["llm_provider"] == out.get("provider")
                    and c["deep_think_llm"] == out.get("model")), None)
        try:
            stages = json.dumps({r: list(v) for r, v in roles.resolve(ran).items()}) if ran else None
        except Exception:  # the record must never fail the run
            stages = None
        row = (uuid.uuid4().hex, record["ticker"], record["as_of"], record["started_at"], now_utc(), status,
               stable_hash(chain), first["llm_provider"], first["deep_think_llm"],
               out.get("provider") if out else None, out.get("model") if out else None,
               int(bool(out and out.get("fell_back"))), json.dumps(list(getattr(job, "attempts", []) or [])),
               stages, json.dumps(list(record["analysts"])), stable_hash(record["evidence"]),
               out.get("how") if out else None, out.get("rating") if out else None,
               ((out.get("final") or {}).get("action") if out else None), (error or None) and error[:2000],
               code_version())
        try:
            with transaction(self.store.con):
                self.store.con.execute("INSERT INTO agent_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
                                       " ?, ?, ?, ?)", row)
        except Exception as exc:  # noqa: BLE001  (provenance is best effort; the run's own result stands)
            logger.warning("could not record the agent run: %s", exc)

    def _try_agents(self, job, ticker, day, analysts, folder, est, evidence, attempts, errors) -> dict:
        from tradingagents.ledger.ai_gateway import AIBlocked, AIInProgress
        from tradingagents.optimizer.ai_views import analyze

        from datetime import UTC

        cooling = dict(self.store.get("global", "ai_cooldowns") or {})
        now = datetime.now(UTC)
        cooling = {k: v for k, v in cooling.items() if datetime.fromisoformat(v["until"]) > now}

        from tradingagents.webapp.keys import ALT_KEYS

        def models(cfg):
            return list(dict.fromkeys(m for m in (cfg.get("deep_think_llm"), cfg.get("quick_think_llm")) if m))

        def keyed(cfg):
            """The cooldown name of the key an attempt uses: the provider, or ``provider#2`` for its second key."""
            return cfg["llm_provider"] + ("#2" if cfg.get("alt_api_key") else "")

        def cool(cfg):
            # A limit on the key, on its free models (when the run would use one), or on either model the agents
            # would call, stops the whole attempt.
            free = any(str(m).endswith(":free") for m in models(cfg))
            return cooling.get(keyed(cfg)) or (cooling.get(f"{keyed(cfg)}/*:free") if free else None) or next(
                (cooling[k] for k in (f"{keyed(cfg)}/{m}" for m in models(cfg)) if k in cooling), None)

        def with_second_keys(attempts):
            """Each attempt, then the same on the provider's second key (if set): tried only after a limit."""
            for cfg in attempts:
                yield cfg
                env = ALT_KEYS.get(cfg["llm_provider"])
                if env and os.environ.get(env) and not cfg.get("alt_api_key"):
                    yield {**cfg, "alt_api_key": os.environ[env]}

        attempts = list(with_second_keys(attempts))
        limited = False   # whether the last attempt stopped on a limit (only then is a second key worth trying)

        for n, cfg in enumerate(attempts, 1):
            if evidence:
                cfg = {**cfg, "quant_evidence": {ticker.upper(): evidence}}
            who = f"{cfg['llm_provider']}{' second key' if cfg.get('alt_api_key') else ''} ({cfg['deep_think_llm']})"
            if cfg.get("alt_api_key") and not limited:
                continue   # the first key worked or failed for another reason: the second key would not help
            hit = cool(cfg)
            if hit:   # it would fail the same way: not tried, and not counted against the budget
                until = datetime.fromisoformat(hit["until"]).astimezone().strftime("%H:%M")
                job.attempts.append({"provider": cfg["llm_provider"], "model": cfg["deep_think_llm"],
                                     "state": "skipped", "why": f"{hit['why']}; skipped until {until}"})
                errors.append(f"{who}: skipped, {hit['why']}")
                limited = True
                continue
            limited = False
            # One retry at most inside the provider's client: a limit error is not sent again and again.
            if cfg.get("llm_max_retries") in (None, ""):
                cfg = {**cfg, "llm_max_retries": 1}
            job.attempts.append({"provider": cfg["llm_provider"], "model": cfg["deep_think_llm"], "state": "trying"})
            job.progress = (f"{ticker} as of {day}: try {n} of {len(attempts)} on {who}"
                            f"{' after ' + str(n - 1) + ' failed' if n > 1 else ''}; a few minutes")
            try:
                tag = f"{ticker} · {cfg['llm_provider']}"
                result, how = analyze(ticker, day, cfg, analysts, self.gateway(), folder, est_usd=est, allow_paid=True,
                                      progress=lambda text, tag=tag: job.say(f"{tag} · {text}"))
            except AIInProgress:
                raise
            except AIBlocked as exc:
                # Switched off stops here. Over the dollar budget or a model with no price, the next provider
                # may still run (a free model holds no dollars).
                if str(exc).startswith("paid AI is switched off"):
                    job.attempts[-1].update(state="failed", why=str(exc)[:120])
                    raise
                errors.append(f"{who}: {exc}")
                job.attempts[-1].update(state="failed", why=str(exc)[:120])
            except Exception as exc:  # out of credit, bad key, rate limit, unreachable: the next provider, if any
                logger.warning("analysis on %s failed: %s", who, exc)
                errors.append(f"{who}: {_short_error(exc)}")
                job.attempts[-1].update(state="failed", why=_short_error(exc))
                held = _cooldown(keyed(cfg), cfg["deep_think_llm"], str(exc), datetime.now(UTC))
                if held:
                    limited = True
                    # Which of the run's models hit the limit is not known, so a model limit holds them all.
                    keys = [held[0]] if held[0] in (keyed(cfg), f"{keyed(cfg)}/*:free") else                         [f"{keyed(cfg)}/{m}" for m in models(cfg)]
                    for k in keys:
                        cooling[k] = {"until": held[1].isoformat(), "why": held[2]}
                    with self.lock:
                        self.store.set("global", "ai_cooldowns", cooling)
            else:
                job.attempts[-1]["state"] = "ok"
                job.log += [f"{ticker}: {e} -> moved on" for e in errors]
                out = {**result, "how": how, "ticker": ticker, "provider": cfg["llm_provider"],
                       "model": cfg["deep_think_llm"], "fell_back": bool(errors), "attempts": list(job.attempts)}
                out["final"] = self._write_decision(ticker, day, folder, out, cfg)
                plans = result.get("entry_plans") or {}
                if plans.get("trader") or plans.get("final"):
                    out["entry_plans"] = self._record_entry_plans(
                        ticker, day, folder, plans.get("trader"), plans.get("final"), result.get("entry_reference"),
                        source=who, evidence={"model_decision": out["final"].get("decision_id")})
                return out
            job.log.append(f"{ticker}: {errors[-1]}")
        # One short line per provider; the raw errors are in the log.
        raise ValueError("No AI provider could run the agents: " + "; ".join(dict.fromkeys(errors)))

    def manual_prompt(self, ticker: str, day: str) -> dict:
        """The TradingAgents team's own prompts for running the analysis in a chat app; calls no AI."""
        from tradingagents.agents.manual import PROMPT_VERSION, build_prompt
        from tradingagents.dataflows.config import set_config
        from tradingagents.dataflows.symbols import safe_ticker_component

        ticker, day = ticker.strip().upper(), self._check_day(day)
        set_config(self.config)
        prompt = build_prompt(ticker, day, self.cache_dir)
        # Kept under its content ID so a pasted answer can be tied to the exact prompt it replied to.
        pid = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        prompts = Path(self.config["results_dir"]) / safe_ticker_component(ticker) / day / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        if not (prompts / f"{pid}.txt").exists():
            (prompts / f"{pid}.txt").write_text(prompt, encoding="utf-8")
        try:
            import tiktoken

            tokens, how = len(tiktoken.get_encoding("o200k_base").encode(prompt)), "o200k_base tokenizer"
        except Exception:
            tokens, how = len(prompt) // 4, "estimate (characters / 4)"
        return {"ticker": ticker, "date": day, "prompt": prompt, "chars": len(prompt), "tokens_est": tokens,
                "tokens_how": how, "version": PROMPT_VERSION, "prompt_id": pid}

    def manual_answer(self, body: dict) -> dict:
        """A chat app's answer to the manual prompt: saved as a report, its rating recorded, the final decision written."""
        from tradingagents.agents.manual import decision_markdown, parse_answer
        from tradingagents.dataflows.symbols import safe_ticker_component
        from tradingagents.memory import TradingMemoryLog

        ticker = str(body.get("ticker") or "").strip().upper()
        day = self._check_day(str(body.get("date") or ""))
        text = str(body.get("text") or "")
        source = str(body.get("source") or "ChatGPT Plus (pasted)")[:60]
        if not ticker or len(text.strip()) < 40:
            raise ValueError("paste the whole answer: all twelve responses, each under its '## Response N' heading")
        from tradingagents.agents.manual import decision_from_steps, final_state

        # The per-step answer (prompt v2): the same final state an API run has, saved as the same report files.
        staged = final_state(text, ticker, day)
        decision = (decision_from_steps(staged[1]) if staged else None) or parse_answer(text)
        from tradingagents.agents.manual import PROMPT_VERSION
        from tradingagents.optimizer.ai_views import fresh_report_folder

        day_dir = Path(self.config["results_dir"]) / safe_ticker_component(ticker) / day
        # The prompt the answer replied to: the latest one copied for this ticker and day.
        prompts = day_dir / "prompts"
        saved = sorted(prompts.glob("*.txt"), key=lambda f: f.stat().st_mtime) if prompts.is_dir() else []
        pid = saved[-1].stem if saved else None
        folder = fresh_report_folder(day_dir / "reports")   # an earlier run is archived, never overwritten
        (folder / "5_portfolio").mkdir(parents=True, exist_ok=True)
        (folder / "chat_answer.md").write_text(
            f"# Answer from {source}\n\nPrompt format: {PROMPT_VERSION}; prompt {pid or 'unknown'}\n\n{text}\n",
            encoding="utf-8")
        if pid:   # the prompt the answer was given, kept with it
            (folder / "prompt.txt").write_text(saved[-1].read_text(encoding="utf-8"), encoding="utf-8")
        if staged and staged[0]["final_trade_decision"]:
            from tradingagents.reporting import write_report_tree

            write_report_tree(staged[0], ticker, folder)          # 1_analysts ... 5_portfolio, complete_report
            md = staged[0]["final_trade_decision"]
        else:
            md = decision_markdown(decision)
            (folder / "5_portfolio" / "decision.md").write_text(md, encoding="utf-8")
        log = TradingMemoryLog(self.config)
        before = {(e["date"], e["ticker"].upper()) for e in log.load_entries()}
        log.store_decision(ticker, day, md, decision["rating"])
        recorded = (day, ticker) not in before
        final = self._write_decision(ticker, day, folder, {"rating": decision["rating"], "fell_back": False,
                                                           "abstained": decision.get("insufficient_evidence")},
                                     None, models=source)
        trader = decision.get("trader") if isinstance(decision.get("trader"), dict) else None
        provenance = {"source": source, "prompt_version": PROMPT_VERSION, "prompt_id": pid,
                      "answer_form": "per-step (API form)" if staged else "single JSON block",
                      "steps_answered": staged[1]["steps"] if staged else None,
                      "schema_notes": staged[1]["notes"] if staged else None,
                      "rating": decision["rating"], "rating_recorded": recorded, "trader": trader,
                      "model_decision": final.get("decision_id"),
                      "saved_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        (folder / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
        return {"ticker": ticker, "date": day, "rating": decision["rating"], "final": final, "recorded": recorded,
                "prompt_id": pid, "trader": trader,
                "note": None if recorded else "a rating for this ticker and day was already on record (it stays; "
                                                 "the verdict uses that one)"}

    def _record_entry_plans(self, ticker: str, day: str, folder: Path, trader: dict | None, final: dict | None,
                            ref: dict | None, source: str, evidence: dict | None = None) -> dict:
        """Check a run's entry plans against the local quote and this account, and save them (append-only).
        A failure here never loses the run: the plans stay in the run's entry_plan.json."""
        from tradingagents.agents import entry_plan as EP

        try:
            ref = ref if ref and ref.get("session") else EP.reference_quote(ticker, day, self.cache_dir)
            account = EP.account_view(self.store, ticker, _today(), self.cache_dir)
            return EP.record(self.store, ticker, day, str(folder), source, ref, account, trader or None,
                             final or None, evidence)
        except Exception as exc:
            logger.warning("entry plans for %s on %s not recorded: %s", ticker, day, exc)
            return {"error": str(exc)[:200]}

    def entry_plans(self, ticker: str) -> dict:
        """Every recorded entry plan for a ticker, by run: current, superseded or expired."""
        from tradingagents.agents.entry_plan import plans_for

        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("which ticker?")
        return {"ticker": ticker, "runs": plans_for(self.store.con, ticker), "today": _today()}

    _INTRADAY: dict = {}

    def intraday(self, symbol: str, span: str) -> dict:
        """Intraday prices for the 1D (5-minute bars) and 5D (30-minute bars) chart, with SPY, from Yahoo; cached for
        5 minutes. Display only: no model reads them."""
        import time as _time

        import pandas as pd
        import yfinance as yf

        symbol = symbol.strip().upper()
        if span not in ("1d", "5d") or not re.fullmatch(r"[A-Z0-9.\-^]{1,12}", symbol):
            raise ValueError("span is 1d or 5d, with a ticker")
        key = (symbol, span)
        hit = self._INTRADAY.get(key)
        if hit and _time.time() - hit[0] < 300:
            return hit[1]
        interval = "5m" if span == "1d" else "30m"
        frames = {t: yf.Ticker(t).history(period=span, interval=interval)["Close"].dropna() for t in (symbol, "SPY")}
        close = frames[symbol]
        if close.empty:
            raise ValueError("Yahoo has no intraday prices for it now")
        spy = frames["SPY"].reindex(close.index).ffill()
        local = close.index.tz_convert("America/New_York") if close.index.tz is not None else close.index
        out = {"dates": [d.strftime("%Y-%m-%d %H:%M") for d in local], "close": [round(float(v), 4) for v in close],
               "spy": [None if pd.isna(v) else round(float(v), 4) for v in spy], "interval": interval}
        self._INTRADAY[key] = (_time.time(), out)
        return out

    def entry_alerts(self) -> dict:
        """The watched entry plans against the latest close. The price part of an entry condition is checkable: a range
        is met when the close is inside it, a limit when the close is at or below it, a market entry at once. The
        agents' free-text trigger is shown, not checked. Closes are daily, so this is checked after each close."""
        from tradingagents.agents.entry_plan import plans_for

        # Every stock with a recent plan; only a current, unexpired buy plan (Buy or Overweight) is watched.
        try:
            recent = [r[0].upper() for r in self.store.con.execute(
                "SELECT DISTINCT ticker FROM entry_plans WHERE created_at >= ?",
                ((date.today() - timedelta(days=30)).isoformat(),))]
        except Exception:   # an older database without the table
            recent = []
        current = {}
        for t in dict.fromkeys([*recent, *[str(x).upper() for x in (self.store.get("global", "entry_alerts") or [])]]):
            run = next((r for r in plans_for(self.store.con, t) if r.get("current")), None)   # {final, trader, current}
            row = (run or {}).get("final") or (run or {}).get("trader")
            if row and not row.get("expired") and str(row.get("action") or "").lower() in ("buy", "overweight"):
                current[t] = row
        watched = list(current)
        prices = latest_prices(self.cache_dir, watched) if watched else {}
        out = []
        for t in watched:
            run = current[t]
            plan = (run or {}).get("plan") or {}
            close = prices.get(t)
            lo, hi, px = plan.get("entry_range_low"), plan.get("entry_range_high"), plan.get("entry_price")
            kind = str(plan.get("entry_type") or "").lower()
            met, rule = None, "no entry price in the plan"
            if close is not None and plan:
                if lo is not None and hi is not None:
                    met, rule = float(lo) <= close <= float(hi), f"close inside {lo}–{hi}"
                elif px is not None and "limit" in kind:
                    met, rule = close <= float(px), f"close at or below {px}"
                elif px is not None:
                    met, rule = True, "market entry"
            out.append({"ticker": t, "met": met, "close": close, "rule": rule, "trigger": plan.get("trigger"),
                        "entry_price": px, "range": [lo, hi] if lo is not None else None,
                        "analysis_date": (run or {}).get("analysis_date"), "expires": plan.get("expires_at"),
                        "has_plan": bool(plan)})
        return {"watching": watched, "alerts": out, "checked": _today()}

    def _write_decision(self, ticker: str, day: str, folder: Path, result: dict, cfg: dict | None,
                        models: str | None = None) -> dict:
        """The run's final decision as one action, written first in its reports (0_FINAL_DECISION.md)."""
        from tradingagents.llm_clients import roles
        from tradingagents.stocks.daily import CASH_FUNDS
        from tradingagents.stocks.verdict import ai_decision

        held = bool(self.store.ledger.state("combined", _today()).positions.get(ticker)) and ticker not in CASH_FUNDS
        ai = ai_decision({"rating": result["rating"], "date": day, "abstained": result.get("abstained")}, held)
        # The model run as of the analysis day: the latest decision with data through that day, never a later one.
        found = self.store.con.execute("SELECT id, info_cutoff FROM stock_decisions WHERE info_cutoff <= ?"
                                       " ORDER BY info_cutoff DESC, created_at DESC LIMIT 1", (day,)).fetchone()
        board = self.signals(ticker, found["id"]) if found else {"rows": []}
        row = (board["rows"] or [{}])[0]
        rule = (row.get("verdict") or {}) if found else {}
        used = [models] if models else sorted({f"{p}/{m}" for p, m in roles.resolve(cfg).values()})
        pm = folder / "5_portfolio" / "decision.md"
        model = rule.get("model") or {}
        final = rule.get("action") or "WATCH"            # what the evidence gate allows
        view = rule.get("label") or final                # the model's view, qualified when not validated
        lines = [f"# FINAL DECISION: {final}", "",
                 f"**{ticker}**, as of {day}. Decided by the app's own {model.get('horizon', 20)}-session model, "
                 f"not by the agents: {rule.get('summary', 'no model run with data through this day')} "
                 f"({'you hold it' if held else 'you do not hold it'}).", "",
                 f"- **Model view**: {view}" + ("" if view == final else " (the evidence gate turns it into "
                                                                         f"{final}: the model is not validated)"),
                 f"- **Model run used**: decision {found['id'][:8]} with data through {found['info_cutoff']}"
                 if found else "- **Model run used**: none with data through this day"]
        if model:
            fund = model.get("kind") == "fund"
            m = measures("fund" if fund else "stock")
            lines += [f"- **Model rank**: #{model['rank']} of {model['of']} {'funds' if fund else 'S&P 500 stocks'} today",
                      f"- **{m['expected']['label']}** ({model.get('horizon', 20)} sessions): {model['expected_net']:+.2%}, "
                      f"{m['probability']['label']} {model['p_win']:.0%}"
                      + (f", P(large loss) {model['p_risk']:.0%}" if model.get("p_risk") is not None else "")]
        lines += [f"- **The agents' view** (shown, not used): rating {result['rating']}, i.e. {ai['action']}"
                  + (f"; {'agrees' if ai['action'] == final else 'differs'}" if model else ""),
                  f"- **Agent models**: {', '.join(used)}" + (" (after a fallback)" if result.get("fell_back") else ""),
                  "- The model is experimental: its ranking showed a small edge in testing, but trading on it did not "
                  "beat holding SPY. You place any order.", ""]
        if pm.is_file():
            lines += ["## The portfolio manager's decision", "", pm.read_text(encoding="utf-8")]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "0_FINAL_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"action": final, "view": view, "agents": ai["action"], "rating": result["rating"], "held": held,
                "model": model, "decision_id": found["id"] if found else None}

    def start_analysis(self, body: dict) -> bool:
        """The agents' full analysis of one ticker as of a chosen day (paid: asked for by a click)."""
        g = self.store.settings("global")
        ticker = str(body.get("ticker") or "").strip().upper()
        if not ticker:
            raise ValueError("no ticker")
        day = self._check_day(str(body.get("date") or _today()))
        analysts = self._analysts(body, g)
        force = bool(body.get("force"))
        return self.analysis.start(lambda job: self._run_agents(job, ticker, day, analysts, g, force))

    # --- AI decisions: the agents' BUY / HOLD / SELL on a shortlist, next to the rules and the economy ---

    def _last_session(self) -> str:
        from tradingagents.stocks import daily

        dec = daily.load(self.store.con)
        if dec and dec.get("info_cutoff"):
            return dec["info_cutoff"]
        d = date.today()
        while d.weekday() >= 5:
            d = date.fromordinal(d.toordinal() - 1)
        return d.isoformat()

    def ai_decisions(self) -> dict:
        """The shortlist the agents would analyze, each with its latest agent rating, its rule verdict and the
        AI decision, plus the latest economic brief: reads only."""
        from tradingagents.optimizer.universe import describe
        from tradingagents.stocks.daily import CASH_FUNDS
        from tradingagents.stocks.verdict import ai_decision

        g = self.store.settings("global")
        board = {r["ticker"]: r for r in self.signals()["rows"]}
        positions = self.store.ledger.state("combined", _today()).positions
        held = sorted(t for t, q in positions.items() if q and t not in CASH_FUNDS)
        trading = sorted(t for t, q in self.store.ledger.state("trading", _today()).positions.items()
                         if q and t not in CASH_FUNDS)
        # BUY verdicts in the model's own order (its rank decides the verdict).
        by_rank = sorted((r for r in board.values() if (r.get("verdict") or {}).get("model")),
                         key=lambda r: r["verdict"]["model"]["rank"])
        buys = [r["ticker"] for r in by_rank if r["verdict"]["verdict"] == "BUY"]   # the model's view (labelled)
        paper = [t for t, r in board.items() if (r.get("decision") or {}).get("paper_action") == "PAPER ENTER"]
        reports_root = Path(self.config["results_dir"])
        rows, seen = [], set()
        # Stocks the agents analyzed in the last 30 days stay listed whatever their other signals say.
        from tradingagents.memory import TradingMemoryLog
        from tradingagents.stocks.verdict import RATING_DAYS, rating_on_record

        cutoff = (date.today() - timedelta(days=RATING_DAYS)).isoformat()
        last_rated = {}
        for e in TradingMemoryLog(self.config).load_entries():   # oldest first: the newest wins
            if e.get("rating") and e.get("date", "") >= cutoff:
                last_rated[e["ticker"].upper()] = rating_on_record(e)
        rated = sorted(last_rated, key=lambda t: last_rated[t]["date"], reverse=True)
        for t, why in [*((t, "you hold it") for t in held), *((t, "analyzed by the agents") for t in rated),
                       *((t, "model verdict BUY") for t in buys), *((t, "top of the daily decision") for t in paper)]:
            if t in seen:
                continue
            seen.add(t)
            r = board.get(t) or {}
            rating = r.get("rating") or last_rated.get(t)   # stocks outside the signal board keep their rating
            day_dir = reports_root / t / rating["date"] / "reports" if rating else None
            rows.append({"ticker": t, "name": describe(t), "why": why, "held": t in positions and bool(positions[t]),
                         "in_trading": t in trading, "rating": rating,
                         "ai": ai_decision(rating, t in held) if rating else None,
                         "verdict": (r.get("verdict") or {}).get("action"),
                         "verdict_label": (r.get("verdict") or {}).get("label"),
                         "verdict_summary": (r.get("verdict") or {}).get("summary"),
                         "reports": bool(day_dir and day_dir.is_dir())})
        brief_dir = reports_root / "market"
        briefs = sorted(brief_dir.glob("brief_*.json")) if brief_dir.is_dir() else []
        brief = json.loads(briefs[-1].read_text(encoding="utf-8")) if briefs else None
        gw = self.gateway().remaining()
        return {"day": self._last_session(), "rows": rows, "default_pick": int(g["ai_shortlist"]),
                "analysts": g["ai_analysts"], "brief": brief and {k: brief.get(k) for k in ("as_of", "summary", "trends", "watch", "model")},
                "budget": {"daily_usd": g["ai_budget"]["daily_usd"], "monthly_usd": g["ai_budget"]["monthly_usd"],
                           "left_today": gw["daily_usd"], "left_month": gw["monthly_usd"],
                           "per_run_usd": g["ai_estimates"]["rating"]},
                "provider": {k: self.agent_config(g)[k] for k in ("llm_provider", "quick_think_llm", "deep_think_llm")},
                "job": self.analysis.state()}

    def start_ai_decisions(self, body: dict) -> bool:
        """The agents on each chosen ticker in turn, as of the last session (paid: asked for by a click).
        A ticker already analyzed that day with the same settings is reused at no charge."""
        from tradingagents.ledger.ai_gateway import AIBlocked

        g = self.store.settings("global")
        tickers = [str(t).strip().upper() for t in body.get("tickers") or [] if str(t).strip()]
        by_ai = bool(body.get("ai_screen"))      # the AI screens the market and names the tickers itself
        if not tickers and not by_ai:
            raise ValueError("pick at least one ticker")
        if len(tickers) > 20:
            raise ValueError("at most 20 tickers per run")
        day = self._check_day(str(body.get("date") or self._last_session()))
        analysts = self._analysts(body, g)
        if body.get("ai_pick"):      # "Ask the AI to pick": once a calendar day, unless some of it failed
            asked = g.get("ai_pick_state") or {}
            if asked.get("date") == _today() and asked.get("tickers") and \
                    all(self._agents_ran_today(t) for t in asked["tickers"]):
                raise ValueError(f"The AI already picked today ({', '.join(asked.get('tickers') or [])}); "
                                 "it can be asked again tomorrow")

        def work(job):
            done, failed = {}, {}
            if by_ai:
                tickers[:] = self._ai_screen(job, day, g)
            for i, t in enumerate(tickers, 1):
                job.log.append(f"{i}/{len(tickers)} {t}: started")
                try:
                    out = self._run_agents(job, t, day, analysts, g)
                except AIBlocked as exc:   # the budget refuses every further ticker too
                    failed[t] = str(exc)
                    job.log.append(f"{t}: stopped: {exc}")
                    break
                except Exception as exc:  # one ticker's failure stays with that ticker
                    failed[t] = str(exc)[:300]
                    job.log.append(f"{t}: failed: {str(exc)[:200]}")
                    continue
                done[t] = out["rating"]
                job.log.append(f"{t}: {out['rating']} ({'reused' if out['how'] == 'cache' else 'new'})")
            return {"ticker": None, "done": done, "failed": failed, "date": day}

        started = self.analysis.start(work)
        if started and body.get("ai_pick") and not by_ai:
            with self.lock:
                self.store.set("global", "ai_pick_state", {"date": _today(), "tickers": tickers,
                                                          "asked_at": datetime.now().isoformat(timespec="minutes")})
        return started

    AI_RUN = 5   # the full team analyses the AI screen's top five

    def _ai_screen(self, job, day: str, g: dict) -> list[str]:
        """The AI's own first screen of the Screener's ranked stocks, saved; returns its top ``AI_RUN`` to analyse."""
        from tradingagents.research.prices import bars
        from tradingagents.stocks import ai_screen

        rows = [r for r in ((self.screen_state()["latest"] or {}).get("rows") or []) if r.get("rank")]
        if not rows:
            raise ValueError("the Screener has no ranked stocks yet: build it first")
        job.progress, job.fraction = f"AI screen: the analysts rank {len(rows)} stocks from their fact sheet", 0.02
        try:
            close = bars([r["ticker"] for r in rows], self.cache_dir)["Close"]
        except Exception:   # no prices: the sheet says so per stock
            close = None
        from datetime import UTC

        now = datetime.now(UTC)
        cooling = {k: v for k, v in (self.store.get("global", "ai_cooldowns") or {}).items()
                   if datetime.fromisoformat(v["until"]) > now}
        errors, out = [], None
        for cfg in self.agent_attempts(g):
            prov, model = cfg["llm_provider"], cfg["quick_think_llm"]
            if prov in cooling or f"{prov}/{model}" in cooling or (model.endswith(":free") and f"{prov}/*:free" in cooling):
                continue                                   # paused: it would fail the same way
            try:
                out = ai_screen.screen(rows, close, day, cfg, self.gateway())
                break
            except Exception as exc:  # the next model in the agents' order
                errors.append(f"{prov}/{model}: {str(exc)[:160]}")
                held = _cooldown(prov, model, str(exc), now)
                if held:
                    cooling[held[0]] = {"until": held[1].isoformat(), "why": held[2]}
        if out is None:
            raise ValueError("No AI model could run the screen: " + ("; ".join(errors) or "every model is paused"))
        tickers = [p["ticker"] for p in out["picks"][:self.AI_RUN]]
        job.log.append(f"AI screen ({out['how']}): {', '.join(p['ticker'] for p in out['picks'])}")
        with self.lock:
            self.store.set("global", "ai_screen_state", {**out, "date": _today()})
            self.store.set("global", "ai_pick_state", {"date": _today(), "tickers": tickers,
                                                      "asked_at": datetime.now().isoformat(timespec="minutes")})
        return tickers

    def request_log(self, ticker: str, day: str, run: str | None = None) -> dict:
        """Every model request of a saved agent run and its reply (read from the run's folder); ``run`` names an
        earlier, archived run of that day."""
        from tradingagents.dataflows.symbols import safe_ticker_component
        from tradingagents.llm_clients.request_log import read

        self._check_day(day)
        if run and not re.fullmatch(r"archive/\d{8}T\d{6}(_\d+)?", run):
            raise ValueError("unknown run")
        folder = Path(self.config["results_dir"]) / safe_ticker_component(ticker.upper()) / day / (run or "reports")
        return {"ticker": ticker.upper(), "date": day, "run": run, "requests": read(folder)}

    def set_ai_budget(self, body: dict) -> dict:
        budget = dict(self.store.get("global", "ai_budget"))
        for k in ("daily_usd", "monthly_usd"):
            if k in body:
                v = float(body[k])
                if not 0 <= v <= 1000:
                    raise ValueError("a budget is between $0 and $1,000")
                budget[k] = v
        if budget["daily_usd"] > budget["monthly_usd"]:
            raise ValueError("the daily budget cannot exceed the monthly one")
        self.store.set("global", "ai_budget", budget)
        return budget

    def screen_state(self) -> dict:
        from tradingagents.stocks import screen

        return {"latest": screen.latest(Path(self.config["results_dir"])), "job": self.screen.state()}

    def start_screen(self) -> bool:
        from tradingagents.stocks import screen
        from tradingagents.stocks.data import parse_tickers

        tickers = sorted(set(get_sp500_members(self.cache_dir)) |
                         set(parse_tickers(" ".join(self.store.get("global", "stock_watchlist") or []))))

        def work(job):
            job.progress, job.fraction = f"scoring {len(tickers)} stocks (cached data is reused)", 0.02
            out = screen.build(self.cache_dir, tickers, Path(self.config["results_dir"]),
                               progress=lambda m: _counted(job, m, 0.02, 0.96))
            job.fraction = 1.0
            return {"stocks": out["stocks"], "errors": len(out["errors"])}

        return self.screen.start(work)

    def decisions_state(self) -> dict:
        """The latest saved decision, the history, the shadow record and the capital view: reads only."""
        from tradingagents.stocks import daily

        shadow = self.store.con.execute("SELECT as_of, state_json FROM shadow_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        rec = json.loads(shadow["state_json"]).get("record") if shadow else None
        return {"latest": daily.load(self.store.con), "history": daily.history(self.store.con),
                "shadow": rec, "capital": self.capital_state(),
                "job": self._job_or_morning(self.decisions, "Daily decision")}

    def start_decisions(self, body: dict) -> bool:
        from tradingagents.stocks import daily

        train = bool(body.get("train"))

        def work(job):
            def say(stage, frac=None):
                if frac is not None:
                    job.fraction = max(job.fraction or 0.0, min(float(frac), 1.0))   # the bar never goes back
                job.progress = stage
            rec = daily.run(self.store, self.cache_dir, date.today(), body.get("policy") or "barrier_20", train, say,
                            force=bool(body.get("force")))
            job.fraction = 1.0
            return {"id": rec["id"], "mode": rec["mode"], "summary": rec.get("summary"), "reused": bool(rec.get("reused"))}

        return self.decisions.start(work)

    def start_zoo(self, body: dict | None = None) -> bool:
        from tradingagents.stocks import model_zoo

        force = bool((body or {}).get("force"))

        def work(job):
            def say(stage, frac=None):
                if frac is not None:
                    job.fraction = max(job.fraction or 0.0, min(float(frac), 1.0))
                job.progress = stage
            return model_zoo.run(self.store, self.cache_dir, say, force=force)

        return self.zoo.start(work)

    def zoo_view(self, ticker: str | None = None) -> dict:
        """The latest run of every model: all stocks, or one stock's forecasts with each model's tested record."""
        from tradingagents.stocks import model_zoo

        run = model_zoo.latest(self.store.con)
        state = self._job_or_morning(self.zoo, "Every model")
        if not run:
            return {"available": False, "job": state}
        from tradingagents.model_outputs import OUTPUTS

        from tradingagents.stocks import daily

        described = model_zoo.describe(run, daily.load(self.store.con))
        run = {**run, "stocks": described.pop("stocks")}
        base = {"available": True, "job": state, **{k: run[k] for k in ("id", "created_at", "info_cutoff",
                                                                          "tested", "members", "notes", "version")},
                **described, "ensemble": run.get("ensemble"),
                "outputs": {k: OUTPUTS[k].as_dict() for k in ("zoo.pct", "zoo.forecast_log_excess",
                                                              "zoo.chronos_log_return", "zoo.ensemble_expected_net",
                                                              "zoo.mean_pct")}}
        if ticker:
            t = ticker.strip().upper()
            return {**base, "ticker": t, "forecasts": run["stocks"].get(t)}
        return {**base, "stocks": run["stocks"], "watchlist": run.get("watchlist", []),
                "names": {t: describe(t) for t in run["stocks"]}}

    def capital_state(self) -> dict:
        from tradingagents.optimizer.data import latest_prices

        tickers = {e["ticker"] for e in self.store.ledger.events() if e["ticker"]}
        prices = latest_prices(self.cache_dir, tickers) if tickers else {}
        today = _today()
        from tradingagents.ledger import income

        return {"settings": capital.snapshot(self.store), "history": capital.history(self.store, 20),
                "started": self.store.get("global", "income_start"),
                "budget": income.budget(self.store, self.cache_dir, date.today()),
                "value": round(self.store.ledger.state(None, today, prices).equity, 2)}

    def live_records(self) -> list[dict]:
        """Every ETF model and horizon with saved runs, scored on what happened after each run."""
        from tradingagents.research.engine import live_record

        con = self.store.con
        pairs = con.execute("SELECT model, horizon, COUNT(DISTINCT info_cutoff) AS n FROM forecast_runs"
                            " WHERE universe = 'etf' GROUP BY model, horizon ORDER BY horizon, model").fetchall()
        market_store = local_store.store(self.cache_dir)
        key = json.dumps([market_store.bars_checked_at() if market_store else None,
                          con.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0]])
        return self._remember("live_records", key, lambda: [
            {k: v for k, v in live_record(con, self.cache_dir, r["model"], r["horizon"]).items() if k != "dates"}
            for r in pairs])

    def forecasts(self, query: dict) -> dict:
        from tradingagents.research.engine import latest_run, load_run

        con = self.store.con
        runs = [dict(r) for r in con.execute(
            "SELECT id, created_at, model, horizon, info_cutoff, universe, backend, seconds, peak_memory_mb,"
            " train_start, train_end, calibration_json FROM forecast_runs ORDER BY created_at DESC LIMIT 30")]
        for r in runs:
            r["calibration"] = json.loads(r.pop("calibration_json"))
        universe = query.get("universe") or "etf"
        chosen = (load_run(con, query["run"]) if query.get("run")
                  else latest_run(con, int(query.get("horizon") or 0) or None, universe=universe))
        if chosen:
            chosen["calibration"] = json.loads(chosen.pop("calibration_json"))
            chosen["config"] = json.loads(chosen.pop("config_json"))
            for f in chosen["forecasts"]:
                f["flags"] = json.loads(f.pop("flags_json"))
                f["name"] = describe(f["ticker"])
        trials = [dict(r) for r in con.execute("SELECT id, created_at, name, universe, horizon, first_test, last_test,"
                                               " scores_json, by_period_json, seconds FROM research_trials ORDER BY id")]
        for t in trials:
            t["scores"] = json.loads(t.pop("scores_json"))
            t["by_period"] = json.loads(t.pop("by_period_json"))
        return {"runs": runs, "run": chosen, "trials": trials, "job": self.forecast.state(),
                "settings": {s: {k: self.store.get(s, k) for k in ("return_model", "forecast_horizon", "forecast_model")}
                             for s in SLEEVES}}

    # --- one ticker, everything known about it ---

    def report_roots(self) -> list[Path]:
        return [Path(self.config["results_dir"]).resolve(), (self.portfolio_path.parent / "results").resolve()]

    def ticker(self, symbol: str) -> dict:
        """Everything about one instrument, section by section; a section that fails says why
        and the rest still show. Each source is cached by the local store, so opening the
        page again costs little."""
        import numpy as np

        from tradingagents.dataflows.vendors.yahoo import history as yh
        from tradingagents.dataflows.vendors.yahoo.news import ticker_articles
        from tradingagents.research.prices import bars

        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 15 or not all(c.isalnum() or c in ".-^=" for c in symbol):
            raise ValueError(f"not a ticker: {symbol!r}")
        out: dict = {"symbol": symbol, "name": describe(symbol)}

        def section(name, fn):
            try:
                out[name] = fn()
            except Exception as exc:  # one source down must not blank the page
                logger.info("ticker %s %s failed: %s", symbol, name, exc)
                out[name] = {"error": str(exc)[:300]}

        def price():
            import pandas as pd

            b = bars([symbol, "SPY"], self.cache_dir)
            close = b["Close"][symbol].dropna()
            if close.empty:
                raise ValueError("Yahoo has no prices for it")
            spy = b["Close"]["SPY"].reindex(close.index).ffill()
            r = np.log(close).diff()
            def ret(n):
                return float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n else None
            year = close.iloc[-252:]
            weekly = close.iloc[::5] if len(close) < 2600 else close.iloc[-2600::5]
            return {"last": float(close.iloc[-1]), "as_of": str(close.index[-1].date()),
                    "first": str(close.index[0].date()),
                    "returns": {"1d": ret(1), "1w": ret(5), "1m": ret(21), "3m": ret(63), "1y": ret(252),
                                "3y": ret(756), "5y": ret(1260)},
                    "high_52w": float(year.max()), "low_52w": float(year.min()),
                    "vol_3m": float(r.iloc[-63:].std() * np.sqrt(252)),
                    "max_drawdown": float(1 - (close / close.cummax()).min()),
                    "volume_1m": float(b["Volume"][symbol].dropna().iloc[-21:].mean())
                    if b["Volume"][symbol].notna().any() else None,
                    "chart": {"dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
                              "close": [round(float(v), 4) for v in weekly],
                              "spy": [round(float(v), 4) for v in spy.loc[weekly.index]]},
                    "daily": {"dates": [d.strftime("%Y-%m-%d") for d in close.index[-1300:]],
                              "close": [round(float(v), 4) for v in close.iloc[-1300:]],
                              "spy": [None if pd.isna(v) else round(float(v), 4) for v in spy.iloc[-1300:]]}}

        def mine():
            today = _today()
            held = {}
            for sl in (*SLEEVES, "combined"):
                q = self.store.ledger.state(sl, today).positions.get(symbol)
                if q:
                    held[sl] = q
            history = [{"session": r["session"], "status": r["status"], "target": r["targets"].get(symbol)}
                       for r in recommendations.history(self.store.con, self.store.ledger.account_id)
                       if symbol in r["targets"]]
            row = self.last_rows.get(symbol)
            return {"quantity": held, "plan": row and {k: row[k] for k in ("current", "target", "trade", "worst_loss",
                                                                             "expected", "reason", "daily_buy")},
                    "official": history}

        def forecast():
            """The standard rule's estimate (always), and each horizon's latest model forecast: the
            stored one when the ticker was in the run, otherwise computed now with the run's model."""
            from tradingagents.optimizer.data import compute_stats, fetch_closes
            from tradingagents.optimizer.plan import growth_excess
            from tradingagents.research.engine import forecast_one, latest_run, ranks

            settings = OptimizerSettings()
            closes = fetch_closes([symbol, settings.cash_ticker, settings.cash_proxy], required=[symbol],
                                  cache_dir=self.cache_dir)
            stats = compute_stats(closes, [settings.cash_ticker, symbol], settings.cash_ticker, settings.lookback_years,
                                  settings.short_history_years, settings.short_history_floor, {},
                                  settings.cash_proxy)
            g = float(growth_excess(stats, settings)[0])
            rule = {"growth_above_cash": g, "growth": g + stats.cash_return, "cash": stats.cash_return,
                    "volatility": float(stats.vol[0]), "worst_drop": float(stats.worst_loss[0]),
                    "assumed_drop": bool(stats.floored[0]), "years": float(stats.years[0])}
            # A company is forecast by the stock model when one has been run; a fund by the ETF model.
            profile = out.get("profile") if isinstance(out.get("profile"), dict) else {}
            universes = ("stock", "etf") if profile.get("quoteType") == "EQUITY" else ("etf",)
            models = []
            for h in (5, 20, 60):
                # A stock run that ranks nothing says nothing about this company: the ETF run stands in.
                run = next((r for u in universes if (r := latest_run(self.store.con, horizon=h, universe=u))
                            and (u == "etf" or ranks(r))), None)
                if not run:
                    continue
                f = next((x for x in run["forecasts"] if x["ticker"] == symbol), None)
                if f:
                    models.append({"horizon": h, "model": run["model"], "universe": run["universe"],
                                   "info_cutoff": run["info_cutoff"],
                                   "on_demand": False, "flags": json.loads(f.get("flags_json") or "[]"),
                                   **{k: f[k] for k in ("mean", "median", "q05", "q95", "p_gain", "estimation_sd")}})
                    continue
                key = ("forecast_one", run["id"], symbol)
                if key not in self._memo:
                    self._memo[key] = ("", forecast_one(self.store.con, self.cache_dir, run, symbol))
                if self._memo[key][1]:
                    models.append(self._memo[key][1])
            return {"rule": rule, "models": models}

        def analysts():
            actions = yh.upgrades(symbol, self.cache_dir)
            recent = actions.iloc[::-1].head(40) if len(actions) else actions
            return {"current": yh.current_analysts(symbol, self.cache_dir),
                    "actions": [{"date": str(i)[:10], **{k: (None if isinstance(v, float) and v != v else v)
                                                         for k, v in row.items()}}
                                for i, row in recent.iterrows()],
                    "actions_total": int(len(actions))}

        def earnings():
            e = yh.earnings(symbol, self.cache_dir)
            if not len(e):
                return []
            rows = e.iloc[::-1].head(16)
            return [{"date": str(i)[:10], "estimate": r.get("EPS Estimate"), "reported": r.get("Reported EPS"),
                     "surprise_pct": r.get("Surprise(%)")} for i, r in rows.iterrows()]

        def reports():
            found = []
            for root in self.report_roots():
                folder = root / symbol
                if not folder.is_dir():
                    continue
                for day in sorted(folder.iterdir(), reverse=True):
                    # The day's latest run, then each earlier run of that day (archived, never overwritten).
                    earlier = sorted((day / "archive").iterdir(), reverse=True) if (day / "archive").is_dir() else []
                    for rep, run in [(day / "reports", None)] + [(a, f"archive/{a.name}") for a in earlier]:
                        if rep.is_dir():
                            from tradingagents.llm_clients.request_log import FILE

                            calls = rep / FILE
                            found.append({"date": day.name, "run": run,
                                          "requests": sum(1 for _ in calls.open(encoding="utf-8")) if calls.is_file() else 0,
                                          "files": [
                                {"name": " / ".join(f.relative_to(rep).with_suffix("").parts).replace("_", " "),
                                 "path": str(f.resolve())} for f in sorted(rep.rglob("*.md"))]})
            from tradingagents.memory import TradingMemoryLog
            from tradingagents.stocks.verdict import ai_decision, rating_on_record

            ratings = [rating_on_record(e) for e in TradingMemoryLog(self.config).load_entries()
                       if e.get("ticker", "").upper() == symbol]
            held = bool(self.store.ledger.state("combined", _today()).positions.get(symbol))
            by_date = {r["date"]: r for r in ratings}
            for run in found:
                if run["run"]:   # an archived run: its own final decision only; the rating on record is the latest's
                    for root in self.report_roots():
                        first = root / symbol / run["date"] / run["run"] / "0_FINAL_DECISION.md"
                        if first.is_file():
                            head = first.read_text(encoding="utf-8").splitlines()[0]
                            run["decision"] = head.split(":", 1)[1].strip() if head.startswith("# FINAL DECISION:") else None
                            break
                    continue
                if run["date"] in by_date:
                    run["rating"] = by_date[run["date"]]["rating"]
                    run["agents"] = ai_decision(by_date[run["date"]], held)["action"]
                # The run's final decision is the model's verdict written in its first report.
                for root in self.report_roots():
                    first = root / symbol / run["date"] / "reports" / "0_FINAL_DECISION.md"
                    if first.is_file():
                        head = first.read_text(encoding="utf-8").splitlines()[0]
                        if head.startswith("# FINAL DECISION:"):
                            run["decision"] = head.split(":", 1)[1].strip()
                        break
            return {"runs": found, "ratings": ratings}

        section("price", price)
        section("profile", lambda: yh.profile(symbol, self.cache_dir))
        section("mine", mine)
        section("forecast", forecast)
        section("news", lambda: ticker_articles(symbol))
        section("analysts", analysts)
        section("earnings", earnings)
        quote_type = (out.get("profile") or {}).get("quoteType") if isinstance(out.get("profile"), dict) else None

        def corporate():
            from tradingagents.research.corporate import summarize

            last = out["price"].get("last") if isinstance(out.get("price"), dict) else None
            return summarize(yh.corporate_actions(symbol, self.cache_dir), date.today(), last, fund=quote_type == "ETF")

        section("corporate", corporate)
        if quote_type == "ETF":
            out["financials"] = {"note": "Funds have no company financial statements."}
            out["moat"] = {"note": "Funds have no moat of their own; look at the companies they hold."}
        else:
            section("financials", lambda: yh.financials(symbol, self.cache_dir))
            if isinstance(out.get("financials"), dict) and "error" not in out["financials"]:
                from tradingagents.research.moat import moat
                profile = out.get("profile") if isinstance(out.get("profile"), dict) else {}
                section("moat", lambda: moat(out["financials"], profile.get("sector"), profile.get("industry")))

                def quality():
                    from tradingagents.research.quality import checklist

                    closes = bars([symbol], self.cache_dir)["Close"][symbol]
                    dividends = (out.get("corporate") or {}) if isinstance(out.get("corporate"), dict) else {}
                    dividends = [[p["date"], p["amount"]] for p in (dividends.get("dividends") or {}).get("payments", [])]
                    financial = str((out.get("moat") or {}).get("basis", "")).startswith("bank")
                    analysts = (out.get("analysts") or {}).get("current") if isinstance(out.get("analysts"), dict) else None
                    return checklist(out["financials"], profile, closes, dividends, analysts, financial)

                section("quality", quality)
                section("moat_sources", lambda: self.moat_sources(symbol))
        section("reports", reports)
        return out

    # --- moat sources: one paid AI reading of the latest 10-K, saved per filing ---

    def _moat_path(self, symbol: str, accession: str) -> Path:
        return Path(self.config["results_dir"]) / "moat" / f"{symbol}_{accession}.json"

    def moat_sources(self, symbol: str) -> dict:
        """The saved reading for the company's newest saved filing, if any, and the job's state."""
        folder = Path(self.config["results_dir"]) / "moat"
        saved = sorted(folder.glob(f"{symbol}_*.json"), key=lambda f: f.stat().st_mtime) if folder.is_dir() else []
        job = self.moat_jobs.get(symbol)
        return {"saved": json.loads(saved[-1].read_text(encoding="utf-8")) if saved else None,
                "job": job.state() if job else None}

    def start_moat_sources(self, symbol: str) -> bool:
        from tradingagents.dataflows.config import run_config
        from tradingagents.dataflows.vendors.sec_edgar import business_section
        from tradingagents.dataflows.vendors.yahoo import history as yh
        from tradingagents.research.moat import moat
        from tradingagents.research.moat_ai import moat_sources

        symbol = symbol.upper()
        job = self.moat_jobs.setdefault(symbol, Job())
        g = self.store.settings("global")

        def work(job):
            job.progress = "reading the latest 10-K from SEC EDGAR"
            with run_config(self.config):
                filing = business_section(symbol)
            if filing is None:
                raise ValueError(f"{symbol} has no 10-K on SEC EDGAR (a fund or a foreign filer), so there is nothing to read")
            path = self._moat_path(symbol, filing["accession"])
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            profile = yh.profile(symbol, self.cache_dir)
            checks = moat(yh.financials(symbol, self.cache_dir), profile.get("sector"), profile.get("industry"))
            numbers = "\n".join(f"- {c['name']}: {c['value']} (needs {c['rule']}): "
                                 f"{'pass' if c['passed'] else 'fail' if c['passed'] is False else 'no data'}"
                                 for c in checks["checks"])
            job.progress = "asking the AI to rate the moat's sources from the filing"
            result = moat_sources(symbol, profile.get("longName") or symbol, filing, numbers, self.context_config(),
                                  gateway=self.gateway(), allow_paid=True, fallbacks=tuple(g["ai_fallbacks"]))
            save_json(path, result)
            return result

        return job.start(work)

    def report(self, path: str, suffix: str = ".md") -> str:
        """A saved analysis report, only from the report folders."""
        target = Path(path).resolve()
        if target.suffix != suffix or not any(root in target.parents for root in self.report_roots()):
            raise ValueError("not a saved report")
        return target.read_text(encoding="utf-8")

    def _last_agent_run(self, ticker: str) -> dict | None:
        """The newest routed run of the agents on a ticker (``agent_runs``): status, when, which model, why it failed."""
        try:
            row = self.store.con.execute("SELECT as_of, finished_at, status, actual_provider, actual_model, attempts_json,"
                                         " error FROM agent_runs WHERE ticker = ? AND run_id NOT IN"
                                         " (SELECT key FROM hidden_outputs WHERE kind = 'agent_run')"
                                         " ORDER BY finished_at DESC LIMIT 1", (ticker,)).fetchone()
        except Exception:   # an older database without the table
            return None
        if not row:
            return None
        tried = json.loads(row["attempts_json"] or "[]")
        why = next((a.get("why") for a in reversed(tried) if a.get("state") == "failed" and a.get("why")), None)
        return {"as_of": row["as_of"], "finished_at": row["finished_at"], "status": row["status"],
                "model": f"{row['actual_provider']}/{row['actual_model']}" if row["actual_provider"] else None,
                "why": why or (row["error"] or "")[:120] or None}

    def latest_agent_report(self, ticker: str) -> dict:
        """The newest saved TradingAgents run for a ticker: its day, final decision text and report files."""
        from tradingagents.dataflows.symbols import safe_ticker_component

        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("no ticker")
        found = []
        for root in self.report_roots():
            base = root / safe_ticker_component(ticker)
            if base.is_dir():
                found += [d / "reports" for d in base.iterdir() if (d / "reports" / "complete_report.md").is_file()]
        last_run = self._last_agent_run(ticker)
        if not found:
            return {"ticker": ticker, "available": False, "last_run": last_run}
        folder = max(found, key=lambda f: (f.parent.name, (f / "complete_report.md").stat().st_mtime))
        final = folder / "0_FINAL_DECISION.md"
        text = final.read_text(encoding="utf-8") if final.is_file() else ""
        # The agents' own call, as the final-decision file states it (the heading is the internal model's verdict).
        seen = re.search(r"rating (\w+), i\.e\. ([A-Z]+)", text)
        used = re.search(r"\*\*Agent models\*\*: ([^\n(]+)", text)
        rating, action = seen.groups() if seen else (None, None)
        pm = folder / "5_portfolio" / "decision.md"
        if rating is None and pm.is_file():      # an older run: the portfolio manager's rating only
            got = re.search(r"\*\*Rating\*\*:\s*(\w+)", pm.read_text(encoding="utf-8"))
            rating = got.group(1) if got else None
        return {"ticker": ticker, "available": True, "date": folder.parent.name,
                "agent_rating": rating, "agent_action": action,
                "agent_models": used.group(1).strip() if used else None, "last_run": last_run,
                "saved_at": datetime.fromtimestamp((folder / "complete_report.md").stat().st_mtime).isoformat(timespec="minutes"),
                "final": text or None,
                "markdown": str((folder / "complete_report.md").resolve()),
                "html": str((folder / "complete_report.html").resolve()) if (folder / "complete_report.html").is_file() else None}

    # --- research ---

    def market(self) -> dict:
        """Recent moves from closing prices, and today's saved explanation if there is one."""
        def build():
            sp500 = get_sp500_members(self.cache_dir)
            themes = [t for group in THEME_GROUPS.values() for t in group]
            closes = fetch_closes([*market_tickers(), *sp500, *themes], required=["SPY"], cache_dir=self.cache_dir)
            return market_overview(closes, sp500)

        market_store = local_store.store(self.cache_dir)
        overview = self._remember("market", json.dumps([market_store.bars_checked_at() if market_store else None,
                                                        _today()]), build)
        path = Path(self.config["results_dir"]) / "market" / f"brief_{overview['as_of']}.json"
        brief = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return {"overview": overview, "brief": brief, "explain": self.explain.state()}

    # --- writes to the books ---

    def ledger_write(self, path: str, body: dict) -> dict:
        """One user action on the books. ``op_id`` names the action: the page sends the same id
        when it retries, so a retry is answered with the first event (``applied: false``);
        a request without one is a new action, never merged with an identical earlier one."""
        led, day = self.store.ledger, body.get("date") or _today()
        op_id = _op_id(body)
        with self.lock:
            applied = led.find_operation(op_id) is None and led.find_operation(op_id, "out") is None
            if path == "/api/ledger/contribution":
                out = {"event": led.contribute(_number(body, "amount"), day, sleeve=body.get("sleeve", "trading"),
                                               settle_date=body.get("settle_date"), note=body.get("note"),
                                               op_id=op_id)}
            elif path == "/api/ledger/fill":
                event, warnings = led.fill(body["sleeve"], body["side"], str(body["ticker"]),
                                           _number(body, "quantity"), _number(body, "price"), day,
                                           fee=_number(body, "fee", 0.0), note=body.get("note"),
                                           strict=not body.get("broker_record"), op_id=op_id)
                out = {"event": event, "warnings": warnings}
            elif path == "/api/ledger/transfer":
                out = {"transfer": led.transfer(body["from"], body["to"], day, ticker=body.get("ticker"),
                                                quantity=_number(body, "quantity", 0.0),
                                                cash=_number(body, "cash", 0.0),
                                                price=_number(body, "price", None),
                                                note=body.get("note"), op_id=op_id)}
            elif path == "/api/ledger/import":
                rows = csv_import.preview(led, body["csv"], body.get("sleeve", "trading"))
                out = {"rows": [asdict(r) for r in rows]}
                if body.get("commit"):
                    out["result"] = csv_import.commit(led, rows, body.get("sleeve", "trading"))
            else:
                raise ValueError(f"unknown ledger action {path}")
            if path != "/api/ledger/import" or body.get("commit"):
                self.after_ledger_change()
        if path != "/api/ledger/import":
            out.update(op_id=op_id, applied=applied)
        return out

    def save_holdings(self, body: dict) -> dict:
        """The Settings holdings editor: share counts, and settled cash when it changed.

        Answers with what the database now holds, so the page shows saved values, not
        what it sent. A cash change must say whether it is a deposit/withdrawal or a
        correction (see ``Store.adjust_cash``); without that it is refused, not guessed.
        """
        wanted = {str(p["ticker"]).strip().upper(): _number(p, "quantity") for p in body.get("positions", [])}
        day = _today()
        cash_out = None
        target = _number(body, "cash", None)
        with self.lock:
            with transaction(self.store.con):     # both or neither
                events = self.store.adjust_holdings(wanted, day, note=body.get("note"))
                now = self.store.ledger.state("trading", day).cash.settled
                if target is not None and abs(target - now) >= 0.005:
                    if body.get("cash_mode") not in Store.CASH_MODES:
                        raise ValueError("say whether the cash change is money in or out of the IRA "
                                         "(a deposit or withdrawal) or a correction of the books")
                    cash_out = self.store.adjust_cash(target, day, body["cash_mode"], note=body.get("cash_note"),
                                                      op_id=_op_id(body))
            self.after_ledger_change()
            state = self.store.ledger.state("trading", day)
        return {"ok": True, "adjustments": events, "cash_change": cash_out,
                "saved": {"positions": [{"ticker": t, "quantity": q} for t, q in sorted(state.positions.items())],
                          "cash_settled": round(state.cash.settled, 2),
                          "cash_unsettled": round(state.cash.unsettled, 2),
                          "cash_reserved": round(state.cash.reserved, 2)}}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.info(fmt, *args)

        def _send(self, status, body: bytes, content_type: str):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                # The page cancelled this read (a newer one replaced it): nothing to answer.
                self.close_connection = True

        def _json(self, payload, status=HTTPStatus.OK):
            self._send(status, json.dumps(_finite(payload), default=str, allow_nan=False).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length < 0 or length > MAX_BODY:
                raise ValueError(f"the request body is larger than {MAX_BODY // (1024 * 1024)} MB")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the request body must be a JSON object")
            return body

        def _page(self, name: str):
            """A page, with this run's write token in it (see ``write_refusal``)."""
            html = (STATIC / name).read_bytes().replace(
                b"<head>", b'<head>\n<meta name="csrf-token" content="' + app.csrf_token.encode() + b'">', 1)
            self._send(HTTPStatus.OK, html, "text/html; charset=utf-8")

        def _started(self, started: bool, what: str):
            # 409 says exactly what is busy, so the page can say so instead of guessing.
            if started:
                self._json({"started": True}, HTTPStatus.ACCEPTED)
            else:
                self._json({"started": False, "error": f"{what} is already running"}, HTTPStatus.CONFLICT)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._page("index.html")
                elif url.path in ("/favicon.svg", "/favicon.ico"):
                    self._send(HTTPStatus.OK, (STATIC / "favicon.svg").read_bytes(), "image/svg+xml")
                elif url.path == "/logo.svg":
                    self._send(HTTPStatus.OK, (STATIC / "logo.svg").read_bytes(), "image/svg+xml")
                elif url.path.startswith("/ticker/"):
                    self._page("ticker.html")
                elif url.path == "/api/ticker":
                    self._json(app.ticker(query.get("symbol", "")))
                elif url.path == "/api/sector-forecast":
                    self._json(_finite(app.sector_forecast()))
                elif url.path == "/api/portfolio-view":
                    self._json(_finite(app.portfolio_view()))
                elif url.path == "/api/model-zoo":
                    self._json(_finite(app.zoo_view(query.get("ticker"))))
                elif url.path == "/api/track-record":
                    from tradingagents.stocks import track_record
                    self._json(_finite(track_record.build(app.store.con, app.cache_dir, config=app.config)))
                elif url.path == "/api/agent-report":
                    self._json(app.latest_agent_report(query.get("ticker", "")))
                elif url.path == "/report":
                    self._send(HTTPStatus.OK, app.report(query.get("path", ""), ".html").encode(), "text/html; charset=utf-8")
                elif url.path == "/api/report":
                    self._send(HTTPStatus.OK, app.report(query.get("path", "")).encode(), "text/plain; charset=utf-8")
                elif url.path == "/api/state":
                    context, cash = app.store.portfolio("combined", _today())
                    config = read_config(app.universe_path)
                    self._json({"portfolio": context.model_dump(exclude_none=True), "universe": config,
                                "cash": {"settled": round(cash.settled, 2), "unsettled": round(cash.unsettled, 2),
                                         "reserved": round(cash.reserved, 2)},
                                "groups": GROUPS, "watchlist": app.store.get("global", "stock_watchlist") or []})
                elif url.path == "/api/trade":
                    self._json(_finite(app.trade()))
                elif url.path == "/api/income":
                    self._json(_finite(app.income()))
                elif url.path == "/api/rotation":
                    self._json(_finite(app.rotation()))
                elif url.path == "/api/keys":
                    from tradingagents.webapp import keys
                    self._json({"keys": keys.listing(), "file": keys.env_path().name})
                elif url.path == "/api/model-record":
                    self._json(_finite(app.model_record()))
                elif url.path == "/api/ledger":
                    self._json(app.books())
                elif url.path == "/api/recommendations":
                    self._json(recommendations.history(app.store.con, app.store.ledger.account_id,
                                                       query.get("sleeve")))
                elif url.path == "/api/recommendation":
                    rec = recommendations.load(app.store.con, query["id"])
                    rec["decisions"] = recommendations.decisions(app.store.con, query["id"])
                    self._json(rec)
                elif url.path == "/api/ai":
                    self._json(app.ai_usage())
                elif url.path == "/api/forecasts/combined":
                    self._json(app.combined_forecasts(int(query.get("horizon") or 20), query.get("universe") or "etf"))
                elif url.path == "/api/screen/quality":
                    self._json(app.screen_state())
                elif url.path == "/api/signals":
                    self._json(app.signals(query.get("ticker")))
                elif url.path == "/api/analyze":
                    self._json(app.analysis_state())
                elif url.path == "/api/intraday":
                    self._json(app.intraday(query.get("symbol", ""), query.get("span", "1d")))
                elif url.path == "/api/entry-alerts":
                    self._json(app.entry_alerts())
                elif url.path == "/api/entry-plans":
                    self._json(app.entry_plans(query.get("ticker", "")))
                elif url.path == "/api/manual-prompt":
                    self._json(app.manual_prompt(query.get("ticker", ""), query.get("date", "")))
                elif url.path == "/api/ai-decisions":
                    self._json(app.ai_decisions())
                elif url.path == "/api/requests":
                    self._json(app.request_log(query.get("ticker", ""), query.get("date", ""), query.get("run") or None))
                elif url.path == "/api/decisions":
                    self._json(app.decisions_state())
                elif url.path == "/api/capital":
                    self._json(app.capital_state())
                elif url.path == "/api/stocks":
                    self._json(app.stock_run(query.get("run")))
                elif url.path == "/api/stocks/outcomes":
                    self._json(app.stock_outcomes())
                elif url.path == "/api/forecasts/live":
                    self._json({"records": app.live_records(), "daily": app.daily.state()})
                elif url.path == "/api/forecasts":
                    self._json(app.forecasts(query))
                elif url.path == "/api/schedule":
                    self._json(_finite(app.schedule_state()))
                elif url.path == "/api/model-outputs":
                    from tradingagents.model_outputs import catalog
                    self._json(catalog())
                elif url.path == "/api/hardware":
                    from tradingagents.research.compute import hardware
                    self._json(hardware())
                elif url.path == "/api/ratings":
                    self._json({**app.ai.state(), "ratings": app.ratings, "date": app.ratings_date})
                elif url.path == "/api/market":
                    self._json(app.market())
                elif url.path == "/api/market/explain":
                    self._json(app.explain.state())
                elif url.path == "/api/moat/sources":
                    self._json(app.moat_sources(query["symbol"].upper()))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # a price outage is shown on the page, not a dropped request
                logger.exception("request failed")
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            refused = write_refusal(self.headers, app.csrf_token, self.server.server_address[1])
            if refused:
                try:      # read what was sent (within the limit), or the refusal arrives as a reset connection
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if 0 <= length <= MAX_BODY:
                    self.rfile.read(length)
                else:
                    self.close_connection = True
                self._json({"error": refused}, HTTPStatus.FORBIDDEN)
                return
            try:
                body = self._body()
                if self.path.startswith("/api/ledger/"):
                    self._json(app.ledger_write(self.path, body))
                elif self.path == "/api/ratings":
                    self._started(app.start_ai(body.get("analysts")), "Rating")
                elif self.path == "/api/forecasts/run":
                    self._started(app.start_forecast(body), "A forecast run")
                elif self.path == "/api/portfolio":
                    self._json(app.save_holdings(body))
                elif self.path == "/api/settings":
                    with app.lock:
                        for key, value in (body.get("values") or {}).items():
                            app.store.set(body.get("scope", "global"), key, value)
                    self._json({"ok": True})
                elif self.path == "/api/universe":
                    groups = [g for g in body.get("groups", []) if g in GROUPS]
                    extra = [str(t).strip().upper() for t in body.get("extra", []) if str(t).strip()]
                    exclude = body.get("exclude", read_config(app.universe_path).get("exclude", []))
                    app.universe_path.write_text(json.dumps({"groups": groups, "extra": list(dict.fromkeys(extra)),
                                                             "exclude": [str(t).strip().upper() for t in exclude]},
                                                            indent=2), encoding="utf-8")
                    self._json({"ok": True})
                elif self.path == "/api/market/explain":
                    self._started(app.start_explain(), "The market explanation")
                elif self.path == "/api/screen/quality":
                    self._started(app.start_screen(), "The quality screen")
                elif self.path == "/api/analyze":
                    self._started(app.start_analysis(body), "An agent analysis")
                elif self.path == "/api/agent-runs/dismiss":
                    with app.lock:
                        self._json(app.dismiss_agent_failures(str(body.get("ticker") or "")))
                elif self.path == "/api/agent-report/delete":
                    with app.lock:
                        self._json(app.delete_agent_report(str(body.get("ticker") or ""), str(body.get("date") or ""),
                                                           body.get("run") or None))
                elif self.path == "/api/manual-answer":
                    self._json(app.manual_answer(body))
                elif self.path == "/api/ai-decisions":
                    self._started(app.start_ai_decisions(body), "An agent analysis")
                elif self.path == "/api/ai-budget":
                    with app.lock:
                        self._json(app.set_ai_budget(body))
                elif self.path == "/api/keys":
                    from tradingagents.webapp import keys
                    with app.lock:
                        saved = keys.save(str(body.get("env") or ""), str(body.get("value") or ""))
                        app.store.set("global", "ai_cooldowns", {})    # a new or topped-up key is tried again
                    self._json({"saved": saved, "keys": keys.listing()})
                elif self.path == "/api/sector-forecast":
                    self._started(app.start_sector(), "The sector model")
                elif self.path == "/api/jobs/dismiss":
                    name = str(body.get("job") or "")
                    job = getattr(app, name, None) if name in JOB_NAMES else None
                    if not isinstance(job, Job):
                        raise ValueError(f"unknown job {name!r}")
                    job.dismiss()
                    self._json(job.state())
                elif self.path == "/api/jobs/stop":
                    name = str(body.get("job") or "")
                    job = getattr(app, name, None) if name in JOB_NAMES else None
                    if not isinstance(job, Job):
                        raise ValueError(f"unknown job {name!r}")
                    if not job.stop_waiting():
                        raise ValueError("that job is not stuck: it is not running, or it reported progress in the "
                                         f"last {Job.STUCK_AFTER // 60} minutes")
                    self._json(job.state())
                elif self.path == "/api/schedule/run":
                    self._started(app.start_morning(body), "The morning run")
                elif self.path == "/api/model-zoo/run":
                    self._started(app.start_zoo(body), "A run of every model")
                elif self.path == "/api/track-record/hide":
                    from tradingagents.stocks import track_record
                    track_record.hide(app.store.con, str(body.get("key") or ""))
                    self._json({"hidden": True})
                elif self.path == "/api/track-record/unhide":
                    from tradingagents.stocks import track_record
                    self._json({"restored": track_record.unhide(app.store.con, str(body.get("key") or ""))})
                elif self.path == "/api/track-record/restore":
                    from tradingagents.stocks import track_record
                    self._json({"restored": track_record.restore_all(app.store.con)})
                elif self.path == "/api/decisions/run":
                    self._started(app.start_decisions(body), "A decision run")
                elif self.path == "/api/capital":
                    with app.lock:
                        capital.update(app.store, body.get("values") or {}, source="user")
                    self._json(app.capital_state())
                elif self.path == "/api/stocks/run":
                    self._started(app.start_stocks(body), "A stock batch")
                elif self.path == "/api/stocks/ideas":
                    from tradingagents.stocks.ideas import make

                    self._json(make(app.store.con, app.store, app.cache_dir, body.get("run"), int(body.get("horizon") or 20)))
                elif self.path == "/api/stocks/decide":
                    from tradingagents.stocks.ideas import decide

                    decide(app.store.con, body["idea"], body["ticker"].upper(), body["decision"], body.get("note"))
                    self._json({"recorded": True})
                elif self.path == "/api/stocks/watchlist":
                    from tradingagents.stocks.data import parse_tickers

                    app.store.set("global", "stock_watchlist", parse_tickers(" ".join(body.get("tickers") or [])))
                    self._json({"watchlist": app.store.get("global", "stock_watchlist")})
                elif self.path == "/api/stocks/news-ai":
                    self._json(app.stock_news_ai(body))
                elif self.path == "/api/moat/sources":
                    self._started(app.start_moat_sources(body["symbol"]), "The moat reading")
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, TypeError, KeyError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # a vendor outage is shown on the page, not a dropped connection
                logger.exception("request failed")
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(portfolio_path: str, schedule_path: str, port: int = 8765, config: dict | None = None,
          universe_path: str | None = None):
    portfolio, schedule = Path(portfolio_path).resolve(), Path(schedule_path).resolve()
    if not portfolio.exists():
        raise ValueError(f"portfolio file {portfolio} does not exist")
    universe = Path(universe_path).resolve() if universe_path else portfolio.parent / "my_universe.json"
    app = App(portfolio, schedule, universe, {**DEFAULT_CONFIG, **(config or {})})
    try:
        app.start_daily_record()
    except Exception:  # the record is a bonus: the app starts without it
        logger.exception("daily Chronos-2 record could not start")
    if os.environ.get("TRADINGAGENTS_NO_SCHEDULE") != "1":
        app.start_scheduler()
    return _Server(("127.0.0.1", port), make_handler(app))


class _Server(ThreadingHTTPServer):
    # On Windows, address reuse lets a second server bind the same port beside a running
    # one, and requests then reach whichever answers: an old copy with old code. Binding
    # exclusively makes a second start fail instead.
    allow_reuse_address = os.name != "nt"
