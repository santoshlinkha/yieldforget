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
from tradingagents.ledger.store import Store
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
        self.attempts, self.events, self.gen = [], [], 0
        self.started_at = self.touched_at = 0.0

    @property
    def progress(self) -> str:
        return self._progress

    @progress.setter
    def progress(self, text: str):
        """Every progress line also goes to the short live list the page shows."""
        self._progress, self.touched_at = text, time.time()
        self.events = [*self.events[-7:], {"at": time.strftime("%H:%M:%S"), "text": str(text)[:140]}]

    def start(self, work) -> bool:
        if self.running:
            return False
        self.gen += 1
        gen = self.gen
        self.running, self.result, self.error, self.log, self.attempts, self.events = True, None, None, [], [], []
        self.started_at = time.time()
        self.progress = "starting"

        def run():
            result, error = None, None
            try:
                result = work(self)
            except Exception as exc:  # shown on the page rather than lost in a thread
                logger.exception("background job failed")
                error = str(exc)
            finally:
                if gen == self.gen:   # a run the page stopped waiting for never overwrites a newer one
                    self.result, self.error, self.running = result, error, False

        threading.Thread(target=run, daemon=True).start()
        return True

    def stop_waiting(self) -> bool:
        """Give up on a run that has made no progress for ``STUCK_AFTER`` seconds. Python cannot kill the thread:
        it may still finish in the background, but its result is ignored and a new run can start."""
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
                "elapsed": round(now - self.started_at) if self.started_at else None,
                "idle": round(now - self.touched_at) if self.running else None,
                "stuck": bool(self.running and now - self.touched_at >= self.STUCK_AFTER)}


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


JOB_NAMES = ("ai", "explain", "forecast", "daily", "stocks", "decisions", "screen", "analysis")


def _short_error(exc: Exception) -> str:
    """A provider failure in a few words for the status line."""
    text = str(exc)
    for needle, short in (("429", "rate limited"), ("rate", "rate limited"), ("quota", "out of quota"),
                          ("401", "bad key"), ("403", "not allowed"), ("404", "model not available"),
                          ("insufficient", "out of credit"), ("timed out", "timed out"), ("tool", "no tool support")):
        if needle in text.lower():
            return short
    return text[:80]


def _today() -> str:
    return date.today().isoformat()


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
        self.ai, self.backtest, self.tracker, self.explain, self.forecast = Job(), Job(), Job(), Job(), Job()
        self.moat_jobs: dict[str, Job] = {}
        self.daily = Job()   # the daily Chronos-2 record, apart from the runs started on the page
        self.stocks = Job()  # a stock research batch
        self.decisions = Job()  # the daily stock decision (and, when asked, this week's model fit)
        self.screen = Job()     # the quality and Wall Street screen
        self.analysis = Job()   # one ticker's agent run for a chosen day, on request
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
        pot = income.budget(self.store, self.cache_dir, today)
        health = self._remember("income-health", self._inputs("income-health"),
                                lambda: income.health(self.store, self.cache_dir, today))
        books = self.books()["sleeves"]["combined"]
        sgov = (books["positions"].get(income.RESERVE) or {}).get("value") or 0.0
        return {"budget": pot, "health": health, "sgov_value": sgov,
                "projection": income.projection(capital.load(self.store), health, sgov)}

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
        out.update({"income": inc, "account": books["sleeves"]["combined"], "job": self.decisions.state(),
                    "freshness": {"prices_checked": market_store.bars_checked_at() if market_store else None,
                                  "last_event": books["events"][0]["recorded_at"] if books["events"] else None}})
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
        """The configs an agent run tries in order: the configured one, then the free Gemini models when the
        spending policy lists Google as a fallback and a Google key is set."""
        config = self.agent_config(g)
        attempts = [config]
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
        # Then the free OpenRouter models, every role on one model, when the key is set.
        if "openrouter" in g["ai_fallbacks"] and os.environ.get("OPENROUTER_API_KEY"):
            for model in g["ai_openrouter_models"]:
                if (config["llm_provider"], config["deep_think_llm"]) != ("openrouter", model):
                    attempts.append({**config, "llm_provider": "openrouter", "quick_think_llm": model,
                                     "deep_think_llm": model, "backend_url": None, "agent_models": {}})
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

    def start_explain(self) -> bool:
        g = self.store.settings("global")

        def work(job):
            job.progress = "reading prices"
            overview = self.market()["overview"]
            job.progress = "reading headlines and asking the AI"
            brief = market_brief(overview, self.config, gateway=self.gateway(), allow_paid=True,
                                 fallbacks=tuple(g["ai_fallbacks"]))
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
            with ThreadPoolExecutor(max_workers=min(3, len(combos))) as pool:
                futures = {pool.submit(run_forecast, self.store.con, self.cache_dir, tickers, horizon=h, model=m,
                                       universe=universe, device_setting=device): (m, h) for m, h in combos}
                for fut in as_completed(futures):
                    m, h = futures[fut]
                    done += 1
                    try:
                        runs.append(fut.result())
                    except Exception as exc:  # one failing model leaves the others' results
                        logger.exception("forecast %s at %s sessions failed", m, h)
                        failed.append({"model": m, "horizon": h, "error": str(exc)})
                    job.progress = f"{done} of {len(combos)} done on {UNIVERSES[universe]} (last: {m}, {h} sessions)"
            return {"universe": universe, "batch": compare_runs(runs), "failed": failed}

        return self.forecast.start(work)

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
            for h in due:
                job.progress = f"recording Chronos-2's {h}-session forecasts"
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
            progress = Progress(lambda stage: setattr(job, "progress", stage))
            job.log = progress.tickers       # per-stock status, read while it runs
            return run_batch(self.store.con, self.cache_dir, names, sources, config, progress)

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
            out[t] = read_news(t, f.get("name") or t, run["info_cutoff"], news["items"], numbers, self.config,
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
        for r in out["rows"]:
            t = r["ticker"]
            row = rows.get(t) or (self._screen_row(t) if ticker else None)
            r["screen"] = row and {k: row.get(k) for k in ("quality_score", "quality_rank", "analyst_mean",
                                                            "analyst_label", "target_upside", "rank")}
            r["rating"] = ratings.get(t)
            # The stock model's score when it has one, else the fund models' rank among funds.
            score = scores.get(t) or funds.get(t)
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
                             "expected_net": round(r["combined"]["mean"], 5), "p_win": round(r["combined"]["p_gain"], 4),
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

    def _run_agents(self, job, ticker: str, day: str, analysts: list[str], g: dict) -> dict:
        """The agents on one ticker as of ``day``, trying each configured provider in turn (paid: only from a click)."""
        from tradingagents.dataflows.symbols import safe_ticker_component
        from tradingagents.ledger.ai_gateway import AIBlocked, AIInProgress
        from tradingagents.optimizer.ai_views import analyze

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
        for n, cfg in enumerate(attempts, 1):
            if evidence:
                cfg = {**cfg, "quant_evidence": {ticker.upper(): evidence}}
            who = f"{cfg['llm_provider']} ({cfg['deep_think_llm']})"
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
                # Over budget or switched off stops here; a model with no price is a setup gap another
                # provider can cover.
                if not str(exc).startswith("no price"):
                    job.attempts[-1].update(state="failed", why=str(exc)[:120])
                    raise
                errors.append(f"{who}: {exc}")
                job.attempts[-1].update(state="failed", why=str(exc)[:120])
            except Exception as exc:  # out of credit, bad key, rate limit, unreachable: the next provider, if any
                logger.warning("analysis on %s failed: %s", who, exc)
                errors.append(f"{who}: {str(exc)[:200]}")
                job.attempts[-1].update(state="failed", why=_short_error(exc))
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
        raise ValueError("No AI provider could run the agents. " + " | ".join(errors))

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
            raise ValueError("paste the whole answer, ending with its JSON block")
        decision = parse_answer(text)
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
            lines += [f"- **Model rank**: #{model['rank']} of {model['of']} {'funds' if fund else 'S&P 500 stocks'} today",
                      f"- **Expected return**: {model['expected_net']:+.2%} {'above cash' if fund else 'after costs'}, "
                      f"P(profit) {model['p_win']:.0%}"
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
        return self.analysis.start(lambda job: self._run_agents(job, ticker, day, analysts, g))

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
        if not tickers:
            raise ValueError("pick at least one ticker")
        if len(tickers) > 20:
            raise ValueError("at most 20 tickers per run")
        day = self._check_day(str(body.get("date") or self._last_session()))
        analysts = self._analysts(body, g)

        def work(job):
            done, failed = {}, {}
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

        return self.analysis.start(work)

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
            job.progress = f"scoring {len(tickers)} stocks (cached data is reused)"
            out = screen.build(self.cache_dir, tickers, Path(self.config["results_dir"]),
                               progress=lambda m: setattr(job, "progress", m))
            return {"stocks": out["stocks"], "errors": len(out["errors"])}

        return self.screen.start(work)

    def decisions_state(self) -> dict:
        """The latest saved decision, the history, the shadow record and the capital view: reads only."""
        from tradingagents.stocks import daily

        shadow = self.store.con.execute("SELECT as_of, state_json FROM shadow_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        rec = json.loads(shadow["state_json"]).get("record") if shadow else None
        return {"latest": daily.load(self.store.con), "history": daily.history(self.store.con),
                "shadow": rec, "capital": self.capital_state(), "job": self.decisions.state()}

    def start_decisions(self, body: dict) -> bool:
        from tradingagents.stocks import daily

        train = bool(body.get("train"))

        def work(job):
            def say(stage):
                job.progress = stage
            rec = daily.run(self.store, self.cache_dir, date.today(), body.get("policy") or "barrier_20", train, say)
            return {"id": rec["id"], "mode": rec["mode"], "summary": rec["summary"]}

        return self.decisions.start(work)

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
                              "spy": [round(float(v), 4) for v in spy.loc[weekly.index]]}}

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
            result = moat_sources(symbol, profile.get("longName") or symbol, filing, numbers, self.config,
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
        if not found:
            return {"ticker": ticker, "available": False}
        folder = max(found, key=lambda f: (f.parent.name, (f / "complete_report.md").stat().st_mtime))
        final = folder / "0_FINAL_DECISION.md"
        return {"ticker": ticker, "available": True, "date": folder.parent.name,
                "saved_at": datetime.fromtimestamp((folder / "complete_report.md").stat().st_mtime).isoformat(timespec="minutes"),
                "final": final.read_text(encoding="utf-8") if final.is_file() else None,
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
        led, day = self.store.ledger, body.get("date") or _today()
        with self.lock:
            if path == "/api/ledger/contribution":
                out = {"event": led.contribute(float(body["amount"]), day, sleeve=body.get("sleeve", "trading"),
                                               settle_date=body.get("settle_date"), note=body.get("note"))}
            elif path == "/api/ledger/fill":
                event, warnings = led.fill(body["sleeve"], body["side"], body["ticker"], float(body["quantity"]),
                                           float(body["price"]), day, fee=float(body.get("fee") or 0.0),
                                           note=body.get("note"), strict=not body.get("broker_record"))
                out = {"event": event, "warnings": warnings}
            elif path == "/api/ledger/transfer":
                out = {"transfer": led.transfer(body["from"], body["to"], day, ticker=body.get("ticker"),
                                                quantity=float(body.get("quantity") or 0.0),
                                                cash=float(body.get("cash") or 0.0),
                                                price=float(body["price"]) if body.get("price") else None,
                                                note=body.get("note"))}
            elif path == "/api/ledger/import":
                rows = csv_import.preview(led, body["csv"], body.get("sleeve", "trading"))
                out = {"rows": [asdict(r) for r in rows]}
                if body.get("commit"):
                    out["result"] = csv_import.commit(led, rows, body.get("sleeve", "trading"))
            else:
                raise ValueError(f"unknown ledger action {path}")
            if path != "/api/ledger/import" or body.get("commit"):
                self.after_ledger_change()
        return out


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.info(fmt, *args)

        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=HTTPStatus.OK):
            self._send(status, json.dumps(_finite(payload), default=str, allow_nan=False).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

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
                    self._send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path in ("/favicon.svg", "/favicon.ico"):
                    self._send(HTTPStatus.OK, (STATIC / "favicon.svg").read_bytes(), "image/svg+xml")
                elif url.path.startswith("/ticker/"):
                    self._send(HTTPStatus.OK, (STATIC / "ticker.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/ticker":
                    self._json(app.ticker(query.get("symbol", "")))
                elif url.path == "/api/agent-report":
                    self._json(app.latest_agent_report(query.get("ticker", "")))
                elif url.path == "/report":
                    self._send(HTTPStatus.OK, app.report(query.get("path", ""), ".html").encode(), "text/html; charset=utf-8")
                elif url.path == "/api/report":
                    self._send(HTTPStatus.OK, app.report(query.get("path", "")).encode(), "text/plain; charset=utf-8")
                elif url.path == "/api/state":
                    context, _ = app.store.portfolio("combined", _today())
                    config = read_config(app.universe_path)
                    self._json({"portfolio": context.model_dump(exclude_none=True), "universe": config,
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
            try:
                body = self._body()
                if self.path.startswith("/api/ledger/"):
                    self._json(app.ledger_write(self.path, body))
                elif self.path == "/api/ratings":
                    self._started(app.start_ai(body.get("analysts")), "Rating")
                elif self.path == "/api/forecasts/run":
                    self._started(app.start_forecast(body), "A forecast run")
                elif self.path == "/api/portfolio":
                    wanted = {p["ticker"].upper(): float(p["quantity"]) for p in body.get("positions", [])}
                    with app.lock:
                        events = app.store.adjust_holdings(wanted, _today(), note=body.get("note"))
                        app.after_ledger_change()
                    self._json({"ok": True, "adjustments": events})
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
                    self._json({"saved": saved, "keys": keys.listing()})
                elif self.path == "/api/jobs/stop":
                    name = str(body.get("job") or "")
                    job = getattr(app, name, None) if name in JOB_NAMES else None
                    if not isinstance(job, Job):
                        raise ValueError(f"unknown job {name!r}")
                    if not job.stop_waiting():
                        raise ValueError("that job is not stuck: it is not running, or it reported progress in the "
                                         f"last {Job.STUCK_AFTER // 60} minutes")
                    self._json(job.state())
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
    return _Server(("127.0.0.1", port), make_handler(app))


class _Server(ThreadingHTTPServer):
    # On Windows, address reuse lets a second server bind the same port beside a running
    # one, and requests then reach whichever answers: an old copy with old code. Binding
    # exclusively makes a second start fail instead.
    allow_reuse_address = os.name != "nt"
