"""A local web app for the daily optimization: one page and a small JSON API.

It runs on this machine because it needs the price feed and the LLM keys in
.env, and it serves only localhost: the page edits your records and can spend
LLM credit, so nothing else on the network should reach it.

Records live in ``portfolio.sqlite3`` next to the holdings file (see
``tradingagents.ledger``): sleeves, every recorded fill and transfer, official
recommendations and AI usage. A plan computed for the page is a **preview** and
is not stored; **saving** makes it the official recommendation for its session.
Paid AI runs only on an explicit POST, through the budgeted gateway.

Prices are downloaded once per day into the cache directory and reused. AI
ratings and backtests run in background threads, since they take minutes, and
the page polls for their progress.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from dataclasses import asdict
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tradingagents.dataflows import local_store
from tradingagents.dataflows.vendors.index_members import get_sp500_members
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.ledger import imports as csv_import
from tradingagents.ledger import capital, recommendations
from tradingagents.ledger.accounting import SLEEVES
from tradingagents.ledger.ai_gateway import Budget, Gateway, Policy, Price
from tradingagents.ledger.store import Store
from tradingagents.optimizer.ai_views import collect_ratings
from tradingagents.optimizer.backtest import render_backtest, run_backtest
from tradingagents.optimizer.data import fetch_closes, latest_prices
from tradingagents.optimizer.market import market_brief, market_overview, market_tickers, save_json
from tradingagents.optimizer.plan import (
    OptimizerSettings,
    build_plan,
    daily_orders,
    load_schedule,
    render_plan,
    render_suggestion,
)
from tradingagents.optimizer.risk import RISK_MODELS, levels_json, risk_level
from tradingagents.optimizer.tracker import archive_ledger, read_ledger, start_ledger, update_ledger, write_ledger
from tradingagents.optimizer.universe import (
    ETFS,
    GROUPS,
    THEME_GROUPS,
    describe,
    read_config,
    tickers_for,
)
from tradingagents.portfolio import load_portfolio

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


class Job:
    """A background task the page polls: its progress, result or error."""

    def __init__(self):
        self.running, self.progress, self.result, self.error, self.log = False, "", None, None, []

    def start(self, work) -> bool:
        if self.running:
            return False
        self.running, self.progress, self.result, self.error, self.log = True, "starting", None, None, []

        def run():
            try:
                self.result = work(self)
            except Exception as exc:  # shown on the page rather than lost in a thread
                logger.exception("background job failed")
                self.error = str(exc)
            finally:
                self.running = False

        threading.Thread(target=run, daemon=True).start()
        return True

    def state(self) -> dict:
        return {"running": self.running, "progress": self.progress, "result": self.result,
                "error": self.error, "log": self.log}


def _finite(value):
    """NaN and infinity as null: they are not JSON, and a browser refuses the whole response."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


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
        self.ratings: dict[str, str] = {}
        self.ratings_date: str | None = None
        self.last_targets: list[str] = []
        self.last_rows: dict[str, dict] = {}
        self._memo: dict[str, tuple[str, object]] = {}   # what was computed, and from which inputs

    def read_json(self, path: Path, default):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def universe(self) -> list[str]:
        return tickers_for(read_config(self.universe_path), self.cache_dir)

    @staticmethod
    def sleeve(body: dict) -> str:
        sleeve = body.get("sleeve") or "investing"
        if sleeve not in SLEEVES:
            raise ValueError(f"a plan is made for one sleeve ({', '.join(SLEEVES)}), not {sleeve}")
        return sleeve

    # --- settings ---

    def risk(self, sleeve: str = "investing", level: int | None = None) -> dict:
        """The sleeve's risk level (or the given one), model and growth target, and every level."""
        s = self.store.settings(sleeve)
        return {**levels_json()[risk_level(level or s["risk_level"]).level - 1], "levels": levels_json(),
                "sleeve": sleeve, "model": s["risk_model"], "models": list(RISK_MODELS),
                "target_growth": s["target_growth"]}

    def settings(self, body: dict) -> OptimizerSettings:
        s = self.store.settings(self.sleeve(body))
        level = risk_level(body["risk_level"] if body.get("risk_level") else s["risk_level"])
        settings = OptimizerSettings(loss_limit=level.loss_limit, risk_model=s["risk_model"],
                                     target_growth=s["target_growth"])
        if body.get("max_budget_share"):
            settings.max_budget_share = float(body["max_budget_share"])
        if "loss_limit" in body:   # a one-off override from the command line, never saved
            settings.loss_limit = None if body["loss_limit"] is None else float(body["loss_limit"])
        if s["return_model"] == "forecast":
            from tradingagents.research.engine import expected_returns, latest_run

            # The model chosen for the sleeve, not whichever forecast happened to run last.
            run = latest_run(self.store.con, horizon=s["forecast_horizon"], model=s["forecast_model"])
            if run is None:
                raise ValueError(f"The {self.sleeve(body)} sleeve is set to use {s['forecast_model']} forecasts over "
                                 f"{s['forecast_horizon']} sessions, but there is no such run. Run one under Forecasts, "
                                 f"or switch the plan back to the standard rule.")
            run["calibration"] = json.loads(run["calibration_json"])
            settings.return_model, settings.forecast_horizon, settings.forecast_run = "forecast", run["horizon"], run["id"]
            settings.forecast_mu = expected_returns(run, [f["ticker"] for f in run["forecasts"]])
        return settings

    def gateway(self) -> Gateway:
        g = self.store.settings("global")
        prices = {k: Price(**v) for k, v in g["ai_prices"].items()}
        return Gateway(self.store.con, Policy(Budget(**g["ai_budget"]), prices, list(g["ai_fallbacks"])), self.owner)

    # --- planning ---

    def _closes(self, context, schedule, universe, settings):
        owned = list(dict.fromkeys([*(p.ticker.upper() for p in context.positions), *schedule,
                                    settings.cash_ticker]))
        proxies = [p for t, p in settings.proxies.items() if t in owned or t in universe]
        return fetch_closes([*owned, *universe, *proxies, settings.cash_proxy, *settings.scenario_references],
                            required=owned, cache_dir=self.cache_dir)

    def _inputs(self, *parts) -> str:
        """A fingerprint of everything a computed page section depends on: the stored prices,
        the books, the settings and the files it reads. Same fingerprint, same answer."""
        market_store = local_store.store(self.cache_dir)
        last_event = self.store.con.execute("SELECT MAX(id) FROM events").fetchone()[0]
        voids = self.store.con.execute("SELECT COUNT(*) FROM event_voids").fetchone()[0]
        files = [f.stat().st_mtime if f.exists() else 0 for f in (self.schedule_path, self.universe_path)]
        settings = {sl: self.store.settings(sl) for sl in SLEEVES}
        return json.dumps([market_store.bars_checked_at() if market_store else None, last_event, voids, files,
                           settings, self.ratings_date, _today(), *parts], sort_keys=True, default=str)

    def _remember(self, name: str, inputs: str, make):
        hit = self._memo.get(name)
        if hit and hit[0] == inputs:
            return hit[1]
        value = make()
        self._memo[name] = (inputs, value)
        return value

    def compute(self, body: dict, ratings: dict | None = None) -> dict:
        """The sleeve's plan and orders, and everything needed to save them; nothing is stored.
        Computed again only when an input changed (see ``_inputs``)."""
        key = self._inputs(body, ratings, self.ratings if body.get("use_ai") else None)
        return self._remember(f"plan:{json.dumps(body, sort_keys=True, default=str)}", key,
                              lambda: self._compute(body, ratings))

    def _compute(self, body: dict, ratings: dict | None = None) -> dict:
        sleeve = self.sleeve(body)
        context, cash = self.store.portfolio(sleeve, _today())
        if not context.positions and cash.total <= 0:
            raise ValueError(f"The {sleeve} sleeve holds nothing yet, so there is nothing to plan. Fund it from "
                             f"Investing first (Overview > Fund the Trading sleeve).")
        # Recurring contributions go to Investing unless configured otherwise.
        schedule = (load_schedule(self.schedule_path) if self.schedule_path.exists() else {}) \
            if sleeve == "investing" else {}
        settings = self.settings(body)
        if ratings is None:
            ratings = self.ratings if body.get("use_ai") and self.ratings_date == _today() else None
        universe = self.universe() if body.get("scan", True) else []
        closes = self._closes(context, schedule, universe, settings)
        plan = build_plan(context, schedule, settings, ratings=ratings, universe=universe, closes=closes)
        self.last_targets = [r.ticker for r in plan.rows if r.target > 1e-4]
        self.last_rows = {r.ticker: asdict(r) for r in [*plan.rows, plan.cash]}
        # Buys may spend only settled cash not already reserved (T+1, cash account).
        suggestion = daily_orders(plan, settled_cash=max(cash.available, 0.0))
        return {"sleeve": sleeve, "plan": plan, "suggestion": suggestion, "settings": settings, "closes": closes,
                "universe": universe, "ratings": ratings, "cash": cash, "body": body}

    def save_official(self, c: dict) -> dict:
        """Store a computed plan as the official recommendation for its session, and its report."""
        plan, suggestion, settings, sleeve = c["plan"], c["suggestion"], c["settings"], c["sleeve"]
        market_store = local_store.store(self.cache_dir)
        fetched = (market_store.bars_checked_at() if market_store else None) or datetime.now().astimezone().isoformat()
        optimized = [r.ticker for r in plan.rows if not r.screened]
        tickers = [*optimized, plan.cash.ticker, settings.cash_proxy, *settings.scenario_references,
                   *settings.proxies.values()]
        with self.lock:
            rec = recommendations.save(
                self.store.con, account_id=self.store.ledger.account_id, sleeve=sleeve, plan=plan,
                # The capital amounts in force are frozen with the record: later edits never change it.
                suggestion=suggestion, settings={**asdict(settings), "capital": capital.snapshot(self.store)},
                closes=c["closes"], universe=c["universe"],
                info_cutoff=c["closes"].index[-1].date().isoformat(), market_data_at=fetched,
                ai={"ratings": c["ratings"] or {}, "ratings_date": self.ratings_date}, price_tickers=tickers)
        # Next to the records it belongs to, so a second set of books never mixes its reports in.
        folder = self.store.folder / "recommendations"
        folder.mkdir(parents=True, exist_ok=True)
        report = folder / f"{rec['session']}_{sleeve}_{rec['id'][:8]}.md"
        report.write_text(f"Official recommendation {rec['id']} for {rec['session']} ({sleeve}); "
                          f"earliest execution {rec['earliest_execution']}.\n\n"
                          + render_suggestion(suggestion, _today()) + "\n" + render_plan(plan), encoding="utf-8")
        return {**rec, "report": str(report)}

    def plan(self, body: dict, save: bool = False) -> dict:
        """A preview of the sleeve's plan for the page, or with ``save`` its official recommendation."""
        c = self.compute(body)
        data = asdict(c["plan"])
        data["settings"]["frontier"] = list(c["settings"].frontier)
        cash = c["cash"]
        out = {"date": _today(), "sleeve": c["sleeve"], "plan": data,
               "risk": self.risk(c["sleeve"], body.get("risk_level")), "suggestion": asdict(c["suggestion"]),
               "ai_used": bool(c["ratings"]), "preview": not save,
               "cash": {**asdict(cash), "available": cash.available},
               "info_cutoff": c["closes"].index[-1].date().isoformat()}
        if save:
            out["saved"] = self.save_official(c)
        out.update(self.overview_extras(c))
        return out

    def overview_extras(self, c: dict) -> dict:
        """What the overview shows beside the plan: the session its orders are for, recurring
        investments, the account by sleeve, and how fresh each input is."""
        import pytz

        from tradingagents.optimizer.auto_invest import recommend

        now = datetime.now(pytz.utc)
        session = recommendations.session_for(now)
        extras = {"orders_session": session, "orders_session_is_today": session == _today(),
                  "returns_from": ({"model": "forecast", "run": c["settings"].forecast_run,
                                    "forecast_model": self.store.get(c["sleeve"], "forecast_model"),
                                    "horizon": c["settings"].forecast_horizon}
                                   if c["settings"].return_model == "forecast" else {"model": "vol_prior"})}
        if c["sleeve"] == "investing":
            current = load_schedule(self.schedule_path) if self.schedule_path.exists() else {}
            # The planned Investing purchase per session (Settings > Capital), not a deposit.
            budget = capital.load(self.store).daily_investment_usd
            extras["auto_invest"] = asdict(recommend(c["plan"], budget, current))
        books = self.books()
        extras["sleeves"] = {k: {"equity": v["equity"], "cash": v["cash"], "positions": len(v["positions"])}
                             for k, v in books["sleeves"].items()}
        last_rec = next(iter(recommendations.history(self.store.con, self.store.ledger.account_id, limit=1)), None)
        run = self.store.con.execute("SELECT created_at, model, horizon, info_cutoff FROM forecast_runs"
                                     " WHERE universe = 'etf' ORDER BY created_at DESC LIMIT 1").fetchone()
        market_store = local_store.store(self.cache_dir)
        extras["freshness"] = {
            "prices_through": c["closes"].index[-1].date().isoformat(),
            "prices_checked": market_store.bars_checked_at() if market_store else None,
            "last_official": {k: last_rec[k] for k in ("id", "session", "published_at", "status")} if last_rec else None,
            "last_forecast": dict(run) if run else None,
            "last_event": books["events"][0]["recorded_at"] if books["events"] else None,
        }
        return extras

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

        def work(job):
            ratings, failures = collect_ratings(tickers, self.config, analysts, gateway=self.gateway(),
                                                allow_paid=True, est_usd=g["ai_estimates"]["rating"],
                                                progress=lambda t, s: job.log.append(f"{t}: {s}"))
            self.ratings, self.ratings_date = ratings, _today()
            job.log += [f"{t} failed: {why}" for t, why in failures]
            return {"ratings": ratings, "date": self.ratings_date, "failures": len(failures)}

        return self.ai.start(work)

    def ai_usage(self) -> dict:
        rows = [dict(r) for r in self.store.con.execute("SELECT * FROM ai_usage ORDER BY id DESC LIMIT 50")]
        return {"remaining": self.gateway().remaining(), "recent": rows,
                "settings": self.store.settings("global")}

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
            tickers = [t for t in dict.fromkeys([*ETFS, *read_config(self.universe_path)["extra"], *held])
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
        tickers = [t for t in dict.fromkeys([*ETFS, *read_config(self.universe_path)["extra"], *held])
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
        return {"settings": capital.snapshot(self.store), "history": capital.history(self.store, 20),
                "proposal": capital.reallocation(self.store, today, prices),
                "daily_plan": capital.daily_plan(self.store, date.today())}

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
                    rep = day / "reports"
                    if rep.is_dir():
                        found.append({"date": day.name, "files": [{"name": f.stem.replace("_", " "),
                                                                   "path": str(f.resolve())}
                                                                  for f in sorted(rep.glob("*.md"))]})
            from tradingagents.memory import TradingMemoryLog
            ratings = [{"date": e["date"], "rating": e["rating"]} for e in TradingMemoryLog(self.config).load_entries()
                       if e.get("ticker", "").upper() == symbol]
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

    def report(self, path: str) -> str:
        """A saved analysis report, only from the report folders."""
        target = Path(path).resolve()
        if target.suffix != ".md" or not any(root in target.parents for root in self.report_roots()):
            raise ValueError("not a saved report")
        return target.read_text(encoding="utf-8")

    # --- research ---

    def start_backtest(self, body: dict) -> bool:
        settings = self.settings(body)
        start = body.get("start") or "2012-01-03"
        context, _ = self.store.portfolio(self.sleeve(body), _today())

        def work(job):
            job.progress = "downloading prices"
            held = [p.ticker.upper() for p in context.positions]
            # ETFs and your holdings only: today's S&P 500 list replayed into the past
            # would only contain the companies that went on to win (survivorship).
            extra = read_config(self.universe_path)["extra"]
            candidates = list(dict.fromkeys([*ETFS, *extra, *held]))
            proxies = [p for t, p in settings.proxies.items() if t in candidates]
            closes = fetch_closes([*candidates, *proxies, settings.cash_proxy, settings.cash_ticker, "VTI", "SPY",
                                   *settings.scenario_references],
                                  required=[settings.cash_proxy], cache_dir=self.cache_dir)
            prices = {t: float(closes[t].dropna().iloc[-1]) for t in held if t in closes}
            values = {p.ticker.upper(): p.quantity * prices.get(p.ticker.upper(), 0.0) for p in context.positions}
            total = sum(values.values()) + float(context.cash or 0.0)

            def progress(n, of):
                job.progress = f"rebalance {n} of {of}"

            # Shown against VTI, SPY and SGOV only, each starting from the sleeve's value today.
            result = run_backtest(closes, [t for t in candidates if t in closes], settings, start=start,
                                  progress=progress)
            data = asdict(result)
            data["capital"] = total
            data["sgov_start"] = str(closes["SGOV"].first_valid_index().date()) if "SGOV" in closes else None
            data["report"] = render_backtest(result)
            return data

        return self.backtest.start(work)

    # --- paper tracker (replaced by the Day 1 record in phase 6) ---

    def _tracker_closes(self, ledger: dict | None, universe: list[str], settings: OptimizerSettings):
        held = [p.ticker.upper() for p in load_portfolio(self.portfolio_path).positions]
        books = [t for b in (ledger or {}).get("books", {}).values() for t in b["shares"]]
        schedule = list(load_schedule(self.schedule_path)) if self.schedule_path.exists() else []
        proxies = list(settings.proxies.values())
        return fetch_closes([*held, *books, *schedule, *universe, *proxies, settings.cash_proxy, settings.cash_ticker,
                             *settings.scenario_references],
                            required=[settings.cash_ticker], cache_dir=self.cache_dir)

    def _start_work(self, body: dict):
        settings = self.settings(body)

        def work(job):
            job.progress = "downloading prices"
            universe = self.universe()
            closes = self._tracker_closes(None, universe, settings)
            ledger = start_ledger(load_portfolio(self.portfolio_path), settings, closes, body.get("start") or None)
            archive_ledger(self.tracker_path)  # a restart keeps the old record beside the new one
            return self._run_tracker(job, ledger, universe, settings, closes)

        return work

    def _update_work(self, ledger: dict):
        def work(job):
            job.progress = "downloading prices"
            settings = self.settings({})
            universe = self.universe()
            closes = self._tracker_closes(ledger, universe, settings)
            return self._run_tracker(job, ledger, universe, settings, closes)

        return work

    def start_tracker(self, body: dict) -> bool:
        return self.tracker.start(self._start_work(body))

    def update_tracker(self) -> bool:
        ledger = read_ledger(self.tracker_path)
        return ledger is not None and self.tracker.start(self._update_work(ledger))

    def track_now(self) -> dict:
        """Record every final day not yet in the ledger, starting it today if there is none."""
        ledger = read_ledger(self.tracker_path)
        return (self._update_work(ledger) if ledger else self._start_work({}))(Job())

    def _run_tracker(self, job, ledger, universe, settings, closes):
        # Another process (a second tab's server, the CLI) may be on the same days.
        if not self.store.claim("paper-tracker", self.owner):
            raise ValueError("the paper tracker is being updated by another process; try again in a minute")
        try:
            schedule = load_schedule(self.schedule_path) if self.schedule_path.exists() else {}

            def progress(n, of):
                job.progress = f"day {n} of {of}"

            ledger = update_ledger(ledger, schedule, settings, closes, universe, progress=progress)
            ledger["risk_level"] = self.store.get("investing", "risk_level")
            write_ledger(self.tracker_path, ledger)
            return {"days": len(ledger["history"]), "last_day": ledger["last_day"]}
        finally:
            self.store.release("paper-tracker", self.owner)

    def tracker_state(self) -> dict:
        ledger = read_ledger(self.tracker_path)
        if ledger:
            names = {t for b in ledger["books"].values() for t in b["shares"]}
            prices = latest_prices(self.cache_dir, names)
            ledger = {**ledger, "holdings": {
                name: sorted(([t, round(q, 6), round(q * prices.get(t, 0.0), 2), describe(t)]
                              for t, q in book["shares"].items()), key=lambda x: -x[2])
                for name, book in ledger["books"].items()}}
        return {**self.tracker.state(), "ledger": ledger}

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
                out = {"event": led.contribute(float(body["amount"]), day, sleeve=body.get("sleeve", "investing"),
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
                rows = csv_import.preview(led, body["csv"], body.get("sleeve", "investing"))
                out = {"rows": [asdict(r) for r in rows]}
                if body.get("commit"):
                    out["result"] = csv_import.commit(led, rows, body.get("sleeve", "investing"))
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
                elif url.path == "/api/report":
                    self._send(HTTPStatus.OK, app.report(query.get("path", "")).encode(), "text/plain; charset=utf-8")
                elif url.path == "/api/state":
                    context, _ = app.store.portfolio("combined", _today())
                    config = read_config(app.universe_path)
                    self._json({"portfolio": context.model_dump(exclude_none=True),
                                "schedule": app.read_json(app.schedule_path, {"daily": {}, "weekly": {}}),
                                "universe": config, "groups": GROUPS,
                                "risk": app.risk(query.get("sleeve", "investing"))})
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
                elif url.path == "/api/backtest":
                    self._json(app.backtest.state())
                elif url.path == "/api/tracker":
                    self._json(app.tracker_state())
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
                if self.path == "/api/plan":
                    self._json(app.plan(body))
                elif self.path == "/api/recommendations/save":
                    self._json(app.plan(body, save=True))
                elif self.path == "/api/decision":
                    if body.get("decision") not in ("accepted", "rejected", "deferred", "partial"):
                        raise ValueError("decision must be accepted, rejected, deferred or partial")
                    with app.lock:
                        decision = recommendations.decide(app.store.con, body["id"], body["decision"], body.get("note"))
                    self._json({"decision": decision, "note": "Recorded. Nothing is bought or sold until you "
                                                              "record the fills."})
                elif self.path.startswith("/api/ledger/"):
                    if self.path == "/api/ledger/trading-preview":
                        tickers = {e["ticker"] for e in app.store.ledger.events() if e["ticker"]}
                        self._json(app.store.trading_assignment_preview(
                            float(body.get("amount") or capital.load(app.store).trading_capital_usd),
                            latest_prices(app.cache_dir, tickers), _today()))
                    else:
                        self._json(app.ledger_write(self.path, body))
                elif self.path == "/api/ratings":
                    self._started(app.start_ai(body.get("analysts")), "Rating")
                elif self.path == "/api/backtest":
                    self._started(app.start_backtest(body), "A backtest")
                elif self.path == "/api/forecasts/run":
                    self._started(app.start_forecast(body), "A forecast run")
                elif self.path == "/api/portfolio":
                    wanted = {p["ticker"].upper(): float(p["quantity"]) for p in body.get("positions", [])}
                    with app.lock:
                        events = app.store.adjust_holdings(wanted, _today(), note=body.get("note"))
                        app.after_ledger_change()
                    self._json({"ok": True, "adjustments": events})
                elif self.path == "/api/schedule":
                    for period in ("daily", "weekly"):
                        for amount in (body.get(period) or {}).values():
                            float(amount)
                    app.schedule_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
                    self._json({"ok": True})
                elif self.path == "/api/risk":
                    sleeve = app.sleeve(body)
                    with app.lock:
                        if body.get("level"):
                            app.store.set(sleeve, "risk_level", risk_level(int(body["level"])).level)
                        if body.get("model"):
                            if body["model"] not in RISK_MODELS:
                                raise ValueError(f"risk model must be one of {', '.join(RISK_MODELS)}")
                            app.store.set(sleeve, "risk_model", body["model"])
                        if body.get("return_model"):
                            if body["return_model"] not in ("vol_prior", "forecast"):
                                raise ValueError("return model must be vol_prior or forecast")
                            app.store.set(sleeve, "return_model", body["return_model"])
                        if body.get("forecast_model"):
                            from tradingagents.research.engine import factory
                            factory(body["forecast_model"])   # refuses an unknown model
                            app.store.set(sleeve, "forecast_model", body["forecast_model"])
                        if body.get("forecast_horizon"):
                            if int(body["forecast_horizon"]) not in (5, 20, 60):
                                raise ValueError("the forecast horizon is 5, 20 or 60 sessions")
                            app.store.set(sleeve, "forecast_horizon", int(body["forecast_horizon"]))
                        if "target_growth" in body:
                            target = body["target_growth"]
                            if target is not None and not -0.5 < float(target) < 1.0:
                                raise ValueError("growth target must be a yearly rate such as 0.08")
                            app.store.set(sleeve, "target_growth", None if target is None else float(target))
                    self._json(app.risk(sleeve))
                elif self.path == "/api/settings":
                    with app.lock:
                        for key, value in (body.get("values") or {}).items():
                            app.store.set(body.get("scope", "global"), key, value)
                    self._json({"ok": True})
                elif self.path == "/api/universe":
                    groups = [g for g in body.get("groups", []) if g in GROUPS]
                    extra = [str(t).strip().upper() for t in body.get("extra", []) if str(t).strip()]
                    app.universe_path.write_text(json.dumps({"groups": groups, "extra": list(dict.fromkeys(extra))},
                                                            indent=2), encoding="utf-8")
                    self._json({"ok": True})
                elif self.path == "/api/market/explain":
                    self._started(app.start_explain(), "The market explanation")
                elif self.path == "/api/decisions/run":
                    self._started(app.start_decisions(body), "A decision run")
                elif self.path == "/api/capital":
                    with app.lock:
                        capital.update(app.store, body.get("values") or {}, source="user")
                    self._json(app.capital_state())
                elif self.path == "/api/capital/transfer":
                    # An accepted reallocation: recorded as an ordinary settled-cash transfer.
                    with app.lock:
                        app.store.ledger.transfer(body["from"], body["to"], _today(), cash=float(body["cash"]),
                                                  note="capital reallocation (accepted proposal)")
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
                elif self.path == "/api/tracker/start":
                    # Starting archives nothing by itself, but a page that failed to read the record
                    # must not replace it: an existing record is only replaced after "Start over".
                    if read_ledger(app.tracker_path) is not None:
                        raise ValueError("a paper record already exists; use Start over to begin a new one")
                    self._started(app.start_tracker(body), "The tracker")
                elif self.path == "/api/tracker/update":
                    if read_ledger(app.tracker_path) is None:
                        raise ValueError("there is no paper record to update yet")
                    self._started(app.update_tracker(), "The tracker")
                elif self.path == "/api/tracker/reset":
                    if app.tracker.running:
                        raise ValueError("the tracker is updating; try again when it finishes")
                    archived = archive_ledger(app.tracker_path)
                    self._json({"ok": True, "archived_to": str(archived) if archived else None})
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
