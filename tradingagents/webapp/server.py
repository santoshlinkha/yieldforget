"""A local web app for the daily optimization: one page and a small JSON API.

It runs on this machine because it needs the price feed and the LLM keys in
.env, and it serves only localhost: the page edits your holdings files and can
spend LLM credit, so nothing else on the network should reach it.

Prices are downloaded once per day into the cache directory and reused. AI
ratings and backtests run in background threads, since they take minutes, and
the page polls for their progress. Ratings are kept in the memory log, so a
second request on the same day reads them back for free.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tradingagents.dataflows.vendors.index_members import get_sp500_members
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.optimizer.ai_views import DEFAULT_ANALYSTS, collect_ratings
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
from tradingagents.optimizer.risk import RISK_MODELS, levels_json, read_choice, risk_level, write_choice
from tradingagents.optimizer.tracker import archive_ledger, read_ledger, start_ledger, update_ledger, write_ledger
from tradingagents.optimizer.universe import (
    ETFS,
    GROUPS,
    THEME_GROUPS,
    describe,
    read_config,
    tickers_for,
)
from tradingagents.portfolio import PortfolioContext, load_portfolio

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


class App:
    """What the handlers share: the files, the cache and the background jobs."""

    def __init__(self, portfolio_path: Path, schedule_path: Path, universe_path: Path, config: dict):
        self.portfolio_path = portfolio_path
        self.schedule_path = schedule_path
        self.universe_path = universe_path
        self.config = config
        self.cache_dir = Path(config["data_cache_dir"]) / "optimizer"
        self.tracker_path = portfolio_path.parent / "paper_tracker.json"
        self.risk_path = portfolio_path.parent / "my_risk.json"
        self.ai, self.backtest, self.tracker, self.explain = Job(), Job(), Job(), Job()
        self.ratings: dict[str, str] = {}
        self.ratings_date: str | None = None
        self.last_targets: list[str] = []

    def read_json(self, path: Path, default):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def universe(self) -> list[str]:
        return tickers_for(read_config(self.universe_path), self.cache_dir)

    def risk(self, level: int | None = None) -> dict:
        """The given risk level, or the saved one, with the saved model and growth target and
        every level for the page."""
        choice = read_choice(self.risk_path)
        return {**levels_json()[risk_level(level or choice.level).level - 1], "levels": levels_json(),
                "model": choice.model, "models": list(RISK_MODELS), "target_growth": choice.target_growth}

    def settings(self, body: dict) -> OptimizerSettings:
        """Settings at the requested risk level, or the saved one, with the saved model and target."""
        choice = read_choice(self.risk_path)
        level = risk_level(body["risk_level"] if body.get("risk_level") else choice.level)
        settings = OptimizerSettings(loss_limit=level.loss_limit, risk_model=choice.model,
                                     target_growth=choice.target_growth)
        if body.get("max_budget_share"):
            settings.max_budget_share = float(body["max_budget_share"])
        return settings

    def plan(self, body: dict) -> dict:
        portfolio = load_portfolio(self.portfolio_path)
        schedule = load_schedule(self.schedule_path) if self.schedule_path.exists() else {}
        settings = self.settings(body)
        today = date.today().isoformat()
        ratings = self.ratings if body.get("use_ai") and self.ratings_date == today else None
        universe = self.universe() if body.get("scan", True) else []

        plan = build_plan(portfolio, schedule, settings, ratings=ratings, universe=universe,
                          cache_dir=self.cache_dir)
        self.last_targets = [r.ticker for r in plan.rows if r.target > 1e-4]
        # The book's cash is the account's settled cash: the only money a buy may spend (T+1).
        suggestion = daily_orders(plan, settled_cash=float(portfolio.cash or 0.0))

        out_dir = Path(self.config["results_dir"]) / "optimizer"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = out_dir / f"{today}.md"
        saved.write_text(render_suggestion(suggestion, today) + "\n" + render_plan(plan), encoding="utf-8")

        data = asdict(plan)
        data["settings"]["frontier"] = list(settings.frontier)
        return {"date": today, "plan": data, "risk": self.risk(body.get("risk_level")), "suggestion": asdict(suggestion), "saved_to": str(saved),
                "ai_used": bool(ratings)}

    def start_ai(self, analysts) -> bool:
        portfolio = load_portfolio(self.portfolio_path)
        cash = OptimizerSettings().cash_ticker
        # Holdings and whatever the last plan wanted to buy: rating the whole scan
        # list would cost one full analysis per fund.
        tickers = [t for t in dict.fromkeys([*(p.ticker.upper() for p in portfolio.positions),
                                             *self.last_targets]) if t != cash]

        def work(job):
            ratings, failures = collect_ratings(tickers, self.config, analysts,
                                                progress=lambda t, s: job.log.append(f"{t}: {s}"))
            self.ratings, self.ratings_date = ratings, date.today().isoformat()
            job.log += [f"{t} failed: {why}" for t, why in failures]
            return {"ratings": ratings, "date": self.ratings_date, "failures": len(failures)}

        return self.ai.start(work)

    def start_backtest(self, body: dict) -> bool:
        settings = self.settings(body)
        start = body.get("start") or "2012-01-03"
        portfolio = load_portfolio(self.portfolio_path)

        def work(job):
            job.progress = "downloading prices"
            held = [p.ticker.upper() for p in portfolio.positions]
            # ETFs and your holdings only: today's S&P 500 list replayed into the past
            # would only contain the companies that went on to win (survivorship).
            extra = read_config(self.universe_path)["extra"]
            candidates = list(dict.fromkeys([*ETFS, *extra, *held]))
            proxies = [p for t, p in settings.proxies.items() if t in candidates]
            closes = fetch_closes([*candidates, *proxies, settings.cash_proxy, settings.cash_ticker, "VTI", "BND",
                                   *settings.scenario_references],
                                  required=[settings.cash_proxy], cache_dir=self.cache_dir)
            prices = {t: float(closes[t].dropna().iloc[-1]) for t in held if t in closes}
            values = {p.ticker.upper(): p.quantity * prices.get(p.ticker.upper(), 0.0) for p in portfolio.positions}
            total = sum(values.values()) + float(portfolio.cash or 0.0)
            mix = {t: v / total for t, v in values.items() if t != settings.cash_ticker and v > 0} if total else {}

            def progress(n, of):
                job.progress = f"rebalance {n} of {of}"

            result = run_backtest(closes, [t for t in candidates if t in closes], settings, start=start,
                                  current_mix=mix, progress=progress)
            data = asdict(result)
            data["report"] = render_backtest(result)
            return data

        return self.backtest.start(work)

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
        schedule = load_schedule(self.schedule_path) if self.schedule_path.exists() else {}

        def progress(n, of):
            job.progress = f"day {n} of {of}"

        ledger = update_ledger(ledger, schedule, settings, closes, universe, progress=progress)
        ledger["risk_level"] = read_choice(self.risk_path).level
        write_ledger(self.tracker_path, ledger)
        return {"days": len(ledger["history"]), "last_day": ledger["last_day"]}

    def market(self) -> dict:
        """Recent moves from closing prices, and today's saved explanation if there is one."""
        sp500 = get_sp500_members(self.cache_dir)
        themes = [t for group in THEME_GROUPS.values() for t in group]
        closes = fetch_closes([*market_tickers(), *sp500, *themes], required=["SPY"], cache_dir=self.cache_dir)
        overview = market_overview(closes, sp500)
        path = Path(self.config["results_dir"]) / "market" / f"brief_{overview['as_of']}.json"
        brief = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return {"overview": overview, "brief": brief, "explain": self.explain.state()}

    def start_explain(self) -> bool:
        def work(job):
            job.progress = "reading prices"
            overview = self.market()["overview"]
            job.progress = "reading headlines and asking the AI"
            brief = market_brief(overview, self.config)
            save_json(Path(self.config["results_dir"]) / "market" / f"brief_{overview['as_of']}.json", brief)
            return {"as_of": overview["as_of"]}

        return self.explain.start(work)

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
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                config = read_config(app.universe_path)
                self._json({"portfolio": app.read_json(app.portfolio_path, {"positions": []}),
                            "schedule": app.read_json(app.schedule_path, {"daily": {}, "weekly": {}}),
                            "universe": config, "groups": GROUPS, "risk": app.risk()})
            elif self.path == "/api/ratings":
                self._json({**app.ai.state(), "ratings": app.ratings, "date": app.ratings_date})
            elif self.path == "/api/backtest":
                self._json(app.backtest.state())
            elif self.path == "/api/tracker":
                self._json(app.tracker_state())
            elif self.path == "/api/market":
                try:
                    self._json(app.market())
                except Exception as exc:  # a price outage is shown in the section, not a dropped request
                    logger.exception("market overview failed")
                    self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            elif self.path == "/api/market/explain":
                self._json(app.explain.state())
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                body = self._body()
                if self.path == "/api/plan":
                    self._json(app.plan(body))
                elif self.path == "/api/ratings":
                    started = app.start_ai(body.get("analysts") or list(DEFAULT_ANALYSTS))
                    self._json({"started": started}, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)
                elif self.path == "/api/backtest":
                    started = app.start_backtest(body)
                    self._json({"started": started}, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)
                elif self.path == "/api/portfolio":
                    PortfolioContext.model_validate(body)  # refuse a book the optimizer cannot read
                    app.portfolio_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
                    self._json({"ok": True})
                elif self.path == "/api/schedule":
                    for period in ("daily", "weekly"):
                        for amount in (body.get(period) or {}).values():
                            float(amount)
                    app.schedule_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
                    self._json({"ok": True})
                elif self.path == "/api/risk":
                    kwargs = {"target_growth": body["target_growth"]} if "target_growth" in body else {}
                    write_choice(app.risk_path, level=int(body["level"]) if body.get("level") else None,
                                 model=body.get("model"), **kwargs)
                    self._json(app.risk())
                elif self.path == "/api/universe":
                    groups = [g for g in body.get("groups", []) if g in GROUPS]
                    extra = [str(t).strip().upper() for t in body.get("extra", []) if str(t).strip()]
                    app.universe_path.write_text(json.dumps({"groups": groups, "extra": list(dict.fromkeys(extra))},
                                                            indent=2), encoding="utf-8")
                    self._json({"ok": True})
                elif self.path == "/api/market/explain":
                    started = app.start_explain()
                    self._json({"started": started}, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)
                elif self.path == "/api/tracker/start":
                    started = app.start_tracker(body)
                    self._json({"started": started}, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)
                elif self.path == "/api/tracker/update":
                    started = app.update_tracker()
                    self._json({"started": started}, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT)
                elif self.path == "/api/tracker/reset":
                    if app.tracker.running:
                        raise ValueError("the tracker is updating; try again when it finishes")
                    archived = archive_ledger(app.tracker_path)
                    self._json({"ok": True, "archived_to": str(archived) if archived else None})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, TypeError) as exc:
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
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
