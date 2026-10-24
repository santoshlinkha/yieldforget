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
import os
import threading
import uuid
from dataclasses import asdict
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tradingagents.dataflows.vendors.index_members import get_sp500_members
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.ledger import imports as csv_import
from tradingagents.ledger import recommendations
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
        self.ai, self.backtest, self.tracker, self.explain = Job(), Job(), Job(), Job()
        self.ratings: dict[str, str] = {}
        self.ratings_date: str | None = None
        self.last_targets: list[str] = []

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

    def compute(self, body: dict, ratings: dict | None = None) -> dict:
        """The sleeve's plan and orders, and everything needed to save them; nothing is stored."""
        sleeve = self.sleeve(body)
        context, cash = self.store.portfolio(sleeve, _today())
        if not context.positions and cash.total <= 0:
            raise ValueError(f"The {sleeve} sleeve holds nothing yet. Fund it with a transfer from Investing "
                             f"(Portfolio > Fund Trading) before planning it.")
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
        # Buys may spend only settled cash not already reserved (T+1, cash account).
        suggestion = daily_orders(plan, settled_cash=max(cash.available, 0.0))
        return {"sleeve": sleeve, "plan": plan, "suggestion": suggestion, "settings": settings, "closes": closes,
                "universe": universe, "ratings": ratings, "cash": cash, "body": body}

    def save_official(self, c: dict) -> dict:
        """Store a computed plan as the official recommendation for its session, and its report."""
        plan, suggestion, settings, sleeve = c["plan"], c["suggestion"], c["settings"], c["sleeve"]
        cache_file = self.cache_dir / f"closes_{_today()}.pkl"
        fetched = datetime.fromtimestamp(cache_file.stat().st_mtime).astimezone().isoformat() \
            if cache_file.exists() else datetime.now().astimezone().isoformat()
        optimized = [r.ticker for r in plan.rows if not r.screened]
        tickers = [*optimized, plan.cash.ticker, settings.cash_proxy, *settings.scenario_references,
                   *settings.proxies.values()]
        with self.lock:
            rec = recommendations.save(
                self.store.con, account_id=self.store.ledger.account_id, sleeve=sleeve, plan=plan,
                suggestion=suggestion, settings=settings, closes=c["closes"], universe=c["universe"],
                info_cutoff=c["closes"].index[-1].date().isoformat(), market_data_at=fetched,
                ai={"ratings": c["ratings"] or {}, "ratings_date": self.ratings_date}, price_tickers=tickers)
        folder = Path(self.config["results_dir"]) / "optimizer"
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
        return out

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
            closes = fetch_closes([*candidates, *proxies, settings.cash_proxy, settings.cash_ticker, "VTI", "BND",
                                   *settings.scenario_references],
                                  required=[settings.cash_proxy], cache_dir=self.cache_dir)
            prices = {t: float(closes[t].dropna().iloc[-1]) for t in held if t in closes}
            values = {p.ticker.upper(): p.quantity * prices.get(p.ticker.upper(), 0.0) for p in context.positions}
            total = sum(values.values()) + float(context.cash or 0.0)
            mix = {t: v / total for t, v in values.items() if t != settings.cash_ticker and v > 0} if total else {}

            def progress(n, of):
                job.progress = f"rebalance {n} of {of}"

            result = run_backtest(closes, [t for t in candidates if t in closes], settings, start=start,
                                  current_mix=mix, progress=progress)
            data = asdict(result)
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
        sp500 = get_sp500_members(self.cache_dir)
        themes = [t for group in THEME_GROUPS.values() for t in group]
        closes = fetch_closes([*market_tickers(), *sp500, *themes], required=["SPY"], cache_dir=self.cache_dir)
        overview = market_overview(closes, sp500)
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
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

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
                            float(body.get("amount") or app.store.get("trading", "initial_allocation")),
                            latest_prices(app.cache_dir, tickers), _today()))
                    else:
                        self._json(app.ledger_write(self.path, body))
                elif self.path == "/api/ratings":
                    self._started(app.start_ai(body.get("analysts")), "Rating")
                elif self.path == "/api/backtest":
                    self._started(app.start_backtest(body), "A backtest")
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
                elif self.path == "/api/tracker/start":
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
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
