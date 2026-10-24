import sys

import typer

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to analyze, e.g. NVDA or 0700.HK; skips the prompt"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models, depth and language also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show}
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run: market, sentiment, news, fundamentals; omit for all the asset type allows"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {"asset_type": kind.value, "portfolio": book, "run_id": run_id, "progress": show_progress,
              "selected_analysts": [a.value for a in chosen]}

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


@app.command()
def suggest(
    portfolio: str = typer.Option("my_portfolio.json", "--portfolio",
                                  help="Holdings file; its folder holds portfolio.sqlite3, the records"),
    schedule: str = typer.Option("my_buys.json", "--schedule", help="JSON file with recurring buys"),
    sleeve: str = typer.Option("investing", "--sleeve", help="investing or trading"),
    risk: int | None = typer.Option(None, "--risk", min=1, max=10,
                                    help="Risk level 1-10 for the sleeve, saved for next time; the saved level if omitted"),
    max_loss: float | None = typer.Option(None, "--max-loss",
                                          help="Stress-loss limit in percent for this run only; 0 for none"),
    model: str | None = typer.Option(None, "--model",
                                     help="How the stress loss is measured, saved for the sleeve: worst_case "
                                          "(every holding at its worst drop at once) or scenarios (the worst past "
                                          "crash replayed on the mix; broke its limit in backtests)"),
    target_growth: float | None = typer.Option(None, "--target-growth",
                                               help="Expected growth to aim for, percent a year (8 for 8%), saved "
                                                    "for the sleeve; the least risky mix that reaches it, within the "
                                                    "risk level. 0 clears it"),
    scan: bool = typer.Option(True, "--scan/--no-scan", help="Also consider the funds in the scan list"),
    universe: str = typer.Option("my_universe.json", "--universe", help="Scan list file; the built-in list if absent"),
    ai: bool = typer.Option(False, "--ai", help="Pay for AI ratings of the holdings (budgeted, cached) and tilt toward them"),
    analysts: str | None = typer.Option(None, "--analysts", help="Analysts for --ai, comma-separated; one by default"),
    save: bool = typer.Option(False, "--save", help="Save the plan as the official recommendation for its session"),
):
    """Print today's orders and the optimized mix behind them. A preview unless --save."""
    from datetime import date
    from pathlib import Path

    from rich.markdown import Markdown

    from tradingagents.optimizer.ai_views import collect_ratings
    from tradingagents.optimizer.plan import render_plan, render_suggestion
    from tradingagents.optimizer.risk import RISK_MODELS, risk_level
    from tradingagents.webapp.server import App

    try:
        portfolio_path = Path(portfolio).resolve()
        app_ = App(portfolio_path, Path(schedule).resolve(), Path(universe).resolve(), DEFAULT_CONFIG)
        store = app_.store
        if risk:
            store.set(sleeve, "risk_level", risk)
        if model:
            if model not in RISK_MODELS:
                raise ValueError(f"--model must be one of {', '.join(RISK_MODELS)}")
            store.set(sleeve, "risk_model", model)
        if target_growth is not None:
            store.set(sleeve, "target_growth", target_growth / 100 if target_growth else None)
        s = store.settings(sleeve)
        level = risk_level(s["risk_level"])
        body = {"sleeve": sleeve, "scan": scan}
        ratings = None
        if ai:
            g = store.settings("global")
            context, _ = store.portfolio(sleeve, date.today())
            tickers = [p.ticker.upper() for p in context.positions if p.ticker.upper() != "SGOV"][:int(g["ai_shortlist"])]
            ratings, failures = collect_ratings(
                tickers, DEFAULT_CONFIG,
                [a.strip() for a in analysts.split(",") if a.strip()] if analysts else g["ai_analysts"],
                gateway=app_.gateway(), allow_paid=True, est_usd=g["ai_estimates"]["rating"],
                progress=lambda t, msg: console.print(f"[dim]{t}: {msg}[/dim]"))
            for ticker, reason in failures:
                console.print(f"[yellow]no rating:[/yellow] {ticker}: {reason}")
        if max_loss is not None:
            body["loss_limit"] = max_loss / 100 if max_loss > 0 else None
        c = app_.compute(body, ratings=ratings or {})
        goal = "" if s["target_growth"] is None else f", growth target {s['target_growth']:.1%}"
        limit = c["plan"].settings.loss_limit
        console.print(f"[dim]{sleeve.capitalize()} sleeve, risk level {level.level} ({level.name}), stress-loss limit "
                      f"{'none' if limit is None else f'{limit:.1%}'} ({s['risk_model'].replace('_', ' ')}){goal}. "
                      f"Settled cash ${c['cash'].available:,.2f}, unsettled ${c['cash'].unsettled:,.2f}.[/dim]")
        console.print(Markdown(render_suggestion(c["suggestion"], date.today().isoformat())))
        console.print(Markdown(render_plan(c["plan"])))
        if save:
            rec = app_.save_official(c)
            console.print(f"[green]Saved official recommendation {rec['id']}[/green] for session {rec['session']}, "
                          f"earliest execution {rec['earliest_execution']}"
                          + (f", superseding {rec['supersedes']}" if rec["supersedes"] else "") + ".")
        else:
            console.print("[dim]Preview only: nothing was saved. Add --save to make it the official recommendation.[/dim]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def track(
    portfolio: str = typer.Option("my_portfolio.json", "--portfolio", help="JSON file with holdings and settled cash"),
    schedule: str = typer.Option("my_buys.json", "--schedule", help="JSON file with recurring buys"),
):
    """Record the paper portfolios for every closed trading day not yet recorded (run daily after 4:30 pm ET)."""
    from pathlib import Path

    from tradingagents.webapp.server import App

    portfolio_path = Path(portfolio).resolve()
    app_ = App(portfolio_path, Path(schedule).resolve(), portfolio_path.parent / "my_universe.json", DEFAULT_CONFIG)
    try:
        result = app_.track_now()
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"Paper tracker: {result['days']} days recorded, last {result['last_day']}.")


@app.command()
def web(
    portfolio: str = typer.Option("my_portfolio.json", "--portfolio", help="JSON file with holdings and settled cash"),
    schedule: str = typer.Option("my_buys.json", "--schedule", help="JSON file with recurring buys"),
    port: int = typer.Option(8765, "--port", help="Port on localhost"),
):
    """Open the daily optimizer web app on this machine."""
    import webbrowser

    from tradingagents.webapp import serve

    try:
        server = serve(portfolio, schedule, port)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    url = f"http://127.0.0.1:{port}/"
    console.print(f"Daily optimizer at {url} (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


book_app = typer.Typer(help="Your records: sleeves, cash, fills, transfers, imports and recommendations.",
                       no_args_is_help=True)
app.add_typer(book_app, name="book")
PORTFOLIO_OPT = typer.Option("my_portfolio.json", "--portfolio", help="Holdings file; its folder holds portfolio.sqlite3")


def _book_store(portfolio: str):
    from pathlib import Path

    from tradingagents.ledger.store import Store

    return Store(Path(portfolio).resolve().parent)


def _prices(store):
    from pathlib import Path

    from tradingagents.optimizer.data import latest_prices

    tickers = {e["ticker"] for e in store.ledger.events() if e["ticker"]}
    return latest_prices(Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer", tickers)


def _export(store, portfolio: str):
    from datetime import date
    from pathlib import Path

    store.export_holdings(Path(portfolio).resolve(), date.today())


def _book_errors(fn):
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None
    return wrapped


@book_app.command("status")
@_book_errors
def book_status(portfolio: str = PORTFOLIO_OPT):
    """Holdings, settled and unsettled cash, and value of each sleeve and the whole account."""
    from datetime import date

    from rich.table import Table

    store = _book_store(portfolio)
    prices = _prices(store)
    for sleeve in ("investing", "trading", "combined"):
        s = store.ledger.state(sleeve, date.today(), prices)
        table = Table(title=f"{sleeve.capitalize()}: ${s.equity:,.2f}")
        for col in ("Ticker", "Quantity", "Value"):
            table.add_column(col, justify="left" if col == "Ticker" else "right")
        for t, q in s.positions.items():
            table.add_row(t, f"{q:,.6g}", f"${s.values[t]:,.2f}" if t in s.values else "no price")
        table.add_row("cash settled", "", f"${s.cash.settled:,.2f}")
        table.add_row("cash unsettled", "", f"${s.cash.unsettled:,.2f}")
        if s.cash.reserved:
            table.add_row("cash reserved", "", f"${s.cash.reserved:,.2f}")
        console.print(table)
    console.print("[dim]Sleeves are bookkeeping inside one IRA; the broker's rules apply to the whole account.[/dim]")


@book_app.command("contribute")
@_book_errors
def book_contribute(amount: float, sleeve: str = "investing", day: str = typer.Option(None, "--date"),
                    settle_date: str = typer.Option(None, "--settle-date"), portfolio: str = PORTFOLIO_OPT):
    """Record new money deposited into the IRA (an external flow, not profit)."""
    from datetime import date

    store = _book_store(portfolio)
    store.ledger.contribute(amount, day or date.today(), sleeve=sleeve, settle_date=settle_date)
    _export(store, portfolio)
    console.print(f"Recorded a ${amount:,.2f} contribution to {sleeve}.")


@book_app.command("fill")
@_book_errors
def book_fill(side: str, ticker: str, quantity: float, price: float, sleeve: str = "investing",
              day: str = typer.Option(None, "--date"), fee: float = 0.0,
              broker_record: bool = typer.Option(False, "--broker-record",
                                                 help="Keep it even if the books say it could not be paid for"),
              portfolio: str = PORTFOLIO_OPT):
    """Record a buy or sell that happened at the broker."""
    from datetime import date

    store = _book_store(portfolio)
    _, warnings = store.ledger.fill(sleeve, side.lower(), ticker, quantity, price, day or date.today(), fee=fee,
                                    strict=not broker_record)
    _export(store, portfolio)
    console.print(f"Recorded: {side} {quantity:g} {ticker.upper()} at ${price:,.4f} in {sleeve}.")
    for w in warnings:
        console.print(f"[yellow]{w}[/yellow]")


@book_app.command("transfer")
@_book_errors
def book_transfer(from_sleeve: str, to_sleeve: str, ticker: str = None, quantity: float = 0.0,
                  price: float = None, cash: float = 0.0, day: str = typer.Option(None, "--date"),
                  portfolio: str = PORTFOLIO_OPT):
    """Move shares or settled cash between sleeves. The account total does not change."""
    from datetime import date

    store = _book_store(portfolio)
    if ticker and price is None:
        price = _prices(store).get(ticker.upper())
        if price is None:
            raise ValueError(f"no cached price for {ticker}; pass --price")
    store.ledger.transfer(from_sleeve, to_sleeve, day or date.today(), ticker=ticker, quantity=quantity,
                          cash=cash, price=price)
    _export(store, portfolio)
    what = f"{quantity:g} {ticker.upper()} (${quantity * price:,.2f})" if ticker else f"${cash:,.2f} cash"
    console.print(f"Moved {what} from {from_sleeve} to {to_sleeve}.")


@book_app.command("fund-trading")
@_book_errors
def book_fund_trading(amount: float = typer.Option(None, help="Dollars to assign; the Trading setting if omitted"),
                      confirm: bool = typer.Option(False, "--confirm", help="Make the transfers shown"),
                      portfolio: str = PORTFOLIO_OPT):
    """Preview, then with --confirm make, the transfers that fund the Trading sleeve from Investing."""
    from datetime import date

    store = _book_store(portfolio)
    amount = amount or store.get("trading", "initial_allocation")
    preview = store.trading_assignment_preview(amount, _prices(store), date.today())
    for step in preview["steps"]:
        if "cash" in step:
            console.print(f"- ${step['cash']:,.2f} of settled cash")
        else:
            console.print(f"- {step['quantity']:g} {step['ticker']} (${step['value']:,.2f})")
    if preview["short_by"] > 0:
        console.print(f"[yellow]Short by ${preview['short_by']:,.2f}: Investing has no more settled cash or SGOV. "
                      f"Move other holdings with `book transfer`, or fund less.[/yellow]")
    console.print(f"[dim]{preview['note']}[/dim]")
    if not confirm:
        console.print("[dim]Preview only. Add --confirm to make these transfers.[/dim]")
        return
    for step in preview["steps"]:
        if "cash" in step:
            store.ledger.transfer("investing", "trading", date.today(), cash=step["cash"])
        else:
            store.ledger.transfer("investing", "trading", date.today(), ticker=step["ticker"],
                                  quantity=step["quantity"], price=step["price"])
    _export(store, portfolio)
    console.print("[green]Done.[/green] Trading's value now moves only with its own results and transfers.")


@book_app.command("import-csv")
@_book_errors
def book_import_csv(path: str, sleeve: str = "investing", commit: bool = typer.Option(False, "--commit"),
                    portfolio: str = PORTFOLIO_OPT):
    """Preview Robinhood's activity CSV, then with --commit write the new rows."""
    from pathlib import Path

    from tradingagents.ledger import imports

    store = _book_store(portfolio)
    rows = imports.preview(store.ledger, Path(path).read_text(encoding="utf-8-sig"), sleeve)
    for r in rows:
        status = "duplicate" if r.duplicate else ("skip" if r.reason else "new")
        console.print(f"line {r.line}: {r.action:12} {status:9} {r.trade_date or '':10} {r.ticker or '':6} "
                      f"{r.side or '':4} {r.quantity:g} ${r.amount:,.2f} {r.reason or ''}")
        for w in r.warnings:
            console.print(f"   [yellow]{w}[/yellow]")
    if commit:
        result = imports.commit(store.ledger, rows, sleeve)
        _export(store, portfolio)
        console.print(f"[green]Wrote {result['written']} rows[/green], skipped {result['skipped']}.")
        for w in result["warnings"]:
            console.print(f"[yellow]{w}[/yellow]")
    else:
        console.print("[dim]Preview only. Add --commit to write the new rows.[/dim]")


@book_app.command("recommendations")
@_book_errors
def book_recommendations(sleeve: str = None, portfolio: str = PORTFOLIO_OPT):
    """Official recommendations, newest first."""
    from tradingagents.ledger import recommendations

    store = _book_store(portfolio)
    for r in recommendations.history(store.con, store.ledger.account_id, sleeve):
        m = r["metrics"]
        console.print(f"{r['id'][:8]} {r['sleeve']:9} session {r['session']} published {r['published_at'][:16]} "
                      f"{r['status']:10} growth {m['expected_target']:.1%} stress {m['target_stress']:.1%}")


@book_app.command("decide")
@_book_errors
def book_decide(rec_id: str, decision: str, note: str = None, portfolio: str = PORTFOLIO_OPT):
    """Record accepted / rejected / deferred / partial for a recommendation. Nothing is traded."""
    from tradingagents.ledger import recommendations

    store = _book_store(portfolio)
    matches = [r["id"] for r in recommendations.history(store.con, store.ledger.account_id)
               if r["id"].startswith(rec_id)]
    if len(matches) != 1:
        raise ValueError(f"{rec_id} matches {len(matches)} recommendations")
    recommendations.decide(store.con, matches[0], decision, note)
    console.print(f"Recorded {decision}. Holdings change only when you record the fills.")


@book_app.command("ai")
@_book_errors
def book_ai(portfolio: str = PORTFOLIO_OPT):
    """AI spending against the budget, and whether any paid call can run."""
    from tradingagents.ledger.ai_gateway import Budget, Gateway, Policy

    store = _book_store(portfolio)
    g = store.settings("global")
    left = Gateway(store.con, Policy(Budget(**g["ai_budget"]))).remaining()
    console.print(f"Today ${left['spent_today_usd']:.4f} in {left['requests_today']} requests; month "
                  f"${left['spent_month_usd']:.4f} in {left['requests_month']}. Left today ${left['daily_usd']:.4f}, "
                  f"this month ${left['monthly_usd']:.4f}.")
    if not g["ai_prices"]:
        console.print("[yellow]No model prices are configured, so no paid AI call can run. Set them in the web "
                      "app (Settings > AI) or with POST /api/settings.[/yellow]")


stocks_app = typer.Typer(help="Stock research: multi-ticker forecasts, recommendations and their record.",
                         no_args_is_help=True)
app.add_typer(stocks_app, name="stocks")


def _stock_cache():
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(".env")
    return Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"


@stocks_app.command("run")
@_book_errors
def stocks_run(tickers: list[str] = typer.Argument(None, help="Tickers; also --csv, --watchlist, --holdings"),
               csv: str = typer.Option(None, "--csv", help="A CSV with a ticker or symbol column"),
               watchlist: bool = typer.Option(False, "--watchlist"), holdings: bool = typer.Option(False, "--holdings"),
               horizons: str = typer.Option("5,20,60", help="Trading sessions, comma separated"),
               news: bool = typer.Option(True, "--news/--no-news"), device: str = typer.Option("auto"),
               portfolio: str = PORTFOLIO_OPT):
    """Forecast the stocks at the latest close and save the run (local models; no paid calls)."""
    from pathlib import Path

    from tradingagents.stocks.data import gather
    from tradingagents.stocks.pipeline import BatchConfig, Progress, run_batch

    store = _book_store(portfolio)
    names, sources = gather(store, " ".join(tickers or []), Path(csv).read_text(encoding="utf-8") if csv else "",
                            watchlist, holdings)
    if not names:
        raise ValueError("no tickers: pass some, or --csv, --watchlist or --holdings")
    config = BatchConfig(horizons=tuple(int(h) for h in horizons.split(",")), news=news, device=device)
    out = run_batch(store.con, _stock_cache(), names, sources, config,
                    Progress(lambda stage: console.print(f"[dim]{stage}[/dim]")))
    console.print(f"Run {out['run_id'][:8]}: data to {out['info_cutoff']}, earliest execution "
                  f"{out['earliest_execution']}, {out['seconds']:.0f} s.")
    for t, why in out["rejected"].items():
        console.print(f"[yellow]{t}: {why}[/yellow]")
    stocks_show(run=out["run_id"], horizon=None, portfolio=portfolio)


@stocks_app.command("show")
@_book_errors
def stocks_show(run: str = typer.Option(None, help="Run id or its start (default: the latest)"),
                horizon: int = typer.Option(None, help="One horizon only"), portfolio: str = PORTFOLIO_OPT):
    """A saved run's forecasts, one row per stock and horizon."""
    from rich.table import Table

    from tradingagents.stocks.pipeline import load_run

    store = _book_store(portfolio)
    if run:
        run = next((x["id"] for x in store.con.execute("SELECT id FROM stock_runs") if x["id"].startswith(run)), run)
    r = load_run(store.con, run)
    if r is None:
        raise ValueError("no stock run found")
    table = Table(title=f"Run {r['id'][:8]} - data to {r['info_cutoff']} - thresholds {r['thresholds_version']}")
    for c in ("Ticker", "h", "Net exp.", "5-95%", "P(gain)", "Up/Down/Side", "Break up/down", "Large loss",
              "DD 5%", "News", "Evidence", "Notes"):
        table.add_column(c)

    def pct(x):
        return "-" if x is None else f"{x * 100:+.1f}%"

    for f in r["forecasts"]:
        if horizon and f["horizon"] != horizon:
            continue
        fc = f.get("forecast") or {}
        if f["status"] != "ok" or not fc:
            table.add_row(f["ticker"], str(f["horizon"]), "abstain", "", "", "", "", "", "", "", "",
                          "; ".join(f["reasons"])[:60])
            continue
        n = f.get("news") or {}
        table.add_row(f["ticker"], str(f["horizon"]), pct(fc.get("net")), f"{pct(fc.get('q05'))} to {pct(fc.get('q95'))}",
                      f"{fc.get('p_gain', 0):.0%}",
                      f"{fc.get('trend_up', 0):.0%}/{fc.get('trend_down', 0):.0%}/{fc.get('trend_sideways', 0):.0%}",
                      f"{fc.get('p_up_break', 0):.0%}/{fc.get('p_down_break', 0):.0%}", f"{fc.get('p_large_loss', 0):.0%}",
                      pct(fc.get("dd_q05")),
                      "-" if n.get("sentiment") is None else f"{n['sentiment']:+.2f} ({n['articles']})",
                      (f.get("evidence_status") or {}).get("return", "?"), "; ".join(f["reasons"])[:60])
    console.print(table)


@stocks_app.command("watchlist")
@_book_errors
def stocks_watchlist(add: list[str] = typer.Option(None, "--add"), remove: list[str] = typer.Option(None, "--remove"),
                     portfolio: str = PORTFOLIO_OPT):
    """Show, add to or remove from the saved stock watchlist."""
    from tradingagents.stocks.data import parse_tickers

    store = _book_store(portfolio)
    current = list(store.get("global", "stock_watchlist") or [])
    for t in parse_tickers(" ".join(add or [])):
        if t not in current:
            current.append(t)
    gone = set(parse_tickers(" ".join(remove or [])))
    current = [t for t in current if t not in gone]
    if add or remove:
        store.set("global", "stock_watchlist", current)
    console.print("Watchlist: " + (", ".join(current) or "(empty)"))


@stocks_app.command("ideas")
@_book_errors
def stocks_ideas(horizon: int = 20, run: str = typer.Option(None), portfolio: str = PORTFOLIO_OPT):
    """Reviewable stock recommendations from the latest run, within the Trading sleeve's rules."""
    from tradingagents.stocks.ideas import make

    store = _book_store(portfolio)
    out = make(store.con, store, _stock_cache(), run, horizon)
    console.print(f"Idea {out['id'][:8]} for {out['sleeve']}: ${out['settled_cash']:.2f} settled of "
                  f"${out['sleeve_value']:.2f}; earliest execution {out['earliest_execution']}, hold until {out['hold_until']}.")
    for line in out["lines"]:
        order = line.get("order")
        console.print(f"  {line['ticker']:6} {line['action']}"
                      + (f": buy ${order['dollars']:.2f} ({order['shares']} sh)" if order else ""))
    console.print(f"[dim]{out['note']}[/dim]")


@stocks_app.command("decide")
@_book_errors
def stocks_decide(idea: str, ticker: str, decision: str, note: str = typer.Option(None), portfolio: str = PORTFOLIO_OPT):
    """Record accepted, rejected or deferred for one stock of a recommendation (fills: book fill)."""
    from tradingagents.stocks.ideas import decide

    store = _book_store(portfolio)
    ids = [r["id"] for r in store.con.execute("SELECT id FROM stock_ideas") if r["id"].startswith(idea)]
    if len(ids) != 1:
        raise ValueError(f"{idea} matches {len(ids)} recommendations")
    decide(store.con, ids[0], ticker.upper(), decision, note)
    console.print(f"Recorded {decision} for {ticker.upper()}.")


@stocks_app.command("outcomes")
@_book_errors
def stocks_outcomes(portfolio: str = PORTFOLIO_OPT):
    """Record what happened to matured forecasts; show the live scorecard and the Day-1 comparison."""
    import json

    from tradingagents.stocks.ideas import prospective
    from tradingagents.stocks.outcomes import resolve, scorecard

    store = _book_store(portfolio)
    n = resolve(store.con, _stock_cache())
    console.print(f"{n} outcome(s) recorded.")
    console.print(json.dumps({"scorecard": scorecard(store.con), "prospective": prospective(store.con, _stock_cache())},
                             indent=1, default=str))


if __name__ == "__main__":
    app()
