import sys
import uuid
from enum import StrEnum
from typing import Annotated

import typer
from rich.markup import escape

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.openai_client import SIGNED_IN_PROVIDERS
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
    html: bool | None = typer.Option(
        None, "--html/--no-html",
        help="Save the report as one HTML page too, complete_report.html, without asking (default: yes)",
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
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show, "html": html}
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
def trade(portfolio: str = typer.Option("my_portfolio.json", "--portfolio",
                                        help="Holdings file; its folder holds portfolio.sqlite3")):
    """Today's trades from the internal model: the best stock to buy with the budget, its buy price, what to sell."""
    import json
    from datetime import date
    from pathlib import Path

    from tradingagents.ledger import income
    from tradingagents.ledger.store import Store
    from tradingagents.stocks import daily
    from tradingagents.stocks import trade as T

    store = Store(Path(portfolio).resolve().parent)
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    pot = income.budget(store, cache, date.today())
    out = T.plan(daily.load(store.con), pot["available"])
    out["budget_detail"] = pot
    out["income_health"] = income.health(store, cache, date.today())
    console.print(json.dumps(out, indent=1, default=str))


@app.command()
def web(
    portfolio: str = typer.Option("my_portfolio.json", "--portfolio", help="JSON file with holdings and settled cash"),
    schedule: str = typer.Option("my_buys.json", "--schedule", help="JSON file with recurring buys"),
    port: int = typer.Option(8765, "--port", help="Port on localhost"),
):
    """Open AlphaDesk, the web app, on this machine."""
    import webbrowser

    from tradingagents.webapp import serve

    try:
        server = serve(portfolio, schedule, port)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    url = f"http://127.0.0.1:{port}/"
    console.print(f"AlphaDesk at {url} (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


book_app = typer.Typer(help="Your records: holdings, cash, fills, imports, the money settings and the budget.",
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
    """Holdings, settled and unsettled cash, and the value of the account."""
    from datetime import date

    from rich.table import Table

    store = _book_store(portfolio)
    prices = _prices(store)
    for sleeve in ("combined",):   # one account
        s = store.ledger.state(sleeve, date.today(), prices)
        table = Table(title=f"Account: ${s.equity:,.2f}")
        for col in ("Ticker", "Quantity", "Value"):
            table.add_column(col, justify="left" if col == "Ticker" else "right")
        for t, q in s.positions.items():
            table.add_row(t, f"{q:,.6g}", f"${s.values[t]:,.2f}" if t in s.values else "no price")
        table.add_row("cash settled", "", f"${s.cash.settled:,.2f}")
        table.add_row("cash unsettled", "", f"${s.cash.unsettled:,.2f}")
        if s.cash.reserved:
            table.add_row("cash reserved", "", f"${s.cash.reserved:,.2f}")
        console.print(table)



@book_app.command("contribute")
@_book_errors
def book_contribute(amount: float, sleeve: str = "trading", day: str = typer.Option(None, "--date"),
                    settle_date: str = typer.Option(None, "--settle-date"), portfolio: str = PORTFOLIO_OPT):
    """Record new money deposited into the IRA (an external flow, not profit)."""
    from datetime import date

    store = _book_store(portfolio)
    store.ledger.contribute(amount, day or date.today(), sleeve=sleeve, settle_date=settle_date,
                            op_id=uuid.uuid4().hex)   # each command is its own deposit
    _export(store, portfolio)
    console.print(f"Recorded a ${amount:,.2f} contribution to {sleeve}.")


@book_app.command("fill")
@_book_errors
def book_fill(side: str, ticker: str, quantity: float, price: float, sleeve: str = "trading",
              day: str = typer.Option(None, "--date"), fee: float = 0.0,
              broker_record: bool = typer.Option(False, "--broker-record",
                                                 help="Keep it even if the books say it could not be paid for"),
              portfolio: str = PORTFOLIO_OPT):
    """Record a buy or sell that happened at the broker."""
    from datetime import date

    store = _book_store(portfolio)
    _, warnings = store.ledger.fill(sleeve, side.lower(), ticker, quantity, price, day or date.today(), fee=fee,
                                    strict=not broker_record, op_id=uuid.uuid4().hex)
    _export(store, portfolio)
    console.print(f"Recorded: {side} {quantity:g} {ticker.upper()} at ${price:,.4f} in {sleeve}.")
    for w in warnings:
        console.print(f"[yellow]{w}[/yellow]")


@book_app.command("retire-investing")
@_book_errors
def book_retire_investing(confirm: bool = typer.Option(False, "--confirm", help="Delete the Investing records"),
                          portfolio: str = PORTFOLIO_OPT):
    """Fold the old Investing sleeve into the one account: back up the database, delete its records, restate holdings."""
    import json
    from datetime import date

    store = _book_store(portfolio)
    if not store.con.execute("SELECT 1 FROM sleeves WHERE name = 'investing'").fetchone():
        console.print("There is no Investing sleeve: nothing to do.")
        return
    if not confirm:
        console.print("This deletes every Investing record (after a backup) and restates today's holdings in the one "
                      "account. Run again with --confirm.")
        return
    out = store.retire_investing(date.today())
    _export(store, portfolio)
    console.print(json.dumps(out, indent=1, default=str))


@book_app.command("import-csv")
@_book_errors
def book_import_csv(path: str, sleeve: str = "trading", commit: bool = typer.Option(False, "--commit"),
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


@book_app.command("capital")
@_book_errors
def book_capital(daily: float = typer.Option(None, help="Dollars bought into SGOV each market session"),
                 match: float = typer.Option(None, help="The broker's match on deposits, e.g. 0.03 for 3%"),
                 dry_yield: float = typer.Option(None, "--dry-yield",
                                                 help="SGOV yield under which its interest has dried up, e.g. 0.02"),
                 portfolio: str = PORTFOLIO_OPT):
    """Show or change the money settings, and show the trading budget they give."""
    import json
    from datetime import date
    from pathlib import Path

    from tradingagents.ledger import capital, income

    store = _book_store(portfolio)
    values = {k: v for k, v in (("daily_sgov_usd", daily), ("match_rate", match), ("dry_yield", dry_yield))
              if v is not None}
    snap = capital.update(store, values, source="cli") if values else capital.snapshot(store)
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    console.print(json.dumps({"settings": snap, "budget": income.budget(store, cache, date.today())},
                             indent=1, default=str))


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
                      "app (Settings → Agent models) or with POST /api/settings.[/yellow]")


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


@stocks_app.command("today")
@_book_errors
def stocks_today(train: bool = typer.Option(False, "--train", help="Fit this week's models if they are missing"),
                 policy: str = typer.Option("barrier_20"), as_of: str = typer.Option(None, "--as-of"),
                 portfolio: str = PORTFOLIO_OPT):
    """Today's decision for the Trading sleeve: ENTER / WAIT / HOLD / REDUCE / EXIT (saved, never executed)."""
    from datetime import date

    from tradingagents.stocks import daily

    store = _book_store(portfolio)
    rec = daily.run(store, _stock_cache(), date.fromisoformat(as_of) if as_of else date.today(), policy, train,
                    lambda stage: console.print(f"[dim]{stage}[/dim]"))
    _print_decision(rec)


def _print_decision(rec):
    console.print(f"Decision {rec['id'][:8]} - {rec['mode'].upper()} - data to {rec['info_cutoff']} "
                  f"({rec['freshness_days']} days old), executes {rec['execution_date']}; capital settings "
                  f"v{rec['settings']['capital']['version']}; Trading ${rec['account']['sleeve_value']:,.2f}, "
                  f"settled ${rec['account']['settled_cash']:,.2f}, reserved ${rec['account']['reserved']:,.2f}")
    for x in rec["lots"]:
        console.print(f"  {x['action']:7} {x['ticker']:6} entered {x['entry_date']} deadline {x['deadline']}: "
                      + "; ".join(x["reasons"]))
    for x in rec["entries"][:10]:
        size = f" ${x['dollars']:.2f}" if x.get("dollars") else " (the shadow account sizes it)"
        extra = f" [{x['paper_action']}{size}]" if x.get("paper_action") else ""
        console.print(f"  {x['action']:7} {x['ticker']:6} expected net {x['expected_net']:+.2%} "
                      f"(+/- {x['net_sd']:.1%}), P(win) {x['p_win']:.0%}, P(large loss) {x['p_risk']:.0%}{extra}"
                      + (f" - blocked: {'; '.join(x['blocked'])}" if x["blocked"] else ""))
    for note in rec["limitations"]:
        console.print(f"[dim]{note}[/dim]")


@stocks_app.command("decisions")
@_book_errors
def stocks_decisions(decision: str = typer.Option(None, help="One decision id (default: the history)"),
                     portfolio: str = PORTFOLIO_OPT):
    """Saved decisions (the action history), or one in full."""
    from tradingagents.stocks import daily

    store = _book_store(portfolio)
    if decision:
        ids = [r["id"] for r in store.con.execute("SELECT id FROM stock_decisions") if r["id"].startswith(decision)]
        if len(ids) != 1:
            raise ValueError(f"{decision} matches {len(ids)} decisions")
        _print_decision(daily.load(store.con, ids[0]))
        return
    for h in daily.history(store.con):
        console.print(f"{h['id'][:8]} {h['info_cutoff']} -> {h['execution_date']} {h['mode']:12} {h['summary']} "
                      f"paper enters {h['paper_enters']} (settings v{h['settings_version']})")


@stocks_app.command("calls")
def stocks_calls(ticker: str, day: str = typer.Argument(..., help="The analyzed day, e.g. 2026-10-23"),
                 full: bool = typer.Option(False, "--full", help="Print every message in full")):
    """What an agent run sent to the AI model, request by request, and what came back."""
    from pathlib import Path

    from tradingagents.llm_clients.request_log import read

    calls = read(Path(DEFAULT_CONFIG["results_dir"]) / ticker.upper() / day / "reports")
    if not calls:
        console.print(f"No recorded requests for {ticker.upper()} on {day} (runs before the request log have none).")
        return
    cut = (lambda t: t) if full else (lambda t: t if len(t) <= 600 else t[:600] + f" ... [{len(t) - 600} more chars]")
    for c in calls:
        console.rule(f"#{c['n']} {c.get('agent')} | {c.get('model')} | {c.get('input_tokens')} in / {c.get('output_tokens')} out")
        if c.get("tools_offered"):
            console.print(f"[dim]tools offered: {', '.join(c['tools_offered'])}[/dim]")
        for m in c["sent"]:
            console.print(f"[bold cyan]SENT {m['role']}[/bold cyan]{' ' + m['name'] if m.get('name') else ''}:", markup=True)
            console.print(cut(m.get("content") or ""), markup=False, highlight=False)
            for t in m.get("tool_calls") or []:
                console.print(f"  -> {t['name']}({t['args']})", markup=False)
        r = c.get("received") or {}
        console.print("[bold green]CAME BACK[/bold green]:" if not c.get("error") else f"[red]ERROR: {c['error']}[/red]")
        console.print(cut(r.get("content") or ""), markup=False, highlight=False)
        for t in r.get("tool_calls") or []:
            console.print(f"  -> asks for {t['name']}({t['args']})", markup=False)


@stocks_app.command("screen")
@_book_errors
def stocks_screen(build: bool = typer.Option(False, "--build", help="Fetch (cached) and rebuild the screen first"),
                  top: int = typer.Option(25, help="How many rows to show"),
                  sort: str = typer.Option("rank", help="rank, quality_score, target_upside, analyst_mean, ..."),
                  portfolio: str = PORTFOLIO_OPT):
    """The quality checklist and Wall Street screen of S&P 500 stocks (and the watchlist), ranked."""
    from pathlib import Path

    from tradingagents.dataflows.vendors.index_members import get_sp500_members
    from tradingagents.stocks import screen
    from tradingagents.stocks.data import parse_tickers

    store = _book_store(portfolio)
    results = Path(DEFAULT_CONFIG["results_dir"])
    if build:
        tickers = sorted(set(get_sp500_members(_stock_cache())) |
                         set(parse_tickers(" ".join(store.get("global", "stock_watchlist") or []))))
        out = screen.build(_stock_cache(), tickers, results, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    else:
        out = screen.latest(results)
        if out is None:
            raise ValueError("no screen yet: run with --build")
    ascending = sort in ("rank", "analyst_mean", "forward_pe")
    rows = sorted(out["rows"], key=lambda r: (r.get(sort) is None, (r.get(sort) or 0) * (1 if ascending else -1)))
    console.print(f"{out['stocks']} stocks, data to {out['as_of']}; sorted by {sort}")
    for r in rows[:top]:
        up = "-" if r["target_upside"] is None else f"{r['target_upside']:+.1%}"
        console.print(f"{r['rank']:4} {r['ticker']:6} quality {r['quality_score'] or 0:5.1f}  moat {r['moat_passed']}/{r['moat_scored']}  "
                      f"consensus {r['analyst_mean'] or '-'} ({r['analysts']})  target {up}")


@stocks_app.command("shadow")
@_book_errors
def stocks_shadow(portfolio: str = PORTFOLIO_OPT):
    """The prospective shadow account's record against cash-flow-matched SPY."""
    import json

    store = _book_store(portfolio)
    row = store.con.execute("SELECT as_of, state_json FROM shadow_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        raise ValueError("no shadow record yet: run `stocks today`")
    console.print(json.dumps({"as_of": row["as_of"], **json.loads(row["state_json"])["record"]}, indent=1, default=str))


# The plans `login` and `logout` take: the providers signed in to, not keyed.
Plan = StrEnum("Plan", {name: name for name in SIGNED_IN_PROVIDERS})


@app.command()
def login(plan: Annotated[Plan, typer.Argument(help="The plan to sign in with")]):
    """Sign in with your ChatGPT Plus or Pro plan, for the chatgpt provider."""
    from tradingagents.llm_clients import chatgpt_auth

    try:
        # One unwrapped line, so the sign-in address can be copied whole.
        email = chatgpt_auth.sign_in(
            say=lambda line: console.print(line, soft_wrap=True, markup=False, highlight=False))
    except chatgpt_auth.ChatGPTAuthError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Signed in to ChatGPT as {escape(email)}.[/green]")


@app.command()
def logout(plan: Annotated[Plan, typer.Argument(help="The plan to sign out of")]):
    """Sign out of your ChatGPT plan: revoke this machine's access and forget it."""
    from tradingagents.llm_clients import chatgpt_auth

    outcome = chatgpt_auth.sign_out()
    if outcome == "revoked":
        console.print("Signed out of ChatGPT.")
    elif outcome == "forgotten":
        console.print("[yellow]Signed out here, but OpenAI could not be reached to revoke the sign-in; "
                      "it stays valid at OpenAI until it expires.[/yellow]")
    else:
        console.print("Not signed in to ChatGPT.")


if __name__ == "__main__":
    app()
