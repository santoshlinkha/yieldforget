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
    portfolio: str = typer.Option("my_portfolio.json", "--portfolio", help="JSON file with holdings and settled cash"),
    schedule: str = typer.Option("my_buys.json", "--schedule", help="JSON file with recurring buys"),
    risk: int | None = typer.Option(None, "--risk", min=1, max=10,
                                    help="Risk level 1-10, saved for next time; the saved level if omitted"),
    max_loss: float | None = typer.Option(None, "--max-loss",
                                          help="Stress-loss limit in percent, overriding the risk level; 0 for none"),
    model: str | None = typer.Option(None, "--model",
                                     help="How the stress loss is measured, saved for next time: worst_case "
                                          "(every holding at its worst drop at once) or scenarios (the worst past "
                                          "crash replayed on the mix); the saved one if omitted"),
    target_growth: float | None = typer.Option(None, "--target-growth",
                                               help="Expected growth to aim for, percent a year (8 for 8%), saved "
                                                    "for next time; the least risky mix that reaches it, within the "
                                                    "risk level. 0 clears it"),
    scan: bool = typer.Option(True, "--scan/--no-scan", help="Also consider the funds in the scan list"),
    universe: str = typer.Option("my_universe.json", "--universe", help="Scan list file; the built-in list if absent"),
    ai: bool = typer.Option(False, "--ai", help="Run the agents on each holding and tilt toward their ratings"),
    analysts: str = typer.Option("market,news", "--analysts", help="Analysts for --ai, comma-separated"),
):
    """Print today's orders and the optimized mix behind them."""
    from datetime import date
    from pathlib import Path

    from rich.markdown import Markdown

    from tradingagents.optimizer import OptimizerSettings, build_plan, load_schedule, render_plan
    from tradingagents.optimizer.ai_views import collect_ratings
    from tradingagents.optimizer.plan import daily_orders, render_suggestion
    from tradingagents.optimizer.risk import read_choice, risk_level, write_choice
    from tradingagents.optimizer.universe import load_universe

    try:
        book = load_portfolio(portfolio)
        buys = load_schedule(schedule)
        risk_file = Path(portfolio).resolve().parent / "my_risk.json"
        if risk or model or target_growth is not None:
            kwargs = {} if target_growth is None else {"target_growth": target_growth / 100 if target_growth else None}
            write_choice(risk_file, level=risk, model=model, **kwargs)
        choice = read_choice(risk_file)
        level = risk_level(choice.level)
        limit = level.loss_limit if max_loss is None else (max_loss / 100 if max_loss > 0 else None)
        goal = "" if choice.target_growth is None else f", growth target {choice.target_growth:.1%}"
        console.print(f"[dim]Risk level {level.level} ({level.name}), stress-loss limit "
                      f"{'none' if limit is None else f'{limit:.1%}'} ({choice.model.replace('_', ' ')}){goal}[/dim]")
        settings = OptimizerSettings(loss_limit=limit, risk_model=choice.model, target_growth=choice.target_growth)
        ratings = None
        if ai:
            cash = settings.cash_ticker
            ratings, failures = collect_ratings(
                [p.ticker.upper() for p in book.positions if p.ticker.upper() != cash], DEFAULT_CONFIG,
                [a.strip() for a in analysts.split(",") if a.strip()],
                progress=lambda t, s: console.print(f"[dim]{t}: {s}[/dim]"))
            for ticker, reason in failures:
                console.print(f"[yellow]no rating:[/yellow] {ticker}: {reason}")
        plan = build_plan(book, buys, settings, ratings=ratings,
                          universe=load_universe(universe, Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer") if scan else None,
                          cache_dir=Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(Markdown(render_suggestion(daily_orders(plan, float(book.cash or 0.0)), date.today().isoformat())))
    console.print(Markdown(render_plan(plan)))


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


if __name__ == "__main__":
    app()
