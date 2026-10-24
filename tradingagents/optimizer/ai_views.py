"""AI ratings for the holdings: one TradingAgents run per ticker, reused when it already ran today.

The rating of each run is recorded in the live memory log like any other
analysis, so a second optimization on the same day reads it back instead of
paying for the run again, and the ratings are later settled and reflected on.
"""

from __future__ import annotations

from collections.abc import Callable

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import TradingMemoryLog

# ETFs have no company statements, so the fundamentals analyst has nothing to read.
DEFAULT_ANALYSTS = ("market", "news")


def collect_ratings(tickers, config: dict, analysts=DEFAULT_ANALYSTS, trade_date: str | None = None,
                    progress: Callable[[str, str], None] | None = None) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Ratings by ticker, and the tickers that failed with why. REVIEW counts as no rating."""
    trade_date = trade_date or get_current_date()
    logged = {e["ticker"]: e["rating"] for e in TradingMemoryLog(config).load_entries()
              if e["date"] == trade_date}
    ratings, failures, graph = {}, [], None
    for ticker in tickers:
        if ticker in logged:
            rating = logged[ticker]
            if progress:
                progress(ticker, f"{rating} (already ran today)")
        else:
            if progress:
                progress(ticker, "running agents")
            try:
                graph = graph or TradingAgentsGraph(analysts, config=config)
                _, rating = graph.propagate(ticker, trade_date)
            except Exception as exc:  # one failed ticker must not end the optimization
                failures.append((ticker, str(exc)))
                continue
        if rating and rating != RATING_REVIEW:
            ratings[ticker] = rating
    return ratings, failures
