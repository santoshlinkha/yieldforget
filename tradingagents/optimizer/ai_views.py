"""AI ratings for the holdings: one TradingAgents run per ticker, reused when it already ran today.

The rating of each run is recorded in the live memory log like any other
analysis, so a second optimization on the same day reads it back instead of
paying for the run again, and the ratings are later settled and reflected on.

Paid runs go through the AI gateway (``tradingagents.ledger.ai_gateway``): each
ticker's run is cached by its inputs, claimed once, held to the spending budget
before and during the run, and its token usage recorded. Nothing is paid for
unless the caller passes ``allow_paid=True``, which only an explicit request to
rate does. The default is one analyst, the market (price) analyst; more are an
explicit choice.
"""

from __future__ import annotations

from collections.abc import Callable

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.memory import TradingMemoryLog

# ETFs have no company statements, and every extra analyst multiplies the cost of a run.
DEFAULT_ANALYSTS = ("market",)
RATING_SCHEMA = "rating-v1"
# Rough cost guards for one run, used to reserve budget before it starts.
EST_REQUESTS_PER_ANALYST = 8
EST_REQUESTS_BASE = 10


def collect_ratings(tickers, config: dict, analysts=DEFAULT_ANALYSTS, trade_date: str | None = None,
                    progress: Callable[[str, str], None] | None = None, gateway=None, allow_paid: bool = False,
                    est_usd: float = 0.30) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Ratings by ticker, and the tickers that failed with why. REVIEW counts as no rating.

    Without a gateway only ratings already in the memory log are returned: no paid run
    is started.
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.ledger.ai_gateway import AIBlocked, AIInProgress

    trade_date = trade_date or get_current_date()
    logged = {e["ticker"]: e["rating"] for e in TradingMemoryLog(config).load_entries()
              if e["date"] == trade_date}
    ratings, failures = {}, []
    analysts = tuple(analysts)
    for ticker in tickers:
        if ticker in logged:
            rating = logged[ticker]
            if progress:
                progress(ticker, f"{rating} (already ran today)")
        elif gateway is None:
            failures.append((ticker, "not rated today, and no paid run was requested"))
            continue
        else:
            if progress:
                progress(ticker, "running agents")

            def work(callback, ticker=ticker):
                graph = TradingAgentsGraph(analysts, config=config, callbacks=[callback])
                return {"rating": graph.propagate(ticker, trade_date)[1]}

            provider = config["llm_provider"]
            try:
                result, how = gateway.run(
                    task="rating", provider=provider, model=config["deep_think_llm"], schema_version=RATING_SCHEMA,
                    info_cutoff=trade_date,
                    inputs={"ticker": ticker, "analysts": sorted(analysts), "quick": config["quick_think_llm"],
                            "deep": config["deep_think_llm"], "debate_rounds": config.get("max_debate_rounds")},
                    also_priced=[(provider, config["quick_think_llm"])],
                    work=work, est_usd=est_usd,
                    est_requests=EST_REQUESTS_BASE + EST_REQUESTS_PER_ANALYST * len(analysts),
                    allow_paid=allow_paid)
            except (AIBlocked, AIInProgress) as exc:
                failures.append((ticker, str(exc)))
                if isinstance(exc, AIBlocked):
                    break   # the budget or policy refuses every further ticker too
                continue
            except Exception as exc:  # one failed ticker must not end the optimization
                failures.append((ticker, str(exc)))
                continue
            rating = result["rating"]
            if progress and how == "cache":
                progress(ticker, f"{rating} (cached)")
        if rating and rating != RATING_REVIEW:
            ratings[ticker] = rating
    return ratings, failures
