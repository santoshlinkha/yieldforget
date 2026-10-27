"""yfinance-based news data fetching functions."""

import contextlib
from datetime import UTC, datetime

import yfinance as yf
from dateutil.relativedelta import relativedelta

from tradingagents.dataflows import local_store
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry


def _extract_article_data(article: dict) -> dict:
    """Extract article data from yfinance news format (handles nested 'content' structure).

    Yahoo sends some fields as null rather than leaving them out, so each falls
    back to its default on a null as well as on a missing key (#1458).
    """
    content = article.get("content")
    if isinstance(content, dict):
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        pub_date_str = content.get("pubDate") or ""
        pub_date = None
        if pub_date_str:
            with contextlib.suppress(ValueError, AttributeError):
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))

        return {
            "title": content.get("title") or "No title",
            "summary": content.get("summary") or "",
            "publisher": (content.get("provider") or {}).get("displayName") or "Unknown",
            "link": url_obj.get("url") or "",
            "pub_date": pub_date,
        }
    # Fallback for flat structure. Parse the epoch publish time so flat
    # articles are date-filterable too (otherwise they bypass the
    # historical window and leak future news, #992/#1007).
    pub_date = None
    ts = article.get("providerPublishTime")
    if ts:
        # Epoch seconds are UTC; parse them as UTC-aware so filtering does
        # not shift with the host timezone (#1126).
        with contextlib.suppress(ValueError, OSError, TypeError):
            pub_date = datetime.fromtimestamp(ts, tz=UTC)
    return {
        "title": article.get("title") or "No title",
        "summary": article.get("summary") or "",
        "publisher": article.get("publisher") or "Unknown",
        "link": article.get("link") or "",
        "pub_date": pub_date,
    }


def _search_news(canonical: str, limit: int) -> tuple[list, bool]:
    """Yahoo search's articles tagged with ``canonical``, and whether it knows the symbol.

    Search answers when the quote feed is empty, but its articles are a
    relevance-picked sample, and some are tagged loosely; only those tagged with
    the symbol are about it.
    """
    search = yf_retry(lambda: yf.Search(canonical, news_count=limit))
    if search is None:  # Yahoo answered "not found": its search is down
        return [], False
    tagged = [
        a for a in (search.news or [])
        if canonical in (t.upper() for t in a.get("relatedTickers") or [])
    ]
    knows = any((q.get("symbol") or "").upper() == canonical for q in search.quotes or [])
    return tagged, knows or bool(search.news)


def _format_articles(articles: list[dict]) -> str:
    out = ""
    for data in articles:
        out += f"### {data['title']} (source: {data['publisher']})\n"
        if data["summary"]:
            out += f"{data['summary']}\n"
        if data["link"]:
            out += f"Link: {data['link']}\n"
        out += "\n"
    return out


def _articles(key: str, fetch) -> list[dict]:
    """Parsed articles for one source request, from the local store when it has them
    fresh (see ``local_store``); straight from Yahoo when the store is off."""
    def parsed():
        return [_extract_article_data(a) for a in (fetch() or [])]

    market_store = local_store.store()
    return market_store.articles(key, parsed) if market_store is not None else parsed()


def _ticker_news(canonical: str, limit: int) -> list:
    """A ticker's articles from Yahoo. Its ticker-news call has come back empty for every
    ticker since late 2026, so Yahoo's search news for the symbol fills in, keeping
    the articles Yahoo tags with the ticker when it tags any."""
    news = yf_retry(lambda: yf.Ticker(canonical).get_news(count=limit)) or []
    if news:
        return news
    try:
        found = yf_retry(lambda: yf.Search(query=canonical, news_count=limit, enable_fuzzy_query=False).news) or []
    except Exception:  # the fallback failing leaves what the ticker feed gave: nothing
        return news
    tagged = [a for a in found if canonical in (a.get("relatedTickers") or [])]
    return tagged or found


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """News about ``ticker`` published in the window, from Yahoo Finance.

    The symbol's quote feed comes first. When it is empty, Yahoo search serves
    the articles tagged with the symbol; when it has none, or neither answers,
    Yahoo news is unavailable, which is not an absence of news, and a configured
    next vendor is tried. Search results are a sample, so finding none in the
    window never reads as "no news".
    """
    article_limit = get_config()["news_article_limit"]
    # Query Yahoo with the canonical symbol, like every other yfinance path —
    # a raw broker/forex/crypto alias (XAUUSD, BTCUSD) otherwise silently
    # returns no news. Keep the user's ticker in the report header.
    canonical = normalize_symbol(ticker)
    resolved = "" if canonical == ticker else f" (resolved to {canonical})"
    subject = f"news for {ticker}{resolved}"
    # The quote feed, kept in the local store like every other Yahoo download.
    articles = _articles(f"ticker:{canonical}:{article_limit}",
                         lambda: yf_retry(lambda: yf.Ticker(canonical).get_news(count=article_limit)))
    feed = bool(articles)
    if not feed:
        found, answered = _search_news(canonical, article_limit)
        if not answered:
            raise VendorUnavailableError(f"Yahoo Finance returned no news response for {canonical}")
        if not found:
            # With the quote feed empty, no tagged article says nothing about the
            # company's news (Yahoo tags no article with most non-US listings).
            raise VendorUnavailableError(f"Yahoo Finance has no news articles tagged {canonical}")
        articles = [_extract_article_data(a) for a in found]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    # Keep only articles within the requested window (look-ahead safe).
    in_range = [a for a in articles if in_window(a["pub_date"], start_dt, end_dt)]
    if in_range:
        return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{_format_articles(in_range)}"

    if not feed:
        return (f"<Yahoo Finance news unavailable for {start_date}..{end_date}: Yahoo search "
                f"returns only a sample of recent articles about {canonical}, so this is not "
                f"an absence of {subject}>")
    gap = coverage_gap(
        (a["pub_date"] for a in articles),
        start_date, end_date, "Yahoo Finance news", subject,
    )
    return gap or f"No news found for {ticker}{resolved} between {start_date} and {end_date}"


def get_global_news_yfinance(
    as_of_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Args:
        as_of_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back. ``None`` falls back to
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles to return. ``None`` falls back to
            ``global_news_article_limit`` from the active config.

    Returns:
        Formatted string containing global news articles
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    search_queries = config["global_news_queries"]

    curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    in_window_news = []
    seen_titles = set()

    for query in search_queries:
        found = _articles(f"search:{query}:{limit}", lambda q=query: yf_retry(lambda: yf.Search(
            query=q,
            news_count=limit,
            enable_fuzzy_query=True,
        ).news))

        for data in found:
            # Window first: the limit counts what the run may read, so an
            # out-of-window article must not spend the budget or cut the
            # remaining searches short (#1356). Flat articles are filtered
            # on the same rule, so none can leak future news (#1007).
            if not in_window(data["pub_date"], start_dt, curr_dt):
                continue
            if data["title"] and data["title"] not in seen_titles:
                seen_titles.add(data["title"])
                in_window_news.append(data)

        if len(in_window_news) >= limit:
            break

    news_str = _format_articles(in_window_news[:limit])

    # Nothing fell inside the window -> say so rather than return an
    # empty-bodied report (#993).
    if not news_str:
        # Results merge several fuzzy searches, so their timestamps prove no
        # continuous coverage; judge the window against the present only.
        gap = coverage_gap((), start_date, as_of_date, "Yahoo Finance global news", "market news")
        return gap or f"No global news found between {start_date} and {as_of_date}"

    return f"## Global Market News, from {start_date} to {as_of_date}:\n\n{news_str}"


def get_market_headlines(as_of_date: str, queries: list[str], look_back_days: int = 7,
                         per_query: int = 10, limit: int = 30) -> list[dict]:
    """Recent market headlines as records (title, publisher, link, date, summary), newest first.

    For callers that show the sources beside what they say about them; the
    agents' tools use the formatted ``get_global_news_yfinance`` instead.
    """
    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=look_back_days)
    found, seen = [], set()
    for query in queries:
        articles = _articles(f"search:{query}:{per_query}", lambda q=query: yf_retry(
            lambda: yf.Search(query=q, news_count=per_query, enable_fuzzy_query=True).news))
        for data in articles:
            if not in_window(data["pub_date"], start_dt, end_dt) or data["title"] in seen:
                continue
            seen.add(data["title"])
            found.append(data)
    found.sort(key=lambda d: d["pub_date"] or datetime.min.replace(tzinfo=UTC), reverse=True)
    return [{**d, "pub_date": d["pub_date"].isoformat() if d["pub_date"] else None} for d in found[:limit]]


def ticker_articles(ticker: str, limit: int = 30) -> list[dict]:
    """A ticker's news as records (title, publisher, link, pub_date, summary), newest first,
    including every article the local store has kept for it from earlier fetches."""
    canonical = normalize_symbol(ticker)
    articles = _articles(f"ticker:{canonical}:{limit}", lambda: _ticker_news(canonical, limit))
    articles = sorted(articles, key=lambda d: d["pub_date"] or datetime.min.replace(tzinfo=UTC), reverse=True)
    return [{**d, "pub_date": d["pub_date"].isoformat() if d["pub_date"] else None} for d in articles]
