"""Ticker news: collected with its times, deduplicated, tagged with events, scored for sentiment locally.

Collection. Headlines come from Yahoo Finance (``yahoo.news.ticker_articles``, which
keeps what it has seen). Each is stored once in ``stock_news`` with its publication
time and the time this app first saw it; a re-fetch never rewrites either. Yahoo's
history is only recent, so news is collected **prospectively**: backtests have no
news, the models are not trained on it, and every summary says since when news has
been collected.

Deduplication. The same story from several outlets is linked to the first one seen:
an identical normalized headline is the same record, and one sharing 80% of its
words within three days is marked ``duplicate_of`` it.

Relevance. A ticker's feed carries stories about other companies. A headline that
names the ticker or the company *mentions* it; one that names it in its first 40
characters is *about* it. Sentiment is averaged over headlines about the stock only;
mentions are listed but not scored into it.

Events, by rule (cheap, transparent): earnings, guidance, financing, buybacks and
dividends, litigation, regulation, management changes, mergers and acquisitions,
analyst actions, products and contracts. **Kind**: *opinion* (questions, "should you",
predictions, lists of stocks to buy), *fact* (reports, announces, files, raises, cuts,
approves, sues...), else *unclear*. A rule tags; it does not understand.

Sentiment, on this machine: FinBERT (ProsusAI/finbert, a BERT model tuned on
financial text) when it can be loaded, else a small Loughran-McDonald word list,
named in each score. Score = P(positive) - P(negative), from -1 to 1, on the headline
and the start of its summary. Each article is scored once and the score stored.

Missing news is not neutral sentiment: a stock with no relevant headline in the
window gets no sentiment, and says so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

EVENT_RULES = {
    "earnings": r"\b(earnings|results|eps|beats?|miss(es|ed)?|quarterly (profit|revenue|sales)|q[1-4]\b)",
    "guidance": r"\b(guidance|outlook|raises (its )?(full[- ]year )?(view|forecast)|cuts (its )?forecast|reaffirm\w*)",
    "financing": r"\b(offering|notes due|convertible|debt sale|share sale|private placement|credit facility|raises \$)",
    "buyback_dividend": r"\b(buyback|repurchase|dividend)",
    "litigation": r"\b(lawsuit|sues|sued|class action|settles?|settlement|verdict|litigation)",
    "regulation": r"\b(fda|antitrust|ftc|doj|regulator\w*|approval|approves?|ban|tariffs?|sanctions?|recall|probe)",
    "management": r"\b(ceo|cfo|chief executive|resigns?|steps down|appoints?|succession)",
    "mna": r"\b(acquir\w+|merger|takeover|buyout|spin[- ]?off|divest\w*)",
    "analyst": r"\b(upgrades?|downgrades?|price target|initiates coverage|overweight|underweight|outperform)",
    "product_contract": r"\b(launch\w*|unveil\w*|contract|partnership|agreement)",
}
OPINION = re.compile(r"(\?|^(why|is|should|could|will|can|what|how)\b|\bshould you\b|\bbuy now\b|\bprediction\b|"
                     r"\bopinion\b|\bhere'?s\b|\b(best|top) (\d+ )?\w* ?stocks?\b|\bmight\b|\bcould\b|\bset to\b|"
                     r"\bpoised\b|\bbetter buy\b)", re.I)
FACT = re.compile(r"\b(reports?|reported|announces?|announced|files?|filed|raises|cuts|lowers|agrees?|agreed|"
                  r"completes?|completed|sued|approves?|approved|appoints?|names|posts?|beats?|misses|declares?|"
                  r"signs?|acquires?)\b", re.I)
POSITIVE = {"beat", "beats", "surge", "surges", "soar", "soars", "record", "raises", "raised", "upgrade", "upgrades",
            "growth", "gain", "gains", "jumps", "rally", "profit", "strong", "approval", "approves", "wins", "outperform"}
NEGATIVE = {"miss", "misses", "falls", "fall", "plunge", "plunges", "drop", "drops", "cut", "cuts", "lowers",
            "downgrade", "downgrades", "loss", "losses", "lawsuit", "probe", "recall", "weak", "decline", "declines",
            "slump", "warning", "warns", "bankruptcy", "fraud", "layoffs", "resigns"}
FINBERT = "ProsusAI/finbert"

_model_lock = threading.Lock()
_model = {"pipe": None, "name": None, "tried": False}


def _norm(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (title or "").lower())).strip()


def article_id(ticker: str, title: str) -> str:
    return hashlib.sha256(f"{ticker.upper()}|{_norm(title)}".encode()).hexdigest()[:24]


def events_in(text: str) -> list[str]:
    return [k for k, rx in EVENT_RULES.items() if re.search(rx, text or "", re.I)]


def kind_of(title: str) -> str:
    if OPINION.search(title or ""):
        return "opinion"
    return "fact" if FACT.search(title or "") else "unclear"


def _named(text: str, ticker: str, name: str | None) -> int | None:
    """Where the ticker or the company's first distinctive word first appears in ``text``, or None."""
    spots = [m.start() for m in [re.search(rf"\b{re.escape(ticker)}\b", text)] if m]
    words = re.sub(r"[^A-Za-z0-9 &.-]", "", name or "").split()
    key = next((w for w in words if len(w) > 2 and w.lower() not in ("the", "inc", "corp", "company")), None)
    if key and (m := re.search(rf"\b{re.escape(key)}", text, re.I)):
        spots.append(m.start())
    return min(spots) if spots else None


def relevance(title: str, summary: str, ticker: str, name: str | None) -> str:
    """'about' (named in the headline's first 40 characters), 'mentions' (named somewhere), or 'other'."""
    at = _named(title or "", ticker, name)
    if at is not None and at < 40:
        return "about"
    return "mentions" if at is not None or _named(summary or "", ticker, name) is not None else "other"


def _sentiment_model():
    with _model_lock:
        if not _model["tried"]:
            _model["tried"] = True
            try:
                from transformers import pipeline

                from tradingagents.research.compute import cuda_usable

                _model["pipe"] = pipeline("text-classification", model=FINBERT, top_k=None,
                                          device=0 if cuda_usable() else -1)
                _model["name"] = FINBERT
            except Exception as exc:  # no download, no torch: the word list stands in, and says so
                logger.warning("FinBERT unavailable, using the word list: %s", exc)
                _model["name"] = "lm-wordlist"
        return _model["pipe"], _model["name"]


def score_texts(texts: list[str]) -> tuple[list[float], str]:
    """Sentiment from -1 to 1 for each text, and the model that scored it."""
    pipe, name = _sentiment_model()
    if pipe is not None:
        out = []
        for res in pipe([t[:512] for t in texts], batch_size=16, truncation=True):
            p = {r["label"].lower(): r["score"] for r in res}
            out.append(float(p.get("positive", 0) - p.get("negative", 0)))
        return out, name
    scores = []
    for t in texts:
        words = re.findall(r"[a-z]+", t.lower())
        pos, neg = sum(w in POSITIVE for w in words), sum(w in NEGATIVE for w in words)
        scores.append((pos - neg) / max(pos + neg, 1))
    return scores, name


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / max(len(x | y), 1)


def store(con, ticker: str, articles: list[dict], name: str | None = None, now: datetime | None = None) -> int:
    """Insert the articles not seen before (first-seen time = now) and score them; returns how many were new."""
    from tradingagents.ledger.db import transaction

    now = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    recent = con.execute("SELECT id, title, published_at FROM stock_news WHERE ticker = ? AND duplicate_of IS NULL"
                         " ORDER BY published_at DESC LIMIT 300", (ticker,)).fetchall()
    known = {r["id"] for r in con.execute("SELECT id FROM stock_news WHERE ticker = ?", (ticker,))}
    new = []
    for a in articles:
        aid = article_id(ticker, a.get("title", ""))
        if aid in known or not a.get("title"):
            continue
        known.add(aid)
        dup = None
        published = a.get("pub_date")
        for r in recent:
            close_in_time = not (published and r["published_at"]) or abs(
                (datetime.fromisoformat(published) - datetime.fromisoformat(r["published_at"])).total_seconds()) < 3 * 86400
            if close_in_time and _jaccard(_norm(a["title"]), _norm(r["title"])) >= 0.8:
                dup = r["id"]
                break
        text = f"{a['title']}. {(a.get('summary') or '')[:300]}"
        new.append({"id": aid, "title": a["title"], "publisher": a.get("publisher"), "url": a.get("link"),
                    "published_at": published, "dup": dup, "kind": kind_of(a["title"]),
                    "events": events_in(text), "text": text,
                    "relevance": relevance(a["title"], a.get("summary"), ticker, name)})
    if not new:
        return 0
    scores, model = score_texts([n["text"] for n in new])
    with transaction(con):
        for n, s in zip(new, scores, strict=True):
            con.execute("INSERT OR IGNORE INTO stock_news VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (n["id"], ticker, n["title"], n["publisher"], n["url"], n["published_at"], now, n["dup"],
                         n["kind"], json.dumps({"events": n["events"], "relevance": n["relevance"]}), s, model))
    return len(new)


def collect(con, stocks: dict[str, dict], workers: int = 4, fetch=None) -> dict[str, str | int]:
    """Fetch and store news for each stock, four at a time; a failure is recorded for that stock only."""
    from tradingagents.dataflows.vendors.yahoo.news import ticker_articles

    fetch = fetch or ticker_articles
    got, out = {}, {}

    def one(t):
        try:
            return t, fetch(t), None
        except Exception as exc:
            return t, None, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, articles, error in pool.map(one, list(stocks)):
            if error is not None:
                out[t] = f"news failed: {str(error)[:120]}"
            else:
                got[t] = articles or []
    for t, articles in got.items():   # scoring and writing on one thread: one model, one connection
        out[t] = store(con, t, articles, stocks[t].get("name"))
    return out


def summary(con, ticker: str, cutoff: datetime, days: int = 7) -> dict:
    """The stock's news in the ``days`` before ``cutoff``: relevant, deduplicated headlines first seen
    by then, their average sentiment, and the events they carry."""
    since = (cutoff - timedelta(days=days)).isoformat()
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM stock_news WHERE ticker = ? AND duplicate_of IS NULL AND first_seen_at <= ?"
        " AND COALESCE(published_at, first_seen_at) >= ? ORDER BY COALESCE(published_at, first_seen_at) DESC",
        (ticker, cutoff.isoformat(), since))]
    first = con.execute("SELECT MIN(first_seen_at) FROM stock_news WHERE ticker = ?", (ticker,)).fetchone()[0]
    items = []
    for r in rows:
        meta = json.loads(r["events_json"])
        if meta.get("relevance") == "other":
            continue
        items.append({"relevance": meta.get("relevance"), "title": r["title"], "publisher": r["publisher"], "url": r["url"], "published_at": r["published_at"],
                      "first_seen_at": r["first_seen_at"], "kind": r["kind"], "events": meta.get("events", []),
                      "sentiment": r["sentiment"]})
    about = [i for i in items if i["relevance"] == "about"]
    scored = [i["sentiment"] for i in about if i["sentiment"] is not None]
    facts = [i["sentiment"] for i in about if i["sentiment"] is not None and i["kind"] == "fact"]
    events = sorted({e for i in about for e in i["events"]})
    return {"window_days": days, "articles": len(about), "mentions": len(items) - len(about),
            "sentiment": sum(scored) / len(scored) if scored else None,
            "sentiment_facts_only": sum(facts) / len(facts) if facts else None,
            "positive": sum(s > 0.2 for s in scored), "negative": sum(s < -0.2 for s in scored),
            "events": events, "items": items[:12], "collected_since": first,
            "model": rows[0]["sentiment_model"] if rows else None,
            "note": None if about else f"no headline about it in {days} days: sentiment unknown, not neutral"}
