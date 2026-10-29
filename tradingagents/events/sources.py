"""Public sources of release calendars and policy / geopolitical indices. Each adapter downloads once, parses into
plain rows and leaves storage to ``store``; ``sync`` runs them all and records each one's outcome.

Verified 2026-10-29 (formats as downloaded that day):

- BLS release schedule, ``https://www.bls.gov/schedule/news_release/bls.ics``: iCalendar, one VEVENT per release with
  a stable UID, ``DTSTART;TZID=US-Eastern`` (the release minute) and the release's name. BLS refuses a request whose
  User-Agent carries no contact address; the SEC adapter's setting (``SEC_EDGAR_USER_AGENT``) is reused.
- BEA release schedule, ``https://www.bea.gov/news/schedule``: an HTML table of date, time (Eastern) and release.
- FOMC calendar, ``https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm``: meetings by year; the statement
  is released at 14:00 Eastern on the meeting's last day (Federal Reserve practice; an unscheduled meeting is marked
  on the page and kept with its own status).
- Geopolitical Risk index (Caldara and Iacoviello), ``https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls``:
  daily GPRD with acts and threats, from 1985. The site updates it weekly and values can be revised.
- Economic Policy Uncertainty (Baker, Bloom and Davis), ``https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx``:
  the monthly news-based US index from 1985. "Free to use with attribution to the authors, the paper and the website."

Not yet implemented (each would be its own adapter): Federal Register documents, Trade Policy Uncertainty, OFAC,
BIS, USITC, GDELT, ALFRED vintages of releases, an earnings-calendar provider. ``SOURCES`` lists only what exists.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import store as ST

EXTRACTION = "events-sources-v1"
NY = ZoneInfo("America/New_York")

URLS = {"bls": "https://www.bls.gov/schedule/news_release/bls.ics",
        "bea": "https://www.bea.gov/news/schedule",
        "fomc": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "gpr": "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
        "epu": "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"}

# Which releases matter to a market-wide calendar, by the words in their names.
BLS_TYPES = (("cpi", r"^consumer price index"), ("employment", r"^employment situation"),
             ("ppi", r"^producer price index"), ("jolts", r"^job openings and labor turnover"),
             ("eci", r"^employment cost index"), ("productivity", r"^productivity and costs"))
BEA_TYPES = (("gdp", r"^gdp"), ("pce", r"^personal income and outlays"), ("trade", r"^u\.s\. international trade"))


def _get(url: str) -> bytes:
    import requests

    from tradingagents.dataflows.vendors.sec_edgar import _user_agent

    r = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=60)
    r.raise_for_status()
    if b"Access Denied" in r.content[:2000] and url.endswith(".ics"):
        raise PermissionError("the source refused the request (it wants a contact address in SEC_EDGAR_USER_AGENT)")
    return r.content


def _utc(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")


def classify(name: str, table) -> str | None:
    low = name.strip().lower()
    return next((kind for kind, pattern in table if re.search(pattern, low)), None)


# --- BLS ---

def parse_bls(ics: bytes) -> list[dict]:
    text = ics.decode("utf-8", "replace").replace("\r\n ", "").replace("\n ", "")
    out = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        fields = {}
        for line in block.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.split(";")[0].strip()] = (key, value.strip())
        summary = fields.get("SUMMARY", ("", ""))[1].replace("\\,", ",")
        kind = classify(summary, BLS_TYPES)
        if not kind or "DTSTART" not in fields:
            continue
        key, value = fields["DTSTART"]
        when = datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=NY) if "T" in value else \
            datetime.strptime(value[:8], "%Y%m%d").replace(hour=8, minute=30, tzinfo=NY)
        out.append({"source_event_id": fields.get("UID", ("", f"{kind}-{value}"))[1], "event_type": kind,
                    "source_url": URLS["bls"], "evidence_excerpt": summary[:300], "scheduled_at": _utc(when),
                    "original_timezone": "America/New_York", "status": "scheduled", "countries": "US",
                    "reliability": "official schedule"})
    return out


# --- BEA ---

def parse_bea(html: bytes) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out, year = [], None
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Year (\d{4})", text)
    year = int(m.group(1)) if m else date.today().year
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        joined = " | ".join(cells)
        dm = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})",
                       joined)
        tm = re.search(r"(\d{1,2}):(\d{2}) ([AP]M)", joined)
        title = cells[-1]
        kind = classify(re.sub(r"^News\s*", "", title), BEA_TYPES)
        if not (dm and tm and kind):
            continue
        hour = int(tm.group(1)) % 12 + (12 if tm.group(3) == "PM" else 0)
        when = datetime.strptime(f"{dm.group(1)} {dm.group(2)} {year}", "%B %d %Y").replace(
            hour=hour, minute=int(tm.group(2)), tzinfo=NY)
        ref = re.search(r",\s*(.+)$", title)
        out.append({"source_event_id": f"{kind}:{title}", "event_type": kind, "source_url": URLS["bea"],
                    "evidence_excerpt": title[:300], "scheduled_at": _utc(when), "reference_period": ref.group(1) if ref else None,
                    "original_timezone": "America/New_York", "status": "scheduled", "countries": "US",
                    "reliability": "official schedule"})
    return out


# --- FOMC ---

def parse_fomc(html: bytes) -> list[dict]:
    """Each meeting's statement: 14:00 New York on the meeting's last day. Footnoted meetings (``*``: with a Summary of
    Economic Projections) are kept; an unscheduled one keeps its own label."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out = []
    for panel in soup.select("div.panel"):
        head = panel.find(["h4", "h5"])
        ym = re.search(r"(\d{4}) FOMC Meetings", head.get_text(" ", strip=True) if head else "")
        if not ym:
            continue
        year = int(ym.group(1))
        for row in panel.select("div.fomc-meeting, div.row.fomc-meeting"):
            month_el = row.select_one(".fomc-meeting__month")
            date_el = row.select_one(".fomc-meeting__date")
            if not (month_el and date_el):
                continue
            month = month_el.get_text(" ", strip=True).split("/")[-1].strip()
            days = date_el.get_text(" ", strip=True)
            last = re.findall(r"\d{1,2}", days)
            if not last:
                continue
            # A meeting spanning two months ("Apr/May 30-1") ends in the second month.
            try:
                end = datetime.strptime(f"{month[:3]} {last[-1]} {year}", "%b %d %Y")
            except ValueError:
                continue
            when = end.replace(hour=14, tzinfo=NY)
            label = "unscheduled" if "unscheduled" in days.lower() else ("with projections" if "*" in days else "")
            out.append({"source_event_id": f"fomc:{year}:{month}:{days}", "event_type": "fomc_statement",
                        "source_url": URLS["fomc"], "evidence_excerpt": f"FOMC meeting {month} {days} {year} {label}".strip(),
                        "scheduled_at": _utc(when), "original_timezone": "America/New_York", "status": "scheduled",
                        "countries": "US", "reliability": "official calendar"})
    return out


# --- indices ---

def parse_gpr(xls: bytes) -> list[tuple[str, str, float, str]]:
    df = pd.read_excel(io.BytesIO(xls))
    day = pd.to_datetime(df["DAY"].astype(str), format="%Y%m%d", errors="coerce")
    rows = []
    for col, series in (("GPRD", "gpr_daily"), ("GPRD_ACT", "gpr_act"), ("GPRD_THREAT", "gpr_threat")):
        for d, v in zip(day, pd.to_numeric(df[col], errors="coerce"), strict=True):
            if pd.notna(d) and pd.notna(v):
                rows.append((series, d.date().isoformat(), float(v), "index (1985-2019 = 100)"))
    return rows


def parse_epu(xlsx: bytes) -> list[tuple[str, str, float, str]]:
    df = pd.read_excel(io.BytesIO(xlsx))
    year = pd.to_numeric(df["Year"], errors="coerce")
    month = pd.to_numeric(df["Month"], errors="coerce")
    value = pd.to_numeric(df["News_Based_Policy_Uncert_Index"], errors="coerce")
    rows = []
    for y, m, v in zip(year, month, value, strict=True):
        if pd.notna(y) and pd.notna(m) and pd.notna(v) and 1985 <= y <= 2100:
            rows.append(("epu_monthly", date(int(y), int(m), 1).isoformat(), float(v), "index (1985-2009 = 100)"))
    return rows


CALENDARS = {"bls": parse_bls, "bea": parse_bea, "fomc": parse_fomc}
INDICES = {"gpr": parse_gpr, "epu": parse_epu}
SOURCES = (*CALENDARS, *INDICES)


def sync(cache_dir, get=_get, only: tuple[str, ...] | None = None) -> list[dict]:
    """Fetch each source once and append what is new. A source that fails is recorded and the others still run."""
    con = ST.connect(cache_dir)
    report = []
    for name in only or SOURCES:
        try:
            blob = get(URLS[name])
            h = ST.digest(blob)
            if name in CALENDARS:
                parsed = CALENDARS[name](blob)
                if not parsed:
                    raise ValueError("no events parsed: the source's format may have changed")
                n = ST.add_events(con, name, parsed, EXTRACTION)
            else:
                parsed = INDICES[name](blob)
                if not parsed:
                    raise ValueError("no observations parsed: the source's format may have changed")
                n = ST.add_observations(con, name, URLS[name], parsed, h, EXTRACTION)
            ST.log_fetch(con, name, "ok", f"{len(parsed)} parsed", n, h)
            report.append({"source": name, "status": "ok", "parsed": len(parsed), "added": n})
        except PermissionError as exc:
            ST.log_fetch(con, name, "unavailable", str(exc))
            report.append({"source": name, "status": "unavailable", "detail": str(exc)})
        except Exception as exc:  # one source failing never stops the others
            ST.log_fetch(con, name, "error", f"{type(exc).__name__}: {exc}")
            report.append({"source": name, "status": "error", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"})
    con.close()
    return report


def _unused(_: timedelta) -> None:   # keep the import explicit for type checkers
    return None
