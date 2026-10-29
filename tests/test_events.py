"""The event store: append-only versions, reschedules, revised index values, strict vs research availability, and a
source that fails without stopping the others. Synthetic inputs only."""

import sqlite3

import numpy as np
import pytest

from tradingagents.events import features as F
from tradingagents.events import sources as S
from tradingagents.events import store as ST

pytestmark = pytest.mark.unit

ICS = b"""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:cpi-oct
DTSTART;TZID=US-Eastern:20261110T083000
SUMMARY:Consumer Price Index
END:VEVENT
BEGIN:VEVENT
UID:jobs-nov
DTSTART;TZID=US-Eastern:20261106T083000
SUMMARY:Employment Situation
END:VEVENT
BEGIN:VEVENT
UID:metro
DTSTART;TZID=US-Eastern:20261104T100000
SUMMARY:Metropolitan Area Employment and Unemployment (Monthly)
END:VEVENT
END:VCALENDAR
"""


def test_bls_calendar_keeps_the_market_releases_in_utc():
    ev = {e["source_event_id"]: e for e in S.parse_bls(ICS)}
    assert set(ev) == {"cpi-oct", "jobs-nov"}                               # the metro release is not market-wide
    assert ev["cpi-oct"]["event_type"] == "cpi" and ev["cpi-oct"]["scheduled_at"] == "2026-11-10T13:30:00Z"


def test_events_are_append_only_and_a_reschedule_is_a_new_version(tmp_path):
    con = ST.connect(tmp_path)
    first = S.parse_bls(ICS)
    assert ST.add_events(con, "bls", first, "t", seen_at="2026-10-01T00:00:00Z") == 2
    assert ST.add_events(con, "bls", first, "t", seen_at="2026-10-02T00:00:00Z") == 0      # unchanged: nothing added
    moved = [dict(e, scheduled_at="2026-11-13T13:30:00Z") if e["source_event_id"] == "cpi-oct" else e for e in first]
    assert ST.add_events(con, "bls", moved, "t", seen_at="2026-10-20T00:00:00Z") == 1
    rows = con.execute("SELECT status, scheduled_at, revision_of FROM events WHERE source_event_id = 'cpi-oct' "
                       "ORDER BY row_id").fetchall()
    assert [r["status"] for r in rows] == ["scheduled", "rescheduled"] and rows[1]["revision_of"]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE events SET status = 'x'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM events")

    # Strict replay sees the version known at each cutoff: the old date before the reschedule, the new one after.
    before = F.calendar_features(con, "2026-10-15", 20, "strict")
    after = F.calendar_features(con, "2026-10-21", 20, "strict")
    assert before["sessions_to_cpi"] == F.sessions_between(F.pd.Timestamp("2026-10-15").date(),
                                                           F.pd.Timestamp("2026-11-10").date())
    assert after["sessions_to_cpi"] == F.sessions_between(F.pd.Timestamp("2026-10-21").date(),
                                                          F.pd.Timestamp("2026-11-13").date())
    assert F.calendar_features(con, "2026-09-01", 20, "strict")["releases_in_window"] == 0   # not received yet


def test_research_mode_assumes_a_lead_and_is_marked_exploratory(tmp_path):
    con = ST.connect(tmp_path)
    ST.add_events(con, "bls", S.parse_bls(ICS), "t", seen_at="2026-10-28T00:00:00Z")      # received late
    strict = F.calendar_features(con, "2026-10-20", 20, "strict")
    research = F.calendar_features(con, "2026-10-20", 20, "research")
    assert strict["releases_in_window"] == 0 and not strict["exploratory"]
    assert research["cpi_in_window"] == 1 and research["employment_in_window"] == 1 and research["exploratory"]
    early = F.calendar_features(con, "2026-09-01", 20, "research")                       # beyond the 30-day lead
    assert np.isnan(early["sessions_to_cpi"])


def test_a_revised_index_value_is_a_new_vintage_and_the_past_keeps_the_old_one(tmp_path):
    con = ST.connect(tmp_path)
    rows = [("gpr_daily", f"2026-09-{d:02d}", 100.0 + d, "idx") for d in range(1, 29)]
    ST.add_observations(con, "gpr", "u", rows, "h1", "t", seen_at="2026-10-01T00:00:00Z")
    revised = [(s, d, v + (50 if d == "2026-09-28" else 0), u) for s, d, v, u in rows]
    assert ST.add_observations(con, "gpr", "u", revised, "h2", "t", seen_at="2026-10-10T00:00:00Z") == 1
    old = F.series_as_of(con, "gpr_daily", "2026-10-05", "strict")
    new = F.series_as_of(con, "gpr_daily", "2026-10-12", "strict")
    assert old.iloc[-1] == 128.0 and new.iloc[-1] == 178.0                  # the revision is not seen before it came
    assert F.series_as_of(con, "gpr_daily", "2026-09-15", "strict").empty   # nothing received by then
    research = F.series_as_of(con, "gpr_daily", "2026-09-15", "research")
    assert research.index.max() <= F.pd.Timestamp("2026-09-08")             # the 7-day publication lag


def test_index_features_flag_missing_series(tmp_path):
    con = ST.connect(tmp_path)
    out = F.index_features(con, "2026-10-20", "strict")
    assert out["gpr_missing"] == 1 and out["epu_missing"] == 1 and np.isnan(out["gpr_z"])


def test_one_source_failing_or_refusing_does_not_stop_the_others(tmp_path):
    def get(url):
        if url == S.URLS["bls"]:
            raise PermissionError("refused")
        if url == S.URLS["bea"]:
            raise TimeoutError("slow")
        if url == S.URLS["fomc"]:
            return b"<html></html>"
        raise RuntimeError("down")

    report = {r["source"]: r for r in S.sync(tmp_path, get=get)}
    assert report["bls"]["status"] == "unavailable" and report["bea"]["status"] == "error"
    assert report["fomc"]["status"] == "error" and "format" in report["fomc"]["detail"]   # parsed nothing: said so
    assert {s["source"] for s in ST.status(ST.connect(tmp_path))} == set(S.SOURCES)
