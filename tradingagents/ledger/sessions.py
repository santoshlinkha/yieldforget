"""NYSE trading sessions: which dates trade, and the next or previous session.

Settlement (T+1 counts sessions, not days) and the no-day-trading rule both need
real sessions. The rules below are the exchange's full-day holidays since 2022
(Juneteenth included), with its weekend observance: a Saturday holiday closes the
Friday before, except New Year's Day, and a Sunday holiday closes the Monday
after. Unscheduled closures, such as a national day of mourning, are not
predictable; ``EXTRA_CLOSURES`` lists the known ones, and a date with real
closing prices always wins over the rules when a price index is given.

Some sessions close early, at 1:00 pm (``early_closes``); anything timed from the
close (an order cutoff, when a day's bar is final) uses ``session_close``.

This is US equities and options. Futures sessions are not calendar dates: they
open the evening before, and their calendar lives with the futures code.
"""

from __future__ import annotations

from datetime import date, time, timedelta
from functools import lru_cache

EXTRA_CLOSURES = {
    *(date(2001, 9, d) for d in (11, 12, 13, 14)),   # September 11
    date(2004, 6, 11),   # President Reagan
    date(2007, 1, 2),    # President Ford
    date(2012, 10, 29), date(2012, 10, 30),          # Hurricane Sandy
    date(2018, 12, 5),   # President G. H. W. Bush
    date(2025, 1, 9),    # President Carter
}


def _easter(year: int) -> date:
    """Western Easter (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    return date(year, month, (h + m - 7 * n + 114) % 31 + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date, saturday_rule: bool = True) -> date | None:
    if day.weekday() == 5:
        return day - timedelta(days=1) if saturday_rule else None
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[date]:
    days = {
        _observed(date(year, 1, 1), saturday_rule=False),       # New Year's Day
        _nth_weekday(year, 1, 0, 3),                             # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),                             # Washington's Birthday
        _easter(year) - timedelta(days=2),                       # Good Friday
        _last_weekday(year, 5, 0),                               # Memorial Day
        _observed(date(year, 7, 4)),                             # Independence Day
        _nth_weekday(year, 9, 0, 1),                             # Labor Day
        _nth_weekday(year, 11, 3, 4),                            # Thanksgiving
        _observed(date(year, 12, 25)),                           # Christmas
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))                   # Juneteenth
    # New Year's Day on a Saturday is not made up the Friday before (it is in the old year).
    return frozenset(d for d in days if d is not None) | {d for d in EXTRA_CLOSURES if d.year == year}


REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
EXTRA_EARLY_CLOSES: set[date] = set()   # announced one-off half days, beyond the rules below


@lru_cache(maxsize=None)
def early_closes(year: int) -> frozenset[date]:
    """Sessions that close at 1:00 pm New York: the day before Independence Day, the day after
    Thanksgiving and Christmas Eve, each when it is itself a session."""
    days = {date(year, 7, 3), _nth_weekday(year, 11, 3, 4) + timedelta(days=1), date(year, 12, 24)}
    return frozenset(d for d in days | {d for d in EXTRA_EARLY_CLOSES if d.year == year} if is_session(d))


def session_close(day) -> time:
    """The regular-trading close of a session, New York time."""
    day = as_date(day)
    return EARLY_CLOSE if day in early_closes(day.year) else REGULAR_CLOSE


def as_date(day) -> date:
    if isinstance(day, date):
        return day
    return date.fromisoformat(str(day)[:10])


def is_session(day) -> bool:
    day = as_date(day)
    return day.weekday() < 5 and day not in holidays(day.year)


def next_session(day, n: int = 1) -> date:
    """The ``n``-th session after ``day`` (``day`` itself does not count)."""
    day = as_date(day)
    while n > 0:
        day += timedelta(days=1)
        if is_session(day):
            n -= 1
    return day


def previous_session(day, n: int = 1) -> date:
    day = as_date(day)
    while n > 0:
        day -= timedelta(days=1)
        if is_session(day):
            n -= 1
    return day


def session_on_or_after(day) -> date:
    day = as_date(day)
    return day if is_session(day) else next_session(day)


def sessions_between(start, end) -> int:
    """Sessions after ``start`` up to and including ``end``; 0 for the same day."""
    start, end = as_date(start), as_date(end)
    n, day = 0, start
    while day < end:
        day += timedelta(days=1)
        n += is_session(day)
    return n
