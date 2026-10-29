"""Point-in-time events: release calendars and policy / geopolitical indices, stored so that a past date sees only what
was known then.

- ``store``: the append-only event store (``<data cache>/optimizer/events.sqlite3``): scheduled releases and index
  observations, each with when its source published it, when this app first saw it, and a content hash; a changed
  schedule or a revised value is a new row, never an edit.
- ``sources``: adapters for public sources (BLS and BEA release schedules, the FOMC calendar, the Caldara-Iacoviello
  geopolitical risk index, the Baker-Bloom-Davis economic policy uncertainty index). Each reports ``ok``,
  ``unavailable`` (with the reason) or ``error``; one failing never stops the others.
- ``features``: market-wide calendar and index features for a decision cutoff and holding window, in two modes:
  ``strict`` (only what this app had received by the cutoff: live replay) and ``research`` (archived history with
  documented publication-lag assumptions; marked exploratory, since today's download is not evidence of what was
  known in the past).

    python -m tradingagents.events sync      # fetch every source once, append what is new
    python -m tradingagents.events status    # what each source holds and how fresh it is
"""
