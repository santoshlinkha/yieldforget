"""Entry plans: the Trader proposes how to enter, the risk debate challenges it, the Portfolio Manager
approves, revises or rejects it, and the app checks it locally before anyone treats it as actionable.

The model writes the plan; it never decides whether the plan can be carried out. ``validate`` does, from
local data only and with no model call:

- **Reference quote**: the close the plan is anchored on, as traded (not dividend- or split-adjusted), with
  its session, currency and how its basis was verified (``reference_quote``). A plan priced against an
  unverified basis is blocked.
- **Prices**: positive, a range ordered low to high, an entry inside its range, in the quote's currency,
  near the reference close, and not on the adjusted basis by mistake.
- **Account**: long-only (no entry from a Sell, no short), settled cash only after reservations, the
  position and whole-IRA concentration caps from the capital settings, the minimum order.
- **Timing**: no execution before the session after the reference close, an expiry after it, a plan past
  its expiry is expired; no day trading (no entry on a day the stock is sold, and no exit on the entry day).

A blocked plan is kept with its reasons, never dropped or repaired. Persistence (``record``) is
append-only: each run stores the Trader's proposal and the Portfolio Manager's final plan with the
approval that links them, the cutoff, the quote, the evidence, and the capital settings version. Which plan
is current, superseded or expired is worked out when the plans are read (``plans_for``).
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime
from pathlib import Path

ENTRY_PLAN_VERSION = "entry-plan-v1"
MAX_GAP = 0.20            # a plan priced further than this from the reference close is blocked
WARN_GAP = 0.08           # ...and further than this, warned
MAX_EXPIRY_SESSIONS = 20  # one holding horizon
NOT_GIVEN = "Insufficient data"


# --- the reference quote ---

def _cached_actions(ticker: str, cache_dir) -> tuple[dict | None, str | None]:
    """Dividends and splits from the local store only (newest snapshot of either kind), and when fetched."""
    try:
        from tradingagents.dataflows import local_store

        st = local_store.store(cache_dir)
        row = st.con.execute("SELECT payload, fetched_at FROM snapshots WHERE ticker = ? AND kind IN"
                             " ('corporate_actions', 'actions') ORDER BY fetched_at DESC LIMIT 1",
                             (ticker,)).fetchone() if st else None
        if row is not None:
            return json.loads(row["payload"]), row["fetched_at"]
    except Exception:  # an unreadable cache is missing actions, not an error
        pass
    return None, None


def _currency(ticker: str, cache_dir) -> str | None:
    try:
        from tradingagents.agents.packet import _cached_profile

        profile, _ = _cached_profile(ticker, cache_dir)
        return (profile or {}).get("currency")
    except Exception:
        return None


def reference_quote(ticker: str, cutoff: str, cache_dir) -> dict:
    """The last close on or before ``cutoff`` as it traded, from the local price store (never a download).

    The store keeps closes adjusted for later dividends and splits. On the store's latest session the two
    agree; for an earlier session the later dividends and splits are undone from the cached corporate
    actions, and only when those were fetched after the store's last session is the basis verified."""
    import pandas as pd

    from tradingagents.research.prices import bars
    from tradingagents.research.quality import raw_closes

    ticker = ticker.upper()
    out = {"ticker": ticker, "cutoff": cutoff, "session": None, "price": None, "adjusted": None,
           "currency": _currency(ticker, cache_dir), "basis": "as traded", "verified": False,
           "how": None, "source": "local price store (Yahoo daily bars)", "at": "the close (16:00 New York)"}
    empty = {f: pd.DataFrame() for f in ("Close", "High", "Low", "Volume")}
    try:
        b = bars([ticker], cache_dir, download=lambda *a, **k: empty)
        close = b["Close"][ticker].dropna() if b and ticker in b.get("Close", {}) else pd.Series(dtype=float)
    except Exception as exc:
        return {**out, "how": f"no prices in the local store ({str(exc)[:80]})"}
    upto = close[close.index <= pd.Timestamp(cutoff)]
    if upto.empty:
        return {**out, "how": "no stored close on or before the cutoff"}
    session, adjusted = upto.index[-1], float(upto.iloc[-1])
    out.update({"session": str(session.date()), "adjusted": round(adjusted, 4)})
    if session == close.index[-1]:
        return {**out, "price": round(adjusted, 4), "verified": True,
                "how": "the store's latest session: its adjusted close is the close as traded"}
    actions, fetched = _cached_actions(ticker, cache_dir)
    if actions is None or not fetched or fetched[:10] < str(close.index[-1].date()):
        return {**out, "how": "later sessions are stored but no corporate actions fetched since: the basis "
                              "cannot be verified"}
    split_adjusted = float(raw_closes(close, actions.get("dividends") or []).loc[session])
    later = [(str(d)[:10], float(r)) for d, r in actions.get("splits") or []
             if r and str(session.date()) < str(d)[:10] <= str(close.index[-1].date())]
    price = split_adjusted * math.prod(r for _, r in later)
    n_div = sum(1 for d, _ in actions.get("dividends") or [] if str(d)[:10] > str(session.date()))
    return {**out, "price": round(price, 4), "verified": True,
            "how": f"undid {n_div} later dividend(s) and {len(later)} later split(s) from corporate actions "
                   f"fetched {fetched[:10]}"}


def render_reference(ref: dict | None) -> str:
    """The quote as one line for the prompts that need it (Trader, risk analysts, Portfolio Manager)."""
    from tradingagents.ledger.sessions import next_session

    if not ref or ref.get("price") is None:
        return ("Entry reference: no verified as-traded close is available; an entry plan must be wait or "
                "unavailable.")
    return (f"Entry reference: {ref['ticker']} closed at {ref['price']} {ref.get('currency') or ''} on "
            f"{ref['session']} (as traded, not adjusted). Price any entry plan against it; the earliest "
            f"execution is {next_session(ref['session'])}.").replace("  ", " ")


# The prompt lines, kept here so the three prompts and the token measurement use the same text.
TRADER_INSTRUCTION = (
    "Also give entry_plan: status ready (enter from earliest_execution), conditional (only if trigger "
    "happens), wait (not now; trigger = what would make you reconsider) or unavailable. Prices are as-traded "
    "numbers in the reference currency near the entry reference close, never percentages; give earliest_execution "
    "and expires_at as YYYY-MM-DD and the evidence_ids you relied on. ")
RISK_INSTRUCTION = ("Challenge the trader's entry plan too: its price or range against the entry reference, "
                    "its trigger, timing and expiry.")
PM_INSTRUCTION = (
    "- **Entry plan**: approve the trader's entry plan, revise it, or reject it (then status wait or "
    "unavailable, with the reason); fill entry_plan with the final plan and entry_plan_review with "
    "approved / revised / rejected. Same price rules as the trader: as traded, near the entry reference.")


# --- the account ---

def account_view(store, ticker: str, as_of, cache_dir=None) -> dict:
    """What the plan is checked against: the trading budget (SGOV's interest and the match, less what is in
    stocks) after reservations, its value, what it holds, today's fills of this stock, the whole IRA's
    exposure to it, and the settings version."""
    from tradingagents.ledger import capital, income
    from tradingagents.optimizer.data import latest_prices
    from tradingagents.stocks.daily import rules

    r = rules(store)
    sleeve = r["sleeve"]
    state = store.ledger.state(sleeve, as_of)
    combined = store.ledger.state("combined", as_of)
    tickers = set(state.positions) | set(combined.positions)
    prices = latest_prices(cache_dir, tickers) if cache_dir and tickers else {}

    def worth(positions):
        return sum(q * float(prices.get(t) or 0.0) for t, q in positions.items() if q)

    fills = [e for e in store.ledger.events(None, as_of) if e["kind"] == "fill" and e["ticker"] == ticker.upper()]
    snap = capital.snapshot(store)
    budget = income.budget(store, cache_dir, as_of)["available"] if cache_dir else 0.0
    stocks = {t: q for t, q in state.positions.items() if t != income.RESERVE}
    return {"sleeve": sleeve, "available": max(0.0, budget - state.cash.reserved), "reserved": state.cash.reserved,
            "sleeve_value": budget + worth(stocks), "held": float(state.positions.get(ticker.upper(), 0.0)),
            "names_held": sum(1 for q in stocks.values() if q),
            "ira_total": combined.cash.total + worth(combined.positions),
            "ira_in_ticker": float(combined.positions.get(ticker.upper(), 0.0)) * float(prices.get(ticker.upper()) or 0.0),
            "sold_on": sorted({e["trade_date"] for e in fills if e["quantity"] < 0}),
            "bought_on": sorted({e["trade_date"] for e in fills if e["quantity"] > 0}),
            "rules": {k: r[k] for k in ("max_position_share", "max_names", "min_order", "combined_name_cap") if k in r},
            "settings": {"version": snap["version"], "hash": snap["hash"],
                         "daily_sgov_usd": snap.get("daily_sgov_usd")}}


# --- validation ---

def _day(text) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10]) if text else None
    except ValueError:
        return None


def validate(plan: dict | None, action: str | None, ref: dict, account: dict, today: date | None = None,
             stop_loss: float | None = None, price_target: float | None = None) -> dict:
    """Whether ``plan`` can be carried out in this account, and why not. ``action`` is the side it serves
    (the Trader's Buy / Hold / Sell or the Portfolio Manager's rating). Nothing is repaired: each problem
    is a block (the plan cannot be used) or a warning (it can, with care)."""
    from tradingagents.ledger.sessions import is_session, next_session, sessions_between

    today = today or date.today()
    blocks, warnings = [], []
    out = {"version": ENTRY_PLAN_VERSION, "checked_on": today.isoformat(), "blocks": blocks, "warnings": warnings,
           "sizing": None, "levels": None, "earliest_allowed": None}
    if not plan:
        return {**out, "state": "unavailable", "feasible": False, "blocks": ["no entry plan was given"]}
    status = str(plan.get("status") or "").lower()
    if status in ("wait", "unavailable"):
        if status == "wait" and not plan.get("trigger"):
            warnings.append("wait without a reconsideration condition")
        return {**out, "state": "waiting" if status == "wait" else "unavailable", "feasible": False}
    if status not in ("ready", "conditional"):
        return {**out, "state": "blocked", "feasible": False, "blocks": [f"unknown status {status or NOT_GIVEN}"]}

    side = str(action or "").title()
    if side not in ("Buy", "Overweight"):
        blocks.append(f"the action is {side or 'not given'}: an entry needs Buy or Overweight (long-only: "
                      "no short sale, no entry from a Sell)")
    # The quote: fresh, as traded, verified.
    if ref.get("price") is None or not ref.get("verified"):
        blocks.append(f"no verified as-traded reference price ({ref.get('how') or NOT_GIVEN})")
    ref_day = _day(ref.get("session"))
    allowed = next_session(max(d for d in (ref_day, _day(ref.get("cutoff"))) if d)) if ref_day else None
    out["earliest_allowed"] = allowed.isoformat() if allowed else None
    if ref_day and _day(ref.get("cutoff")) and sessions_between(ref_day, _day(ref["cutoff"])) > 1:
        blocks.append(f"the reference close ({ref_day}) is {sessions_between(ref_day, _day(ref['cutoff']))} "
                      "sessions older than the analysis cutoff")
    if ref_day and sessions_between(ref_day, today) > 5:
        warnings.append(f"the reference close is {sessions_between(ref_day, today)} sessions old: check the "
                        "price before acting")

    kind = str(plan.get("entry_type") or "").lower()
    price, low, high = plan.get("entry_price"), plan.get("entry_range_low"), plan.get("entry_range_high")
    if kind not in ("market", "limit", "conditional"):
        blocks.append(f"entry type {kind or NOT_GIVEN}: one of market, limit, conditional")
    if kind == "limit" and price is None and (low is None or high is None):
        blocks.append("a limit entry needs a price or a range")
    if (kind == "conditional" or status == "conditional") and not plan.get("trigger"):
        blocks.append("a conditional entry needs its trigger")
    for name, v in (("entry price", price), ("range low", low), ("range high", high)):
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0):
            blocks.append(f"{name} {v} is not a positive price")
    if low is not None and high is not None and isinstance(low, (int, float)) and isinstance(high, (int, float)):
        if low > high:
            blocks.append(f"range {low}-{high} is not ordered low to high")
        elif price is not None and not low <= price <= high:
            blocks.append(f"entry price {price} is outside its range {low}-{high}")
    if plan.get("currency") and ref.get("currency") and str(plan["currency"]).upper() != str(ref["currency"]).upper():
        blocks.append(f"currency {plan['currency']} differs from the quote's {ref['currency']}")
    elif not plan.get("currency"):
        warnings.append(f"no currency given: read as the quote's ({ref.get('currency') or NOT_GIVEN})")
    used = price if price is not None else (high if high is not None else None)
    ref_px = ref.get("price")
    if kind == "market" and used is None:
        used = ref_px
    if used is not None and ref_px and isinstance(used, (int, float)) and used > 0:
        gap = used / ref_px - 1
        if abs(gap) > MAX_GAP:
            blocks.append(f"priced {gap:+.0%} from the reference close {ref_px}: check the basis and the number")
        elif abs(gap) > WARN_GAP:
            warnings.append(f"priced {gap:+.0%} from the reference close {ref_px}")
        if kind == "limit" and gap > 0.02:
            warnings.append("a buy limit above the last close would fill at once, at the market")
        adj = ref.get("adjusted")
        if adj and abs(adj / ref_px - 1) > 0.01 and abs(used / adj - 1) < 0.005 and abs(gap) > 0.01:
            blocks.append(f"the price matches the adjusted close {adj}, not the as-traded close {ref_px}")

    first, expires = _day(plan.get("earliest_execution")), _day(plan.get("expires_at"))
    if plan.get("earliest_execution") and first is None:
        blocks.append(f"earliest execution {plan['earliest_execution']} is not a date")
    if first is None and allowed:
        first = allowed
        warnings.append(f"no earliest execution given: {allowed}, the session after the reference close")
    if first and allowed and first < allowed:
        blocks.append(f"earliest execution {first} is before {allowed}, the first session after the reference close")
    if first and not is_session(first):
        blocks.append(f"earliest execution {first} is not a trading session")
    if plan.get("expires_at") and expires is None:
        blocks.append(f"expiry {plan['expires_at']} is not a date")
    if expires is None:
        warnings.append("no expiry given: re-check the plan each session")
    elif first and expires < first:
        blocks.append(f"expires {expires}, before its earliest execution {first}")
    elif first and sessions_between(first, expires) > MAX_EXPIRY_SESSIONS:
        warnings.append(f"expires {sessions_between(first, expires)} sessions after it starts, longer than "
                        f"one {MAX_EXPIRY_SESSIONS}-session horizon")
    if first and first.isoformat() in account.get("sold_on", []):
        blocks.append(f"the stock was sold on {first}: buying back the same day is a day trade")
    warnings.append("no exit on the entry session (no day trading): a stop or target applies from the next "
                    "session")

    # Sizing: settled cash after reservations, within the position, name and whole-IRA caps.
    r = account.get("rules") or {}
    if used is not None and isinstance(used, (int, float)) and used > 0 and account:
        cap = r.get("max_position_share", 0.25) * account["sleeve_value"]
        slots = max(0, r.get("max_names", 5) - account["names_held"]) if not account["held"] else 1
        dollars = min(cap, account["available"] / max(slots, 1)) if slots else 0.0
        if account["held"]:
            warnings.append(f"already held ({account['held']:g} shares): this adds to the position")
        if not slots:
            blocks.append(f"name limit: {r.get('max_names')} names already held")
        if dollars < r.get("min_order", 1.0):
            blocks.append(f"settled cash: ${account['available']:,.2f} available after ${account['reserved']:,.2f} "
                          "reserved")
        ira_total = account.get("ira_total") or 0.0
        if ira_total and (account.get("ira_in_ticker", 0.0) + dollars) / ira_total > r.get("combined_name_cap", 1.0):
            blocks.append(f"IRA concentration: more than {r['combined_name_cap']:.0%} of the whole IRA in "
                          f"{ref.get('ticker')}")
        out["sizing"] = {"dollars": round(dollars, 2), "shares": round(dollars / used, 6), "price_used": used,
                         "available": round(account["available"], 2), "reserved": round(account["reserved"], 2),
                         "position_cap": round(cap, 2), "settings_version": (account.get("settings") or {}).get("version")}
        levels = {"stop_loss": stop_loss, "price_target": price_target}
        if stop_loss is not None and stop_loss >= used:
            warnings.append(f"stop-loss {stop_loss} is not below the entry {used}")
        if price_target is not None and price_target <= used:
            warnings.append(f"price target {price_target} is not above the entry {used}")
        if stop_loss is not None and price_target is not None and stop_loss < used < price_target:
            levels["reward_to_risk"] = round((price_target - used) / (used - stop_loss), 2)
        if stop_loss is not None and stop_loss < used:
            levels["loss_at_stop_usd"] = round(out["sizing"]["shares"] * (used - stop_loss), 2)
        out["levels"] = levels

    state = "blocked" if blocks else ("expired" if expires and expires < today else
                                      ("conditional" if status == "conditional" else "actionable"))
    return {**out, "state": state, "feasible": not blocks and state != "expired"}


# --- persistence ---

def record(store, ticker: str, analysis_date: str, run_ref: str | None, source: str, ref: dict, account: dict,
           trader: dict | None, final: dict | None, evidence: dict | None = None, today: date | None = None) -> dict:
    """Validate and save one run's plans: the Trader's proposal and the Portfolio Manager's final plan,
    linked by the approval. ``trader``/``final``: ``{"action"|"rating", "plan", "review", "stop_loss",
    "price_target"}`` (either may be None: a chat answer has no separate trader). Append-only."""
    from tradingagents.ledger.db import now_utc, transaction

    rows, parent = [], None
    with transaction(store.con):
        for stage, part in (("trader", trader), ("portfolio_manager", final)):
            if part is None:
                continue
            side = part.get("action") or part.get("rating")
            check = validate(part.get("plan"), side, ref, account, today, part.get("stop_loss"), part.get("price_target"))
            review = part.get("review") or ("proposed" if stage == "trader" else None)
            pid = uuid.uuid4().hex[:16]
            plan = part.get("plan")
            ev = {**(evidence or {}), "evidence_ids": list((plan or {}).get("evidence_ids") or [])}
            store.con.execute(
                "INSERT INTO entry_plans (id, created_at, ticker, analysis_date, run_ref, source, stage, review,"
                " parent_id, action, plan_json, reference_json, validation_json, settings_json, evidence_json,"
                " plan_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pid, now_utc(), ticker.upper(), analysis_date, run_ref, source, stage, review, parent, side,
                 json.dumps(plan) if plan is not None else None, json.dumps(ref), json.dumps(check),
                 json.dumps({"settings": account.get("settings"), "rules": account.get("rules"),
                             "sleeve": account.get("sleeve")}), json.dumps(ev), ENTRY_PLAN_VERSION))
            rows.append({"id": pid, "stage": stage, "review": review, "validation": check})
            if stage == "trader":
                parent = pid
    return {"plans": rows}


def plans_for(con, ticker: str, today: date | None = None) -> list[dict]:
    """Every run's plans for ``ticker``, newest first. The newest final plan is current; every earlier one
    is marked superseded by the plan after it; a plan past its expiry is expired, whatever it was."""
    today = today or date.today()
    rows = [dict(r) for r in con.execute("SELECT * FROM entry_plans WHERE ticker = ? ORDER BY created_at DESC, rowid DESC",
                                         (ticker.upper(),))]
    for r in rows:
        for k in ("plan", "reference", "validation", "settings", "evidence"):
            raw = r.pop(f"{k}_json")
            r[k] = json.loads(raw) if raw else None
        expires = _day((r["plan"] or {}).get("expires_at"))
        r["expired"] = bool(expires and expires < today)
    finals = [r for r in rows if r["stage"] == "portfolio_manager"]
    for newer, older in zip(finals, finals[1:]):
        older["superseded_by"] = newer["id"]
    by_parent = {r["parent_id"]: r for r in finals if r["parent_id"]}
    runs = []
    for r in finals:
        trader = next((t for t in rows if t["id"] == r["parent_id"]), None)
        runs.append({"final": r, "trader": trader, "current": "superseded_by" not in r and not r["expired"]})
    for t in rows:   # a trader proposal without a final plan (the run stopped before the manager)
        if t["stage"] == "trader" and t["id"] not in by_parent:
            runs.append({"final": None, "trader": t, "current": False})
    runs.sort(key=lambda run: (run["final"] or run["trader"])["created_at"], reverse=True)
    return runs


def from_state(state: dict) -> dict:
    """The structured plans a finished agent run left in its state (empty when the models gave none)."""
    return {"trader": state.get("trader_entry") or None, "final": state.get("final_entry") or None}


def save_run_file(folder: Path, state_plans: dict, ref: dict | None) -> None:
    """The run's plans beside its reports, as the models gave them (validation is the database's)."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "entry_plan.json").write_text(json.dumps({"version": ENTRY_PLAN_VERSION, "reference": ref,
                                                              **state_plans, "saved_at": datetime.now().isoformat(
                                                                  timespec="seconds")}, indent=1), encoding="utf-8")
