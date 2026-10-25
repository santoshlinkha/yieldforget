"""Every paid AI request goes through here: cached, budgeted, claimed once, and recorded.

- **Cache.** A result is keyed by the task, provider, model, prompt/schema version,
  information cutoff and a hash of the inputs that matter to that task, and
  nothing else: no request timestamp, so asking again reuses the answer, and a
  change to an input that matters asks anew.
- **Claim.** Before a paid job starts, its key is claimed in the database in the
  same transaction that checks the budget. A second request for the same key
  while it runs gets "in progress" instead of paying twice; a claim whose
  owner stopped heartbeating expires and can be taken over.
- **Budget.** Daily and monthly limits on dollars and requests are checked before
  dispatch, counting what has been spent, what running jobs have reserved, and
  this job's estimate. During the job, every model request is checked again by
  a callback, so a run that costs more than expected stops at the limit instead
  of after it. A model with no configured price cannot be called at all: an
  unknown cost cannot be held to a budget.
- **Record.** Every model request's tokens (cached and reasoning tokens where the
  provider reports them), latency and estimated cost are written to ``ai_usage``.
- **Opt-in only.** Nothing here runs unless the caller passes ``allow_paid=True``,
  which only an explicit user action does. Page loads, polling, replay and tests
  never do, and ``TRADINGAGENTS_NO_PAID_AI=1`` refuses every paid call outright.

AI and data costs are kept here, not in the brokerage books.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .db import now_utc, transaction

CLAIM_TTL = timedelta(minutes=20)


class AIBlocked(RuntimeError):
    """A paid request was refused before it was sent: budget, no price, no opt-in, or kill switch."""


class AIInProgress(RuntimeError):
    """The same job is already running; its result will be cached when it finishes."""


@dataclass
class Budget:
    daily_usd: float = 1.00
    monthly_usd: float = 10.00
    daily_requests: int = 200
    monthly_requests: int = 2000


@dataclass
class Price:
    """US dollars per million tokens."""
    input: float
    output: float
    cached_input: float | None = None


@dataclass
class Policy:
    budget: Budget = field(default_factory=Budget)
    prices: dict[str, Price] = field(default_factory=dict)     # "provider/model" -> price
    # Providers a task may fall back to, in order; empty: the configured provider only.
    fallbacks: list[str] = field(default_factory=list)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def cache_key(task: str, provider: str, model: str, schema_version: str, info_cutoff: str, inputs: dict) -> str:
    return stable_hash({"task": task, "provider": provider, "model": model, "schema": schema_version,
                        "cutoff": info_cutoff, "inputs": stable_hash(inputs)})


def _month_start(now: datetime) -> str:
    return now.strftime("%Y-%m-01")


class Gateway:
    def __init__(self, con, policy: Policy | None = None, owner: str | None = None):
        self.con, self.policy = con, policy or Policy()
        self.owner = owner or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

    # --- spending ---

    def spent(self, since: str) -> tuple[float, int]:
        row = self.con.execute("SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM ai_usage WHERE at >= ?",
                               (since,)).fetchone()
        return float(row[0]), int(row[1])

    def reserved(self) -> tuple[float, int]:
        cutoff = (datetime.now(UTC) - CLAIM_TTL).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        row = self.con.execute("SELECT COALESCE(SUM(reserved_usd), 0), COALESCE(SUM(reserved_requests), 0)"
                               " FROM ai_jobs WHERE status = 'claimed' AND heartbeat_at >= ?", (cutoff,)).fetchone()
        return float(row[0]), int(row[1])

    def remaining(self) -> dict:
        now = datetime.now(UTC)
        day_usd, day_n = self.spent(now.strftime("%Y-%m-%d"))
        month_usd, month_n = self.spent(_month_start(now))
        res_usd, res_n = self.reserved()
        b = self.policy.budget
        return {"daily_usd": b.daily_usd - day_usd - res_usd, "monthly_usd": b.monthly_usd - month_usd - res_usd,
                "daily_requests": b.daily_requests - day_n - res_n,
                "monthly_requests": b.monthly_requests - month_n - res_n,
                "spent_today_usd": day_usd, "spent_month_usd": month_usd,
                "requests_today": day_n, "requests_month": month_n, "budget": b.__dict__}

    def price(self, provider: str, model: str) -> Price:
        p = self.policy.prices.get(f"{provider}/{model}")
        if p is None:
            raise AIBlocked(f"no price is configured for {provider}/{model}, so its cost cannot be held to the "
                            f"budget; add it under Settings → Agent models before using it")
        return p

    def cost(self, provider: str, model: str, input_tokens: int, output_tokens: int, cached: int = 0) -> float:
        p = self.price(provider, model)
        cached_rate = p.cached_input if p.cached_input is not None else p.input
        return ((input_tokens - cached) * p.input + cached * cached_rate + output_tokens * p.output) / 1e6

    # --- cache ---

    def cached(self, key: str) -> Any | None:
        row = self.con.execute("SELECT output_json FROM ai_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    # --- running a job ---

    def run(self, *, task: str, provider: str, model: str, schema_version: str, info_cutoff: str, inputs: dict,
            work: Callable[[UsageCallback], Any], est_usd: float, est_requests: int,
            allow_paid: bool = False, also_priced: list[tuple[str, str]] = ()) -> tuple[Any, str]:
        """``(result, "cache" | "paid")``. ``work`` receives the callback to hand to the LLM client;
        ``also_priced`` lists any other models the work calls, which need a price too."""
        key = cache_key(task, provider, model, schema_version, info_cutoff, inputs)
        hit = self.cached(key)
        if hit is not None:
            return hit, "cache"
        if os.environ.get("TRADINGAGENTS_NO_PAID_AI") == "1":
            raise AIBlocked("paid AI is switched off (TRADINGAGENTS_NO_PAID_AI=1)")
        if not allow_paid:
            raise AIBlocked(f"{task} is not cached and paid AI was not requested")
        for p, m in [(provider, model), *also_priced]:
            self.price(p, m)
        self._claim(key, task, est_usd, est_requests)
        callback = UsageCallback(self, key, task, provider, model, models=[(provider, model), *also_priced])
        try:
            result = work(callback)
        except BaseException as exc:
            self._finish(key, "failed", str(exc)[:500])
            raise
        with transaction(self.con):
            self.con.execute("INSERT OR REPLACE INTO ai_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (key, task, provider, model, schema_version, info_cutoff, stable_hash(inputs),
                              json.dumps(result, default=str), now_utc()))
            self._finish(key, "done", None)
        return result, "paid"

    def _claim(self, key: str, task: str, est_usd: float, est_requests: int):
        with transaction(self.con):
            row = self.con.execute("SELECT status, heartbeat_at, owner FROM ai_jobs WHERE key = ?", (key,)).fetchone()
            stale = (datetime.now(UTC) - CLAIM_TTL).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            if row and row["status"] == "claimed" and row["heartbeat_at"] >= stale:
                raise AIInProgress(f"{task} is already running ({row['owner']})")
            left = self.remaining()
            for name, need in (("daily_usd", est_usd), ("monthly_usd", est_usd),
                               ("daily_requests", est_requests), ("monthly_requests", est_requests)):
                if left[name] < need:
                    raise AIBlocked(f"{task} needs about {need:g} of the {name.replace('_', ' ')} budget; "
                                    f"{max(left[name], 0):g} is left")
            self.con.execute("INSERT OR REPLACE INTO ai_jobs VALUES (?, ?, 'claimed', ?, ?, ?, ?, ?, NULL, NULL)",
                             (key, task, self.owner, est_usd, est_requests, now_utc(), now_utc()))

    def _finish(self, key: str, status: str, error: str | None):
        with transaction(self.con):
            self.con.execute("UPDATE ai_jobs SET status = ?, finished_at = ?, error = ?, reserved_usd = 0,"
                             " reserved_requests = 0 WHERE key = ?", (status, now_utc(), error, key))

    def heartbeat(self, key: str):
        with transaction(self.con):
            self.con.execute("UPDATE ai_jobs SET heartbeat_at = ? WHERE key = ? AND owner = ?",
                             (now_utc(), key, self.owner))

    def record(self, key: str | None, task: str, provider: str, model: str, *, input_tokens=0, output_tokens=0,
               cached=0, reasoning=0, latency_ms=None, status="ok", retry=0, note=None) -> float | None:
        try:
            cost = self.cost(provider, model, input_tokens, output_tokens, cached)
        except AIBlocked:
            cost = None
        with transaction(self.con):
            self.con.execute(
                "INSERT INTO ai_usage (job_key, task, provider, model, at, input_tokens, output_tokens,"
                " cached_input_tokens, reasoning_tokens, latency_ms, cost_usd, status, retry, note)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, task, provider, model, now_utc(), input_tokens, output_tokens, cached, reasoning,
                 latency_ms, cost, status, retry, note))
        return cost


def _usage(response) -> dict:
    """Token counts, and the model when reported, from a LangChain LLMResult."""
    out = {"input": 0, "output": 0, "cached": 0, "reasoning": 0, "model": None}
    for gens in getattr(response, "generations", []) or []:
        for g in gens:
            message = getattr(g, "message", None)
            out["model"] = out["model"] or (getattr(message, "response_metadata", None) or {}).get("model_name")
            meta = getattr(message, "usage_metadata", None) or {}
            out["input"] += meta.get("input_tokens", 0)
            out["output"] += meta.get("output_tokens", 0)
            out["cached"] += (meta.get("input_token_details") or {}).get("cache_read", 0)
            out["reasoning"] += (meta.get("output_token_details") or {}).get("reasoning", 0)
    if getattr(response, "llm_output", None):
        out["model"] = out["model"] or response.llm_output.get("model_name")
    if not out["input"] and getattr(response, "llm_output", None):
        usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
        out["input"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        out["output"] = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return out


try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object


class UsageCallback(BaseCallbackHandler):
    """Checks the budget before every model request and records each one's usage after it."""

    raise_error = True   # a refusal must stop the run, not be logged and ignored

    def __init__(self, gateway: Gateway, key: str, task: str, provider: str, model: str,
                 models: list[tuple[str, str]] | None = None):
        self.gateway, self.key, self.task, self.provider, self.model = gateway, key, task, provider, model
        # Every (provider, model) the work may call: a run can put agent roles on different providers.
        self.models = list(dict.fromkeys(models or [(provider, model)]))
        self._started: dict = {}
        self._asked: dict = {}     # run_id -> the model the request named
        self.requests = 0

    def _which(self, *names) -> tuple[str, str]:
        """The (provider, model) a request belongs to, from the model names it reported; a versioned
        name ("deepseek-v4-flash-2026-07-01") matches its base ID. Unknown: the job's own model."""
        for name in filter(None, names):
            for p, m in self.models:
                if name == m:
                    return p, m
            for p, m in sorted(self.models, key=lambda pm: -len(pm[1])):
                if name.startswith(m) or name.endswith("/" + m):
                    return p, m
        return self.provider, self.model

    def _note_request(self, run_id, kwargs):
        params = kwargs.get("invocation_params") or {}
        self._asked[run_id] = params.get("model") or params.get("model_name")

    def _check(self, run_id):
        left = self.gateway.remaining()
        # This job's own reservation is in ``reserved``; spending within it is allowed.
        if min(left["daily_usd"], left["monthly_usd"]) < -self._own_reserved() or \
                min(left["daily_requests"], left["monthly_requests"]) < 0 - self._own_requests():
            raise AIBlocked(f"{self.task} reached the AI budget mid-run; stopped before the next request")
        self.gateway.heartbeat(self.key)
        self._started[run_id] = time.monotonic()

    def _own_reserved(self) -> float:
        row = self.gateway.con.execute("SELECT reserved_usd FROM ai_jobs WHERE key = ?", (self.key,)).fetchone()
        return float(row[0]) if row else 0.0

    def _own_requests(self) -> int:
        row = self.gateway.con.execute("SELECT reserved_requests FROM ai_jobs WHERE key = ?", (self.key,)).fetchone()
        return int(row[0]) if row else 0

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._check(run_id)
        self._note_request(run_id, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._check(run_id)
        self._note_request(run_id, kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self.requests += 1
        u = _usage(response)
        started = self._started.pop(run_id, None)
        # Priced as the model that answered (a run can use a quick and a deep model, on several providers).
        provider, model = self._which(self._asked.pop(run_id, None), u["model"])
        self.gateway.record(self.key, self.task, provider, model, input_tokens=u["input"],
                            output_tokens=u["output"], cached=u["cached"], reasoning=u["reasoning"],
                            latency_ms=int((time.monotonic() - started) * 1000) if started else None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._started.pop(run_id, None)
        provider, model = self._which(self._asked.pop(run_id, None))
        self.gateway.record(self.key, self.task, provider, model, status="error", note=str(error)[:300])
