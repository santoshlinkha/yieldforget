"""AI request budgets hold at exactly their limit, with parallel requests and failed ones.

Synthetic: a priced fake model, usage written by hand; no model is called.
"""

import uuid

import pytest

from tradingagents.ledger import ai_gateway
from tradingagents.ledger.accounting import Ledger
from tradingagents.ledger.db import connect

pytestmark = pytest.mark.unit


class _Answer:
    """A LangChain-like result with token usage."""
    llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}, "model_name": "test-model"}
    generations = []


def _gw(tmp_path, price=(1.0, 4.0), **budget):
    led = Ledger(connect(tmp_path / "t.sqlite3"))
    led.ensure_account()
    policy = ai_gateway.Policy(budget=ai_gateway.Budget(**budget),
                               prices={"openai/test-model": ai_gateway.Price(input=price[0], output=price[1])})
    return ai_gateway.Gateway(led.con, policy)


def _run(gw, requests, est_requests=1, est_usd=0.0, answer=True):
    """Start ``requests`` model calls inside one job; how many were allowed before the budget stopped them."""
    allowed = []

    def work(cb):
        ids = [uuid.uuid4() for _ in range(requests)]
        try:
            for rid in ids:
                cb.on_chat_model_start({}, [], run_id=rid)
                allowed.append(rid)
                if answer:
                    cb.on_llm_end(_Answer(), run_id=rid)
        except ai_gateway.AIBlocked:
            pass
        return {"n": len(allowed)}

    gw.run(task="t", provider="openai", model="test-model", schema_version="v1", info_cutoff="2026-10-16",
           inputs={"k": uuid.uuid4().hex}, work=work, est_usd=est_usd, est_requests=est_requests, allow_paid=True)
    return len(allowed)


@pytest.fixture(autouse=True)
def _paid_ai_allowed(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_NO_PAID_AI", raising=False)


def test_a_limit_of_one_request_allows_exactly_one(tmp_path):
    gw = _gw(tmp_path, daily_requests=1, monthly_requests=1)
    assert _run(gw, 3) == 1
    assert gw.remaining()["daily_requests"] == 0


def test_a_limit_of_three_allows_three_across_jobs(tmp_path):
    gw = _gw(tmp_path, daily_requests=3, monthly_requests=100)
    assert _run(gw, 2, est_requests=1) == 2
    assert _run(gw, 5, est_requests=1) == 1


def test_parallel_requests_not_yet_answered_count_against_the_limit(tmp_path):
    gw = _gw(tmp_path, daily_requests=2, monthly_requests=100)
    assert _run(gw, 4, est_requests=2, answer=False) == 2       # four analysts start at once; two may go


def test_failed_requests_still_count(tmp_path):
    gw = _gw(tmp_path, daily_requests=2, monthly_requests=100)

    def work(cb):
        a, b = uuid.uuid4(), uuid.uuid4()
        cb.on_chat_model_start({}, [], run_id=a)
        cb.on_llm_error(RuntimeError("429"), run_id=a)
        cb.on_chat_model_start({}, [], run_id=b)
        cb.on_llm_end(_Answer(), run_id=b)
        with pytest.raises(ai_gateway.AIBlocked):
            cb.on_chat_model_start({}, [], run_id=uuid.uuid4())
        return {}

    gw.run(task="t", provider="openai", model="test-model", schema_version="v1", info_cutoff="2026-10-16",
           inputs={"k": 1}, work=work, est_usd=0.0, est_requests=1, allow_paid=True)


def test_no_dollars_left_stops_paid_models_but_not_free_ones(tmp_path):
    gw = _gw(tmp_path, daily_usd=0.0004, monthly_usd=10.0, daily_requests=100, monthly_requests=100)
    # A request's cost is only known once answered: 0.0001 left still lets one more go (to 0.0006), then no more.
    assert _run(gw, 5) == 2
    free = _gw(tmp_path / "free", price=(0.0, 0.0), daily_usd=0.0, monthly_usd=0.0,
               daily_requests=3, monthly_requests=3)
    assert _run(free, 5) == 3                            # free models are held by the request count only
