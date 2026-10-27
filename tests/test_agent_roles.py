"""Agent roles on their own providers: resolution, validation, the graph's clients, and usage pricing."""

import copy

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import roles

pytestmark = pytest.mark.unit

BASE = {"llm_provider": "openai", "quick_think_llm": "q", "deep_think_llm": "d"}


def test_roles_default_to_the_quick_or_deep_model_and_move_when_mapped():
    out = roles.resolve(BASE)
    assert out["bull"] == ("openai", "q") and out["portfolio_manager"] == ("openai", "d")
    moved = roles.resolve({**BASE, "agent_models": {"bear": {"provider": "Google", "model": "g-flash"}}})
    assert moved["bear"] == ("google", "g-flash") and moved["bull"] == ("openai", "q")
    assert roles.models_used({**BASE, "agent_models": {"bear": {"provider": "google", "model": "g"}}}) == \
        [("google", "g"), ("openai", "d"), ("openai", "q")]


@pytest.mark.parametrize("bad, why", [({"judge": {"provider": "openai", "model": "x"}}, "unknown agent role"),
                                       ({"bull": {"provider": "nope", "model": "x"}}, "unknown provider"),
                                       ({"bull": {"provider": "openai", "model": ""}}, "no model")])
def test_bad_mappings_are_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        roles.validate(bad)


def test_the_suggestion_puts_opposing_sides_on_different_providers():
    assert roles.suggest("openai", ["openai"]) == {}
    two = roles.suggest("openai", ["google", "openai"])
    assert two["bull"]["provider"] == "openai" and two["bear"]["provider"] == "google"
    assert two["aggressive"]["provider"] != two["conservative"]["provider"]
    three = roles.suggest("openai", ["openai", "google", "deepseek"])
    assert three["research_manager"]["provider"] not in (three["bull"]["provider"], three["bear"]["provider"])


def test_the_graph_builds_one_client_per_provider_and_model(tmp_path, monkeypatch):
    from tradingagents.graph import trading_graph

    made = []

    class Client:
        def __init__(self, provider, model, base_url, **kwargs):
            made.append((provider, model, base_url))
            self.llm = f"{provider}/{model}"

        def get_llm(self):
            return self.llm

    from tradingagents.llm_clients import factory

    fake = lambda provider, model, base_url=None, **k: Client(provider, model, base_url, **k)  # noqa: E731
    monkeypatch.setattr(trading_graph, "create_llm_client", fake)
    monkeypatch.setattr(factory, "create_llm_client", fake)   # the quick and deep tiers (upstream's tier clients)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "r"), data_cache_dir=str(tmp_path / "c"), memory_log_path=str(tmp_path / "m.md"),
               backend_url="http://own-endpoint", agent_models={"bear": {"provider": "google", "model": "g-flash"},
                                                                "conservative": {"provider": "google", "model": "g-flash"}})
    graph = trading_graph.TradingAgentsGraph(config=cfg)
    assert graph.llms["bear"] == graph.llms["conservative"] == "google/g-flash"
    assert graph.llms["bull"] == f"openai/{cfg['quick_think_llm']}"
    assert sorted(made) == sorted(set(made))                                   # one client each
    assert ("google", "g-flash", None) in made                                 # another provider never gets this backend URL
    assert graph.run_settings()["agent_models"]["bear"]["provider"] == "google"


def test_usage_is_priced_under_the_provider_that_answered(tmp_path):
    from types import SimpleNamespace

    from tradingagents.ledger.ai_gateway import Gateway, Policy, Price, UsageCallback
    from tradingagents.ledger.store import Store

    store = Store(tmp_path)
    prices = {"openai/q": Price(1.0, 2.0), "google/g-flash": Price(10.0, 20.0)}
    gw = Gateway(store.con, Policy(prices=prices))
    cb = UsageCallback(gw, None, "rating", "openai", "q", models=[("openai", "q"), ("google", "g-flash")])
    message = SimpleNamespace(response_metadata={"model_name": "g-flash-2026-10-01"},
                              usage_metadata={"input_tokens": 1_000_000, "output_tokens": 0})
    cb._note_request("r1", {"invocation_params": {"model": "g-flash"}})
    cb.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output=None), run_id="r1")
    row = store.con.execute("SELECT provider, model, cost_usd FROM ai_usage").fetchone()
    assert (row[0], row[1], row[2]) == ("google", "g-flash", 10.0)


def test_the_setting_is_validated_when_stored(tmp_path):
    from tradingagents.ledger.store import Store

    store = Store(tmp_path)
    store.set("global", "ai_agent_models", {"bull": {"provider": "DeepSeek", "model": "deepseek-flash"}})
    assert store.get("global", "ai_agent_models") == {"bull": {"provider": "deepseek", "model": "deepseek-flash"}}
    with pytest.raises(ValueError):
        store.set("global", "ai_agent_models", {"bull": {"provider": "nope", "model": "x"}})
