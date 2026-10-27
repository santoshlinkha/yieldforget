"""Which model each agent role runs on.

By default every role uses the run's provider: the managers (research and portfolio)
the deep model, everyone else the quick one. ``config["agent_models"]`` moves any role
to its own provider and model, e.g. the bull on OpenAI and the bear on Gemini so the
debate is not one model arguing with itself:

    {"bull": {"provider": "openai", "model": "gpt-6-luna"},
     "bear": {"provider": "google", "model": "gemini-3-flash-preview"}}

Roles missing from the mapping keep the default.
"""

from __future__ import annotations

from .api_key_env import PROVIDER_API_KEY_ENV

# role -> (label, stage, default tier)
ROLES: dict[str, tuple[str, str, str]] = {
    "market": ("Market analyst", "analysts", "quick"),
    "social": ("Sentiment analyst", "analysts", "quick"),
    "news": ("News analyst", "analysts", "quick"),
    "fundamentals": ("Fundamentals analyst", "analysts", "quick"),
    "bull": ("Bull researcher", "research debate", "quick"),
    "bear": ("Bear researcher", "research debate", "quick"),
    "research_manager": ("Research manager", "decision", "deep"),
    "trader": ("Trader", "decision", "quick"),
    "aggressive": ("Aggressive risk analyst", "risk debate", "quick"),
    "neutral": ("Neutral risk analyst", "risk debate", "quick"),
    "conservative": ("Conservative risk analyst", "risk debate", "quick"),
    "portfolio_manager": ("Portfolio manager", "decision", "deep"),
    "reflector": ("Reflection on past calls", "memory", "quick"),
}


def validate(mapping) -> dict[str, dict[str, str]]:
    """The mapping cleaned to {role: {"provider", "model"}}; raises ValueError on an unknown role or provider."""
    out = {}
    for role, spec in (mapping or {}).items():
        if role not in ROLES:
            raise ValueError(f"unknown agent role {role!r}; roles are {', '.join(ROLES)}")
        if not spec:
            continue   # an empty entry means the default
        provider = str(spec.get("provider") or "").strip().lower()
        model = str(spec.get("model") or "").strip()
        if provider not in PROVIDER_API_KEY_ENV:
            raise ValueError(f"unknown provider {provider!r} for {role}")
        if not model:
            raise ValueError(f"no model given for {role}")
        out[role] = {"provider": provider, "model": model}
    return out


def tier_model(config: dict, tier: str) -> tuple[str, str]:
    """(provider, model) of the quick or deep tier: the tier's own provider, else the run's."""
    from .factory import tier_provider

    return tier_provider(config, tier), config[f"{tier}_think_llm"]


def resolve(config: dict) -> dict[str, tuple[str, str]]:
    """(provider, model) for every role."""
    overrides = validate(config.get("agent_models"))
    out = {}
    for role, (_, _, tier) in ROLES.items():
        spec = overrides.get(role)
        out[role] = (spec["provider"], spec["model"]) if spec else tier_model(config, tier)
    return out


def models_used(config: dict, roles=None) -> list[tuple[str, str]]:
    """The distinct (provider, model) pairs the given roles (default: all) call, in a fixed order."""
    resolved = resolve(config)
    return sorted({resolved[r] for r in (roles or resolved)})


def first_model(provider: str, tier: str) -> str | None:
    """The catalog's first listed model for a provider and tier (quick or deep), if it lists one."""
    from .model_catalog import get_model_options

    try:
        ids = [m for _, m in get_model_options(provider, tier) if m != "custom"]
    except (KeyError, ValueError):
        return None
    return ids[0] if ids else None


def suggest(default_provider: str, providers: list[str]) -> dict[str, dict[str, str]]:
    """A split of the debates across the providers given (those with keys), the default first.

    Each debate puts its opposing sides on different providers (bull vs bear; aggressive vs
    conservative) so one model does not argue with itself; a third provider, when there is one,
    takes the neutral seat and the research manager, so the judge of the bull/bear debate is
    neither side. The analysts, trader and portfolio manager stay on the default. One provider:
    nothing to split.
    """
    order = [p for p in dict.fromkeys([default_provider.lower(), *providers])
             if p in providers and first_model(p, "quick")]
    if len(order) < 2:
        return {}
    a, b = order[0], order[1]
    c = order[2] if len(order) > 2 else None
    seats = {"bull": (a, "quick"), "bear": (b, "quick"), "aggressive": (a, "quick"), "conservative": (b, "quick")}
    if c:
        seats |= {"neutral": (c, "quick"), "research_manager": (c, "deep")}
    return {role: {"provider": p, "model": first_model(p, tier) or first_model(p, "quick")}
            for role, (p, tier) in seats.items()}
