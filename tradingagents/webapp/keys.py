"""API keys the app may use, read and changed from the Settings page, stored in the local ``.env``.

- Only known key names can be written (the LLM providers' and the data vendors'), so the page cannot
  set arbitrary environment variables.
- The page never receives a key back: only whether it is set and a masked form (``sk-or-…3e9d``).
- A change is written atomically to ``.env`` (git-ignored) and applied to the running app at once.
- An empty value removes the key.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

DATA_KEYS = {"FRED_API_KEY": "FRED (economic data)", "ALPHA_VANTAGE_API_KEY": "Alpha Vantage (market data)",
             "TYPESAFE_API_KEY": "TypeSafe (post screening)"}


CONTEXT_KEY = "GEMINI_CONTEXT_API_KEY"
CONTEXT_KEYS = {CONTEXT_KEY: "Google Gemini for non-agent work (Trend page brief, ticker news, moat)"}
# A second key per provider, tried on the same model when the first key hits a limit.
ALT_KEYS = {"openrouter": "OPENROUTER_API_KEY_2"}
ALT_LABELS = {"OPENROUTER_API_KEY_2": "OpenRouter, second key (used when the first hits a limit)"}


def known() -> dict[str, str]:
    """{env name: label} for every key the page may set."""
    out = {env: provider for provider, env in PROVIDER_API_KEY_ENV.items() if env}
    return {**out, **DATA_KEYS, **CONTEXT_KEYS, **ALT_LABELS}


def env_path() -> Path:
    from dotenv import find_dotenv

    found = find_dotenv(usecwd=True)
    return Path(found) if found else Path.cwd() / ".env"


def mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 10:
        return "•" * len(v)
    head = v[: v.find("-", 3) + 1] if v.startswith("sk-") and v.find("-", 3) > 0 else v[:3]
    return f"{head}…{v[-4:]}"


def listing() -> list[dict]:
    rows = []
    for env, label in known().items():
        value = os.environ.get(env) or ""
        rows.append({"env": env, "label": label, "set": bool(value), "masked": mask(value) if value else None})
    return sorted(rows, key=lambda r: (not r["set"], r["label"]))


def save(env: str, value: str, path: Path | None = None) -> dict:
    """Set (or with an empty value remove) one key in ``.env`` and in this process."""
    if env not in known():
        raise ValueError(f"{env} is not a key this app uses")
    value = (value or "").strip()
    if re.search(r"[\s\"'#]", value):
        raise ValueError("a key cannot contain spaces, quotes or #")
    path = path or env_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [line for line in lines if not re.match(rf"\s*(export\s+)?{re.escape(env)}\s*=", line)]
    if value:
        lines.append(f"{env}={value}")
    tmp = path.with_suffix(".env.tmp") if path.suffix else path.with_name(path.name + ".tmp")
    try:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    # Only once the file holds it: a failed save leaves this process as it was.
    if value:
        os.environ[env] = value
    else:
        os.environ.pop(env, None)
    return {"env": env, "set": bool(value), "masked": mask(value) if value else None}
