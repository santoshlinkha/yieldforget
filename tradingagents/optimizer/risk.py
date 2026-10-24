"""Risk levels: a 1-10 dial in place of a raw loss-limit percentage.

Each level is a stress-loss limit: the share of the whole portfolio it may lose
if every holding repeats its worst historical drop at once. Level 10 has no
stress limit at all; only the per-fund caps still apply. The chosen level is
saved next to the holdings, so the web app and the CLI use the same one.

The same file keeps how the stress loss is measured (``model``, see plan.py) and
an optional growth target, so the levels keep their meaning under either model:
a level is always the most the portfolio may lose in a stress test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiskLevel:
    level: int
    name: str
    loss_limit: float | None   # None: no stress limit


RISK_LEVELS = (
    RiskLevel(1, "Capital preservation", 0.01),
    RiskLevel(2, "Very conservative", 0.025),
    RiskLevel(3, "Conservative", 0.05),
    RiskLevel(4, "Cautious", 0.10),
    RiskLevel(5, "Moderate", 0.15),
    RiskLevel(6, "Balanced", 0.20),
    RiskLevel(7, "Growth", 0.30),
    RiskLevel(8, "Aggressive", 0.40),
    RiskLevel(9, "Very aggressive", 0.50),
    RiskLevel(10, "Maximum growth", None),
)
# Used until a level is saved: the 2.5% the paper tracker was started at.
DEFAULT_LEVEL = 2
# How the stress loss is measured; the first is the default.
RISK_MODELS = ("worst_case", "scenarios")


@dataclass(frozen=True)
class RiskChoice:
    level: int
    model: str = RISK_MODELS[0]
    target_growth: float | None = None    # expected growth per year aimed for; None: none


def risk_level(level: int) -> RiskLevel:
    if not 1 <= int(level) <= len(RISK_LEVELS):
        raise ValueError(f"risk level must be 1 to {len(RISK_LEVELS)}, not {level}")
    return RISK_LEVELS[int(level) - 1]


def _read(path: str | Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_level(path: str | Path) -> int:
    """The saved level, or the default when none is saved yet."""
    try:
        return risk_level(_read(path)["level"]).level
    except (ValueError, KeyError, TypeError):
        return DEFAULT_LEVEL


def read_choice(path: str | Path) -> RiskChoice:
    """Level, model and growth target as saved; the defaults for anything missing or invalid."""
    data = _read(path)
    model = data.get("model") if data.get("model") in RISK_MODELS else RISK_MODELS[0]
    target = data.get("target_growth")
    target = float(target) if isinstance(target, (int, float)) and not isinstance(target, bool) else None
    return RiskChoice(read_level(path), model, target)


def write_level(path: str | Path, level: int) -> RiskLevel:
    chosen = risk_level(level)
    Path(path).write_text(json.dumps({**_read(path), "level": chosen.level}, indent=2), encoding="utf-8")
    return chosen


_UNSET = object()


def write_choice(path: str | Path, level: int | None = None, model: str | None = None,
                 target_growth=_UNSET) -> RiskChoice:
    """Save what is given and keep the rest; ``target_growth=None`` clears the target."""
    data = _read(path)
    if level is not None:
        data["level"] = risk_level(level).level
    if model is not None:
        if model not in RISK_MODELS:
            raise ValueError(f"risk model must be one of {', '.join(RISK_MODELS)}, not {model}")
        data["model"] = model
    if target_growth is not _UNSET:
        if target_growth is None:
            data.pop("target_growth", None)
        elif not -0.5 < float(target_growth) < 1.0:
            raise ValueError(f"growth target must be a yearly rate such as 0.08, not {target_growth}")
        else:
            data["target_growth"] = float(target_growth)
    data.setdefault("level", DEFAULT_LEVEL)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return read_choice(path)


def levels_json() -> list[dict]:
    return [asdict(r) for r in RISK_LEVELS]
