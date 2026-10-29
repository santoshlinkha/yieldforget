"""How much one new position may be: the one rule the daily decision and the Trade view both use.

A new name gets the smaller of its cap (``max_position_share`` of the sleeve: the trading budget
plus the stocks held) and an equal share of what is left of the budget across the open name slots.
The Trade view re-applies the rule to the budget as it is now (cash may have moved since the
decision), and never sizes above what the saved decision allowed.
"""

from __future__ import annotations

import math


def position_size(rules: dict, sleeve_value: float | None, budget: float, slots: int) -> float:
    """Dollars for the next new name. ``sleeve_value`` None (an old record without it) drops the cap."""
    if slots <= 0 or not budget or budget <= 0:
        return 0.0
    cap = rules["max_position_share"] * sleeve_value if sleeve_value is not None else math.inf
    return max(0.0, min(cap, budget / slots))
