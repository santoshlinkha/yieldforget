"""python -m tradingagents.events sync | status | features [YYYY-MM-DD] [strict|research]"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path


def main(argv: list[str]) -> int:
    from tradingagents.dataflows.config import get_config

    cache = Path(get_config()["data_cache_dir"]) / "optimizer"
    cmd = argv[0] if argv else "status"
    if cmd == "sync":
        from .sources import sync

        report = sync(cache)
        print(json.dumps(report, indent=1))
        return 0 if all(r["status"] == "ok" for r in report) else 1
    if cmd == "status":
        from . import store

        con = store.connect(cache)
        print(json.dumps(store.status(con), indent=1))
        return 0
    if cmd == "features":
        from .features import market_event_features

        day = argv[1] if len(argv) > 1 else date.today().isoformat()
        mode = argv[2] if len(argv) > 2 else "strict"
        from datetime import UTC, datetime

        now = datetime.now(UTC).replace(tzinfo=None) if mode == "strict" else None    # a live view: known by now
        print(json.dumps(market_event_features(cache, day, 20, mode, now), indent=1, default=float))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
