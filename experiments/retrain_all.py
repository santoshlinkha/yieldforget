"""Retrain every internal model from scratch, keeping everything that existed before.

    python -m experiments.retrain_all            (from the repository root, with the app's venv)

1. Backup: copies the model caches (decision, zoo, signal, forecast and sector models, the training fingerprints)
   and the stacking study's out-of-fold files to ``backups/retrain_<date>/``, plus a consistent snapshot of the
   app's database. Saved runs and decisions in the database are append-only: the new ones are added beside them.
2. The ensemble: every member's out-of-fold scores recomputed for every year (the cached ones are moved into the
   backup, not deleted), then the weights refitted (Chronos is skipped, as in the morning run: it is reported, not
   weighted).
3. Every model in the zoo refitted (TimesFM 3.0 too when its license was accepted).
4. The daily decision's own models refitted: an earlier fit is not reused, even for the same training data.
5. The signals batch (5/20/60 sessions) refitted for the S&P 500.
6. The sector model and the fund forecasts run again.

Each step logs ``HH:MM:SS`` lines; one failing step is reported and the next one still runs. Nothing here places
an order or calls a paid service.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def say(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def main() -> int:
    from tradingagents.dataflows.config import get_config
    from tradingagents.ledger.store import Store

    config = get_config()
    cache = Path(config["data_cache_dir"]) / "optimizer"
    store = Store(ROOT)
    device = store.get("global", "compute_backend")
    backup = ROOT / "backups" / f"retrain_{date.today().isoformat()}"
    backup.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    # --- 1. backup ---
    say(f"backing up to {backup}")
    for name in ("decision_models", "zoo_models", "stock_models", "forecast_models", "sector_model"):
        src = cache / name
        if src.exists():
            shutil.copytree(src, backup / "cache" / name, dirs_exist_ok=True)
    if (cache / "training.duckdb").exists():
        shutil.copy2(cache / "training.duckdb", backup / "cache" / "training.duckdb")
    out = ROOT / "experiments" / "output"
    (backup / "experiments_output").mkdir(exist_ok=True)
    for f in out.glob("*.json"):
        shutil.copy2(f, backup / "experiments_output" / f.name)
    db = ROOT / "portfolio.sqlite3"
    with sqlite3.connect(db) as src, sqlite3.connect(backup / "portfolio.sqlite3") as dst:
        src.backup(dst)                                    # a consistent copy even while the app is running
    say("backup done")

    def step(name, fn):
        say(f"step: {name}")
        t0 = time.time()
        try:
            fn()
            say(f"done: {name} ({(time.time() - t0) / 60:.1f} min)")
        except Exception:
            failed.append(name)
            say(f"FAILED: {name}\n{traceback.format_exc()}")

    # --- 2. the ensemble, every fold from scratch ---
    def ensemble():
        moved = backup / "oof_cache"
        moved.mkdir(exist_ok=True)
        for f in out.glob("oof_*.pkl"):
            if not f.stem.endswith("_quick"):
                shutil.move(str(f), moved / f.name)       # kept in the backup; the study recomputes them
        proc = subprocess.run([sys.executable, "-m", "experiments.stacking_study"], cwd=ROOT,
                              env={**os.environ, "STACKING_SKIP_CHRONOS": "1"})
        if proc.returncode:
            raise RuntimeError(f"stacking study exited with {proc.returncode}")
    step("ensemble weights (out-of-fold scores recomputed)", ensemble)

    # --- 3. every model ---
    from tradingagents.stocks import model_zoo
    step("every model refitted", lambda: model_zoo.run(store, cache, lambda t, *_: say(f"  zoo: {t}"),
                                                        force=True, refit=True))

    # --- 4. the daily decision, its own models refitted ---
    def decision():
        from tradingagents.stocks import daily, training_store

        real_latest, real_find = daily.latest_fit, training_store.find
        daily.latest_fit = lambda *a, **k: None            # no earlier bundle reused ...
        training_store.find = lambda *a, **k: None         # ... nor an earlier fit of the same training data
        try:
            daily.run(store, cache, date.today(), "barrier_20", True, lambda t, *_: say(f"  decision: {t}"), force=True)
        finally:
            daily.latest_fit, training_store.find = real_latest, real_find
    step("daily decision refitted", decision)

    # --- 5. the signals batch ---
    def signals():
        from tradingagents.optimizer.universe import get_sp500_members
        from tradingagents.stocks.pipeline import BatchConfig, Progress, run_batch

        moved = backup / "stock_models_moved"
        moved.mkdir(exist_ok=True)
        for f in (cache / "stock_models").glob("*.pkl"):
            shutil.move(str(f), moved / f.name)           # kept; the batch fits them again
        names = sorted(get_sp500_members(cache))
        run_batch(store.con, cache, names, {t: "sp500" for t in names},
                  BatchConfig(horizons=(5, 20, 60), news=False, device=device), Progress(lambda t: say(f"  signals: {t}")),
                  force=True)
    step("signals batch refitted (5/20/60 sessions)", signals)

    # --- 6. sector model and fund forecasts ---
    from tradingagents.research import sector_model
    step("sector model", lambda: sector_model.run(cache, store, lambda t: say(f"  sector: {t}"), force=True))

    def funds():
        from tradingagents.optimizer.universe import etf_list, read_config
        from tradingagents.research.engine import CHRONOS, run_forecast

        context, _ = store.portfolio("combined", date.today().isoformat())
        held = [p.ticker.upper() for p in context.positions]
        etfs = [t for t in dict.fromkeys([*etf_list(read_config(ROOT / "my_universe.json")), *held])
                if t not in ("SGOV", "BIL")]
        model = (store.settings("trading") or {}).get("forecast_model") or "xgb_quantile"
        for h, m in ((20, CHRONOS), (60, CHRONOS), (20, model)):
            say(f"  funds: {m}, {h} sessions")
            run_forecast(store.con, cache, etfs, horizon=h, model=m, universe="etf", device_setting=device, force=True)
    step("fund forecasts", funds)

    say("finished" + (f" with failures: {', '.join(failed)}" if failed else ": every step succeeded"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
