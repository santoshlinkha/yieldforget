"""Modern challengers against the baseline rankers on identical stock rows, recorded whatever they show.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe experiments/challenger_stock_study.py
Fixed before results: a sample of 150 of today's S&P 500 members with 10+ years of prices (seed 11;
more survivorship-biased than the main study, and labelled so); decision dates every 20 sessions from
2014; horizon 20; folds of two years from 2016; development to June 2024, exploratory after. Trained
models (rankers, PatchTST) refit each fold on rows whose labels ended before it; pretrained models
(Chronos-2, TimesFM 2.5, Kronos-mini) are zero-shot and retrospective, so their scores are
exploration only (contamination unknown or high).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
HERE = Path(__file__).parent / "output"
EDGES = ["2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-07-01"]
DEV_END = pd.Timestamp("2024-06-30")


def log(*m):
    print(time.strftime("%H:%M:%S"), *m, flush=True)


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version
    from tradingagents.ledger.store import Store
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates
    from tradingagents.research.walkforward import _ic_by_date, block_bootstrap
    from tradingagents.stocks import challengers as CH
    from tradingagents.stocks import data as D
    from tradingagents.stocks import macro as M
    from tradingagents.stocks import panel as P
    from tradingagents.stocks.adapters import Momentum, Ranker, WithinDateLinear

    t0 = time.time()
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    data = D.load([], cache)
    rng = np.random.default_rng(11)
    long = [t for t, i in data.info.items() if not i.get("removed") and t in data.close and data.close[t].notna().sum() > 2520]
    sample = sorted(rng.choice(sorted(long), size=150, replace=False))
    data.info = {t: data.info[t] for t in sample}
    data.spans = None                                    # the sample is its own reference universe
    dates = decision_dates(data.close.index, start="2014-01-01", every=20)
    panel = P.build(data, 20, dates=dates, macro=M.features(dates, cache))
    panel = panel[panel["in_index"] | True]
    feats = P.features_for()
    device = "cuda" if cuda_usable() else "cpu"
    log("sample", len(sample), "panel", panel.shape, "device", device)
    ohlcv = CH.fetch_opens(sample, cache)
    log("opens for", len(ohlcv), "stocks")
    close, volume = data.close[sample], data.volume[sample]
    edges = [pd.Timestamp(e) for e in EDGES] + [panel.index.get_level_values("date").max() + pd.Timedelta(days=1)]
    trained = {"momentum": Momentum, "linear_xs": WithinDateLinear, "xgb_rank": lambda: Ranker(device),
               "patchtst": lambda: CH.PatchTST(close, volume, device=device)}
    pretrained = {"chronos2": lambda: CH.Chronos2Score(close, 20, device),
                  "timesfm25": lambda: CH.TimesFM25Score(close, 20, device),
                  "kronos_mini": lambda: CH.KronosScore(ohlcv, 20, data.close.index, cache, device)}
    d = panel.index.get_level_values("date")
    preds, costs = [], {}
    for start, end in zip(edges, edges[1:], strict=False):
        test = panel[(d >= start) & (d < end) & panel["label"].notna()]
        train = panel[(pd.to_datetime(panel["label_end"]) < start) & panel["label"].notna()]
        if test.empty:
            continue
        frame = test[["label"]].copy()
        for name, make in {**trained, **pretrained}.items():
            a = make()
            status, why = a.status()
            if status != "available":
                costs.setdefault(name, {"status": why})
                continue
            t = time.time()
            if name in trained:
                a.fit(train, feats)
            frame[name] = a.predict(test)
            c = costs.setdefault(name, {"seconds": 0.0, "peak_mb": 0.0, "device": device})
            c["seconds"] += time.time() - t
            c["peak_mb"] = max(c["peak_mb"], (getattr(a, "last", {}) or {}).get("peak_mb", 0.0))
            c["describe"] = a.describe()
            log(start.date(), name, f"{time.time() - t:.0f}s")
        preds.append(frame)
    pred = pd.concat(preds)
    pred.to_pickle(HERE / "challenger_stock_pred.pkl")
    out = {"sample": len(sample), "survivorship": "biased (today's members with 10+ years of prices)",
           "costs": costs, "availability": CH.availability(cache), "parts": {}}
    pd_ = pred.index.get_level_values("date")
    for part, sel in (("development", pd_ <= DEV_END), ("exploratory", pd_ > DEV_END)):
        res = {}
        for m in [c for c in pred.columns if c != "label"]:
            ic = _ic_by_date(pred[sel][["label"]].assign(mean=pred[sel][m]), "mean")
            res[m] = {"ic": float(ic.mean()) if len(ic) else None,
                      "ci": list(block_bootstrap(ic.to_numpy(), block=2)) if len(ic) >= 4 else [None, None],
                      "ranked_dates": int(ic.size), "dates": int(pred[sel].index.get_level_values("date").nunique()),
                      "coverage": float(pred[sel][m].notna().mean())}
        out["parts"][part] = res
    out["seconds"] = time.time() - t0
    store = Store(Path("my_portfolio.json").resolve().parent)
    with transaction(store.con):
        store.con.execute(
            "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
            " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now_utc(), "challenger_stock_study:h20", "sp500_sample150", 20, EDGES[0], str(pd_.max().date()),
             json.dumps({"sample": sample, "features": feats, "edges": EDGES, "smoke": False}), code_version(),
             json.dumps(out, default=str), "{}", out["seconds"]))
    (HERE / "challenger_stock_study.json").write_text(json.dumps(out, indent=1, default=str))
    log("wrote", HERE / "challenger_stock_study.json")


if __name__ == "__main__":
    main()
