"""The Mamba sequence model (``stocks.seq_model``) on the model study's exact folds and evaluation; its result is added
to ``experiments/output/model_zoo.json`` beside the other models.

    QUICK=1 .venv/Scripts/python.exe -m experiments.mamba_study     # 60 stocks
    .venv/Scripts/python.exe -m experiments.mamba_study             # S&P 500 members
"""
import json
import os
import time
from pathlib import Path

import pandas as pd

from experiments.model_zoo_study import OUT, crossfit, evaluate, log

KEY = "mamba (64-day sequences + 54 features)"


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.research.compute import cuda_usable
    from tradingagents.research.features import decision_dates
    from tradingagents.stocks import data as D, macro as M, panel as P, study as S
    from tradingagents.stocks.positions import POLICIES, executable_prices
    from tradingagents.stocks.seq_model import MambaRanker, SequenceBank

    t0 = time.time()
    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    device = "cuda" if cuda_usable() else "cpu"
    data = D.load([], cache)
    if os.getenv("QUICK") == "1":
        data.info = {t: data.info[t] for t in list(data.info)[:60]}
    dates = decision_dates(data.close.index, start="2006-01-01", every=S.EVERY)
    panel = P.build(data, 20, dates=dates, macro=M.features(dates, cache))
    panel = panel[panel["in_index"]]
    stocks = list(data.stocks)
    px, div = executable_prices(data.close[stocks], data.actions)
    tr = S.attach_trades(panel.assign(in_index=True), px, data.low[stocks], data.close[stocks], div, POLICIES["barrier_20"])
    net = tr["net"].where(tr["complete"])
    bank = SequenceBank(data)
    log("panel", panel.shape, "bank", bank.x.shape, device)
    feats = P.features_for()
    edges = S.fold_edges(panel.index.get_level_values("date"))

    def fit_predict(train, test):
        log("fold from", str(test.index.get_level_values("date").min().date()), "train rows", len(train))
        return MambaRanker(bank, device).fit(train, feats).predict(test)

    score = crossfit(panel, edges, fit_predict, pd.to_datetime(panel["label_end"]))
    result = evaluate(score, net)
    log("mamba", json.dumps(result))
    if os.getenv("QUICK") != "1":
        study = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {"models": {}}
        study.setdefault("models", {})[KEY] = {**result, "seconds": round(time.time() - t0)}
        OUT.write_text(json.dumps(study, indent=1, default=float))
    log("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
