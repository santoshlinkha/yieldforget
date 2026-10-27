"""Model C variants on the policy study's exact folds and holdout: symmetric vs asymmetric barriers, and one-step
vs two-step (fill, then outcome) models for the limit entry.

    PYTHONPATH=. TRADINGAGENTS_CACHE_DIR=<cache copy> python experiments/policy_variants.py
"""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / "TradingAgents" / ".env")


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.stocks import policies as PO
    from tradingagents.stocks import policy_study as S
    from tradingagents.stocks.decision_layer import LayerConfig
    from tradingagents.stocks.study import block_length

    cache = Path(DEFAULT_CONFIG["data_cache_dir"]) / "optimizer"
    base = dict(PO.POLICY_SET)
    out = {}
    for barrier, (pk, rk) in (("symmetric 1.0/1.0", (1.0, 1.0)), ("asymmetric target 1.75 / stop 1.25", (1.75, 1.25))):
        for k in ("next_close", "limit_dip"):
            PO.POLICY_SET[k] = replace(base[k], profit_k=pk, risk_k=rk)
        t = time.time()
        S.KINDS = ("lgbm", "lgbm_seq")
        ds = S.dataset(cache, lambda m: print(time.strftime("%H:%M:%S"), barrier, m, flush=True))
        wf = S.walk_forward(ds)
        labs = ds["labels"]
        realized = {p: labs[p]["excess"] for p in S.POLICIES}
        vix = ds["rows"]["vix"]
        res = {"classification": {}, "strategies": {}}
        for (kind, p), pred in wf["preds"].items():
            for part, sel in (("development", pred["fold"] != "holdout"), ("holdout", pred["fold"] == "holdout")):
                c = S.class_scores(pred[sel], wf["base"][p], labs[p], p, vix)
                res["classification"][f"{kind}:{p}:{part}"] = {
                    "skill": {k2: v["skill"] for k2, v in c["per_class"].items()}, "ic": c["mean_excess_ic"],
                    "q10_coverage": c["q10_coverage"], "n": c["n"]}
        block = block_length(np.zeros(1), 25, S.EVERY)
        for kind in S.KINDS:
            nc, lim = wf["preds"][(kind, "next_close")], wf["preds"][(kind, "limit_dip")]
            dev = nc.index.get_level_values("date") < S.HOLDOUT_START
            tried = {m: S.per_date(S.choose(nc[dev], lim, m, LayerConfig()), realized).mean() for m in S.MARGINS}
            best = max(tried, key=tried.get)
            for part, sel in (("development", dev), ("holdout", ~dev)):
                pdx = S.per_date(S.choose(nc[sel], lim, best, LayerConfig()), realized)
                res["strategies"][f"{kind}:{part}"] = {**S.strategy_stats(pdx, block), "margin": best}
        res["seconds"] = round(time.time() - t)
        out[barrier] = res
        print(json.dumps(res, default=float)[:1500], flush=True)
    for k, v in base.items():
        PO.POLICY_SET[k] = v
    Path("experiments/output").mkdir(parents=True, exist_ok=True)
    Path("experiments/output/policy_variants.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
