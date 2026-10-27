"""The policy study: are the outcome models good enough to decide buy now / buy on a limit / cash?

Fixed before results (``DESIGN``):

- Rows: point-in-time S&P 500 members on decision dates every ``every`` sessions (10) from 2006;
  features from ``panel`` (known at the decision close); labels from ``policies`` (triple barrier,
  execution after the cutoff, no-fills, gaps, delistings).
- Development: yearly walk-forward test folds 2014..2024, common calendar boundaries for all stocks.
  For a fold starting at S: training rows are those whose label ended before S (purged; a 20-25
  session label never reaches into the test year). The last two years of them are the **calibration**
  slice (purged the same way against the slice's start); the models are fitted on the rows before it.
- **Holdout**: 2025-01-01 to the latest complete label. Scored once, after everything below was frozen
  on development. Earlier studies in this repository inspected prices from July 2024 on for *other*
  models, so this period is untouched by this study's choices but not unseen by its author.
- Decision layer on development only: the buy margin is picked from ``MARGINS`` (4 values; the count is
  reported) by mean per-decision excess over cash; the risk budget is configured, not tuned.
- Portfolio per decision date: at most 5 names, equal slots, an unfilled slot is cash. Each date's
  result is the mean excess over cash of its 5 slots; dates overlap (windows of 20-25 sessions every
  10), so intervals use a block bootstrap with blocks covering the overlap.
- Benchmarks on the same dates and rows: the existing ranker rule (XGBoost learning-to-rank, cross-fitted
  on the same folds, mapped expected net >= 0.5%, top 5, next-close entry: the deployed model without its
  P(profit) and P(large loss) gates), the linear baseline through the same layer, 12-1 momentum top 5,
  SPY bought at the next close for 20 sessions, and cash (0).
- **Validated** only if on the holdout: Brier skill above zero for both P(target first) and P(stop first)
  against the training base rates, the 10th-percentile excess covers 5-15% of outcomes, and the LightGBM
  layer's mean per-decision excess over cash has a block-bootstrap 95% interval above zero and is not
  below the existing rule's. Otherwise "insufficient evidence", and the layer abstains live.

Meta-labeling: a take/skip model on the existing ranker's out-of-fold candidates was already studied
(``TradeQuality`` in ``study``; no version beat the base rate), so it is not added. Agent decisions have
no
history; they are recorded prospectively (memory log, saved runs) for a later study.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_layer import LayerConfig
from .outcome_models import OutcomeModel, brier, classes_for, log_loss, outcome_class, reliability
from .policies import POLICY_SET, label, spec_dict

DESIGN_VERSION = "policy-study-v1"
EVERY = 10
DEV_FIRST, HOLDOUT_START = 2014, pd.Timestamp("2025-01-01")
CAL_YEARS = 2
MARGINS = (0.0, 0.005, 0.01, 0.02)
POLICIES = ("next_close", "limit_dip")
KINDS = ("lgbm", "linear")
SLOTS = 5


# --- data ---

def dataset(cache_dir: Path, progress=None) -> dict:
    """Panel rows (members only), features and the policies' labels."""
    from tradingagents.research.features import decision_dates

    from . import data as D, macro as M, panel as P
    from .positions import executable_prices

    say = progress or (lambda *_: None)
    say("loading prices")
    data = D.load([], cache_dir)
    sessions = data.close.index
    dates = decision_dates(sessions, start="2006-01-01", every=EVERY)
    say("building features")
    panel = P.build(data, 20, dates=dates, macro=M.features(dates, cache_dir))
    rows = panel[panel["in_index"]].copy()
    stocks = list(data.stocks)
    px, div = executable_prices(data.close[stocks], data.actions)
    say("labelling the policies")
    labels = {p: label(rows[["ewma_vol_daily", "cost_rt"]], px, data.low[stocks], data.close[stocks], div,
                       data.close[D.CASH], POLICY_SET[p]) for p in POLICIES}
    spy, cash = data.close["SPY"], data.close[D.CASH].reindex(sessions).ffill()
    return {"rows": rows, "features": P.features_for(), "labels": labels, "sessions": sessions, "spy": spy,
            "cash": cash, "survivorship": data.survivorship}


def spy_excess(dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex, spy: pd.Series, cash: pd.Series, h=20) -> pd.Series:
    """SPY bought at the next close and held h sessions, over cash for the same days."""
    s, c = spy.reindex(sessions).ffill(), cash.reindex(sessions).ffill()
    idx = sessions.get_indexer(dates)
    out = {}
    for d, i in zip(dates, idx, strict=True):
        if i < 0 or i + 1 + h >= len(sessions):
            continue
        a, b = i + 1, i + 1 + h
        out[d] = float(s.iloc[b] / s.iloc[a] - 1 - (c.iloc[b] / c.iloc[a] - 1))
    return pd.Series(out)


# --- folds ---

def folds(dates: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in range(DEV_FIRST, HOLDOUT_START.year)]
    out.append(("holdout", HOLDOUT_START, dates.max() + pd.Timedelta(days=1)))
    return out


def split(rows: pd.DataFrame, lab: pd.DataFrame, start: pd.Timestamp):
    """(fit, cal) rows for a test period starting at ``start``: both purged by label end."""
    ends = pd.to_datetime(lab["label_end_date"])
    known = lab["outcome"].isin(["target", "stop", "expiry", "nofill", "delisted"])
    before = known & (ends < start)
    cal_start = start - pd.DateOffset(years=CAL_YEARS)
    d = rows.index.get_level_values("date")
    cal = before & (d >= cal_start)
    fit = before & (ends < cal_start)
    return fit.to_numpy(), cal.to_numpy()


def _frame(rows, lab, mask, features):
    return pd.concat([rows.loc[mask, features], lab.loc[mask, ["outcome", "excess"]]], axis=1)


# --- the walk-forward ---

def walk_forward(ds: dict, progress=None) -> dict:
    """Out-of-fold predictions for every (kind, policy) on development folds and the holdout."""
    say = progress or (lambda *_: None)
    rows, feats = ds["rows"], ds["features"]
    d = rows.index.get_level_values("date")
    preds = {(k, p): [] for k in KINDS for p in POLICIES}
    base = {p: [] for p in POLICIES}
    for name, start, end in folds(pd.DatetimeIndex(d.unique())):
        test = ((d >= start) & (d < end))
        if not test.any():
            continue
        for p in POLICIES:
            lab = ds["labels"][p]
            fit, cal = split(rows, lab, start)
            if fit.sum() < 20000 or cal.sum() < 2000:
                continue
            ftr, fcal = _frame(rows, lab, fit, feats), _frame(rows, lab, cal, feats)
            freq = outcome_class(ftr["outcome"]).value_counts(normalize=True)
            base[p].append(pd.DataFrame({f"b_{c}": freq.get(c, 0.0) for c in classes_for(p)},
                                        index=rows.index[test]).assign(fold=name))
            for k in KINDS:
                t0 = time.time()
                m = OutcomeModel(k, p).fit(ftr, fcal, feats)
                out = m.predict(rows.loc[test, feats]).assign(fold=name)
                preds[(k, p)].append(out)
                say(f"{name} {p} {k}: {test.sum()} rows, {time.time() - t0:.0f}s")
    return {"preds": {key: pd.concat(v) for key, v in preds.items() if v},
            "base": {p: pd.concat(v) for p, v in base.items() if v}}


def existing_rule(ds: dict, device: str, progress=None) -> pd.DataFrame:
    """The deployed ranker, cross-fitted on the same folds, and its expected-net map from earlier folds."""
    from .adapters import Ranker, apply_map, score_to_net, within_date_pct

    say = progress or (lambda *_: None)
    rows, feats = ds["rows"], ds["features"]
    lab = ds["labels"]["next_close"]
    d = rows.index.get_level_values("date")
    out = []
    for name, start, end in folds(pd.DatetimeIndex(d.unique())):
        test = (d >= start) & (d < end)
        train = rows[(pd.to_datetime(rows["label_end"]) < start) & rows["label"].notna()]
        if not test.any() or train.empty:
            continue
        r = Ranker(device).fit(train, feats)
        s = r.predict(rows[test])
        out.append(pd.DataFrame({"score": s, "score_pct": within_date_pct(rows[test], s), "fold": name},
                                index=rows.index[test]))
        say(f"{name} ranker")
    oof = pd.concat(out)
    # The expected-net map for each fold from earlier folds' out-of-fold scores whose trades had ended.
    oof["expected_net"] = np.nan
    ends = pd.to_datetime(lab["label_end_date"]).reindex(oof.index)
    net = lab["net"].reindex(oof.index)
    od = oof.index.get_level_values("date")
    for name, start, _ in folds(pd.DatetimeIndex(od.unique())):
        past = (ends < start) & net.notna()
        cur = (oof["fold"] == name).to_numpy()
        if past.sum() < 1000:
            continue
        mp = score_to_net(oof.loc[past.to_numpy(), "score_pct"].to_numpy(), net[past].to_numpy())
        oof.loc[cur, "expected_net"] = apply_map(mp, oof.loc[cur, "score_pct"].to_numpy())
    return oof


# --- the portfolio on each decision date ---

def choose(pred_nc: pd.DataFrame, pred_lim: pd.DataFrame, margin: float, cfg: LayerConfig) -> pd.Series:
    """The layer's choice per row (vectorized ``decision_layer.decide`` for a stock not held)."""
    f = pd.DataFrame({"m_nc": pred_nc["mean_excess"], "q_nc": pred_nc["q10_excess"],
                      "m_lim": pred_lim["mean_excess"].reindex(pred_nc.index),
                      "q_lim": pred_lim["q10_excess"].reindex(pred_nc.index)})
    ok_nc = (f["m_nc"] >= margin) & (f["q_nc"] >= -cfg.downside_floor)
    ok_lim = (f["m_lim"] >= margin) & (f["q_lim"] >= -cfg.downside_floor)
    pick = pd.Series("cash", index=f.index)
    pick[ok_nc] = "next_close"
    better = ok_lim & (~ok_nc | (f["m_lim"] > f["m_nc"]))
    pick[better] = "limit_dip"
    score = np.where(pick == "limit_dip", f["m_lim"], np.where(pick == "next_close", f["m_nc"], -np.inf))
    return pd.DataFrame({"pick": pick, "score": score}, index=f.index)


def per_date(picks: pd.DataFrame, realized: dict[str, pd.Series], slots: int = SLOTS) -> pd.Series:
    """Mean excess over cash of each date's ``slots`` (best picks; an empty slot is cash)."""
    chosen = picks[picks["pick"] != "cash"].copy()
    chosen["excess"] = np.nan
    for p, ser in realized.items():
        sel = chosen["pick"] == p
        chosen.loc[sel, "excess"] = ser.reindex(chosen.index[sel]).to_numpy()
    chosen = chosen.dropna(subset=["excess"])
    dates = picks.index.get_level_values("date").unique()
    if chosen.empty:
        return pd.Series(0.0, index=dates)
    top = chosen.sort_values("score", ascending=False).groupby(level="date").head(slots)
    s = top.groupby(level="date")["excess"].sum() / slots
    return s.reindex(dates).fillna(0.0)


def ci(x: pd.Series, block: int, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    v = x.dropna().to_numpy()
    if len(v) < 2 * block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(v)
    k = int(np.ceil(n / block))
    means = []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, k)
        means.append(np.concatenate([v[s:s + block] for s in starts])[:n].mean())
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def strategy_stats(x: pd.Series, block: int) -> dict:
    lo, hi = ci(x, block)
    return {"dates": int(x.notna().sum()), "mean_excess": float(x.mean()), "ci95": [lo, hi],
            "share_positive": float((x > 0).mean()), "worst": float(x.min()), "p05": float(x.quantile(0.05))}


# --- scoring ---

def class_scores(pred: pd.DataFrame, base: pd.DataFrame, lab: pd.DataFrame, policy: str, vix: pd.Series) -> dict:
    cls = classes_for(policy)
    y = outcome_class(lab["outcome"]).reindex(pred.index).map({c: i for i, c in enumerate(cls)})
    ok = y.notna().to_numpy()
    p = pred.loc[ok, [f"p_{c}" for c in cls]].to_numpy()
    b = base.reindex(pred.index).loc[ok, [f"b_{c}" for c in cls]].to_numpy()
    yy = y[ok].astype(int).to_numpy()
    out = {"n": int(ok.sum()), "brier": brier(p, yy), "brier_base": brier(b, yy), "log_loss": log_loss(p, yy),
           "log_loss_base": log_loss(b, yy), "per_class": {}}
    for j, c in enumerate(cls):
        hit = (yy == j).astype(float)
        bs, bb = float(((p[:, j] - hit) ** 2).mean()), float(((b[:, j] - hit) ** 2).mean())
        out["per_class"][c] = {"brier": bs, "brier_base": bb, "skill": 1 - bs / bb if bb else None,
                               "reliability": reliability(p[:, j], hit)}
    folds_ = pred.loc[ok, "fold"].to_numpy()
    out["by_fold"] = {}
    for f in pd.unique(folds_):
        s = folds_ == f
        out["by_fold"][str(f)] = {"n": int(s.sum()), "skill": 1 - brier(p[s], yy[s]) / brier(b[s], yy[s])}
    v = vix.reindex(pred.index[ok]).to_numpy()
    if np.isfinite(v).any():
        hi = v >= np.nanmedian(v)
        out["by_regime"] = {r: {"n": int(m.sum()), "skill": 1 - brier(p[m], yy[m]) / brier(b[m], yy[m])}
                            for r, m in (("high_vix", hi), ("low_vix", ~hi)) if m.sum() > 100}
    ex = lab["excess"].reindex(pred.index)
    okx = ex.notna()
    pe = pred.loc[okx]
    ic = pd.DataFrame({"m": pe["mean_excess"], "x": ex[okx]}).groupby(level="date").apply(
        lambda g: g["m"].rank().corr(g["x"].rank()) if len(g) > 20 else np.nan)
    out["mean_excess_ic"] = float(ic.mean())
    out["q10_coverage"] = float((ex[okx] < pe["q10_excess"]).mean())
    return out


def run(store, cache_dir: Path, device: str = "cpu", progress=None) -> dict:
    from tradingagents.ledger.db import now_utc, transaction
    from tradingagents.ledger.recommendations import code_version

    from .study import block_length

    say = progress or (lambda *_: None)
    t0 = time.time()
    ds = dataset(cache_dir, say)
    rows = ds["rows"]
    wf = walk_forward(ds, say)
    ex_rule = existing_rule(ds, device, say)
    labs = ds["labels"]
    realized = {p: labs[p]["excess"] for p in POLICIES}
    vix = rows["vix"] if "vix" in rows else pd.Series(np.nan, index=rows.index)
    cfg = LayerConfig()
    report = {"design": DESIGN_VERSION, "policies": spec_dict(), "every": EVERY, "margins_tried": list(MARGINS),
              "survivorship": ds["survivorship"], "classification": {}, "strategies": {}}
    for (k, p), pred in wf["preds"].items():
        for part, sel in (("development", pred["fold"] != "holdout"), ("holdout", pred["fold"] == "holdout")):
            if sel.any():
                report["classification"][f"{k}:{p}:{part}"] = class_scores(pred[sel], wf["base"][p], labs[p], p, vix)
    block = block_length(np.zeros(1), 25, EVERY)
    is_hold = lambda idx: (idx.get_level_values("date") >= HOLDOUT_START)  # noqa: E731
    margins = {}
    for k in KINDS:
        nc, lim = wf["preds"][(k, "next_close")], wf["preds"][(k, "limit_dip")]
        dev = ~is_hold(nc.index)
        tried = {m: per_date(choose(nc[dev], lim, m, cfg), realized).mean() for m in MARGINS}
        best = max(tried, key=tried.get)
        margins[k] = {"chosen": best, "tried": {str(m): float(v) for m, v in tried.items()}}
        for part, sel in (("development", dev), ("holdout", ~dev)):
            if sel.any():
                pdx = per_date(choose(nc[sel], lim, best, cfg), realized)
                report["strategies"][f"{k}_layer:{part}"] = {**strategy_stats(pdx, block), "series": pdx}
    # The existing rule, momentum, SPY and an equal-weight next-close benchmark, on the same dates.
    for part, sel in (("development", ~is_hold(ex_rule.index)), ("holdout", is_hold(ex_rule.index))):
        e = ex_rule[sel]
        pick = pd.DataFrame({"pick": np.where(e["expected_net"] >= 0.005, "next_close", "cash"),
                             "score": e["score"]}, index=e.index)
        report["strategies"][f"existing_rule:{part}"] = {**strategy_stats(per_date(pick, realized), block),
                                                        "series": per_date(pick, realized)}
        mom = rows.loc[e.index, "mom_12_1"]
        mpick = pd.DataFrame({"pick": "next_close", "score": mom}, index=e.index)
        report["strategies"][f"momentum_top5:{part}"] = {**strategy_stats(per_date(mpick, realized), block),
                                                        "series": per_date(mpick, realized)}
        allx = realized["next_close"].reindex(e.index).groupby(level="date").mean()
        report["strategies"][f"all_members_equal:{part}"] = strategy_stats(allx, block)
        sp = spy_excess(pd.DatetimeIndex(e.index.get_level_values("date").unique()), ds["sessions"], ds["spy"], ds["cash"])
        report["strategies"][f"spy:{part}"] = {**strategy_stats(sp, block), "series": sp}
    report["margins"] = margins
    report["status"] = status_from(report)
    # Live bundle: fitted on every labelled row (fit/cal split as in the folds), with the frozen margin.
    say("fitting the live models")
    latest = rows.index.get_level_values("date").max() + pd.Timedelta(days=1)
    live = {}
    for p in POLICIES:
        fit, cal = split(rows, labs[p], latest)
        live[p] = {k: OutcomeModel(k, p).fit(_frame(rows, labs[p], fit, ds["features"]),
                                              _frame(rows, labs[p], cal, ds["features"]), ds["features"])
                   for k in ("lgbm",)}
    cut = str(rows.index.get_level_values("date").max().date())
    bundle = {"version": DESIGN_VERSION, "cutoff": cut, "features": ds["features"], "models": live,
              "margin": margins["lgbm"]["chosen"], "status": report["status"], "config": cfg.as_dict(),
              "policies": spec_dict()}
    import pickle

    folder = Path(cache_dir) / "decision_models"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"policy_bundle_{cut}.pkl"
    path.write_bytes(pickle.dumps(bundle))
    report["bundle"] = str(path)
    report["seconds"] = round(time.time() - t0, 1)
    series = {k: v.pop("series") for k, v in report["strategies"].items() if "series" in v}
    with transaction(store.con):
        cur = store.con.execute(
            "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
            " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now_utc(), "policy_study:h20", "sp500_pit", 20, f"{DEV_FIRST}-01-01", cut,
             json.dumps({"design": DESIGN_VERSION, "every": EVERY, "margins": list(MARGINS), "policies": spec_dict()}),
             code_version(), json.dumps(report, default=float),
             json.dumps({k: {str(i.date()): float(x) for i, x in s.items()} for k, s in series.items()}),
             report["seconds"]))
    report["trial"] = cur.lastrowid
    return report


def status_from(report: dict) -> dict:
    """The fixed promotion rule (module docstring), applied to the holdout."""
    c = report["classification"].get("lgbm:next_close:holdout")
    s = report["strategies"].get("lgbm_layer:holdout")
    e = report["strategies"].get("existing_rule:holdout")
    if not c or not s:
        return {"validated": False, "state": "untested", "why": "no holdout results"}
    fails = []
    for cls in ("target", "stop"):
        sk = c["per_class"][cls]["skill"]
        if sk is None or sk <= 0:
            fails.append(f"P({cls} first) has no Brier skill over the base rate on the holdout ({sk:+.3f})")
    if not 0.05 <= c["q10_coverage"] <= 0.15:
        fails.append(f"the 10th-percentile excess covered {c['q10_coverage']:.0%} of outcomes (needs 5-15%)")
    lo = s["ci95"][0]
    if not (lo == lo and lo > 0):
        fails.append(f"mean excess over cash per decision {s['mean_excess']:+.2%}, 95% interval "
                     f"[{s['ci95'][0]:+.2%}, {s['ci95'][1]:+.2%}] does not exclude zero")
    if e and s["mean_excess"] < e["mean_excess"]:
        fails.append(f"below the existing rule ({s['mean_excess']:+.2%} vs {e['mean_excess']:+.2%})")
    return {"validated": not fails, "state": "validated" if not fails else "not validated",
            "why": "; ".join(fails), "evaluated": str(date.today())}
