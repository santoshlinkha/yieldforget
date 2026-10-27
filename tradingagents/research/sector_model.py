"""Which sector is likely to lead the market over the next 20 sessions, and its probable winning stock.

**Model.** LightGBM LambdaRank (gradient-boosted trees trained to order items within a group; groups are
decision dates, items the 11 SPDR sector funds) predicts the order of the sectors' returns over SPY for
the 20 sessions after the next close. A LightGBM regression on the same rows gives the size of that
excess, shown as an estimate. Features, all known at the decision close:

- the sector against SPY: relative momentum over 5, 21, 63, 126 and 252 sessions (and their rank among
  sectors that day), the relative-rotation ratio and momentum (``rotation.stages``), distance from its
  200-session average, 63-session drawdown, its volatility relative to SPY;
- the market drivers (shared by all sectors; the trees learn each sector's sensitivity through the
  sector code): the 10-year yield and the yield curve, oil, the dollar, credit (high yield against
  Treasuries), the VIX, and SPY's own trend and volatility.

**Validation.** Decision dates every 5 sessions from 2008. Yearly walk-forward test folds 2010-2024
(development); a training row is used only when its label ended before the fold starts (purged).
**Holdout**: 2025-01-01 on, scored once with everything frozen. Measured per date: the rank correlation
of the predicted and realized order (IC), and the excess of the predicted top three sectors over the
equal-weighted eleven. Overlapping 20-session labels every 5 sessions, so intervals use a block bootstrap
(blocks of 4 dates). Benchmarks on the same dates: plain 63-session relative momentum. **Validated** only
when the holdout IC and the top-three excess both have intervals above zero and the model beats momentum.

**Probable winner.** In each sector, the stock the internal stock model ranks highest in its latest daily
decision (GICS sectors of the S&P 500 members). That model has its own (experimental) status.

Prices: the local Yahoo store (adjusted closes). No paid data, no API calls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .rotation import SECTORS, stages

H, EVERY, FIRST_TEST, HOLDOUT = 20, 5, 2010, pd.Timestamp("2025-01-01")
GICS = {"Information Technology": "XLK", "Communication Services": "XLC", "Consumer Discretionary": "XLY",
        "Financials": "XLF", "Health Care": "XLV", "Industrials": "XLI", "Energy": "XLE", "Materials": "XLB",
        "Consumer Staples": "XLP", "Utilities": "XLU", "Real Estate": "XLRE"}
DRIVERS = ["^TNX", "^IRX", "USO", "UUP", "HYG", "IEF", "^VIX", "SPY"]
PARAMS = {"num_leaves": 7, "learning_rate": 0.03, "min_data_in_leaf": 80, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 11, "num_threads": 0}
ROUNDS = 300
VERSION = "sector-lambdarank-v1"


def features(close: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, sector) with every feature and the label (where known)."""
    close = close.ffill()
    spy = np.log(close["SPY"])
    st = stages(close)
    tnx, irx = close["^TNX"], close["^IRX"]
    curve = tnx - irx
    credit = np.log(close["HYG"]) - np.log(close["IEF"])
    macro = pd.DataFrame({
        "tnx_chg_21": tnx.diff(21), "tnx_chg_63": tnx.diff(63), "curve": curve, "curve_chg_21": curve.diff(21),
        "oil_21": np.log(close["USO"]).diff(21), "oil_63": np.log(close["USO"]).diff(63),
        "usd_21": np.log(close["UUP"]).diff(21), "usd_63": np.log(close["UUP"]).diff(63),
        "credit_21": credit.diff(21), "credit_63": credit.diff(63), "vix": close["^VIX"], "vix_chg_21": close["^VIX"].diff(21),
        "spy_63": spy.diff(63), "spy_vol_21": spy.diff().rolling(21).std() * np.sqrt(252)})
    frames = []
    for code, t in enumerate(SECTORS):
        if t not in close:
            continue
        lp = np.log(close[t])
        rel = lp - spy
        f = pd.DataFrame({f"rel_{k}": rel.diff(k) for k in (5, 21, 63, 126, 252)})
        f["ratio"] = st[("ratio", t)] - 100
        f["momentum"] = st[("momentum", t)] - 100
        f["ma200"] = lp - np.log(close[t].rolling(200).mean())
        f["dd_63"] = close[t] / close[t].rolling(63).max() - 1
        f["vol_rel"] = lp.diff().rolling(21).std() / spy.diff().rolling(21).std()
        f["sector"] = code
        # The label: the sector's return over SPY from the next close to H sessions after it.
        f["label"] = rel.shift(-(H + 1)) - rel.shift(-1)
        f["label_end"] = pd.Series(close.index, index=close.index).shift(-(H + 1))
        f = f.join(macro)
        f.index.name = "date"
        f["ticker"] = t
        frames.append(f.dropna(subset=["rel_252", "ratio", "momentum"]).set_index("ticker", append=True))
    panel = pd.concat(frames).sort_index()
    for k in (5, 21, 63, 126, 252):   # where each sector stands among the sectors that day
        panel[f"rank_{k}"] = panel.groupby(level="date")[f"rel_{k}"].rank(pct=True)
    return panel


FEATURES = None   # set on first use: every column but the label


def feature_names(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("label", "label_end")]


def _grades(df: pd.DataFrame) -> np.ndarray:
    pct = df.groupby(level="date")["label"].rank(pct=True)
    return np.clip(np.floor(pct * 5), 0, 4).astype(int).to_numpy()


def fit(train: pd.DataFrame, feats: list[str]) -> dict:
    import lightgbm as lgb

    train = train.sort_index(level="date")
    groups = train.groupby(level="date").size().to_numpy()
    ranker = lgb.train({**PARAMS, "objective": "lambdarank", "metric": "ndcg"},
                       lgb.Dataset(train[feats], _grades(train), group=groups, categorical_feature=["sector"]), ROUNDS)
    reg = lgb.train({**PARAMS, "objective": "regression"},
                    lgb.Dataset(train[feats], train["label"].clip(*train["label"].quantile([.01, .99])),
                                categorical_feature=["sector"]), ROUNDS)
    return {"ranker": ranker, "reg": reg, "features": feats,
            "trained_through": str(train.index.get_level_values("date").max().date())}


def predict(model: dict, rows: pd.DataFrame) -> pd.DataFrame:
    from tradingagents.stocks.training_store import guard

    guard(model.get("trained_through"), rows.index.get_level_values("date"))
    out = pd.DataFrame({"score": model["ranker"].predict(rows[model["features"]]),
                        "expected_excess": model["reg"].predict(rows[model["features"]])}, index=rows.index)
    out["rank_pct"] = out.groupby(level="date")["score"].rank(pct=True)
    return out


def _per_date(scores: pd.Series, label: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"s": scores, "y": label}).dropna()
    rows = []
    for d, g in df.groupby(level="date"):
        if len(g) < 6:
            continue
        top = g.nlargest(3, "s")["y"].mean()
        rows.append({"date": d, "ic": g["s"].rank().corr(g["y"].rank()), "top3_excess": top - g["y"].mean(),
                     "top1_beats_spy": float(g.nlargest(1, "s")["y"].iloc[0] > 0)})
    return pd.DataFrame(rows).set_index("date")


def _ci(x: pd.Series, block: int = 4, reps: int = 2000) -> list[float]:
    v = x.dropna().to_numpy()
    if len(v) < 2 * block:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(0)
    n, k = len(v), int(np.ceil(len(v) / block))
    means = [np.concatenate([v[s:s + block] for s in rng.integers(0, n - block + 1, k)])[:n].mean() for _ in range(reps)]
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _summary(per: pd.DataFrame) -> dict:
    return {"dates": len(per), "ic": float(per["ic"].mean()), "ic_ci": _ci(per["ic"]),
            "top3_excess": float(per["top3_excess"].mean()), "top3_ci": _ci(per["top3_excess"]),
            "top1_beats_spy": float(per["top1_beats_spy"].mean())}


def study(close: pd.DataFrame, progress=None) -> dict:
    """Walk-forward development folds, the holdout, the momentum benchmark and the validation status."""
    say = progress or (lambda *_: None)
    panel = features(close)
    feats = feature_names(panel)
    dates = panel.index.get_level_values("date").unique()
    decision = set(dates[(dates >= "2008-01-01")][::EVERY])
    panel = panel[panel.index.get_level_values("date").isin(decision)]
    labelled = panel[panel["label"].notna()]
    ld = labelled.index.get_level_values("date")
    preds = []
    folds = [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in range(FIRST_TEST, HOLDOUT.year)]
    folds.append(("holdout", HOLDOUT, ld.max() + pd.Timedelta(days=1)))
    for name, start, end in folds:
        test = labelled[(ld >= start) & (ld < end)]
        train = labelled[pd.to_datetime(labelled["label_end"]) < start]
        if test.empty or train.index.get_level_values("date").nunique() < 100:
            continue
        m = fit(train, feats)
        p = predict(m, test).assign(fold=name)
        preds.append(p)
        say(f"sector model: {name}")
    p = pd.concat(preds)
    lab = labelled["label"]
    mom = labelled["rel_63"]
    out = {"version": VERSION, "features": feats, "horizon": H, "every": EVERY, "parts": {}}
    for part, sel in (("development", p["fold"] != "holdout"), ("holdout", p["fold"] == "holdout")):
        idx = p.index[sel]
        out["parts"][part] = {"lambdarank": _summary(_per_date(p.loc[idx, "score"], lab.reindex(idx))),
                              "momentum_63": _summary(_per_date(mom.reindex(idx), lab.reindex(idx)))}
    h = out["parts"].get("holdout", {})
    lr, mo = h.get("lambdarank", {}), h.get("momentum_63", {})
    fails = []
    if not (lr.get("ic_ci", [np.nan])[0] > 0):
        lo, hi = lr.get("ic_ci", [float("nan")] * 2)
        fails.append(f"holdout rank correlation {lr.get('ic', float('nan')):+.3f} (95% interval {lo:+.3f} to {hi:+.3f}) "
                     "includes zero")
    if not (lr.get("top3_ci", [np.nan])[0] > 0):
        lo, hi = lr.get("top3_ci", [float("nan")] * 2)
        fails.append(f"its top 3 sectors beat the average by {lr.get('top3_excess', float('nan')):+.2%} per 20 sessions "
                     f"({lo:+.2%} to {hi:+.2%}), which includes zero")
    if lr.get("top3_excess", -1) <= mo.get("top3_excess", 0):
        fails.append(f"not better than plain momentum ({lr.get('top3_excess', float('nan')):+.2%} vs "
                     f"{mo.get('top3_excess', float('nan')):+.2%})")
    out["status"] = {"validated": not fails, "state": "validated" if not fails else "not validated", "why": "; ".join(fails)}
    return out, panel, feats


def run(cache_dir: Path, store=None, progress=None) -> dict:
    """Study, then fit on every labelled row and forecast the latest date; saved for the page."""
    from tradingagents.research.prices import bars

    say = progress or (lambda *_: None)
    t0 = time.time()
    close = bars(["SPY", *SECTORS, *DRIVERS], cache_dir)["Close"]
    say("sector model: walk-forward study")
    report, panel, feats = study(close, say)
    full = features(close)
    latest = full.index.get_level_values("date").max()
    labelled = panel[panel["label"].notna()]
    model = fit(labelled, feats)
    now = predict(model, full.xs(latest, level="date", drop_level=False)).droplevel("date")
    now = now.sort_values("score", ascending=False)
    report["forecast"] = {"as_of": str(latest.date()), "trained_through": model["trained_through"],
                          "sectors": [{"ticker": t, "name": SECTORS[t], "rank": i + 1,
                                       "rank_pct": round(float(r["rank_pct"]), 3),
                                       "expected_excess": round(float(r["expected_excess"]), 4)}
                                      for i, (t, r) in enumerate(now.iterrows())]}
    report["seconds"] = round(time.time() - t0, 1)
    folder = Path(cache_dir) / "sector_model"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "latest.json").write_text(json.dumps(report, default=float, indent=1), encoding="utf-8")
    if store is not None:
        from tradingagents.ledger.db import now_utc, transaction
        from tradingagents.ledger.recommendations import code_version

        with transaction(store.con):
            store.con.execute(
                "INSERT INTO research_trials (created_at, name, universe, horizon, first_test, last_test, config_json,"
                " code_version, scores_json, by_period_json, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now_utc(), "sector_model:h20", "spdr_sectors", H, f"{FIRST_TEST}-01-01", str(latest.date()),
                 json.dumps({"version": VERSION, "every": EVERY, "params": PARAMS, "rounds": ROUNDS}), code_version(),
                 json.dumps({k: v for k, v in report.items() if k != "forecast"}, default=float), "{}", report["seconds"]))
    return report


def latest(cache_dir: Path) -> dict | None:
    f = Path(cache_dir) / "sector_model" / "latest.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def winners(decision: dict | None, members: dict[str, str]) -> dict[str, dict]:
    """Per sector fund, the stock the internal model ranks highest in the latest decision."""
    scores = (decision or {}).get("scores") or {}
    best = {}
    for t, s in scores.items():
        label = members.get(t) or ""
        gics = label[label.rfind("(") + 1:-1] if label.endswith(")") else ""
        etf = GICS.get(gics)
        if not etf or not s.get("member", True):
            continue
        if etf not in best or s["rank"] < best[etf]["rank"]:
            best[etf] = {"ticker": t, "name": label[:label.rfind("(")].strip(), "rank": s["rank"], "of": s["of"]}
    return best


# --- the pooled structural model: macro-implied push, relative strength, residual reversal ---

def driver_frame(close: pd.DataFrame, cache_dir) -> pd.DataFrame:
    """Daily moves of the 8 drivers of ``rotation`` (the threats index as 0 when it cannot be had)."""
    from .rotation import DRIVERS, driver_moves

    d = driver_moves(close, cache_dir)
    for name in DRIVERS:
        if name not in d:
            d[name] = 0.0
    return d


def structural(close: pd.DataFrame, dates: pd.DatetimeIndex, cache_dir, n: int = 252, lam: float = 1.0) -> pd.DataFrame:
    """Per (date, sector): macro_implied (rolling ridge betas x 20-session EWMA of each driver's move),
    resid_10 (sum of the last 10 unexplained moves) and the stage features."""
    drivers = driver_frame(close, cache_dir).fillna(0.0)
    spy = np.log(close["SPY"]).diff()
    trend = drivers.ewm(span=20, min_periods=10).mean()
    st = stages(close)
    rows = []
    for t in SECTORS:
        if t not in close:
            continue
        y = (np.log(close[t]).diff() - spy)
        for d in dates:
            i = close.index.get_indexer([d])[0]
            if i < n:
                continue
            Y = y.iloc[i - n + 1:i + 1].to_numpy()
            Xm = drivers.iloc[i - n + 1:i + 1].to_numpy()
            ok = np.isfinite(Y) & np.isfinite(Xm).all(axis=1)
            if ok.sum() < n // 2:
                continue
            Xs, Ys = Xm[ok], Y[ok]
            mu, sd = Xs.mean(0), Xs.std(0) + 1e-12
            Z = (Xs - mu) / sd
            beta = np.linalg.solve(Z.T @ Z + lam * len(Z) * 0.01 * np.eye(Z.shape[1]), Z.T @ (Ys - Ys.mean())) / sd
            resid = Ys - Ys.mean() - (Xs - mu) @ beta
            rows.append({"date": d, "ticker": t, "macro_implied": float(trend.iloc[i].to_numpy() @ beta) * 20,
                         "resid_10": float(resid[-10:].sum()),
                         "ratio": float(st[("ratio", t)].iloc[i] - 100), "momentum": float(st[("momentum", t)].iloc[i] - 100)})
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


STRUCT_FEATURES = ["macro_implied", "resid_10", "ratio", "momentum", "rel_63"]


def fit_pooled(train: pd.DataFrame) -> dict:
    from sklearn.linear_model import RidgeCV

    X = train[STRUCT_FEATURES]
    mu, sd = X.mean(), X.std().replace(0, 1)
    m = RidgeCV(alphas=np.logspace(-1, 4, 16)).fit(((X - mu) / sd).fillna(0), train["label"])
    return {"model": m, "mu": mu, "sd": sd, "trained_through": str(train.index.get_level_values("date").max().date())}


def predict_pooled(model: dict, rows: pd.DataFrame) -> pd.Series:
    from tradingagents.stocks.training_store import guard

    guard(model["trained_through"], rows.index.get_level_values("date"))
    X = ((rows[STRUCT_FEATURES] - model["mu"]) / model["sd"]).fillna(0)
    return pd.Series(model["model"].predict(X), index=rows.index)


def compare(close: pd.DataFrame, cache_dir, progress=None) -> dict:
    """LambdaRank vs the pooled structural ridge, on the same dates and folds."""
    say = progress or (lambda *_: None)
    panel = features(close)
    dates = panel.index.get_level_values("date").unique()
    decision = dates[(dates >= "2008-01-01")][::EVERY]
    panel = panel[panel.index.get_level_values("date").isin(decision)]
    say("structural features")
    sf = structural(close, pd.DatetimeIndex(decision), cache_dir)
    panel = panel.join(sf.drop(columns=["ratio", "momentum"]), how="left")
    lab = panel[panel["label"].notna() & panel["macro_implied"].notna()]
    ld = lab.index.get_level_values("date")
    feats = feature_names(panel.drop(columns=["macro_implied", "resid_10"]))
    folds = [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in range(FIRST_TEST, HOLDOUT.year)]
    folds.append(("holdout", HOLDOUT, ld.max() + pd.Timedelta(days=1)))
    lr_, pr_ = [], []
    for name, start, end in folds:
        test = lab[(ld >= start) & (ld < end)]
        train = lab[pd.to_datetime(lab["label_end"]) < start]
        if test.empty or train.index.get_level_values("date").nunique() < 100:
            continue
        lr_.append(predict(fit(train, feats), test)["score"].rename("s").to_frame().assign(fold=name))
        pr_.append(predict_pooled(fit_pooled(train), test).rename("s").to_frame().assign(fold=name))
        say(f"compare {name}")
    out = {}
    for model, frames in (("lambdarank", lr_), ("pooled_ridge", pr_), ("momentum_63", None)):
        p = pd.concat(frames) if frames else pd.concat(lr_)[["fold"]].assign(s=lab["rel_63"])
        for part, sel in (("development", p["fold"] != "holdout"), ("holdout", p["fold"] == "holdout")):
            idx = p.index[sel]
            out.setdefault(part, {})[model] = _summary(_per_date(p.loc[idx, "s"], lab["label"].reindex(idx)))
    return out


def causes_r2(close: pd.DataFrame, cache_dir, window: int = 252) -> dict:
    """Average in-window R squared of each sector's daily excess return on 6 vs 8 drivers (every 21 sessions)."""
    d8 = driver_frame(close, cache_dir)
    d6 = d8.iloc[:, :6]
    spy = np.log(close["SPY"]).diff()
    out = {}
    for label, D in (("6 drivers", d6), ("8 drivers", d8)):
        r2s = []
        for t in SECTORS:
            if t not in close:
                continue
            y = np.log(close[t]).diff() - spy
            for i in range(window, len(close), 21):
                Y, X = y.iloc[i - window:i].to_numpy(), D.iloc[i - window:i].to_numpy()
                ok = np.isfinite(Y) & np.isfinite(X).all(axis=1)
                if ok.sum() < window // 2:
                    continue
                Xs = np.column_stack([np.ones(ok.sum()), X[ok]])
                b, *_ = np.linalg.lstsq(Xs, Y[ok], rcond=None)
                res = Y[ok] - Xs @ b
                r2s.append(1 - res.var() / Y[ok].var())
        out[label] = float(np.mean(r2s))
    return out
