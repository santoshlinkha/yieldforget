"""Modern challengers for the stock ranking, behind the common adapter interface (``adapters``).

Verified 2026-10-24 (Hugging Face API and model cards):

| Model | Checkpoint, revision | Licence | Training data, contamination |
|---|---|---|---|
| Chronos-2 | amazon/chronos-2 @ 29ec376 | Apache-2.0 | Chronos datasets, GIFT-Eval pretrain, synthetic; no cutoff published: **unknown** |
| TimesFM 2.5 (200M) | google/timesfm-2.5-200m-pytorch @ 1d95242 | Apache-2.0 | Google's mixed corpus (GIFT-Eval pretrain, trends, pageviews, synthetic): **unknown** |
| TimesFM 3.0 | google/timesfm-3.0-pytorch @ 43046b8 (2026-08-24) | **TimesFM Non-Commercial v1.0** | GiftEval pretrain minus fev-bench, pageviews to Nov 2023, trends to 2022, synthetic. Not used by default: the owner decides whether personal research fits the licence |
| Kronos-mini (4.1M) | NeoQuasar/Kronos-mini @ f4e6869, tokenizer Kronos-Tokenizer-2k @ 26966d0 | MIT | Trained on financial K-lines from many exchanges: **high** (it has likely seen these prices) |

Kronos's code is not on PyPI: the MIT-licensed ``model/`` package of github.com/shiyu-coder/Kronos is
fetched once at a pinned commit (``KRONOS_COMMIT``) into the cache folder and imported from there.
TimesFM uses the ``timesfm`` package (3.0.2, Apache-2.0), installed alongside the existing PyTorch.

Conventions. Pretrained models are zero-shot: their score is the forecast log return of the close
over the horizon (Chronos-2 and TimesFM: median; Kronos: the mean of sampled paths, which are not
calibrated probabilities). Future covariates: only the trading calendar (known at the cutoff).
Retrospective scores are exploration: every pretrained model may have seen the test years, so only
forecasts recorded prospectively can confirm one. PatchTST is trained here, pooled, chronologically.

Every adapter reports its status, revision, device, runtime and peak memory; a missing dependency
gives a precise ``unavailable`` status and never breaks the application.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import Adapter

REVISIONS = {"chronos2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
             "timesfm25": ("google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"),
             "kronos_mini": ("NeoQuasar/Kronos-mini", "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"),
             "kronos_tokenizer": ("NeoQuasar/Kronos-Tokenizer-2k", "26966d0035065a0cae0ebad7af8ece35bc1fb51c")}
KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


class Measured(Adapter):
    """Runtime and peak memory of the last ``predict`` (CUDA peak on the GPU, tracemalloc on the CPU)."""

    device = "cpu"

    def _measure(self, fn):
        import torch

        t0 = time.time()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            out = fn()
            self.last = {"seconds": round(time.time() - t0, 2), "peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                         "device": self.device}
        else:
            tracemalloc.start()
            out = fn()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.last = {"seconds": round(time.time() - t0, 2), "peak_mb": round(peak / 2 ** 20, 1), "device": "cpu"}
        return out


def _contexts(close: pd.DataFrame, rows: pd.DataFrame, length: int) -> tuple[list[np.ndarray], np.ndarray]:
    """For each (date, ticker) row: the last ``length`` log closes up to and including the date."""
    logp = np.log(close)
    pos = {d: i for i, d in enumerate(close.index)}
    out, last = [], []
    for d, t in rows.index:
        i = pos[d] + 1
        s = logp[t].iloc[max(0, i - length):i].dropna().to_numpy(np.float32)
        out.append(s if len(s) else np.zeros(1, np.float32))
        last.append(s[-1] if len(s) else np.nan)
    return out, np.array(last)


class Chronos2Score(Measured):
    name, kind = "chronos2", "score"

    def __init__(self, close: pd.DataFrame, horizon: int, device="cpu"):
        self.close, self.horizon, self.device = close, horizon, device

    def status(self):
        try:
            importlib.import_module("chronos")
            return "available", ""
        except ImportError as exc:
            return "unavailable", f"chronos-forecasting not installed ({exc})"

    def predict(self, rows):
        from tradingagents.research.foundation import Chronos2

        def run():
            m = Chronos2(self.close, device=self.device).fit(None, [], self.horizon)
            return m._forecast(rows)[:, 1]
        return self._measure(run)

    def describe(self):
        return {"name": self.name, "checkpoint": REVISIONS["chronos2"], "license": "Apache-2.0",
                "contamination": "unknown", **getattr(self, "last", {})}


class TimesFM25Score(Measured):
    name, kind = "timesfm25", "score"
    CONTEXT = 512

    def __init__(self, close: pd.DataFrame, horizon: int, device="cpu", batch=128):
        self.close, self.horizon, self.device, self.batch = close, horizon, device, batch

    def status(self):
        try:
            importlib.import_module("timesfm")
            return "available", ""
        except ImportError as exc:
            return "unavailable", f"timesfm not installed ({exc})"

    def _model(self):
        import timesfm
        import torch

        repo, rev = REVISIONS["timesfm25"]
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=rev)
        if not self.device.startswith("cuda"):
            torch.set_num_threads(max(1, torch.get_num_threads()))
        m.compile(timesfm.ForecastConfig(max_context=self.CONTEXT, max_horizon=max(self.horizon, 64),
                                         normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
                                         per_core_batch_size=self.batch))
        return m

    def predict(self, rows):
        def run():
            model = self._model()
            ctx, last = _contexts(self.close, rows, self.CONTEXT)
            out = np.empty(len(rows))
            for a in range(0, len(ctx), self.batch):
                point, _ = model.forecast(horizon=self.horizon, inputs=[np.exp(c) for c in ctx[a:a + self.batch]])
                out[a:a + self.batch] = np.log(np.maximum(point[:, self.horizon - 1], 1e-9)) - last[a:a + self.batch]
            return out
        return self._measure(run)

    def describe(self):
        return {"name": self.name, "checkpoint": REVISIONS["timesfm25"], "license": "Apache-2.0",
                "contamination": "unknown", **getattr(self, "last", {})}


def kronos_dir(cache_dir: Path) -> Path:
    return Path(cache_dir).parent / "third_party" / f"kronos-{KRONOS_COMMIT[:12]}"


class KronosScore(Measured):
    """Kronos-mini on daily OHLCV (opens from ``opens``; a row without a real open is skipped, never
    filled with a stand-in). Score: the mean of ``samples`` sampled paths' close at the horizon."""
    name, kind = "kronos_mini", "score"
    CONTEXT = 256

    def __init__(self, ohlcv: dict[str, pd.DataFrame], horizon: int, sessions: pd.DatetimeIndex, cache_dir: Path,
                 device="cpu", samples=4, batch=64):
        self.ohlcv, self.horizon, self.sessions, self.cache_dir = ohlcv, horizon, sessions, Path(cache_dir)
        self.device, self.samples, self.batch = device, samples, batch

    def status(self):
        d = kronos_dir(self.cache_dir)
        if not (d / "model" / "kronos.py").exists():
            return "unavailable", (f"Kronos code not present at {d} (fetch github.com/shiyu-coder/Kronos at commit "
                                   f"{KRONOS_COMMIT}); it is not on PyPI")
        try:
            importlib.import_module("einops")
        except ImportError as exc:
            return "unavailable", f"einops not installed ({exc})"
        return "available", ""

    def _predictor(self):
        d = str(kronos_dir(self.cache_dir))
        if d not in sys.path:
            sys.path.insert(0, d)
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

        tok = KronosTokenizer.from_pretrained(REVISIONS["kronos_tokenizer"][0], revision=REVISIONS["kronos_tokenizer"][1])
        model = Kronos.from_pretrained(REVISIONS["kronos_mini"][0], revision=REVISIONS["kronos_mini"][1])
        return KronosPredictor(model, tok, device=self.device, max_context=self.CONTEXT)

    def predict(self, rows):
        def run():
            pred = self._predictor()
            out = np.full(len(rows), np.nan)
            pos = {d: i for i, d in enumerate(self.sessions)}
            jobs = []
            for k, (d, t) in enumerate(rows.index):
                frame = self.ohlcv.get(t)
                if frame is None:
                    continue
                hist = frame.loc[:d].dropna().iloc[-self.CONTEXT:]
                if len(hist) < self.CONTEXT or hist.index[-1] != d:
                    continue
                i = pos[d]
                future = self.sessions[i + 1:i + 1 + self.horizon]
                if len(future) < self.horizon:      # beyond the data: the calendar is not known
                    continue
                jobs.append((k, hist, future))
            for a in range(0, len(jobs), self.batch):
                chunk = jobs[a:a + self.batch]
                res = pred.predict_batch([j[1].reset_index(drop=True) for j in chunk],
                                         [pd.Series(j[1].index) for j in chunk], [pd.Series(j[2]) for j in chunk],
                                         pred_len=self.horizon, T=1.0, top_p=0.9, sample_count=self.samples, verbose=False)
                for (k, hist, _), r in zip(chunk, res, strict=True):
                    out[k] = math.log(max(float(r["close"].iloc[-1]), 1e-9) / float(hist["close"].iloc[-1]))
            return out
        return self._measure(run)

    def describe(self):
        return {"name": self.name, "checkpoint": REVISIONS["kronos_mini"], "tokenizer": REVISIONS["kronos_tokenizer"],
                "code_commit": KRONOS_COMMIT, "license": "MIT", "contamination": "high (trained on market K-lines)",
                "samples": self.samples, **getattr(self, "last", {})}


class PatchTST(Measured):
    """A compact PatchTST: 64 sessions of (log return, |return|, log volume change) cut into 8 patches,
    a 2-layer transformer encoder (d=32, 4 heads), the point-in-time context features joined at the
    head, trained pooled on the within-date demeaned tradable label (Adam, early-stopped on the last
    fifth of training dates, purged). About 30k weights."""
    name, kind = "patchtst", "score"
    L, PATCH = 64, 8

    def __init__(self, close: pd.DataFrame, volume: pd.DataFrame, device="cpu", epochs=6, seed=7):
        self.close, self.volume, self.device, self.epochs, self.seed = close, volume, device, epochs, seed

    def status(self):
        try:
            importlib.import_module("torch")
            return "available", ""
        except ImportError as exc:
            return "unavailable", f"torch not installed ({exc})"

    def _seq(self, rows):
        r = np.log(self.close).diff()
        v = np.log(self.volume.replace(0, np.nan)).diff()
        pos = {d: i for i, d in enumerate(self.close.index)}
        cols = {t: i for i, t in enumerate(self.close.columns)}
        R, V = r.to_numpy(np.float32), v.to_numpy(np.float32)
        out = np.zeros((len(rows), self.L, 3), np.float32)
        for k, (d, t) in enumerate(rows.index):
            i, j = pos[d], cols[t]
            a = max(0, i - self.L + 1)
            seg_r, seg_v = R[a:i + 1, j], V[a:i + 1, j]
            x = np.stack([seg_r, np.abs(seg_r), seg_v], axis=1)
            x = np.nan_to_num(x)
            x = (x - x.mean(0)) / (x.std(0) + 1e-6)
            out[k, -len(x):] = x
        return out

    def _net(self, n_ctx):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.embed = nn.Linear(3 * PatchTST.PATCH, 32)
                s.pos = nn.Parameter(torch.zeros(1, PatchTST.L // PatchTST.PATCH, 32))
                layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True)
                s.enc = nn.TransformerEncoder(layer, 2)
                s.head = nn.Sequential(nn.Linear(32 + n_ctx, 32), nn.GELU(), nn.Linear(32, 1))

            def forward(s, x, c):
                b = x.shape[0]
                p = x.reshape(b, PatchTST.L // PatchTST.PATCH, PatchTST.PATCH * 3)
                h = s.enc(s.embed(p) + s.pos).mean(1)
                return s.head(torch.cat([h, c], 1)).squeeze(1)
        return Net()

    def fit(self, train, features):
        import torch

        from tradingagents.research.models import Design, inner_split

        rows = train[train["label"].notna()].sort_index(level="date")
        y = (rows["label"] - rows.groupby(level="date")["label"].transform("mean")).clip(-0.5, 0.5)
        rows = rows.assign(label=y)
        self.design = Design().fit(rows, features)
        before, valid = inner_split(rows)
        dev = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        net = self._net(len(features)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

        def tensors(f):
            return (torch.tensor(self._seq(f)), torch.tensor(self.design(f)), torch.tensor(f["label"].to_numpy(np.float32)))
        xb, cb, yb = tensors(before)
        xv, cv, yv = tensors(valid)
        best, state = float("inf"), None
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(xb))
            for a in range(0, len(perm), 1024):
                i = perm[a:a + 1024]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(xb[i].to(dev), cb[i].to(dev)), yb[i].to(dev))
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(torch.nn.functional.mse_loss(net(xv.to(dev), cv.to(dev)), yv.to(dev)))
            if v < best:
                best, state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        net.load_state_dict(state)
        self.net, self.dev, self.valid_mse = net.eval(), dev, best
        return self

    def predict(self, rows):
        import torch

        def run():
            with torch.no_grad():
                x, c = torch.tensor(self._seq(rows)), torch.tensor(self.design(rows))
                return np.concatenate([self.net(x[a:a + 4096].to(self.dev), c[a:a + 4096].to(self.dev)).cpu().numpy()
                                       for a in range(0, len(x), 4096)])
        return self._measure(run)

    def describe(self):
        return {"name": self.name, "weights": int(sum(p.numel() for p in self.net.parameters())),
                "valid_mse": getattr(self, "valid_mse", None), **getattr(self, "last", {})}


def availability(cache_dir: Path) -> dict:
    """Every challenger's status, without loading any model."""
    dummy = pd.DataFrame()
    out = {}
    for a in (Chronos2Score(dummy, 20), TimesFM25Score(dummy, 20), KronosScore({}, 20, pd.DatetimeIndex([]), cache_dir),
              PatchTST(dummy, dummy)):
        s, why = a.status()
        out[a.name] = {"status": s, "reason": why}
    out["timesfm30"] = {"status": "not used by default",
                        "reason": "TimesFM 3.0 weights carry a non-commercial licence; enable deliberately"}
    return out


def fetch_opens(tickers: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Daily OHLCV with real opens (split- and dividend-adjusted, Yahoo), for the challenger sample only,
    downloaded once and saved; the main price cache has no opens."""
    import yfinance as yf

    folder = Path(cache_dir) / "ohlcv"
    folder.mkdir(parents=True, exist_ok=True)
    out, missing = {}, []
    for t in tickers:
        p = folder / f"{t}.pkl"
        if p.exists():
            out[t] = pd.read_pickle(p)
        else:
            missing.append(t)
    if missing:
        raw = yf.download(missing, start="2000-01-01", auto_adjust=True, progress=False, group_by="ticker", threads=True)
        for t in missing:
            try:
                f = raw[t].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            except KeyError:
                continue
            f.index = pd.to_datetime(f.index).tz_localize(None)
            f.to_pickle(folder / f"{t}.pkl")
            out[t] = f
    return out


def describe_all(adapters) -> str:
    return json.dumps([a.describe() for a in adapters], default=str)
