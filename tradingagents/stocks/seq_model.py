"""A Mamba (selective state-space) model of each stock's recent daily path, trained to rank stocks.

Mamba reads a sequence in linear time, keeping a learned summary of what came before. Here the official
``mamba-ssm`` CUDA kernels do not build on Windows, so it runs on ``mambapy`` (the same architecture in plain PyTorch,
slower but exact), on the GPU when there is one.

**Input.** For each (date, ticker): the last ``LOOKBACK`` sessions to the decision close of six daily channels, each
known at its close: the stock's log return, SPY's, its sector fund's, the stock less SPY, its high-low range and its
volume against its 20-session average. Returns and ranges are divided by the stock's own volatility over the window
(its log is a seventh, constant channel), so a calm and a wild stock look alike in shape. The 54 panel features of
the date join the sequence summary before the output layer.

**Target.** The stock's 20-session return over cash as a within-date rank, centered (-0.5 to 0.5): the model learns
to order stocks, like the internal ranker, not to guess the market. Loss: mean squared error.

**Training.** Rows in time order; the last 10% of training dates (purged by label end) choose the epoch; at most
``ROWS_PER_EPOCH`` rows sampled per epoch. Small on purpose (``D_MODEL``, ``LAYERS``): financial targets are mostly
noise and a 4 GB GPU must hold it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 64
D_MODEL, LAYERS = 32, 2
EPOCHS, BATCH, ROWS_PER_EPOCH, LR = 3, 512, 60_000, 1e-3
PREDICT_CHUNK = 1024      # the plain-PyTorch scan holds batch x length x 2*D_MODEL x 16 floats: keep it small
CHANNELS = 7


class SequenceBank:
    """Every stock's daily channels as one array (sessions x tickers x channels), built once."""

    def __init__(self, data, start: str = "2004-06-01"):
        close = data.close.loc[start:]
        tickers = list(close.columns)
        lc = np.log(close)
        r = lc.diff()
        spy = r["SPY"] if "SPY" in r else pd.Series(0.0, index=r.index)
        sector = pd.DataFrame(0.0, index=r.index, columns=tickers)
        for t in tickers:
            etf = (data.info.get(t) or {}).get("sector_etf")
            if etf and etf in r:
                sector[t] = r[etf]
        high = data.high.reindex(index=close.index, columns=tickers) if getattr(data, "high", None) is not None else close
        low = data.low.reindex(index=close.index, columns=tickers)
        rng = np.log(high / low).replace([np.inf, -np.inf], np.nan)
        vol = data.volume.reindex(index=close.index, columns=tickers) if getattr(data, "volume", None) is not None else None
        vchg = np.log(vol / vol.rolling(20, min_periods=5).mean()) if vol is not None else r * 0
        chans = [r, pd.DataFrame(np.repeat(spy.to_numpy()[:, None], len(tickers), 1), index=r.index, columns=tickers),
                 sector, r.sub(spy, axis=0), rng, vchg.replace([np.inf, -np.inf], np.nan)]
        self.x = np.stack([c.to_numpy(dtype=np.float32) for c in chans], axis=-1)      # S x T x 6
        self.x = np.nan_to_num(self.x, nan=0.0)
        self.sessions = close.index
        self.col = {t: i for i, t in enumerate(tickers)}

    def batch(self, dates: pd.DatetimeIndex, tickers: list[str]) -> np.ndarray:
        """(rows, LOOKBACK, CHANNELS), each window ending at its date's close, volatility-scaled."""
        p = self.sessions.get_indexer(dates)
        c = np.array([self.col.get(t, -1) for t in tickers])
        ok = (p >= LOOKBACK - 1) & (c >= 0)
        out = np.zeros((len(p), LOOKBACK, CHANNELS), dtype=np.float32)
        if ok.any():
            steps = p[ok, None] + np.arange(-LOOKBACK + 1, 1)[None, :]
            w = self.x[steps, c[ok, None], :]                                          # rows x L x 6
            sd = w[:, :, 0].std(axis=1, keepdims=True) + 1e-4
            w[:, :, [0, 1, 2, 3, 4]] /= sd[:, :, None]
            w[:, :, 5] = np.clip(w[:, :, 5], -3, 3)
            out[ok, :, :6] = np.clip(w, -10, 10)
            out[ok, :, 6] = np.log(sd)
        return out


def _net(n_static: int, device: str):
    import torch
    from mambapy.mamba import Mamba, MambaConfig

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(CHANNELS, D_MODEL)
            self.mamba = Mamba(MambaConfig(d_model=D_MODEL, n_layers=LAYERS))
            self.head = torch.nn.Sequential(torch.nn.Linear(D_MODEL + n_static, 64), torch.nn.GELU(),
                                            torch.nn.Dropout(0.1), torch.nn.Linear(64, 1))

        def forward(self, seq, static):
            h = self.mamba(self.inp(seq))[:, -1, :]                                    # the summary at the last step
            return self.head(torch.cat([h, static], dim=1)).squeeze(1)

    return Net().to(device)


class MambaRanker:
    name = "mamba"

    def __init__(self, bank: SequenceBank, device: str = "cpu", seed: int = 7):
        self.bank, self.device, self.seed = bank, device, seed

    def __getstate__(self):
        """Saved as its weights (the network class is built inside ``_net``), without the price bank."""
        state = {k: v for k, v in self.__dict__.items() if k not in ("net", "bank")}
        if getattr(self, "net", None) is not None:
            state["_net_state"] = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        return state

    def __setstate__(self, state):
        import torch

        weights = state.pop("_net_state", None)
        self.__dict__.update(state)
        self.bank = None                       # given back by the caller (``model_zoo.run``) before predicting
        if weights is not None:
            if str(self.device).startswith("cuda") and not torch.cuda.is_available():
                self.device = "cpu"
            self.net = _net(len(self.features), self.device)
            self.net.load_state_dict(weights)
            self.net.eval()

    def _static(self, rows, fit=False):
        x = rows[self.features].astype(float)
        if fit:
            self.mu, self.sd = x.mean(), x.std().replace(0, 1).fillna(1)
        return np.clip(((x - self.mu) / self.sd).fillna(0).to_numpy(np.float32), -5, 5)

    def _xy(self, rows):
        seq = self.bank.batch(rows.index.get_level_values("date"), list(rows.index.get_level_values("ticker")))
        return seq, self._static(rows)

    def fit(self, train: pd.DataFrame, features: list[str]):
        import torch

        from tradingagents.research.models import inner_split

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.features = list(features)
        train = train[train["label"].notna()].sort_index(level="date")
        self._static(train, fit=True)
        y_all = (train["label"].groupby(level="date").rank(pct=True) - 0.5).to_numpy(np.float32)
        before, valid = inner_split(train)
        pos = {k: i for i, k in enumerate(train.index)}
        tr_i = np.array([pos[k] for k in before.index])
        va_i = np.array([pos[k] for k in valid.index])
        va_i = rng.choice(va_i, size=min(len(va_i), 20_000), replace=False)
        self.net = _net(len(self.features), self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=1e-4)
        vseq, vstat = self._xy(train.iloc[va_i])
        best, best_state = np.inf, None
        for _ in range(EPOCHS):
            self.net.train()
            for b in np.array_split(rng.permutation(tr_i)[:ROWS_PER_EPOCH], max(1, min(len(tr_i), ROWS_PER_EPOCH) // BATCH)):
                seq, stat = self._xy(train.iloc[b])
                pred = self.net(torch.from_numpy(seq).to(self.device), torch.from_numpy(stat).to(self.device))
                loss = torch.mean((pred - torch.from_numpy(y_all[b]).to(self.device)) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_v = float(np.mean((self._predict(vseq, vstat) - y_all[va_i]) ** 2))
            if loss_v < best:
                best, best_state = loss_v, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        if self.device != "cpu":
            torch.cuda.empty_cache()
        return self

    def _predict(self, seq, stat) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seq), PREDICT_CHUNK):
                out.append(self.net(torch.from_numpy(seq[i:i + PREDICT_CHUNK]).to(self.device),
                                    torch.from_numpy(stat[i:i + PREDICT_CHUNK]).to(self.device)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        seq, stat = self._xy(rows)
        return self._predict(seq, stat)
