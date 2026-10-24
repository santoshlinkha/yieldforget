"""One neural sequence challenger: a compact LSTM on each asset's recent daily returns, with the tabular features.

Why this one: the universe is a few dozen ETFs with about 20 years of daily data,
which is far too little for a large transformer (PatchTST, TFT, iTransformer) to
learn more than a small model would. A one-layer LSTM over the last 60 sessions of
the asset's and SPY's returns, each divided by its own trailing volatility, joined
with the same ranked features the trees use through a gate per feature (a light
variable-selection layer), has about ten thousand weights. It is pooled across
assets, like the trees.

Training uses the same rows, folds and labels as every other model: mean
squared error on the clipped label, Adam, early stopping on the newest fifth of
the training dates, a fixed seed, and the CPU or a CUDA GPU (``device``). Inputs
are only returns up to each row's date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Design, Forecaster, clip_label, inner_split

WINDOW = 60


class Sequences:
    """Daily log returns, divided by trailing 63-session volatility, known up to each date."""

    def __init__(self, close: pd.DataFrame, market: pd.Series):
        r = np.log(close).diff()
        m = np.log(market.reindex(close.index)).diff()
        scale = r.rolling(63, min_periods=40).std().shift(1)
        mscale = m.rolling(63, min_periods=40).std().shift(1)
        self.asset = (r / scale).clip(-6, 6).fillna(0.0)
        self.market = (m / mscale).clip(-6, 6).fillna(0.0)
        self.pos = {d: i for i, d in enumerate(close.index)}

    def windows(self, rows: pd.DataFrame) -> np.ndarray:
        """``(rows, WINDOW, 2)``: each row's last WINDOW sessions up to and including its date."""
        a, m = self.asset.to_numpy(np.float32), self.market.to_numpy(np.float32)
        cols = {t: j for j, t in enumerate(self.asset.columns)}
        out = np.zeros((len(rows), WINDOW, 2), dtype=np.float32)
        for k, (d, t) in enumerate(rows.index):
            i = self.pos[d] + 1
            lo = max(0, i - WINDOW)
            out[k, WINDOW - (i - lo):, 0] = a[lo:i, cols[t]]
            out[k, WINDOW - (i - lo):, 1] = m[lo:i]
        return out


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(self, sequences: Sequences, device: str = "cpu", seed: int = 7, hidden: int = 32,
                 epochs: int = 40, batch: int = 512, patience: int = 5):
        self.sequences, self.device, self.seed = sequences, device, seed
        self.hidden, self.epochs, self.batch, self.patience = hidden, epochs, batch, patience

    def _net(self, n_features: int):
        import torch
        from torch import nn

        hidden = self.hidden

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Parameter(torch.zeros(n_features))      # per-feature selection
                self.lstm = nn.LSTM(2, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden + n_features, hidden), nn.GELU(),
                                          nn.Dropout(0.1), nn.Linear(hidden, 1))

            def forward(self, seq, tab):
                _, (h, _) = self.lstm(seq)
                x = torch.cat([h[-1], tab * torch.sigmoid(self.gate)], dim=1)
                return self.head(x).squeeze(1)

        return Net()

    def _tensors(self, rows, labels=True):
        import torch

        seq = torch.from_numpy(self.sequences.windows(rows))
        tab = torch.from_numpy(self.design(rows))
        y = torch.from_numpy(clip_label(rows["label"]).to_numpy(np.float32)) * 100 if labels else None
        return seq, tab, y

    def _train(self, rows, valid, epochs=None):
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        net = self._net(len(self.design.features)).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        seq, tab, y = (t.to(self.device) for t in self._tensors(rows))
        if valid is not None:
            vs, vt, vy = (t.to(self.device) for t in self._tensors(valid))
        gen = torch.Generator().manual_seed(self.seed)
        best, best_epoch, best_state, waited = float("inf"), 0, None, 0
        for epoch in range(epochs or self.epochs):
            net.train()
            order = torch.randperm(len(y), generator=gen).to(self.device)
            for start in range(0, len(y), self.batch):
                idx = order[start:start + self.batch]
                opt.zero_grad()
                loss = torch.mean((net(seq[idx], tab[idx]) - y[idx]) ** 2)
                loss.backward()
                opt.step()
            if valid is None:
                continue
            net.eval()
            with torch.no_grad():
                err = float(torch.mean((net(vs, vt) - vy) ** 2))
            if err < best - 1e-6:
                best, best_epoch, waited = err, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_epoch

    def fit(self, train, features, horizon):
        self.design = Design().fit(train, features)
        before, valid = inner_split(train)
        _, self.best_epochs = self._train(before, valid)
        # Refit on everything for the epochs that worked on the held-back fifth.
        self.net, _ = self._train(train, None, epochs=max(self.best_epochs, 1))
        return self

    def predict(self, rows):
        import torch

        seq, tab, _ = self._tensors(rows, labels=False)
        self.net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), 4096):   # bounded batches keep VRAM use small
                out.append(self.net(seq[start:start + 4096].to(self.device),
                                    tab[start:start + 4096].to(self.device)).cpu().numpy())
        return np.concatenate(out) / 100

    def describe(self):
        import torch

        gates = torch.sigmoid(self.net.gate).detach().cpu().numpy()
        return {"name": self.name, "device": self.device, "seed": self.seed, "hidden": self.hidden,
                "epochs": self.best_epochs, "torch": torch.__version__,
                "weights": int(sum(p.numel() for p in self.net.parameters())),
                "gates": dict(sorted(zip(self.design.features, np.round(gates, 3).tolist(), strict=True),
                                     key=lambda kv: -kv[1]))}
