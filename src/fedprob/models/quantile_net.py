"""Jaringan saraf kecil yang memprediksi beberapa kuantil sekaligus.

Input : histori `lookback` hari (ternormalisasi) + fitur kalender H hari target
Output: (H, Q) kuantil, dijamin terurut (tidak saling silang) dengan cara
        memprediksi median lalu menambah/mengurangi selisih positif (softplus).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import QUANTILES
from ..data.windows import HORIZON, LOOKBACK, N_CAL


class QuantileMLP(nn.Module):
    def __init__(
        self,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        n_cal: int = N_CAL,
        quantiles=QUANTILES,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.quantiles = tuple(quantiles)
        q = torch.tensor(self.quantiles)
        self.register_buffer("q", q)
        self.mid = int(torch.argmin((q - 0.5).abs()))
        self.n_q = len(self.quantiles)
        self.net = nn.Sequential(
            nn.Linear(lookback + horizon * n_cal, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * self.n_q),
        )

    def forward(self, x_hist: torch.Tensor, x_cal: torch.Tensor) -> torch.Tensor:
        # Prediksi relatif terhadap nilai terakhir -> lebih mudah dipelajari
        last = x_hist[:, -1:]
        x = torch.cat([x_hist - last, x_cal.flatten(1)], dim=1)
        raw = self.net(x).view(-1, self.horizon, self.n_q)

        median = raw[..., self.mid] + last
        up = F.softplus(raw[..., self.mid + 1:]).cumsum(-1)
        down = F.softplus(raw[..., : self.mid].flip(-1)).cumsum(-1).flip(-1)
        return torch.cat(
            [median[..., None] - down, median[..., None], median[..., None] + up], dim=-1
        )

    # Antarmuka yang dipakai loop training/federated (sama dengan model diffusion)
    def training_loss(self, x_hist, x_cal, y):
        return pinball_loss(self(x_hist, x_cal), y, self.q)

    def eval_loss(self, x_hist, x_cal, y):
        return self.training_loss(x_hist, x_cal, y)

    def predict_quantiles(self, x_hist, x_cal):
        return self(x_hist, x_cal)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """pred: (B, H, Q), y: (B, H), q: (Q,)"""
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()
