"""Utilitas training & prediksi untuk QuantileMLP (dipakai oleh mode lokal,
terpusat, maupun federated)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data.windows import ClientData, Split
from .models.quantile_net import QuantileMLP, pinball_loss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensors(splits: list[Split]):
    splits = [s for s in splits if len(s) > 0]
    return (
        torch.from_numpy(np.concatenate([s.x_hist for s in splits])),
        torch.from_numpy(np.concatenate([s.x_cal for s in splits])),
        torch.from_numpy(np.concatenate([s.y for s in splits])),
    )


def make_loader(splits: list[Split], batch_size: int = 64, shuffle: bool = True, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*_tensors(splits)), batch_size=batch_size, shuffle=shuffle, generator=g)


def train_epochs(
    model: QuantileMLP,
    loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    prox_ref: dict | None = None,
    mu: float = 0.0,
    opt: torch.optim.Optimizer | None = None,
) -> float:
    """Latih `model` beberapa epoch. Jika `prox_ref` diberikan (FedProx), tambahkan
    penalti mu/2 * ||w - w_global||^2 agar model lokal tidak menjauh dari global."""
    model.train()
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    ref = None
    if prox_ref is not None and mu > 0:
        ref = {k: v.detach() for k, v in prox_ref.items()}
    total, count = 0.0, 0
    for _ in range(epochs):
        for xh, xc, y in loader:
            opt.zero_grad()
            loss = pinball_loss(model(xh, xc), y, model.q)
            if ref is not None:
                prox = sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters())
                loss = loss + 0.5 * mu * prox
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
    return total / max(count, 1)


@torch.no_grad()
def predict(model: QuantileMLP, split: Split) -> np.ndarray:
    """Prediksi kuantil (n, H, Q) di skala ternormalisasi."""
    model.eval()
    if len(split) == 0:
        return np.zeros((0, model.horizon, model.n_q), np.float32)
    return model(torch.from_numpy(split.x_hist), torch.from_numpy(split.x_cal)).numpy()


@torch.no_grad()
def split_loss(model: QuantileMLP, split: Split) -> float:
    if len(split) == 0:
        return float("nan")
    model.eval()
    pred = model(torch.from_numpy(split.x_hist), torch.from_numpy(split.x_cal))
    return float(pinball_loss(pred, torch.from_numpy(split.y), model.q))


def weighted_val_loss(model: QuantileMLP, clients: list[ClientData]) -> float:
    losses = [(split_loss(model, c.val), len(c.val)) for c in clients if len(c.val) > 0]
    return float(sum(l * n for l, n in losses) / sum(n for _, n in losses))


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-3
    batch_size: int = 64
    patience: int = 10
    hidden: int = 128
    seed: int = 0


@dataclass
class TrainResult:
    model: QuantileMLP
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0


def train_with_early_stopping(clients: list[ClientData], cfg: TrainConfig = TrainConfig()) -> TrainResult:
    """Latih satu model pada data gabungan `clients` (1 klien = Local-only,
    semua klien = Centralized). Pilih epoch terbaik berdasarkan loss validasi."""
    set_seed(cfg.seed)
    model = QuantileMLP(hidden=cfg.hidden)
    loader = make_loader([c.train for c in clients], cfg.batch_size, seed=cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    for ep in range(1, cfg.epochs + 1):
        tr = train_epochs(model, loader, 1, opt=opt)
        va = weighted_val_loss(model, clients)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if va < best:
            best, best_state, best_epoch, wait = va, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return TrainResult(model, history, best_epoch)
