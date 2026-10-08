"""Federated learning (FedAvg / FedProx) yang disimulasikan di satu komputer.

Alur per ronde:
  1. Server mengirim bobot model global ke setiap bank (klien).
  2. Tiap bank melatih model itu beberapa epoch HANYA dengan datanya sendiri.
  3. Bank mengirim balik bobot (bukan data!) ke server.
  4. Server merata-ratakan bobot, dibobot jumlah sampel tiap bank -> model global baru.

Data mentah tidak pernah keluar dari bank. FedProx menambahkan penalti agar model
lokal tidak terlalu menjauh dari model global (membantu saat data non-IID).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from ..data.windows import ClientData
from ..models.quantile_net import QuantileMLP
from ..training import (
    make_loader,
    set_seed,
    split_loss,
    train_epochs,
    weighted_val_loss,
)


@dataclass
class FedConfig:
    rounds: int = 30
    local_epochs: int = 2
    lr: float = 1e-3
    batch_size: int = 64
    mu: float = 0.0  # 0 = FedAvg, > 0 = FedProx
    hidden: int = 128
    seed: int = 0


@dataclass
class FedResult:
    model: QuantileMLP
    history: list[dict] = field(default_factory=list)
    best_round: int = 0


def average_weights(states: list[dict], weights: list[float]) -> dict:
    total = sum(weights)
    return {
        k: sum(s[k].float() * (w / total) for s, w in zip(states, weights)).to(states[0][k].dtype)
        for k in states[0]
    }


def run_federated(clients: list[ClientData], cfg: FedConfig = FedConfig(), callback=None, model_fn=None, init_state: dict | None = None) -> FedResult:
    """Jalankan FedAvg/FedProx. Model global terbaik dipilih dari loss validasi
    (rata-rata tertimbang dari loss validasi yang dilaporkan tiap bank).

    `model_fn` opsional untuk arsitektur lain; `init_state` untuk melanjutkan dari bobot tertentu."""
    set_seed(cfg.seed)
    global_model = model_fn() if model_fn is not None else QuantileMLP(hidden=cfg.hidden)
    if init_state is not None:
        global_model.load_state_dict(init_state)
    participants = [c for c in clients if len(c.train) > 0]
    loaders = [make_loader([c.train], cfg.batch_size, seed=cfg.seed + i) for i, c in enumerate(participants)]

    best, best_state, best_round = float("inf"), None, 0
    history = []
    for r in range(1, cfg.rounds + 1):
        states, weights, losses = [], [], []
        for c, loader in zip(participants, loaders):
            local = copy.deepcopy(global_model)
            prox_ref = dict(global_model.named_parameters()) if cfg.mu > 0 else None
            losses.append(train_epochs(local, loader, cfg.local_epochs, cfg.lr, prox_ref=prox_ref, mu=cfg.mu))
            states.append(local.state_dict())
            weights.append(len(c.train))
        global_model.load_state_dict(average_weights(states, weights))

        val = weighted_val_loss(global_model, clients)
        rec = {
            "round": r,
            "train_loss": sum(l * w for l, w in zip(losses, weights)) / sum(weights),
            "val_loss": val,
        }
        rec.update({f"val_{c.name}": split_loss(global_model, c.val) for c in clients})
        history.append(rec)
        if callback is not None:
            callback(rec)
        if val < best:
            best, best_state, best_round = val, copy.deepcopy(global_model.state_dict()), r

    global_model.load_state_dict(best_state)
    return FedResult(global_model, history, best_round)


def fine_tune(model: QuantileMLP, client: ClientData, epochs: int = 5, lr: float = 5e-4, seed: int = 0) -> QuantileMLP:
    """Personalisasi: model global disesuaikan sedikit dengan data bank sendiri.
    Epoch terbaik dipilih dari validasi lokal (termasuk opsi tidak fine-tune)."""
    tuned = copy.deepcopy(model)
    if len(client.train) == 0:
        return tuned
    loader = make_loader([client.train], seed=seed)
    opt = torch.optim.Adam(tuned.parameters(), lr=lr)
    best, best_state = split_loss(tuned, client.val), copy.deepcopy(tuned.state_dict())
    for _ in range(epochs):
        train_epochs(tuned, loader, 1, opt=opt)
        v = split_loss(tuned, client.val)
        if v < best:
            best, best_state = v, copy.deepcopy(tuned.state_dict())
    tuned.load_state_dict(best_state)
    return tuned
