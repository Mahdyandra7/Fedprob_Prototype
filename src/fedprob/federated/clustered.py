"""Clustered Federated Learning: satu model global untuk semua bank tidak cocok jika
pola antar bank sangat berbeda (non-IID). Ide (mengikuti Sattler dkk., 2020):

1. Pemanasan: jalankan FedAvg beberapa ronde -> model global awal.
2. Tiap bank melatih model global itu sebentar dengan datanya sendiri dan mengirim
   *perubahan bobot* (update) -- informasi yang sama dengan FedAvg biasa, tanpa data.
3. Server mengelompokkan bank berdasarkan kemiripan arah update (cosine similarity).
   Bank yang datanya mirip akan "menarik" model ke arah yang mirip.
4. FedAvg dilanjutkan terpisah di tiap kelompok -> satu model per kelompok.

Jumlah kelompok dipilih otomatis dengan silhouette score; jika tidak ada struktur
kelompok yang jelas, semua bank tetap dalam satu kelompok (= FedAvg biasa).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..data.windows import ClientData
from ..training import make_loader, train_epochs
from .fedavg import FedConfig, run_federated


@dataclass
class ClusteredResult:
    models: list  # model per bank
    labels: np.ndarray  # kelompok tiap bank
    similarity: np.ndarray  # matriks cosine similarity antar update bank
    silhouette: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)


def _flat(state: dict) -> torch.Tensor:
    return torch.cat([v.flatten().float() for v in state.values()])


def _silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    ks = np.unique(labels)
    if len(ks) < 2 or len(ks) == len(labels):
        return -1.0
    s = []
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            s.append(0.0)
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == k].mean() for k in ks if k != labels[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def client_updates(clients: list[ClientData], global_model, epochs: int = 3, lr: float = 1e-3, seed: int = 0):
    """Update bobot tiap bank setelah melatih model global secara lokal."""
    g = _flat(global_model.state_dict())
    ups = []
    for i, c in enumerate(clients):
        local = copy.deepcopy(global_model)
        train_epochs(local, make_loader([c.train], seed=seed + i), epochs, lr)
        ups.append((_flat(local.state_dict()) - g).numpy())
    return np.stack(ups)


def cluster_clients(updates: np.ndarray, max_clusters: int = 3, min_silhouette: float = 0.3):
    """Kelompokkan bank dari arah update (hierarchical clustering, jarak = 1 - cosine).

    Ambang silhouette 0.3: pada update acak tanpa struktur, silhouette terbaik bisa
    mencapai ~0.22 (persentil 99, 8 bank) hanya karena kebetulan."""
    u = updates / (np.linalg.norm(updates, axis=1, keepdims=True) + 1e-12)
    sim = u @ u.T
    dist = np.clip(1 - sim, 0, 2)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    best_k, best_s, scores = 1, min_silhouette, {}
    for k in range(2, min(max_clusters, len(updates) - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        scores[k] = _silhouette(dist, lab)
        if scores[k] > best_s:
            best_k, best_s = k, scores[k]
    labels = np.zeros(len(updates), int) if best_k == 1 else fcluster(Z, best_k, criterion="maxclust") - 1
    return labels, sim, scores


def run_clustered(
    clients: list[ClientData],
    cfg: FedConfig = FedConfig(),
    warmup_rounds: int = 5,
    max_clusters: int = 3,
    model_fn=None,
) -> ClusteredResult:
    warm = run_federated(clients, FedConfig(**{**asdict(cfg), "rounds": warmup_rounds}), model_fn=model_fn)
    updates = client_updates(clients, warm.model, seed=cfg.seed)
    labels, sim, scores = cluster_clients(updates, max_clusters)

    models: list = [None] * len(clients)
    history = {"warmup": warm.history}
    rest = FedConfig(**{**asdict(cfg), "rounds": max(1, cfg.rounds - warmup_rounds)})
    for k in np.unique(labels):
        idx = np.flatnonzero(labels == k)
        res = run_federated([clients[i] for i in idx], rest, model_fn=model_fn, init_state=warm.model.state_dict())
        history[f"klaster_{k}"] = res.history
        for i in idx:
            models[i] = res.model
    return ClusteredResult(models, labels, sim, scores, history)
