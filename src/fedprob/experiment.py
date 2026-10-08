"""Satu fungsi untuk menjalankan eksperimen end-to-end: simulasi -> training
berbagai metode -> evaluasi. Dipakai oleh notebook 04/05 dan dashboard."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import QUANTILES
from .data import build_all_clients, get_scenario, oracle_quantiles, simulate
from .data.simulator import SimConfig, SimulationResult
from .data.windows import ClientData
from .federated import FedConfig, fine_tune, run_federated
from .metrics import evaluate
from .models import ets_quantiles, seasonal_naive_quantiles
from .training import TrainConfig, predict, train_with_early_stopping

METHOD_LABELS = {
    "oracle": "Oracle (distribusi sebenarnya)",
    "seasonal_naive": "Seasonal Naive",
    "ets": "ETS (Holt-Winters)",
    "local": "Local-only (tiap bank sendiri)",
    "fedavg": "Federated (FedAvg)",
    "fedprox": "Federated (FedProx)",
    "fedavg_ft": "FedAvg + fine-tune lokal",
    "central": "Centralized (data digabung)",
}

DEFAULT_METHODS = ("oracle", "seasonal_naive", "ets", "local", "fedavg", "fedprox", "fedavg_ft", "central")


@dataclass
class ExperimentResult:
    sim: SimulationResult
    clients: list[ClientData]
    preds: dict[str, list[np.ndarray]]  # metode -> per bank (n_test, H, Q), skala ternormalisasi
    metrics: pd.DataFrame  # baris: (metode, bank)
    histories: dict[str, list[dict]] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Rata-rata metrik per metode (semua bank)."""
        return self.metrics.groupby("method", sort=False).mean(numeric_only=True)

    def pred_original_scale(self, method: str, bank: int) -> np.ndarray:
        return self.clients[bank].denorm(self.preds[method][bank])


def _oracle_preds(sim: SimulationResult, c: ClientData) -> np.ndarray:
    H = c.test.y.shape[1]
    out = np.stack([oracle_quantiles(sim, c.bank, int(t), H, QUANTILES, seed=int(t)) for t in c.test.origins])
    return ((out - c.center) / c.scale).astype(np.float32)


def run_experiment(
    config: SimConfig | str = "normal",
    methods=DEFAULT_METHODS,
    train_cfg: TrainConfig | None = None,
    fed_cfg: FedConfig | None = None,
    fedprox_mu: float = 0.01,
    verbose: bool = True,
    progress=None,
) -> ExperimentResult:
    """Jalankan semua `methods` pada satu skenario.

    `progress(msg)` opsional untuk melaporkan kemajuan (mis. ke dashboard).
    """
    cfg = get_scenario(config) if isinstance(config, str) else config
    train_cfg = train_cfg or TrainConfig()
    fed_cfg = fed_cfg or FedConfig()

    def log(msg):
        if verbose:
            print(msg)
        if progress is not None:
            progress(msg)

    sim = simulate(cfg)
    clients = build_all_clients(sim)
    preds: dict[str, list[np.ndarray]] = {}
    histories: dict[str, list[dict]] = {}
    fedavg_model = None

    for m in methods:
        log(f"-> {METHOD_LABELS.get(m, m)}")
        if m == "oracle":
            preds[m] = [_oracle_preds(sim, c) for c in clients]
        elif m == "seasonal_naive":
            preds[m] = [seasonal_naive_quantiles(c) for c in clients]
        elif m == "ets":
            preds[m] = [ets_quantiles(sim, c) for c in clients]
        elif m == "local":
            out = []
            for c in clients:
                res = train_with_early_stopping([c], train_cfg)
                histories[f"local_{c.name}"] = res.history
                out.append(predict(res.model, c.test))
            preds[m] = out
        elif m == "central":
            res = train_with_early_stopping(clients, train_cfg)
            histories["central"] = res.history
            preds[m] = [predict(res.model, c.test) for c in clients]
        elif m in ("fedavg", "fedprox", "fedavg_ft"):
            name = "fedprox" if m == "fedprox" else "fedavg"
            if name == "fedavg" and fedavg_model is not None:
                model = fedavg_model  # fedavg_ft memakai ulang model FedAvg yang sudah dilatih
            else:
                mu = fedprox_mu if name == "fedprox" else 0.0
                res = run_federated(clients, FedConfig(**{**asdict(fed_cfg), "mu": mu}))
                model = res.model
                histories[name] = res.history
                if name == "fedavg":
                    fedavg_model = model
            if m == "fedavg_ft":
                preds[m] = [predict(fine_tune(model, c, seed=fed_cfg.seed), c.test) for c in clients]
            else:
                preds[m] = [predict(model, c.test) for c in clients]
        else:
            raise ValueError(f"Metode tidak dikenal: {m}")

    rows = []
    for m, plist in preds.items():
        for c, p in zip(clients, plist):
            rows.append({"method": m, "bank": c.name, "size": sim.banks[c.bank].size, **evaluate(c.test.y, p)})
    metrics = pd.DataFrame(rows)
    return ExperimentResult(sim, clients, preds, metrics, histories)


# ---------------------------------------------------------------------------
# Cache hasil ke disk (agar dashboard/demo cepat)
# ---------------------------------------------------------------------------

def _cache_key(cfg: SimConfig, methods, train_cfg, fed_cfg, fedprox_mu) -> str:
    payload = json.dumps(
        [asdict(cfg), list(methods), asdict(train_cfg), asdict(fed_cfg), fedprox_mu],
        sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def run_experiment_cached(
    config: SimConfig | str = "normal",
    methods=DEFAULT_METHODS,
    train_cfg: TrainConfig | None = None,
    fed_cfg: FedConfig | None = None,
    fedprox_mu: float = 0.01,
    cache_dir: str | Path = "results",
    **kw,
) -> ExperimentResult:
    cfg = get_scenario(config) if isinstance(config, str) else config
    train_cfg = train_cfg or TrainConfig()
    fed_cfg = fed_cfg or FedConfig()
    path = Path(cache_dir) / f"exp_{_cache_key(cfg, methods, train_cfg, fed_cfg, fedprox_mu)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_experiment(cfg, methods, train_cfg, fed_cfg, fedprox_mu, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res
