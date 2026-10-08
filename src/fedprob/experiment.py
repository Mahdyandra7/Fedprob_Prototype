"""Satu fungsi untuk menjalankan eksperimen end-to-end: simulasi -> training
berbagai metode -> evaluasi. Dipakai oleh notebook dan dashboard.

Nama metode:
- dasar        : oracle, seasonal_naive, ets
- QuantileMLP  : local, central, fedavg, fedprox, fedavg_ft, clustered
- diffusion    : diffusion_fed, diffusion_central
- conformal    : "<metode>+cqr" atau "<metode>+aci" (mis. "fedavg+aci"), diterapkan
                 di atas metode QuantileMLP/diffusion manapun
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import QUANTILES
from .conformal import aci, cqr
from .data import build_all_clients, get_scenario, oracle_quantiles, simulate
from .data.simulator import SimConfig, SimulationResult
from .data.windows import ClientData
from .federated import FedConfig, fine_tune, run_clustered, run_federated
from .metrics import evaluate
from .models import DiffusionForecaster, ets_quantiles, seasonal_naive_quantiles
from .training import TrainConfig, predict, train_with_early_stopping

METHOD_LABELS = {
    "oracle": "Oracle (distribusi sebenarnya)",
    "seasonal_naive": "Seasonal Naive",
    "ets": "ETS (Holt-Winters)",
    "local": "Local-only (tiap bank sendiri)",
    "fedavg": "Federated (FedAvg)",
    "fedprox": "Federated (FedProx)",
    "fedavg_ft": "FedAvg + fine-tune lokal",
    "clustered": "Clustered FL",
    "central": "Centralized (data digabung)",
    "diffusion_fed": "Diffusion (federated)",
    "diffusion_central": "Diffusion (centralized)",
}
CONFORMAL_LABELS = {"cqr": "CQR", "aci": "ACI adaptif"}

DEFAULT_METHODS = (
    "oracle", "seasonal_naive", "ets", "local", "fedavg", "fedprox", "fedavg_ft",
    "clustered", "central", "fedavg+cqr", "fedavg+aci",
)


def method_label(m: str) -> str:
    if "+" in m:
        base, conf = m.split("+", 1)
        return f"{METHOD_LABELS.get(base, base)} + {CONFORMAL_LABELS.get(conf, conf)}"
    return METHOD_LABELS.get(m, m)


@dataclass
class DiffusionConfig:
    """Pengaturan training model diffusion (lihat models/diffusion.py)."""

    rounds: int = 30
    local_epochs: int = 3
    epochs: int = 150
    patience: int = 25
    n_samples: int = 200
    eval_samples: int = 32
    seed: int = 0


@dataclass
class ExperimentResult:
    sim: SimulationResult
    clients: list[ClientData]
    preds: dict[str, list[np.ndarray]]  # metode -> per bank (n_test, H, Q), skala ternormalisasi
    metrics: pd.DataFrame  # baris: (metode, bank)
    histories: dict[str, list[dict]] = field(default_factory=dict)
    preds_val: dict[str, list[np.ndarray]] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)  # mis. label klaster, sampel jalur diffusion

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
    fedprox_mu: float = 0.1,
    diff_cfg: DiffusionConfig | None = None,
    verbose: bool = True,
    progress=None,
) -> ExperimentResult:
    """Jalankan semua `methods` pada satu skenario.

    `progress(msg)` opsional untuk melaporkan kemajuan (mis. ke dashboard).
    """
    cfg = get_scenario(config) if isinstance(config, str) else config
    train_cfg = train_cfg or TrainConfig()
    fed_cfg = fed_cfg or FedConfig()
    diff_cfg = diff_cfg or DiffusionConfig()

    def log(msg):
        if verbose:
            print(msg)
        if progress is not None:
            progress(msg)

    sim = simulate(cfg)
    clients = build_all_clients(sim)
    preds: dict[str, list[np.ndarray]] = {}
    preds_val: dict[str, list[np.ndarray]] = {}
    histories: dict[str, list[dict]] = {}
    extras: dict = {}
    models: dict[str, list] = {}  # metode -> model per bank (untuk dipakai ulang)

    def use_models(m, per_bank):
        models[m] = per_bank
        preds_val[m] = [predict(mod, c.val) for mod, c in zip(per_bank, clients)]
        preds[m] = [predict(mod, c.test) for mod, c in zip(per_bank, clients)]

    def diff_model():
        return DiffusionForecaster(n_samples=diff_cfg.n_samples, eval_samples=diff_cfg.eval_samples)

    def compute(m):
        if m in preds:
            return
        log(f"-> {method_label(m)}")
        if "+" in m:
            base, conf = m.split("+", 1)
            compute(base)
            out = []
            for c, pv, pt in zip(clients, preds_val[base], preds[base]):
                if conf == "cqr":
                    out.append(cqr(c.val.y, pv, pt))
                elif conf == "aci":
                    out.append(aci(c.val.y, pv, c.val.origins, c.test.y, pt, c.test.origins))
                else:
                    raise ValueError(f"Metode conformal tidak dikenal: {conf}")
            preds[m] = out
        elif m == "oracle":
            preds[m] = [_oracle_preds(sim, c) for c in clients]
        elif m == "seasonal_naive":
            preds[m] = [seasonal_naive_quantiles(c) for c in clients]
        elif m == "ets":
            preds[m] = [ets_quantiles(sim, c) for c in clients]
        elif m == "local":
            per = []
            for c in clients:
                res = train_with_early_stopping([c], train_cfg)
                histories[f"local_{c.name}"] = res.history
                per.append(res.model)
            use_models(m, per)
        elif m == "central":
            res = train_with_early_stopping(clients, train_cfg)
            histories[m] = res.history
            use_models(m, [res.model] * len(clients))
        elif m in ("fedavg", "fedprox"):
            mu = fedprox_mu if m == "fedprox" else 0.0
            res = run_federated(clients, FedConfig(**{**asdict(fed_cfg), "mu": mu}))
            histories[m] = res.history
            use_models(m, [res.model] * len(clients))
        elif m == "fedavg_ft":
            compute("fedavg")
            use_models(m, [fine_tune(models["fedavg"][0], c, seed=fed_cfg.seed) for c in clients])
        elif m == "clustered":
            res = run_clustered(clients, fed_cfg)
            histories.update({f"clustered_{k}": v for k, v in res.history.items()})
            extras["cluster_labels"] = res.labels
            extras["cluster_similarity"] = res.similarity
            use_models(m, res.models)
        elif m == "diffusion_fed":
            torch.manual_seed(diff_cfg.seed)
            fcfg = FedConfig(rounds=diff_cfg.rounds, local_epochs=diff_cfg.local_epochs, seed=diff_cfg.seed)
            res = run_federated(clients, fcfg, model_fn=diff_model)
            histories[m] = res.history
            use_models(m, [res.model] * len(clients))
        elif m == "diffusion_central":
            tcfg = TrainConfig(epochs=diff_cfg.epochs, patience=diff_cfg.patience, seed=diff_cfg.seed)
            res = train_with_early_stopping(clients, tcfg, model_fn=diff_model)
            histories[m] = res.history
            use_models(m, [res.model] * len(clients))
        else:
            raise ValueError(f"Metode tidak dikenal: {m}")

    for m in methods:
        compute(m)

    # Simpan sebagian sampel jalur diffusion (untuk analisis tingkat-jalur & dashboard)
    for m in [m for m in models if m.startswith("diffusion")]:
        extras[f"{m}_paths"] = [
            models[m][0].sample(torch.from_numpy(c.test.x_hist), torch.from_numpy(c.test.x_cal), n_samples=100, seed=1)
            .numpy().astype(np.float32)
            for c in clients
        ]

    rows = []
    for m in methods:
        for c, p in zip(clients, preds[m]):
            rows.append({"method": m, "bank": c.name, "size": sim.banks[c.bank].size, **evaluate(c.test.y, p)})
    metrics = pd.DataFrame(rows)
    return ExperimentResult(sim, clients, {m: preds[m] for m in methods}, metrics, histories,
                            {m: v for m, v in preds_val.items() if m in methods}, extras)


# ---------------------------------------------------------------------------
# Cache hasil ke disk (agar dashboard/demo cepat)
# ---------------------------------------------------------------------------

def _cache_key(*parts) -> str:
    payload = json.dumps([asdict(p) if hasattr(p, "__dataclass_fields__") else p for p in parts],
                         sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def run_experiment_cached(
    config: SimConfig | str = "normal",
    methods=DEFAULT_METHODS,
    train_cfg: TrainConfig | None = None,
    fed_cfg: FedConfig | None = None,
    fedprox_mu: float = 0.1,
    diff_cfg: DiffusionConfig | None = None,
    cache_dir: str | Path = "results",
    **kw,
) -> ExperimentResult:
    cfg = get_scenario(config) if isinstance(config, str) else config
    train_cfg = train_cfg or TrainConfig()
    fed_cfg = fed_cfg or FedConfig()
    diff_cfg = diff_cfg or DiffusionConfig()
    uses_diffusion = any(m.split("+")[0].startswith("diffusion") for m in methods)
    key = _cache_key(cfg, list(methods), train_cfg, fed_cfg, fedprox_mu, diff_cfg if uses_diffusion else None)
    path = Path(cache_dir) / f"exp_{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_experiment(cfg, methods, train_cfg, fed_cfg, fedprox_mu, diff_cfg, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res
