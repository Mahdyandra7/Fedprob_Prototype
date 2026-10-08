"""Pipeline end-to-end studi kasus ATM (KasPintar).

Data NN5 (111 ATM) -> 6 bank fiktif -> forecast kuantil penarikan kumulatif 1..14 hari
-> training Local / federated / terpusat -> ACI satu sisi per ATM -> evaluasi.

Pemilihan model:
- Kandidat federated diambil dari temuan simulasi (FedAvg, FedProx, FedProx+fine-tune,
  Clustered FL), lalu **dipilih berdasarkan CRPS validasi pada data riil** (bukan asumsi).
  Hasilnya: FedAvg (di simulasi yang menang FedProx -- simulasi memberi hipotesis,
  data riil yang memutuskan).
- ACI satu sisi **per bank** (skor semua ATM satu bank digabung): satu-satunya cara yang
  menjaga service level saat Natal/Paskah. ACI per ATM gagal karena sampelnya terlalu sedikit.
- Diffusion tidak dipakai: keputusan isi kas cukup dijawab kuantil total kumulatif.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..conformal import aci_upper_pooled
from ..data.windows import ClientData, build_series_data, calendar_features, merge_clients
from ..federated import FedConfig, fine_tune, run_clustered, run_federated
from ..metrics import evaluate
from ..models.quantile_net import QuantileMLP
from ..realdata.nn5 import NN5Data, holiday_effect, load_nn5
from ..training import TrainConfig, predict, train_with_early_stopping
from .policy import CYCLE, backtest, rule_of_thumb_loads, weekly_origins

Q_ATM = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99)
SERVICE_LEVELS = (0.90, 0.95, 0.98, 0.99)
H_ATM = 14
FED_CANDIDATES = ("fedavg", "fedprox", "fedprox_ft", "clustered")
CACHE_VERSION = 2

METHOD_LABELS = {
    "aturan_praktis": "Aturan praktis (rata-rata 4 minggu x faktor)",
    "seasonal_naive": "Seasonal naive + kuantil residual",
    "local": "Local (tiap bank sendiri)",
    "fedavg": "Federated (FedAvg)",
    "fedprox": "Federated (FedProx)",
    "fedprox_ft": "FedProx + fine-tune per bank",
    "clustered": "Clustered FL",
    "central": "Centralized (data digabung)",
}


def label(m: str) -> str:
    if m.endswith("+aci"):
        return METHOD_LABELS.get(m[:-4], m[:-4]) + " + ACI"
    return METHOD_LABELS.get(m, m)


@dataclass
class ATMConfig:
    val_days: int = 84  # 12 minggu
    test_days: int = 168  # 24 minggu: Des 1997 - Mei 1998 (Natal, Tahun Baru, Paskah)
    cold_bank: int | None = 5  # bank "baru" (Zeta): histori ATM terbatas
    cold_history: int = 84  # hari histori sebelum validasi untuk bank baru
    scale_fraction: float = 0.25
    fedprox_mu: float = 0.1
    hidden: int = 128
    seed: int = 0


@dataclass
class ATMCaseResult:
    config: ATMConfig
    data: NN5Data
    atms: list[ClientData]  # per ATM (target kumulatif)
    val_start: int
    test_start: int
    preds: dict[str, list[np.ndarray]]  # metode -> per ATM (n_test, H, Q), satuan hari-permintaan
    preds_val: dict[str, list[np.ndarray]]
    aci: dict[str, dict[float, list[np.ndarray]]]  # metode -> level -> per ATM (n_test, H) kuantil atas
    metrics: pd.DataFrame
    histories: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    @property
    def test_origins(self) -> np.ndarray:
        return self.atms[0].test.origins

    @property
    def system(self) -> str:
        """Metode yang dipakai sistem KasPintar: model federated terpilih + ACI per bank."""
        return f"{self.extras.get('selected', 'fedavg')}+aci"

    def bank_name(self, b: int) -> str:
        return self.data.banks[b][0]

    def upper(self, method: str, atm: int, level: float) -> np.ndarray:
        """Kuantil atas (n_test, H) satuan hari-permintaan; '+aci' memakai versi terkalibrasi."""
        if method.endswith("+aci"):
            return self.aci[method[:-4]][level][atm]
        return self.preds[method][atm][..., Q_ATM.index(level)]

    def loads(self, method: str, atm: int, level: float = 0.95, factor: float = 1.2) -> np.ndarray:
        """Jumlah isi (unit kas) untuk tiap origin Minggu di periode uji."""
        wk = weekly_origins(self.data.dates, self.test_origins)
        if method == "aturan_praktis":
            return rule_of_thumb_loads(self.data.y[:, atm], self.test_origins[wk], factor)
        return self.upper(method, atm, level)[wk, CYCLE - 1] * self.atms[atm].center

    def demand(self, atm: int) -> tuple[np.ndarray, np.ndarray]:
        """(total 7 hari, harian (n, 7)) aktual untuk tiap origin Minggu."""
        o = self.test_origins[weekly_origins(self.data.dates, self.test_origins)]
        daily = np.stack([self.data.y[t + 1: t + 1 + CYCLE, atm] for t in o])
        return daily.sum(axis=1), daily

    def backtest(self, method: str, level: float = 0.95, factor: float = 1.2, atms=None) -> dict:
        atms = range(len(self.atms)) if atms is None else atms
        L, D, Dd = [], [], []
        for j in atms:
            L.append(self.loads(method, j, level, factor))
            d, dd = self.demand(j)
            D.append(d)
            Dd.append(dd)
        return backtest(np.concatenate(L), np.concatenate(D), np.concatenate(Dd))


# ---------------------------------------------------------------------------

def _seasonal_naive_cumsum(c: ClientData, y: np.ndarray, start: int, val_start: int, split: str) -> np.ndarray:
    """Kumulatif 'minggu lalu di hari yang sama' + kuantil residual dari data train."""
    def point(origins):
        out = np.empty((len(origins), H_ATM))
        for i, t in enumerate(origins):
            vals = [y[t + h - 7 * math.ceil(h / 7)] for h in range(1, H_ATM + 1)]
            out[i] = np.cumsum(vals) / c.center
        return out

    tr = c.train
    resid = tr.y - point(tr.origins)
    rq = np.quantile(resid, Q_ATM, axis=0).T  # (H, Q)
    s = getattr(c, split)
    return (point(s.origins)[:, :, None] + rq[None]).astype(np.float32)


def build_atm_clients(data: NN5Data, cfg: ATMConfig) -> tuple[list[ClientData], int, int]:
    n = len(data.dates)
    test_start = n - cfg.test_days
    val_start = test_start - cfg.val_days
    cal = calendar_features(data.dates, holiday_effect(data.dates))
    atms = []
    for j in range(data.y.shape[1]):
        b = int(data.bank_of[j])
        start = val_start - cfg.cold_history if b == cfg.cold_bank else 0
        atms.append(build_series_data(
            data.y[:, j], cal, start, val_start, test_start, data.names[j], b,
            target="cumsum", scale_fraction=cfg.scale_fraction, horizon=H_ATM,
        ))
    return atms, val_start, test_start


def run_atm_case(
    cfg: ATMConfig | None = None,
    methods=("seasonal_naive", "local", "fedavg", "fedprox", "fedprox_ft", "clustered", "central"),
    aci_methods=("local", "fedavg", "fedprox", "central"),
    train_cfg: TrainConfig | None = None,
    fed_cfg: FedConfig | None = None,
    verbose: bool = True,
) -> ATMCaseResult:
    cfg = cfg or ATMConfig()
    train_cfg = train_cfg or TrainConfig(seed=cfg.seed)
    fed_cfg = fed_cfg or FedConfig(seed=cfg.seed)
    log = print if verbose else (lambda *a: None)

    data = load_nn5()
    atms, val_start, test_start = build_atm_clients(data, cfg)
    n_banks = len(data.banks)
    banks = [merge_clients([atms[j] for j in data.atms_of(b)], data.banks[b][0], b) for b in range(n_banks)]

    def model_fn():
        return QuantileMLP(horizon=H_ATM, quantiles=Q_ATM, hidden=cfg.hidden, residual=False)

    preds, preds_val, histories, extras, models = {}, {}, {}, {}, {}

    def use(m, per_bank_models):
        models[m] = per_bank_models
        preds[m] = [predict(per_bank_models[a.bank], a.test) for a in atms]
        preds_val[m] = [predict(per_bank_models[a.bank], a.val) for a in atms]

    for m in methods:
        log(f"-> {label(m)}")
        if m == "seasonal_naive":
            preds[m] = [_seasonal_naive_cumsum(a, data.y[:, j], 0, val_start, "test") for j, a in enumerate(atms)]
            preds_val[m] = [_seasonal_naive_cumsum(a, data.y[:, j], 0, val_start, "val") for j, a in enumerate(atms)]
        elif m == "local":
            per = []
            for b in banks:
                r = train_with_early_stopping([b], train_cfg, model_fn=model_fn)
                histories[f"local_{b.name}"] = r.history
                per.append(r.model)
            use(m, per)
        elif m == "central":
            r = train_with_early_stopping(banks, train_cfg, model_fn=model_fn)
            histories[m] = r.history
            use(m, [r.model] * n_banks)
        elif m in ("fedavg", "fedprox"):
            mu = cfg.fedprox_mu if m == "fedprox" else 0.0
            r = run_federated(banks, FedConfig(**{**asdict(fed_cfg), "mu": mu}), model_fn=model_fn)
            histories[m] = r.history
            use(m, [r.model] * n_banks)
        elif m == "fedprox_ft":
            base = models["fedprox"][0]
            use(m, [fine_tune(base, b, seed=cfg.seed) for b in banks])
        elif m == "clustered":
            r = run_clustered(banks, fed_cfg, model_fn=model_fn)
            extras["cluster_labels"] = r.labels
            extras["cluster_silhouette"] = r.silhouette
            use(m, r.models)
        else:
            raise ValueError(m)

    log("-> ACI satu sisi per bank")
    aci = {m: {lv: [None] * len(atms) for lv in SERVICE_LEVELS} for m in aci_methods if m in preds}
    for m in aci:
        for lv in SERVICE_LEVELS:
            qi = Q_ATM.index(lv)
            for b in range(n_banks):
                idx = data.atms_of(b)
                adj = aci_upper_pooled(
                    np.stack([atms[j].val.y for j in idx]), np.stack([preds_val[m][j][..., qi] for j in idx]),
                    atms[idx[0]].val.origins,
                    np.stack([atms[j].test.y for j in idx]), np.stack([preds[m][j][..., qi] for j in idx]),
                    atms[idx[0]].test.origins, lv,
                )
                for s_, j in enumerate(idx):
                    aci[m][lv][j] = adj[s_]

    # Seleksi model federated berdasarkan CRPS validasi (horizon 7 hari)
    val_crps = {
        m: float(np.mean([evaluate(a.val.y[:, CYCLE - 1:CYCLE], p[:, CYCLE - 1:CYCLE], Q_ATM)["CRPS"]
                          for a, p in zip(atms, preds_val[m])]))
        for m in preds
    }
    extras["val_crps"] = val_crps
    cands = [m for m in FED_CANDIDATES if m in val_crps]
    if cands:
        extras["selected"] = min(cands, key=val_crps.get)
        log(f"   model terpilih (CRPS validasi): {extras['selected']}")

    rows = []
    for m in preds:
        for j, a in enumerate(atms):
            ev = evaluate(a.test.y[:, CYCLE - 1:CYCLE], preds[m][j][:, CYCLE - 1:CYCLE], Q_ATM)
            row = {"method": m, "atm": a.name, "bank": data.banks[a.bank][0], **ev}
            for lv in SERVICE_LEVELS:
                row[f"atas{int(lv * 100)}"] = float((a.test.y[:, CYCLE - 1] <= preds[m][j][:, CYCLE - 1, Q_ATM.index(lv)]).mean())
                if m in aci:
                    rows_aci = float((a.test.y[:, CYCLE - 1] <= aci[m][lv][j][:, CYCLE - 1]).mean())
                    row[f"atas{int(lv * 100)}_aci"] = rows_aci
            rows.append(row)
    metrics = pd.DataFrame(rows)
    return ATMCaseResult(cfg, data, atms, val_start, test_start, preds, preds_val, aci, metrics, histories, extras)


def run_atm_case_cached(cfg: ATMConfig | None = None, cache_dir: str | Path = "results", **kw) -> ATMCaseResult:
    cfg = cfg or ATMConfig()
    key = hashlib.md5(json.dumps([CACHE_VERSION, asdict(cfg), {k: str(v) for k, v in kw.items() if k != "verbose"}],
                                 sort_keys=True).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"atm_{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_atm_case(cfg, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res
