"""Mengubah series per bank menjadi pasangan (input, target) untuk supervised learning.

Untuk setiap titik asal (origin) t:
- input  : `lookback` nilai terakhir y(t-L+1..t) + fitur kalender hari-hari target
- target : y(t+1..t+H)

Normalisasi per bank *relatif terhadap level saldo*: z = (y - rata2) / (5% x rata2),
dengan rata2 dihitung dari data train bank tsb. Jadi 1 unit = 5% dari saldo rata-rata.
Skala ini sebanding antar bank walau panjang historinya berbeda (normalisasi mean/std
biasa gagal di sini: std dari histori pendek jauh lebih kecil sehingga bank dengan data
sedikit tampak "lebih bergejolak" dan model bersama tidak bisa dipakai lintas bank).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulator import SimulationResult, lebaran_effect, payday_indicator

LOOKBACK = 56
HORIZON = 14
SCALE_FRACTION = 0.05  # 1 unit ternormalisasi = 5% dari saldo rata-rata


def calendar_features(dates: pd.DatetimeIndex, event: np.ndarray | None = None) -> np.ndarray:
    """Fitur kalender yang diketahui di masa depan: hari dalam minggu (one-hot),
    gajian, efek hari besar, dan musim (sin/cos hari dalam tahun).

    `event` = efek hari besar per tanggal; default efek Lebaran (data simulasi).
    Untuk data riil bisa diganti, mis. efek hari libur Inggris pada data ATM NN5."""
    dow = np.eye(7)[dates.dayofweek.values]
    doy = 2 * np.pi * dates.dayofyear.values / 365.25
    event = lebaran_effect(dates) if event is None else event
    return np.column_stack(
        [dow, payday_indicator(dates), event, np.sin(doy), np.cos(doy)]
    ).astype(np.float32)


N_CAL = 11


@dataclass
class Split:
    x_hist: np.ndarray  # (n, L)
    x_cal: np.ndarray  # (n, H, N_CAL)
    y: np.ndarray  # (n, H)
    origins: np.ndarray  # (n,) indeks hari origin

    def __len__(self) -> int:
        return len(self.origins)


@dataclass
class ClientData:
    bank: int
    name: str
    center: float
    scale: float
    train: Split
    val: Split
    test: Split
    target: str = "level"  # "level" = nilai harian, "cumsum" = jumlah kumulatif 1..H hari

    def denorm(self, z: np.ndarray) -> np.ndarray:
        """Input/target level ternormalisasi -> skala asli."""
        return z * self.scale + self.center

    def to_amount(self, t: np.ndarray) -> np.ndarray:
        """Target/prediksi -> skala asli (untuk target kumulatif: dikali rata-rata harian)."""
        return t * self.center if self.target == "cumsum" else self.denorm(t)


def _make_split(z: np.ndarray, cal: np.ndarray, origins: np.ndarray, L: int, H: int,
                target: np.ndarray | None = None) -> Split:
    if len(origins) == 0:
        return Split(np.zeros((0, L), np.float32), np.zeros((0, H, cal.shape[1]), np.float32),
                     np.zeros((0, H), np.float32), origins)
    hist_idx = origins[:, None] + np.arange(-L + 1, 1)[None, :]
    tgt_idx = origins[:, None] + np.arange(1, H + 1)[None, :]
    y = z[tgt_idx] if target is None else target[origins]
    return Split(
        z[hist_idx].astype(np.float32),
        cal[tgt_idx].astype(np.float32),
        y.astype(np.float32),
        origins,
    )


def build_series_data(
    y: np.ndarray,
    cal: np.ndarray,
    start: int,
    val_start: int,
    test_start: int,
    name: str,
    bank: int = 0,
    target: str = "level",
    scale_fraction: float = SCALE_FRACTION,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    test_stride: int = 1,
) -> ClientData:
    """Dataset satu seri dengan split berbasis waktu (tanpa kebocoran):
    - train: target seluruhnya sebelum periode validasi
    - val  : origin & target di periode validasi
    - test : origin di periode uji (tiap `test_stride` hari), target di periode uji

    target="cumsum": target hari ke-h = jumlah y(t+1..t+h) dibagi rata-rata harian train
    (satuan: "hari permintaan rata-rata"). Dipakai untuk keputusan isi ulang kas.
    """
    L, H, n = lookback, horizon, len(y)
    center = float(np.mean(y[start:val_start]))
    scale = scale_fraction * abs(center) + 1e-8
    z = (y - center) / scale

    cum = None
    if target == "cumsum":
        cum = np.full((n, H), np.nan)
        cs = np.concatenate([[0.0], np.nancumsum(y)])
        for h in range(1, H + 1):
            cum[: n - h, h - 1] = (cs[h + 1: n + 1] - cs[1: n - h + 1]) / center
    elif target != "level":
        raise ValueError(f"target tidak dikenal: {target}")

    first = start + L - 1
    train_o = np.arange(first, val_start - H)
    val_o = np.arange(max(first, val_start - 1), test_start - H)
    test_o = np.arange(max(first, test_start - 1), n - H, test_stride)
    return ClientData(
        bank=bank, name=name, center=center, scale=scale,
        train=_make_split(z, cal, train_o, L, H, cum),
        val=_make_split(z, cal, val_o, L, H, cum),
        test=_make_split(z, cal, test_o, L, H, cum),
        target=target,
    )


def build_client_data(
    sim: SimulationResult,
    bank: int,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    test_stride: int = 1,
) -> ClientData:
    """Dataset satu bank dari data simulasi (lihat `build_series_data`)."""
    return build_series_data(
        sim.y[:, bank], calendar_features(sim.dates), int(sim.start_idx[bank]),
        sim.val_start, sim.test_start, sim.banks[bank].name, bank,
        lookback=lookback, horizon=horizon, test_stride=test_stride,
    )


def build_all_clients(sim: SimulationResult, **kw) -> list[ClientData]:
    return [build_client_data(sim, k, **kw) for k in range(sim.config.n_banks)]


def _concat(splits: list[Split]) -> Split:
    return Split(*(np.concatenate([getattr(s, f) for s in splits]) for f in ("x_hist", "x_cal", "y", "origins")))


def merge_clients(parts: list[ClientData], name: str, bank: int = 0) -> ClientData:
    """Gabungkan beberapa seri (mis. semua ATM milik satu bank) menjadi satu klien
    federated. Hanya untuk training; evaluasi tetap per seri."""
    return ClientData(
        bank=bank, name=name, center=float("nan"), scale=float("nan"),
        train=_concat([p.train for p in parts]),
        val=_concat([p.val for p in parts]),
        test=_concat([p.test for p in parts]),
        target=parts[0].target,
    )
