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


def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Fitur kalender yang diketahui di masa depan: hari dalam minggu (one-hot),
    gajian, efek Lebaran, dan musim (sin/cos hari dalam tahun)."""
    dow = np.eye(7)[dates.dayofweek.values]
    doy = 2 * np.pi * dates.dayofyear.values / 365.25
    return np.column_stack(
        [dow, payday_indicator(dates), lebaran_effect(dates), np.sin(doy), np.cos(doy)]
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

    def denorm(self, z: np.ndarray) -> np.ndarray:
        return z * self.scale + self.center


def _make_split(z: np.ndarray, cal: np.ndarray, origins: np.ndarray, L: int, H: int) -> Split:
    if len(origins) == 0:
        return Split(np.zeros((0, L), np.float32), np.zeros((0, H, N_CAL), np.float32),
                     np.zeros((0, H), np.float32), origins)
    hist_idx = origins[:, None] + np.arange(-L + 1, 1)[None, :]
    tgt_idx = origins[:, None] + np.arange(1, H + 1)[None, :]
    return Split(
        z[hist_idx].astype(np.float32),
        cal[tgt_idx].astype(np.float32),
        z[tgt_idx].astype(np.float32),
        origins,
    )


def build_client_data(
    sim: SimulationResult,
    bank: int,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    test_stride: int = 1,
) -> ClientData:
    """Dataset satu bank dengan split berbasis waktu (tanpa kebocoran):
    - train: target seluruhnya sebelum periode validasi
    - val  : origin & target di periode validasi
    - test : origin di periode uji (tiap `test_stride` hari), target di periode uji
    """
    L, H = lookback, horizon
    y = sim.y[:, bank]
    start = int(sim.start_idx[bank])
    vs, ts, n = sim.val_start, sim.test_start, sim.config.n_days

    train_vals = y[start:vs]
    center = float(np.mean(train_vals))
    scale = SCALE_FRACTION * abs(center) + 1e-8
    z = (y - center) / scale
    cal = calendar_features(sim.dates)

    first = start + L - 1
    train_o = np.arange(first, vs - H)
    val_o = np.arange(max(first, vs - 1), ts - H)
    test_o = np.arange(max(first, ts - 1), n - H, test_stride)

    return ClientData(
        bank=bank,
        name=sim.banks[bank].name,
        center=center,
        scale=scale,
        train=_make_split(z, cal, train_o, L, H),
        val=_make_split(z, cal, val_o, L, H),
        test=_make_split(z, cal, test_o, L, H),
    )


def build_all_clients(sim: SimulationResult, **kw) -> list[ClientData]:
    return [build_client_data(sim, k, **kw) for k in range(sim.config.n_banks)]
