"""Simulator saldo dana harian beberapa bank fiktif.

Setiap bank k pada hari t dimodelkan sebagai:

    y_k(t) = S_k * [ D_k(t) + beta_k * M(t) + I_k(t) + sigma_k(t) * eps_k(t) ]

- S_k      : skala/ukuran bank (bank besar vs kecil)
- D_k(t)   : komponen deterministik = level dasar + tren + pola mingguan
             + efek gajian + efek Lebaran + efek shock
- M(t)     : faktor pasar bersama, AR(1) -- dirasakan semua bank
- I_k(t)   : faktor idiosinkratik bank, AR(1)
- sigma_k  : volatilitas noise (bisa naik saat shock)

Parameter `alpha` (heterogenitas) mengatur seberapa jauh parameter tiap bank
menyimpang dari parameter "umum". alpha=0 -> semua bank punya proses yang sama
(hanya beda skala & realisasi acak); alpha=1 -> sangat berbeda (non-IID).

Karena prosesnya diketahui, kita bisa menghitung distribusi prediktif *sebenarnya*
(oracle) lewat simulasi Monte Carlo -- sesuatu yang tidak mungkin pada data riil.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Perkiraan tanggal Idul Fitri (Lebaran) -- kalender hari libur bersifat publik,
# jadi boleh dipakai sebagai fitur oleh model.
LEBARAN_DATES = pd.to_datetime(
    ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31", "2026-03-20", "2027-03-10"]
)

# Pola mingguan "umum" (Senin..Minggu): aktivitas turun di akhir pekan.
_COMMON_WEEKLY = np.array([0.5, 0.3, 0.2, 0.1, 0.6, -0.7, -1.0])


@dataclass
class SimConfig:
    """Konfigurasi simulasi. Lihat `scenarios.py` untuk preset."""

    n_banks: int = 8
    start_date: str = "2023-01-01"
    n_days: int = 1096  # ~3 tahun
    alpha: float = 0.3  # heterogenitas antar bank (0 = identik, 1 = sangat berbeda)
    seed: int = 42

    # Panjang periode validasi & uji (hari, di ujung series)
    val_days: int = 90
    test_days: int = 90

    # Histori terbatas: {indeks bank: jumlah hari histori sebelum periode validasi}
    limited_history: dict[int, int] = field(default_factory=dict)

    # Shock bersama (None = tanpa shock). Hari relatif terhadap awal periode uji.
    shock_day: int | None = None
    shock_depth: float = 12.0  # penurunan level (unit ternormalisasi, basis 100)
    shock_vol: float = 2.5  # pengali volatilitas saat puncak shock
    shock_tau: float = 25.0  # lama pemulihan (hari)

    # Parameter "umum" proses
    base_level: float = 100.0
    trend_per_year: float = 8.0
    weekly_amp: float = 2.0
    payday_amp: float = 3.0
    lebaran_amp: float = 6.0
    market_phi: float = 0.98
    market_sd: float = 0.4
    idio_phi: float = 0.90
    idio_sd: float = 0.5
    noise_sd: float = 1.0


@dataclass
class BankProfile:
    """Parameter spesifik satu bank (hasil penarikan acak dari SimConfig)."""

    name: str
    size: str
    scale: float
    trend: float
    weekly: np.ndarray
    payday_amp: float
    lebaran_amp: float
    beta: float
    idio_phi: float
    idio_sd: float
    noise_sd: float
    shock_sens: float


@dataclass
class SimulationResult:
    config: SimConfig
    dates: pd.DatetimeIndex
    y: np.ndarray  # (n_days, n_banks), NaN sebelum bank mulai punya data
    start_idx: np.ndarray  # indeks hari pertama data tiap bank
    banks: list[BankProfile]
    deterministic: np.ndarray  # D_k(t), ternormalisasi (tanpa skala)
    vol_mult: np.ndarray  # (n_days,) pengali volatilitas
    market: np.ndarray  # M(t)
    idio: np.ndarray  # I_k(t)

    @property
    def val_start(self) -> int:
        return self.config.n_days - self.config.val_days - self.config.test_days

    @property
    def test_start(self) -> int:
        return self.config.n_days - self.config.test_days

    @property
    def bank_names(self) -> list[str]:
        return [b.name for b in self.banks]

    def to_frame(self) -> pd.DataFrame:
        """Data wide: index tanggal, kolom nama bank."""
        return pd.DataFrame(self.y, index=self.dates, columns=self.bank_names)

    def split_of(self, t: int) -> str:
        if t >= self.test_start:
            return "test"
        if t >= self.val_start:
            return "val"
        return "train"


# ---------------------------------------------------------------------------
# Komponen kalender
# ---------------------------------------------------------------------------

def payday_indicator(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 pada sekitar tanggal gajian (tgl >= 25 atau <= 2)."""
    dom = dates.day.values
    return ((dom >= 25) | (dom <= 2)).astype(float)


def lebaran_effect(dates: pd.DatetimeIndex) -> np.ndarray:
    """Penarikan dana menjelang Lebaran: lembah Gaussian ~3 hari sebelum Lebaran.

    Nilai di [-1, 0]; dikalikan amplitudo per bank.
    """
    t = dates.values.astype("datetime64[D]").astype(np.int64)[:, None]
    centers = (LEBARAN_DATES - pd.Timedelta(days=3)).values.astype("datetime64[D]").astype(np.int64)[None, :]
    d = t - centers
    return -np.exp(-0.5 * (d / 6.0) ** 2).sum(axis=1)


def shock_profile(n_days: int, start: int | None, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Bentuk shock: (level di [-1,0], intensitas volatilitas di [0,1]).

    Level turun linear selama 5 hari lalu pulih eksponensial.
    """
    level = np.zeros(n_days)
    intensity = np.zeros(n_days)
    if start is None:
        return level, intensity
    t = np.arange(n_days) - start
    drop = np.clip((t + 1) / 5.0, 0, 1)
    recover = np.where(t >= 5, np.exp(-(t - 5) / tau), 1.0)
    active = t >= 0
    level[active] = -(drop * recover)[active]
    intensity[active] = np.exp(-t[active] / tau)
    return level, intensity


# ---------------------------------------------------------------------------
# Simulasi
# ---------------------------------------------------------------------------

_SIZES = ["besar", "besar", "menengah", "menengah", "menengah", "kecil", "kecil", "kecil"]
_SCALES = {"besar": 50.0, "menengah": 15.0, "kecil": 3.0}


def _draw_banks(cfg: SimConfig, rng: np.random.Generator) -> list[BankProfile]:
    a = cfg.alpha
    banks = []
    for k in range(cfg.n_banks):
        size = _SIZES[k % len(_SIZES)]
        z = rng.standard_normal(10)
        own_weekly = rng.standard_normal(7)
        own_weekly -= own_weekly.mean()
        banks.append(
            BankProfile(
                name=f"Bank {chr(ord('A') + k)}",
                size=size,
                scale=_SCALES[size] * float(np.exp(0.2 * rng.standard_normal())),
                trend=cfg.trend_per_year * (1 + 1.0 * a * z[0]),
                weekly=cfg.weekly_amp * ((1 - a) * _COMMON_WEEKLY + a * 1.5 * own_weekly),
                payday_amp=cfg.payday_amp * max(0.0, 1 + a * z[1]),
                lebaran_amp=cfg.lebaran_amp * max(0.0, 1 + a * z[2]),
                beta=1 + 0.6 * a * z[3],
                idio_phi=float(np.clip(cfg.idio_phi + 0.08 * a * z[4], 0.5, 0.99)),
                idio_sd=cfg.idio_sd * float(np.exp(0.4 * a * z[5])),
                noise_sd=cfg.noise_sd * float(np.exp(0.5 * a * z[6])),
                # bank kecil cenderung lebih rentan saat shock
                shock_sens=(1.5 if size == "kecil" else 1.0) * max(0.2, 1 + 0.5 * a * z[7]),
            )
        )
    return banks


def _ar1(n: int, phi: float, sd: float, rng: np.random.Generator, x0: float = 0.0) -> np.ndarray:
    x = np.empty(n)
    prev = x0
    for t in range(n):
        prev = phi * prev + sd * rng.standard_normal()
        x[t] = prev
    return x


def simulate(cfg: SimConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    banks = _draw_banks(cfg, rng)
    n, K = cfg.n_days, cfg.n_banks

    t_years = np.arange(n) / 365.0
    dow = dates.dayofweek.values
    payday = payday_indicator(dates)
    lebaran = lebaran_effect(dates)
    shock_abs = None if cfg.shock_day is None else n - cfg.test_days + cfg.shock_day
    shock_level, shock_int = shock_profile(n, shock_abs, cfg.shock_tau)
    vol_mult = 1.0 + (cfg.shock_vol - 1.0) * shock_int

    market_stat_sd = cfg.market_sd / np.sqrt(1 - cfg.market_phi**2)
    market = _ar1(n, cfg.market_phi, cfg.market_sd, rng, x0=market_stat_sd * rng.standard_normal())

    D = np.empty((n, K))
    idio = np.empty((n, K))
    y = np.empty((n, K))
    for k, b in enumerate(banks):
        D[:, k] = (
            cfg.base_level
            + b.trend * t_years
            + b.weekly[dow]
            + b.payday_amp * payday
            + b.lebaran_amp * lebaran
            + cfg.shock_depth * b.shock_sens * shock_level
        )
        idio_stat_sd = b.idio_sd / np.sqrt(1 - b.idio_phi**2)
        idio[:, k] = _ar1(n, b.idio_phi, b.idio_sd, rng, x0=idio_stat_sd * rng.standard_normal())
        eps = rng.standard_normal(n)
        y[:, k] = b.scale * (D[:, k] + b.beta * market + idio[:, k] + b.noise_sd * vol_mult * eps)

    start_idx = np.zeros(K, dtype=int)
    val_start = n - cfg.val_days - cfg.test_days
    for k, hist in cfg.limited_history.items():
        if k >= K:
            continue
        start_idx[k] = max(0, val_start - hist)
        y[: start_idx[k], k] = np.nan

    return SimulationResult(cfg, dates, y, start_idx, banks, D, vol_mult, market, idio)


def oracle_quantiles(
    sim: SimulationResult,
    bank: int,
    origin: int,
    horizon: int,
    quantiles,
    n_samples: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Kuantil distribusi prediktif *sebenarnya* untuk y(origin+1 .. origin+horizon).

    Mengetahui state M(origin) dan I_k(origin), kita simulasikan ulang masa depan
    dengan inovasi acak baru. Ini adalah batas terbaik yang mungkin dicapai model.
    Return: array (horizon, n_quantiles) dalam skala asli.
    """
    cfg = sim.config
    b = sim.banks[bank]
    rng = np.random.default_rng(seed)
    m = np.full(n_samples, sim.market[origin])
    i = np.full(n_samples, sim.idio[origin, bank])
    out = np.empty((horizon, len(quantiles)))
    for h in range(1, horizon + 1):
        t = origin + h
        m = cfg.market_phi * m + cfg.market_sd * rng.standard_normal(n_samples)
        i = b.idio_phi * i + b.idio_sd * rng.standard_normal(n_samples)
        eps = rng.standard_normal(n_samples)
        draws = b.scale * (sim.deterministic[t, bank] + b.beta * m + i + b.noise_sd * sim.vol_mult[t] * eps)
        out[h - 1] = np.quantile(draws, quantiles)
    return out
