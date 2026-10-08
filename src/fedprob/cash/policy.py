"""Kebijakan pengisian kas ATM dan backtest siklus mingguan.

Skenario operasional: setiap Senin pagi petugas (CIT) mengisi ATM dengan jumlah L,
yang harus cukup sampai Senin berikutnya (7 hari). Karena penarikan selalu >= 0:

    ATM kehabisan kas dalam siklus  <=>  total penarikan 7 hari > L

Jadi isi untuk *service level* s (peluang tidak kehabisan) adalah kuantil-s dari total
penarikan 7 hari. Model kuantil memprediksi kuantil itu secara langsung.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE = 7  # hari per siklus pengisian


def weekly_origins(dates: pd.DatetimeIndex, origins: np.ndarray, weekday: int = 6) -> np.ndarray:
    """Indeks (posisi dalam `origins`) dari origin hari Minggu: data sampai Minggu
    malam diketahui, pengisian dilakukan Senin pagi."""
    return np.flatnonzero(dates[origins].dayofweek == weekday)


def rule_of_thumb_loads(y: np.ndarray, origins: np.ndarray, factor: float = 1.2, weeks: int = 4) -> np.ndarray:
    """Praktik umum: isi = rata-rata total mingguan `weeks` minggu terakhir x `factor`."""
    out = np.empty(len(origins))
    for i, t in enumerate(origins):
        totals = [y[t - CYCLE * (k + 1) + 1: t - CYCLE * k + 1].sum() for k in range(weeks)]
        out[i] = factor * np.mean(totals)
    return out


def backtest(loads: np.ndarray, demand: np.ndarray, daily: np.ndarray | None = None) -> dict:
    """Evaluasi keputusan isi.

    loads, demand: (n,) isi dan total penarikan aktual per siklus.
    daily: (n, CYCLE) penarikan harian aktual (opsional, untuk hari kehabisan).
    """
    loads, demand = np.asarray(loads, float), np.asarray(demand, float)
    stockout = demand > loads
    idle = np.maximum(loads - demand, 0)
    res = {
        "kehabisan_%": 100 * stockout.mean(),
        "menganggur_%": 100 * idle.sum() / demand.sum(),  # uang menganggur relatif terhadap permintaan
        "menganggur_rata2": idle.mean(),
        "kurang_rata2": np.maximum(demand - loads, 0).mean(),  # permintaan tak terlayani
        "isi_rata2": loads.mean(),
        "n_siklus": len(loads),
    }
    if daily is not None and stockout.any():
        cum = np.cumsum(daily, axis=1)
        day = np.argmax(cum > loads[:, None], axis=1) + 1
        res["hari_habis_rata2"] = float(day[stockout].mean())
    return res


def total_cost(res: dict, cost_of_funds_year: float = 0.15, stockout_penalty: float = 30.0) -> float:
    """Biaya ilustratif per siklus (satuan unit kas):
    uang menganggur x biaya simpan mingguan (biaya dana, asuransi, keamanan) +
    peluang kehabisan x penalti (kunjungan darurat, transaksi hilang, reputasi).
    Asumsi default hanya ilustrasi dan bisa diubah di dashboard."""
    idle_cost = res["menganggur_rata2"] * cost_of_funds_year / 52
    stockout_cost = res["kehabisan_%"] / 100 * stockout_penalty
    return idle_cost + stockout_cost
