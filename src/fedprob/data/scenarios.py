"""Preset skenario simulasi.

Setiap skenario adalah kumpulan parameter `SimConfig` yang menggambarkan satu
"cerita" untuk diuji: kapan federated learning menolong, kapan tidak.
"""

from __future__ import annotations

from dataclasses import replace

from .simulator import SimConfig

SCENARIOS: dict[str, dict] = {
    "normal": dict(
        alpha=0.3,
    ),
    "heterogen": dict(
        alpha=0.9,
    ),
    "data_langka": dict(
        alpha=0.3,
        # tiga bank kecil (F, G, H) hanya punya ~4 bulan histori sebelum validasi
        limited_history={5: 120, 6: 120, 7: 120},
    ),
    "shock": dict(
        alpha=0.3,
        shock_day=10,  # shock terjadi 10 hari setelah periode uji dimulai
    ),
    "bank_baru": dict(
        alpha=0.3,
        # Bank H baru bergabung ~8 bulan sebelum periode validasi
        limited_history={7: 240},
    ),
}

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "normal": "Kondisi dasar: 8 bank dengan pola cukup mirip (alpha=0.3).",
    "heterogen": "Pola tiap bank sangat berbeda (alpha=0.9) -- data non-IID.",
    "data_langka": "Bank kecil F, G, H hanya punya ~4 bulan histori.",
    "shock": "Guncangan bersama (penarikan dana besar + volatilitas naik) di periode uji.",
    "bank_baru": "Bank H baru bergabung dengan histori ~8 bulan.",
}


def get_scenario(name: str, **overrides) -> SimConfig:
    """Ambil SimConfig untuk skenario `name`, dengan parameter yang bisa ditimpa."""
    if name not in SCENARIOS:
        raise KeyError(f"Skenario '{name}' tidak dikenal. Pilihan: {list(SCENARIOS)}")
    return replace(SimConfig(), **{**SCENARIOS[name], **overrides})
