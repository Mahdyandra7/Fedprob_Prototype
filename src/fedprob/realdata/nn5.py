"""Data riil NN5: penarikan tunai harian 111 ATM di Inggris (18 Mar 1996 - 22 Mar 1998).

Sumber: NN5 forecasting competition (2008), via Monash Time Series Forecasting Archive.
Zenodo: https://zenodo.org/records/4656110 -- lisensi CC BY 4.0.
Nilai sudah diskalakan oleh penyelenggara (satuan mata uang tidak disebutkan); di sini
disebut "unit kas".

Catatan data:
- Nilai hilang ("?") dan angka 0 (kemungkinan ATM tidak beroperasi -> permintaan tidak
  teramati) diperlakukan sebagai *tidak teramati* dan diimputasi dengan median hari yang
  sama pada 4 minggu sebelumnya. Posisi imputasi disimpan di `mask` agar bisa dilaporkan.
- 111 ATM dibagi deterministik ke 6 bank fiktif (NN5 tidak menyebut pemilik ATM).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://zenodo.org/records/4656110/files/nn5_daily_dataset_with_missing_values.zip?download=1"
MD5 = "2529489455b2d9f730fbb3c65d39df88"
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "nn5.zip"

# Hari libur bank di Inggris (England & Wales) 1996-1998
UK_HOLIDAYS = pd.to_datetime([
    "1996-01-01", "1996-04-05", "1996-04-08", "1996-05-06", "1996-05-27", "1996-08-26", "1996-12-25", "1996-12-26",
    "1997-01-01", "1997-03-28", "1997-03-31", "1997-05-05", "1997-05-26", "1997-08-25", "1997-12-25", "1997-12-26",
    "1998-01-01", "1998-04-10", "1998-04-13", "1998-05-04", "1998-05-25", "1998-08-31", "1998-12-25", "1998-12-28",
])

# 6 bank fiktif: (nama, jumlah ATM, ukuran)
BANKS = [
    ("Bank Alfa", 30, "besar"),
    ("Bank Beta", 30, "besar"),
    ("Bank Gamma", 17, "menengah"),
    ("Bank Delta", 17, "menengah"),
    ("Bank Epsilon", 9, "kecil"),
    ("Bank Zeta", 8, "kecil (baru)"),
]


@dataclass
class NN5Data:
    dates: pd.DatetimeIndex
    y: np.ndarray  # (n_days, 111) setelah imputasi
    raw: np.ndarray  # (n_days, 111) asli, NaN = hilang
    mask: np.ndarray  # True = nilai diimputasi (hilang atau 0)
    names: list[str]  # T1..T111
    bank_of: np.ndarray  # indeks bank per ATM
    banks: list[tuple]  # BANKS

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.y, index=self.dates, columns=self.names)

    def atms_of(self, b: int) -> np.ndarray:
        return np.flatnonzero(self.bank_of == b)


def download(path: Path = DEFAULT_PATH) -> Path:
    """Unduh zip NN5 dari Zenodo (sekali saja) dan verifikasi checksum MD5."""
    import hashlib

    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            path.write_bytes(r.read())
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != MD5:
        raise ValueError(f"Checksum NN5 tidak cocok ({md5}); hapus {path} lalu unduh ulang.")
    return path


def parse_tsf(text: str) -> tuple[list[str], pd.Timestamp, np.ndarray]:
    """Parser sederhana format .tsf (Monash) untuk seri berpanjang sama."""
    names, values, start = [], [], None
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data:
            continue
        name, ts, vals = line.split(":", 2)
        start = start or pd.Timestamp(ts.split(" ")[0])
        names.append(name)
        values.append([np.nan if v == "?" else float(v) for v in vals.split(",")])
    return names, start, np.array(values, dtype=float).T


def _impute(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = raw.copy()
    mask = np.isnan(y) | (y <= 0)
    y[mask] = np.nan
    n = len(y)
    for j in range(y.shape[1]):
        col = y[:, j]
        overall = np.nanmedian(col)
        for t in np.flatnonzero(mask[:, j]):
            prev = [col[t - 7 * k] for k in range(1, 5) if t - 7 * k >= 0 and not np.isnan(col[t - 7 * k])]
            if not prev:  # awal seri: pakai minggu-minggu berikutnya
                prev = [col[t + 7 * k] for k in range(1, 5) if t + 7 * k < n and not np.isnan(col[t + 7 * k])]
            col[t] = np.median(prev) if prev else overall
    return y, mask


def assign_banks(n_atm: int, seed: int = 2026) -> np.ndarray:
    """Pembagian ATM ke bank secara acak tetapi deterministik."""
    perm = np.random.default_rng(seed).permutation(n_atm)
    bank_of = np.empty(n_atm, dtype=int)
    i = 0
    for b, (_, size, _) in enumerate(BANKS):
        bank_of[perm[i:i + size]] = b
        i += size
    return bank_of


def load_nn5(path: Path = DEFAULT_PATH) -> NN5Data:
    path = download(path)
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    names, start, raw = parse_tsf(text)
    y, mask = _impute(raw)
    dates = pd.date_range(start, periods=len(raw), freq="D")
    return NN5Data(dates, y, raw, mask, names, assign_banks(len(names)), BANKS)


def holiday_effect(dates: pd.DatetimeIndex, holidays: pd.DatetimeIndex = UK_HOLIDAYS,
                   lead: int = 2, width: float = 3.0) -> np.ndarray:
    """Benjolan Gaussian menjelang hari libur (penarikan tunai biasanya naik sebelum libur)."""
    t = dates.values.astype("datetime64[D]").astype(np.int64)[:, None]
    c = (holidays - pd.Timedelta(days=lead)).values.astype("datetime64[D]").astype(np.int64)[None, :]
    return np.exp(-0.5 * ((t - c) / width) ** 2).sum(axis=1)
