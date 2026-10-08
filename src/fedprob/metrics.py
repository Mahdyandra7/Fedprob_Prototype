"""Metrik evaluasi forecast probabilistik.

Konvensi bentuk array:
- y_true : (..., H)
- q_pred : (..., H, Q) dengan kuantil terurut sesuai `quantiles`

Untuk membandingkan bank besar & kecil secara adil, hitung metrik pada skala
ternormalisasi (dibagi std data train bank tsb).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import QUANTILES


def pinball(y_true: np.ndarray, q_pred: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    """Pinball (quantile) loss per kuantil, dirata-rata atas semua titik. Shape (Q,)."""
    q = np.asarray(quantiles)
    diff = y_true[..., None] - q_pred
    loss = np.maximum(q * diff, (q - 1) * diff)
    return loss.reshape(-1, len(q)).mean(axis=0)


def crps_approx(y_true, q_pred, quantiles=QUANTILES) -> float:
    """Aproksimasi CRPS = 2 x rata-rata pinball loss atas kuantil.
    Makin kecil makin baik; menilai akurasi DAN kalibrasi sekaligus."""
    return float(2 * pinball(y_true, q_pred, quantiles).mean())


def _q_index(quantiles, level: float) -> int:
    idx = np.where(np.isclose(np.asarray(quantiles), level))[0]
    if len(idx) == 0:
        raise ValueError(f"Kuantil {level} tidak tersedia")
    return int(idx[0])


def coverage(y_true, q_pred, nominal: float, quantiles=QUANTILES) -> float:
    """Proporsi nilai aktual yang jatuh di dalam interval `nominal` (mis. 0.8 -> q10..q90).
    Model yang terkalibrasi baik: coverage ~= nominal."""
    lo = _q_index(quantiles, round((1 - nominal) / 2, 4))
    hi = _q_index(quantiles, round(1 - (1 - nominal) / 2, 4))
    inside = (y_true >= q_pred[..., lo]) & (y_true <= q_pred[..., hi])
    return float(inside.mean())


def interval_width(q_pred, nominal: float, quantiles=QUANTILES) -> float:
    lo = _q_index(quantiles, round((1 - nominal) / 2, 4))
    hi = _q_index(quantiles, round(1 - (1 - nominal) / 2, 4))
    return float((q_pred[..., hi] - q_pred[..., lo]).mean())


def mae_median(y_true, q_pred, quantiles=QUANTILES) -> float:
    return float(np.abs(y_true - q_pred[..., _q_index(quantiles, 0.5)]).mean())


def evaluate(y_true, q_pred, quantiles=QUANTILES) -> dict[str, float]:
    """Kumpulan metrik standar dalam satu dict."""
    return {
        "MAE": mae_median(y_true, q_pred, quantiles),
        "CRPS": crps_approx(y_true, q_pred, quantiles),
        "Coverage50": coverage(y_true, q_pred, 0.5, quantiles),
        "Coverage80": coverage(y_true, q_pred, 0.8, quantiles),
        "Coverage90": coverage(y_true, q_pred, 0.9, quantiles),
        "Lebar80": interval_width(q_pred, 0.8, quantiles),
    }


def reliability(y_true, q_pred, quantiles=QUANTILES) -> pd.DataFrame:
    """Untuk tiap kuantil q: proporsi y_true <= prediksi kuantil q.
    Model terkalibrasi sempurna berada di garis diagonal (observed == q)."""
    obs = [(y_true <= q_pred[..., i]).mean() for i in range(len(quantiles))]
    return pd.DataFrame({"nominal": quantiles, "observed": obs})
