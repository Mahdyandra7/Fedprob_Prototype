"""Conformal prediction: kalibrasi ulang interval agar coverage-nya "jujur".

Model kuantil bisa terlalu percaya diri (interval terlalu sempit). Conformal
prediction memperbaikinya *setelah* model dilatih, hanya dengan melihat seberapa
sering interval meleset di data kalibrasi:

- **CQR (Conformalized Quantile Regression, Romano dkk. 2019)** -- statik.
  Untuk interval nominal c dengan batas (lo, hi), skor kesesuaian
  s = max(lo - y, y - hi). Ambil kuantil (1-alpha) dari skor di data validasi
  sebagai Q, lalu interval baru = (lo - Q, hi + Q). Q < 0 berarti interval dipersempit.

- **ACI (Adaptive Conformal Inference, Gibbs & Candes 2021)** -- online.
  Saat periode uji berjalan, setiap kali sebuah forecast "terbukti" (tanggal
  targetnya sudah lewat), level alpha diperbarui:
      alpha <- alpha + gamma * (alpha_target - meleset)
  Jika interval sering meleset (mis. saat shock), alpha turun -> interval melebar.
  Skor kalibrasi diambil dari jendela `window` forecast terbaru.

Keduanya dihitung **lokal di tiap bank** (skor tidak pernah dikirim ke server),
sehingga tetap sejalan dengan prinsip federated learning.

Semua array dalam skala ternormalisasi; shape prediksi (n_origin, H, Q).
"""

from __future__ import annotations

import numpy as np

from . import BANDS, QUANTILES


def _band_indices(quantiles=QUANTILES):
    q = np.asarray(quantiles)
    out = []
    for lo, hi, _ in BANDS:
        i, j = int(np.argmin(abs(q - lo))), int(np.argmin(abs(q - hi)))
        out.append((i, j, hi - lo))
    return out


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Kuantil konservatif ((n+1)-koreksi) dari skor."""
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, max(0.0, np.ceil((n + 1) * level) / n))
    return float(np.quantile(scores, level, method="higher"))


def _finalize(q: np.ndarray) -> np.ndarray:
    # Jaga urutan kuantil tetap monoton setelah penyesuaian
    return np.sort(q, axis=-1).astype(np.float32)


def cqr(val_y, val_pred, test_pred, quantiles=QUANTILES, per_horizon: bool = True) -> np.ndarray:
    """CQR statik: kalibrasi pada data validasi, diterapkan ke prediksi uji."""
    out = test_pred.copy()
    H = test_pred.shape[1]
    for i, j, nominal in _band_indices(quantiles):
        s = np.maximum(val_pred[..., i] - val_y, val_y - val_pred[..., j])  # (n_val, H)
        for h in range(H):
            Q = _conformal_quantile(s[:, h] if per_horizon else s.ravel(), nominal)
            out[:, h, i] -= Q
            out[:, h, j] += Q
    return _finalize(out)


def online_bias_shift(y, pred, origins, window: int = 14, quantiles=QUANTILES) -> np.ndarray:
    """Koreksi bias online: geser semua kuantil sebesar median residual (y - median)
    dari `window` forecast terbaru yang sudah terbukti (o' + h <= o). Kausal, tanpa kebocoran.
    Return pergeseran (n, H) yang ditambahkan ke prediksi."""
    mid = int(np.argmin(abs(np.asarray(quantiles) - 0.5)))
    resid = y - pred[..., mid]  # (n, H)
    H = pred.shape[1]
    shift = np.zeros(resid.shape, dtype=np.float32)
    for h in range(H):
        resolve_day = origins + h + 1
        for k, t in enumerate(origins):
            known = resid[resolve_day <= t, h][-window:]
            if len(known):
                shift[k, h] = np.median(known)
    return shift


def aci(
    val_y,
    val_pred,
    val_origins,
    test_y,
    test_pred,
    test_origins,
    quantiles=QUANTILES,
    gamma: float = 0.01,
    window: int = 30,
    bias_window: int | None = None,
) -> np.ndarray:
    """ACI online. Forecast dari origin o pada horizon h baru "terbukti" pada hari o+h,
    jadi pada origin t hanya skor dengan o+h <= t yang boleh dipakai (tanpa kebocoran).

    Jika `bias_window` diisi, prediksi lebih dulu digeser dengan `online_bias_shift`
    (memperbaiki pusat prediksi), lalu ACI mengkalibrasi lebar intervalnya."""
    if bias_window:
        all_o = np.concatenate([val_origins, test_origins])
        all_y = np.concatenate([val_y, test_y])
        all_p = np.concatenate([val_pred, test_pred])
        shifted = all_p + online_bias_shift(all_y, all_p, all_o, bias_window, quantiles)[..., None]
        val_pred, test_pred = shifted[: len(val_origins)], shifted[len(val_origins):]
    out = test_pred.copy()
    H = test_pred.shape[1]
    origins = np.concatenate([val_origins, test_origins])
    ys = np.concatenate([val_y, test_y])
    preds = np.concatenate([val_pred, test_pred])
    n_val = len(val_origins)

    for i, j, nominal in _band_indices(quantiles):
        target = 1 - nominal
        scores = np.maximum(preds[..., i] - ys, ys - preds[..., j])  # (n_all, H)
        for h in range(H):
            hh = h + 1
            alpha = target
            Q_used = {}  # indeks test -> Q yang dipakai (untuk menilai meleset/tidak)
            resolved_ptr = 0
            resolve_day = origins + hh
            order = np.argsort(resolve_day, kind="stable")
            for k, t in enumerate(test_origins):
                # Perbarui alpha dengan semua forecast uji yang terbukti sampai hari t
                while resolved_ptr < len(order) and resolve_day[order[resolved_ptr]] <= t:
                    idx = order[resolved_ptr]
                    if idx >= n_val and (idx - n_val) in Q_used:
                        miss = float(scores[idx, h] > Q_used[idx - n_val])
                        alpha = alpha + gamma * (target - miss)
                    resolved_ptr += 1
                known = resolve_day <= t
                recent = scores[known, h][-window:]
                if alpha <= 0:
                    Q = float(np.max(recent)) * 1.5 if len(recent) else 0.0
                elif alpha >= 1:
                    Q = -np.inf
                else:
                    Q = _conformal_quantile(recent, 1 - alpha)
                Q = max(Q, -0.5 * float(test_pred[k, h, j] - test_pred[k, h, i]))  # interval tidak boleh terbalik
                Q_used[k] = Q
                out[k, h, i] -= Q
                out[k, h, j] += Q
    return _finalize(out)
