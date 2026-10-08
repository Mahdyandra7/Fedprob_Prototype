"""Model pembanding klasik yang juga menghasilkan kuantil.

Semua fungsi bekerja di skala ternormalisasi (sama seperti `ClientData`) dan
mengembalikan array (n_origin, H, Q).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .. import QUANTILES
from ..data.simulator import SimulationResult
from ..data.windows import ClientData, Split


def _seasonal_naive_point(split: Split, horizon: int, season: int = 7) -> np.ndarray:
    L = split.x_hist.shape[1]
    cols = []
    for h in range(1, horizon + 1):
        c = math.ceil(h / season)
        cols.append(split.x_hist[:, L - 1 + h - season * c])
    return np.stack(cols, axis=1)


def seasonal_naive_quantiles(
    data: ClientData, split: str = "test", season: int = 7, quantiles=QUANTILES
) -> np.ndarray:
    """Seasonal naive: nilai minggu lalu di hari yang sama.
    Interval diambil dari kuantil empiris residual di data train (per horizon)."""
    target = getattr(data, split)
    H = target.y.shape[1]
    point = _seasonal_naive_point(target, H, season)
    resid = data.train.y - _seasonal_naive_point(data.train, H, season)  # (n_train, H)
    rq = np.quantile(resid, quantiles, axis=0).T  # (H, Q)
    return point[:, :, None] + rq[None, :, :]


def ets_quantiles(
    sim: SimulationResult,
    data: ClientData,
    split: str = "test",
    fit_days: int = 365,
    quantiles=QUANTILES,
) -> np.ndarray:
    """ETS (Holt-Winters, tren teredam + musiman mingguan, error aditif).
    Dilatih ulang di setiap origin pada `fit_days` hari terakhir. Interval
    dari distribusi Gaussian analitik ETS."""
    from scipy.stats import norm
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    target = getattr(data, split)
    H = target.y.shape[1]
    z_all = (sim.y[:, data.bank] - data.center) / data.scale
    start = int(sim.start_idx[data.bank])
    zq = norm.ppf(quantiles)
    out = np.empty((len(target), H, len(quantiles)))
    for i, t in enumerate(target.origins):
        lo = max(start, t + 1 - fit_days)
        series = pd.Series(z_all[lo: t + 1], index=pd.DatetimeIndex(sim.dates[lo: t + 1], freq="D"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ETSModel(
                series, error="add", trend="add", damped_trend=True,
                seasonal="add", seasonal_periods=7,
            ).fit(disp=False)
            fc = res.get_prediction(start=len(series), end=len(series) + H - 1)
            mean = np.asarray(fc.predicted_mean)
            sd = np.sqrt(np.asarray(fc.var_pred_mean))
        out[i] = mean[:, None] + sd[:, None] * zq[None, :]
    return out
