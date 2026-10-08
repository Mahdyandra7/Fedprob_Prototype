"""Fungsi visualisasi: matplotlib untuk notebook, plotly untuk dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import BANDS, QUANTILES
from .data.simulator import SimulationResult

MAIN = "#2f6fdf"
ACTUAL = "#222222"


def _qi(level: float) -> int:
    return int(np.argmin(np.abs(np.asarray(QUANTILES) - level)))


def stitch_forecasts(q_pred: np.ndarray, origins: np.ndarray, stride: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Gabungkan forecast dari banyak origin menjadi satu garis waktu kontinu:
    diambil origin setiap `stride` hari, dan dari tiap origin dipakai `stride` hari
    pertama forecast-nya. Return (indeks hari, (T, Q))."""
    keep = (origins - origins[0]) % stride == 0
    idx, vals = [], []
    for o, q in zip(origins[keep], q_pred[keep]):
        n = min(stride, len(q))
        idx.extend(range(o + 1, o + 1 + n))
        vals.append(q[:n])
    return np.array(idx), np.concatenate(vals)


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------

def plot_series(sim: SimulationResult, banks=None, normalize: bool = False, ax=None, title=None):
    """Plot series beberapa bank dengan penanda periode validasi & uji."""
    banks = range(sim.config.n_banks) if banks is None else banks
    ax = ax or plt.subplots(figsize=(12, 4))[1]
    for k in banks:
        y = sim.y[:, k]
        if normalize:
            y = (y - np.nanmean(y[: sim.val_start])) / np.nanstd(y[: sim.val_start])
        ax.plot(sim.dates, y, lw=0.8, label=f"{sim.banks[k].name} ({sim.banks[k].size})")
    ax.axvspan(sim.dates[sim.val_start], sim.dates[sim.test_start - 1], color="orange", alpha=0.08, label="validasi")
    ax.axvspan(sim.dates[sim.test_start], sim.dates[-1], color="red", alpha=0.08, label="uji")
    ax.set_title(title or "Data simulasi")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    return ax


def plot_fan(
    sim: SimulationResult,
    bank: int,
    q_pred: np.ndarray,
    origins: np.ndarray,
    stride: int = 7,
    history_days: int = 60,
    ax=None,
    title: str | None = None,
):
    """Fan chart: pita kuantil 50/80/90% + median, dibandingkan nilai aktual.
    `q_pred` di skala asli, shape (n_origin, H, Q)."""
    ax = ax or plt.subplots(figsize=(12, 4))[1]
    idx, q = stitch_forecasts(q_pred, origins, stride)
    d = sim.dates[idx]
    for lo, hi, lab in BANDS:
        ax.fill_between(d, q[:, _qi(lo)], q[:, _qi(hi)], color=MAIN, alpha=0.15, lw=0, label=f"interval {lab}")
    ax.plot(d, q[:, _qi(0.5)], color=MAIN, lw=1.5, label="median")
    h0 = max(0, idx[0] - history_days)
    ax.plot(sim.dates[h0: idx[-1] + 1], sim.y[h0: idx[-1] + 1, bank], color=ACTUAL, lw=0.9, label="aktual")
    ax.axvline(sim.dates[idx[0]], color="grey", ls=":", lw=1)
    ax.set_title(title or f"{sim.banks[bank].name}: forecast probabilistik")
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    return ax


def plot_history(histories: dict[str, list[dict]], key: str = "val_loss", ax=None, title=None):
    """Kurva loss per ronde/epoch untuk beberapa metode."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    for name, h in histories.items():
        df = pd.DataFrame(h)
        x = df["round"] if "round" in df else df["epoch"]
        ax.plot(x, df[key], label=name)
    ax.set_xlabel("ronde / epoch")
    ax.set_ylabel(key)
    ax.set_title(title or "Konvergensi")
    ax.legend(fontsize=8)
    return ax


def plot_reliability(rel: dict[str, pd.DataFrame], ax=None, title=None):
    """Reliability diagram: garis diagonal = kalibrasi sempurna."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="ideal")
    for name, df in rel.items():
        ax.plot(df["nominal"], df["observed"], marker="o", label=name)
    ax.set_xlabel("kuantil nominal")
    ax.set_ylabel("proporsi aktual <= prediksi")
    ax.set_title(title or "Kalibrasi")
    ax.legend(fontsize=8)
    return ax


def plot_metric_bars(metrics: pd.DataFrame, metric: str = "CRPS", methods=None, ax=None, title=None):
    """Bar chart metrik per bank untuk tiap metode."""
    df = metrics if methods is None else metrics[metrics["method"].isin(methods)]
    pivot = df.pivot(index="bank", columns="method", values=metric)
    if methods is not None:
        pivot = pivot[[m for m in methods if m in pivot.columns]]
    ax = pivot.plot.bar(ax=ax, figsize=(12, 4), width=0.8)
    ax.set_title(title or f"{metric} per bank (lebih kecil lebih baik)")
    ax.set_xlabel("")
    ax.legend(fontsize=8, ncol=4)
    return ax


# ---------------------------------------------------------------------------
# plotly (dashboard)
# ---------------------------------------------------------------------------

def plotly_fan(sim: SimulationResult, bank: int, q_pred: np.ndarray, origins: np.ndarray,
               stride: int = 7, history_days: int = 90, title: str | None = None):
    import plotly.graph_objects as go

    idx, q = stitch_forecasts(q_pred, origins, stride)
    d = sim.dates[idx]
    fig = go.Figure()
    h0 = max(0, idx[0] - history_days)
    for lo, hi, lab in BANDS:
        fig.add_trace(go.Scatter(x=d, y=q[:, _qi(hi)], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d, y=q[:, _qi(lo)], fill="tonexty", fillcolor="rgba(47,111,223,0.15)",
                                 line=dict(width=0), name=f"interval {lab}"))
    fig.add_trace(go.Scatter(x=d, y=q[:, _qi(0.5)], line=dict(color=MAIN, width=2), name="median"))
    fig.add_trace(go.Scatter(x=sim.dates[h0: idx[-1] + 1], y=sim.y[h0: idx[-1] + 1, bank],
                             line=dict(color=ACTUAL, width=1), name="aktual"))
    fig.update_layout(title=title or f"{sim.banks[bank].name}: forecast probabilistik",
                      height=380, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    return fig


def plotly_series(sim: SimulationResult, normalize: bool = False):
    import plotly.graph_objects as go

    fig = go.Figure()
    for k, b in enumerate(sim.banks):
        y = sim.y[:, k]
        if normalize:
            y = (y - np.nanmean(y[: sim.val_start])) / np.nanstd(y[: sim.val_start])
        fig.add_trace(go.Scatter(x=sim.dates, y=y, name=f"{b.name} ({b.size})", line=dict(width=1)))
    fig.add_vrect(x0=sim.dates[sim.val_start], x1=sim.dates[sim.test_start - 1], fillcolor="orange",
                  opacity=0.08, line_width=0, annotation_text="validasi")
    fig.add_vrect(x0=sim.dates[sim.test_start], x1=sim.dates[-1], fillcolor="red", opacity=0.08,
                  line_width=0, annotation_text="uji")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plotly_history(histories: dict[str, list[dict]], key: str = "val_loss"):
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, h in histories.items():
        df = pd.DataFrame(h)
        x = df["round"] if "round" in df else df["epoch"]
        fig.add_trace(go.Scatter(x=x, y=df[key], name=name, mode="lines+markers"))
    fig.update_layout(xaxis_title="ronde", yaxis_title=key, height=350, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def _path_stress(paths: np.ndarray, worst_frac: float = 0.05) -> np.ndarray:
    """Indeks jalur 'skenario stres': `worst_frac` jalur dengan titik terendah paling rendah."""
    n_worst = max(1, int(round(worst_frac * len(paths))))
    return np.argsort(paths.min(axis=1))[:n_worst]


def plot_paths(sim: SimulationResult, bank: int, origin: int, paths: np.ndarray, history_days: int = 42,
               ax=None, title: str | None = None, worst_frac: float = 0.05):
    """Sampel jalur masa depan (skala asli) dari satu origin + sorotan skenario stres."""
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    H = paths.shape[1]
    hist = slice(origin - history_days + 1, origin + 1)
    fut = sim.dates[origin + 1: origin + 1 + H]
    for p in paths:
        ax.plot(fut, p, color=MAIN, alpha=0.08, lw=0.8)
    for i in _path_stress(paths, worst_frac):
        ax.plot(fut, paths[i], color="#d62728", alpha=0.7, lw=1)
    ax.plot(sim.dates[hist], sim.y[hist, bank], color=ACTUAL, lw=1.2, label="histori")
    ax.plot(fut, sim.y[origin + 1: origin + 1 + H, bank], color=ACTUAL, lw=1.5, ls="--", label="aktual")
    ax.plot([], [], color=MAIN, label="sampel jalur")
    ax.plot([], [], color="#d62728", label=f"{worst_frac:.0%} skenario terburuk")
    ax.set_title(title or f"{sim.banks[bank].name}: skenario masa depan")
    ax.legend(fontsize=8, loc="upper left")
    return ax


def plotly_paths(sim: SimulationResult, bank: int, origin: int, paths: np.ndarray, history_days: int = 42,
                 worst_frac: float = 0.05, title: str | None = None):
    import plotly.graph_objects as go

    H = paths.shape[1]
    fut = sim.dates[origin + 1: origin + 1 + H]
    hist = slice(origin - history_days + 1, origin + 1)
    fig = go.Figure()
    for k, p in enumerate(paths):
        fig.add_trace(go.Scatter(x=fut, y=p, mode="lines", line=dict(color="rgba(47,111,223,0.12)", width=1),
                                 hoverinfo="skip", showlegend=k == 0, name="sampel jalur"))
    for k, i in enumerate(_path_stress(paths, worst_frac)):
        fig.add_trace(go.Scatter(x=fut, y=paths[i], mode="lines", line=dict(color="rgba(214,39,40,0.7)", width=1),
                                 showlegend=k == 0, name=f"{worst_frac:.0%} skenario terburuk"))
    fig.add_trace(go.Scatter(x=sim.dates[hist], y=sim.y[hist, bank], line=dict(color=ACTUAL, width=2), name="histori"))
    fig.add_trace(go.Scatter(x=fut, y=sim.y[origin + 1: origin + 1 + H, bank],
                             line=dict(color=ACTUAL, width=2, dash="dash"), name="aktual"))
    fig.update_layout(title=title or f"{sim.banks[bank].name}: skenario masa depan", height=420,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig
