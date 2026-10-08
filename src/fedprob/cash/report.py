"""Laporan mingguan untuk operator kas ("Laporan Senin") dan plot studi kasus ATM."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import Q_ATM, SERVICE_LEVELS, ATMCaseResult, label
from .policy import CYCLE, rule_of_thumb_loads, weekly_origins

MAIN, RULE, ACTUAL = "#2f6fdf", "#e07b00", "#222222"


def calibrated_quantiles(res: ATMCaseResult, method: str, atm: int) -> np.ndarray:
    """Kuantil (n_test, H, Q): kuantil model, dengan level 90/95/98/99% diganti versi ACI
    bila metode berakhiran '+aci'. Diurutkan ulang agar tetap monoton."""
    base = method[:-4] if method.endswith("+aci") else method
    q = res.preds[base][atm].copy()
    if method.endswith("+aci"):
        for lv in SERVICE_LEVELS:
            q[..., Q_ATM.index(lv)] = res.aci[base][lv][atm]
    return np.sort(q, axis=-1)


def prob_exceed(q: np.ndarray, x: float) -> float:
    """P(permintaan > x) dari vektor kuantil (interpolasi linear, ekor dipotong di 1%/99%)."""
    return float(1 - np.interp(x, q, Q_ATM, left=0.0, right=1.0))


def weekly_report(res: ATMCaseResult, bank: int, week: int, level: float = 0.95,
                  method: str | None = None, factor: float = 1.2) -> pd.DataFrame:
    """Rekomendasi isi untuk semua ATM satu bank pada minggu ke-`week` periode uji.
    Kolom aktual hanya untuk backtest (di operasi nyata belum diketahui)."""
    method = method or res.system
    wk = weekly_origins(res.data.dates, res.test_origins)
    k = wk[week]
    t = res.test_origins[k]
    rows = []
    for j in res.data.atms_of(bank):
        a = res.atms[j]
        q7 = calibrated_quantiles(res, method, j)[k, CYCLE - 1] * a.center
        rec = res.upper(method, j, level)[k, CYCLE - 1] * a.center
        rule = rule_of_thumb_loads(res.data.y[:, j], np.array([t]), factor)[0]
        actual = res.data.y[t + 1: t + 1 + CYCLE, j].sum()
        rows.append({
            "ATM": a.name,
            "perkiraan median": q7[Q_ATM.index(0.5)],
            "isi rekomendasi": rec,
            "isi aturan praktis": rule,
            "P(habis) jika aturan praktis": prob_exceed(q7, rule),
            "aktual 7 hari": actual,
            "habis (rekomendasi)": actual > rec,
            "habis (aturan)": actual > rule,
        })
    df = pd.DataFrame(rows).sort_values("P(habis) jika aturan praktis", ascending=False)
    df.attrs["minggu"] = f"{res.data.dates[t + 1].date()} s/d {res.data.dates[t + CYCLE].date()}"
    return df


def tradeoff_curve(res: ATMCaseResult, method: str, atms=None) -> pd.DataFrame:
    """Titik (kehabisan %, menganggur %) untuk beberapa service level / faktor."""
    rows = []
    if method == "aturan_praktis":
        for f in np.round(np.arange(1.0, 1.85, 0.05), 2):
            rows.append({"param": f, **res.backtest(method, factor=f, atms=atms)})
    else:
        for lv in SERVICE_LEVELS:
            rows.append({"param": lv, **res.backtest(method, level=lv, atms=atms)})
    return pd.DataFrame(rows)


def idle_at_same_stockout(res: ATMCaseResult, method: str, level: float = 0.95) -> dict:
    """Bandingkan dengan aturan praktis pada tingkat kehabisan yang SAMA (interpolasi kurva
    aturan praktis). Return persen uang menganggur keduanya dan penghematannya."""
    sys = res.backtest(method, level=level)
    rule = tradeoff_curve(res, "aturan_praktis").sort_values("kehabisan_%")
    rule_idle = float(np.interp(sys["kehabisan_%"], rule["kehabisan_%"], rule["menganggur_%"]))
    rule_factor = float(np.interp(sys["kehabisan_%"], rule["kehabisan_%"], rule["param"]))
    return {
        "kehabisan_%": sys["kehabisan_%"],
        "menganggur_sistem_%": sys["menganggur_%"],
        "menganggur_aturan_%": rule_idle,
        "faktor_aturan_setara": rule_factor,
        "penghematan_relatif_%": 100 * (1 - sys["menganggur_%"] / rule_idle),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_cum_fan(res: ATMCaseResult, atm: int, week: int, method: str | None = None,
                 level: float = 0.95, ax=None, factor: float = 1.2):
    """Kumulatif penarikan 14 hari ke depan: pita kuantil, isi rekomendasi, aturan praktis, aktual."""
    method = method or res.system
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    wk = weekly_origins(res.data.dates, res.test_origins)
    k = wk[week]
    t = res.test_origins[k]
    a = res.atms[atm]
    q = calibrated_quantiles(res, method, atm)[k] * a.center  # (H, Q)
    days = res.data.dates[t + 1: t + 1 + q.shape[0]]
    for lo, hi, al in ((0.05, 0.95, 0.15), (0.10, 0.90, 0.2), (0.25, 0.75, 0.3)):
        ax.fill_between(days, q[:, Q_ATM.index(lo)], q[:, Q_ATM.index(hi)], color=MAIN, alpha=al, lw=0)
    ax.plot(days, q[:, Q_ATM.index(0.5)], color=MAIN, label="median kumulatif")
    rec = res.upper(method, atm, level)[k, CYCLE - 1] * a.center
    rule = rule_of_thumb_loads(res.data.y[:, atm], np.array([t]), factor)[0]
    ax.axhline(rec, color=MAIN, ls="--", lw=1.5, label=f"isi rekomendasi ({level:.0%})")
    ax.axhline(rule, color=RULE, ls=":", lw=1.5, label=f"isi aturan praktis (x{factor})")
    ax.axvline(days[CYCLE - 1], color="grey", lw=0.8)
    ax.plot(days, np.cumsum(res.data.y[t + 1: t + 1 + q.shape[0], atm]), color=ACTUAL, marker="o", ms=3, label="aktual")
    ax.set_title(f"{a.name} ({res.bank_name(a.bank)}), minggu {days[0].date()}")
    ax.set_ylabel("penarikan kumulatif (unit kas)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=8, loc="upper left")
    return ax


def plot_tradeoff(res: ATMCaseResult, methods, ax=None, atms=None):
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    for m in methods:
        c = tradeoff_curve(res, m, atms)
        ax.plot(c["kehabisan_%"], c["menganggur_%"], marker="o", label=label(m))
        if m != "aturan_praktis":
            for _, r in c.iterrows():
                ax.annotate(f"{r.param:.0%}", (r["kehabisan_%"], r["menganggur_%"]), fontsize=7,
                            xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("ATM kehabisan kas (% siklus)")
    ax.set_ylabel("uang menganggur (% dari permintaan)")
    ax.set_title("Trade-off: makin ke kiri-bawah makin baik")
    ax.legend(fontsize=8)
    return ax


def weekly_stockout(res: ATMCaseResult, method: str, level: float = 0.95, factor: float = 1.2) -> pd.Series:
    """Persentase ATM kehabisan per minggu periode uji."""
    wk = weekly_origins(res.data.dates, res.test_origins)
    out = np.zeros(len(wk))
    for j in range(len(res.atms)):
        out += res.demand(j)[0] > res.loads(method, j, level, factor)
    idx = res.data.dates[res.test_origins[wk] + 1]
    return pd.Series(100 * out / len(res.atms), index=idx)
