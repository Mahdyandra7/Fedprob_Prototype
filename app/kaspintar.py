"""KasPintar: Sistem Rekomendasi Pengisian Kas ATM (demo studi kasus data riil NN5).

Jalankan dari root project:
    streamlit run app/kaspintar.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedprob.cash.pipeline import Q_ATM, SERVICE_LEVELS, label, run_atm_case_cached
from fedprob.cash.policy import CYCLE, rule_of_thumb_loads, total_cost, weekly_origins
from fedprob.cash.report import calibrated_quantiles, tradeoff_curve, weekly_report, weekly_stockout

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2f6fdf", "#e07b00"

st.set_page_config(page_title="KasPintar", layout="wide")


@st.cache_resource(show_spinner="Memuat model & hasil (pertama kali ~15 menit untuk training)...")
def load():
    return run_atm_case_cached(cache_dir=ROOT / "results", verbose=False)


res = load()
SYS = res.system
wk = weekly_origins(res.data.dates, res.test_origins)
week_labels = [f"{res.data.dates[res.test_origins[k] + 1].date()}" for k in wk]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("KasPintar")
    st.caption("Rekomendasi pengisian kas ATM berbasis *federated probabilistic forecasting*")
    bank = st.selectbox("Bank", range(len(res.data.banks)),
                        format_func=lambda b: f"{res.bank_name(b)} ({len(res.data.atms_of(b))} ATM, {res.data.banks[b][2]})")
    week = st.select_slider("Minggu pengisian (Senin)", options=list(range(len(wk))), value=3,
                            format_func=lambda i: week_labels[i])
    level = st.select_slider("Target service level (peluang TIDAK kehabisan)", SERVICE_LEVELS, value=0.95,
                             format_func=lambda v: f"{v:.0%}")
    factor = st.slider("Faktor aturan praktis (pembanding)", 1.0, 1.8, 1.2, 0.05)
    with st.expander("Asumsi biaya (ilustrasi)"):
        cof = st.slider("Biaya simpan uang menganggur (% per tahun)", 5, 40, 15) / 100
        pen = st.slider("Penalti per kejadian kehabisan (unit kas)", 5, 150, 30)
    st.caption(f"Model: {label(SYS)} · dipilih dari CRPS validasi")

st.title("KasPintar: Rekomendasi Pengisian Kas ATM")
st.caption(
    "Berapa uang yang harus diisi ke tiap ATM minggu ini agar peluang kehabisan sesuai target, "
    "dengan uang menganggur seminimal mungkin? Data riil: 111 ATM (NN5), 6 bank yang berkolaborasi tanpa berbagi data."
)

tab_rep, tab_perf, tab_fed, tab_about = st.tabs(
    ["Laporan Senin", "Kinerja (backtest 23 minggu)", "Kolaborasi federated", "Tentang sistem"]
)

# ---------------------------------------------------------------- laporan senin
with tab_rep:
    rep = weekly_report(res, bank, week, level, SYS, factor)
    risky = rep["P(habis) jika aturan praktis"] > (1 - level) * 2
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total isi rekomendasi", f"{rep['isi rekomendasi'].sum():,.0f}")
    c2.metric("Total isi aturan praktis", f"{rep['isi aturan praktis'].sum():,.0f}",
              delta=f"{rep['isi aturan praktis'].sum() / rep['isi rekomendasi'].sum() - 1:+.0%} vs rekomendasi",
              delta_color="off")
    c3.metric("ATM berisiko bila pakai aturan praktis", f"{int(risky.sum())} / {len(rep)}")
    c4.metric("Kehabisan aktual (rekomendasi vs aturan)",
              f"{int(rep['habis (rekomendasi)'].sum())} vs {int(rep['habis (aturan)'].sum())}")
    st.markdown(f"**Minggu {rep.attrs['minggu']}**: diurutkan dari ATM paling berisiko bila memakai aturan praktis.")
    show = rep.copy()
    st.dataframe(
        show.style.format({c: "{:,.1f}" for c in ["perkiraan median", "isi rekomendasi", "isi aturan praktis", "aktual 7 hari"]})
        .format({"P(habis) jika aturan praktis": "{:.0%}"})
        .background_gradient(subset=["P(habis) jika aturan praktis"], cmap="Reds", vmin=0, vmax=0.5),
        hide_index=True, width="stretch",
    )
    st.caption("Kolom *aktual* & *habis* hanya tersedia karena ini backtest. Pada operasi nyata, operator melihat rekomendasi dan peluang risikonya.")

    atm_names = rep["ATM"].tolist()
    pick = st.selectbox("Detail ATM", atm_names)
    j = res.data.names.index(pick)
    k = wk[week]
    t = res.test_origins[k]
    a = res.atms[j]
    q = calibrated_quantiles(res, SYS, j)[k] * a.center
    days = res.data.dates[t + 1: t + 1 + q.shape[0]]
    fig = go.Figure()
    for lo, hi, al in ((0.05, 0.95, 0.15), (0.10, 0.90, 0.2), (0.25, 0.75, 0.3)):
        fig.add_trace(go.Scatter(x=days, y=q[:, Q_ATM.index(hi)], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=days, y=q[:, Q_ATM.index(lo)], fill="tonexty", line=dict(width=0),
                                 fillcolor=f"rgba(47,111,223,{al})", name=f"rentang {int((hi - lo) * 100)}%"))
    fig.add_trace(go.Scatter(x=days, y=q[:, Q_ATM.index(0.5)], line=dict(color=BLUE), name="median kumulatif"))
    rec = res.upper(SYS, j, level)[k, CYCLE - 1] * a.center
    rule = rule_of_thumb_loads(res.data.y[:, j], np.array([t]), factor)[0]
    fig.add_hline(y=rec, line=dict(color=BLUE, dash="dash"), annotation_text=f"isi rekomendasi {rec:,.0f}")
    fig.add_hline(y=rule, line=dict(color=ORANGE, dash="dot"), annotation_text=f"aturan praktis {rule:,.0f}",
                  annotation_position="bottom right")
    fig.add_vline(x=days[CYCLE - 1], line=dict(color="grey", width=1))
    fig.add_trace(go.Scatter(x=days, y=np.cumsum(res.data.y[t + 1: t + 1 + len(days), j]), mode="lines+markers",
                             line=dict(color="#222"), name="aktual (backtest)"))
    fig.update_layout(height=420, yaxis_title="penarikan kumulatif (unit kas)", margin=dict(l=10, r=10, t=30, b=10),
                      title=f"{pick}: total penarikan 14 hari ke depan")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- kinerja
with tab_perf:
    pol = ["aturan_praktis", "local", "fedavg", SYS]
    rows = {label(m): res.backtest(m, level=level, factor=factor) for m in pol}
    c = st.columns(len(pol))
    for col, m in zip(c, pol):
        r = rows[label(m)]
        col.metric(label(m), f"{r['kehabisan_%']:.1f}% kehabisan", f"{r['menganggur_%']:.1f}% uang menganggur",
                   delta_color="off")
    st.markdown(f"Target service level **{level:.0%}** → kehabisan maksimal **{100 * (1 - level):.0f}%**. "
                f"Aturan praktis memakai faktor **x{factor:.2f}**.")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for m, color in [("aturan_praktis", ORANGE), ("local", "#999"), ("fedavg", "#7fa7ef"), (SYS, BLUE)]:
        cv = tradeoff_curve(res, m)
        fig.add_trace(go.Scatter(x=cv["kehabisan_%"], y=cv["menganggur_%"], mode="lines+markers", name=label(m),
                                 line=dict(color=color), text=[f"{p}" for p in cv["param"]]))
    fig.add_vline(x=100 * (1 - level), line=dict(color="grey", dash="dot"))
    fig.update_layout(title="Trade-off (kiri-bawah lebih baik)", xaxis_title="ATM kehabisan (% siklus)",
                      yaxis_title="uang menganggur (% permintaan)", height=420, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig, width="stretch")
    ws = pd.DataFrame({label(m): weekly_stockout(res, m, level, factor) for m in ["aturan_praktis", SYS]})
    fig2 = go.Figure()
    for col_name, color in zip(ws.columns, [ORANGE, BLUE]):
        fig2.add_trace(go.Scatter(x=ws.index, y=ws[col_name], mode="lines+markers", name=col_name, line=dict(color=color)))
    for h, n in [("1997-12-25", "Natal"), ("1998-04-10", "Paskah")]:
        fig2.add_vline(x=pd.Timestamp(h), line=dict(color="red", dash="dot"))
    fig2.add_hline(y=100 * (1 - level), line=dict(color="grey", dash="dash"))
    fig2.update_layout(title="% ATM kehabisan per minggu (garis merah: Natal & Paskah)", height=420,
                       margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fig2, width="stretch")

    st.subheader("Ilustrasi biaya per ATM per minggu")
    costs = pd.DataFrame({"kebijakan": list(rows),
                          "biaya": [total_cost(r, cof, pen) for r in rows.values()]}).set_index("kebijakan")
    st.bar_chart(costs, height=280)
    st.caption("Biaya = uang menganggur × biaya simpan mingguan + peluang kehabisan × penalti. Asumsi dapat diubah di sidebar.")

# ---------------------------------------------------------------- federated
with tab_fed:
    st.markdown(
        "Setiap bank melatih model **di servernya sendiri**; yang dikirim ke koordinator hanya **bobot model**. "
        "Koordinator merata-ratakan bobot (FedAvg) lalu mengirim model gabungan kembali. **Data transaksi ATM tidak pernah keluar dari bank.**"
    )
    m = res.metrics[res.metrics.method.isin(["local", "fedavg", "central"])]
    piv = m.groupby(["bank", "method"]).CRPS.mean().unstack()[["local", "fedavg", "central"]].rename(columns=label)
    c1, c2 = st.columns(2)
    c1.markdown("**Akurasi forecast per bank (CRPS, lebih kecil lebih baik)**")
    c1.bar_chart(piv, stack=False, height=330)
    rows = []
    for b in range(len(res.data.banks)):
        for mm in ["local+aci", SYS]:
            r = res.backtest(mm, level=level, atms=res.data.atms_of(b))
            rows.append({"bank": res.bank_name(b), "kebijakan": label(mm), "kehabisan %": r["kehabisan_%"]})
    c2.markdown(f"**ATM kehabisan per bank (target {100 * (1 - level):.0f}%)**")
    c2.bar_chart(pd.DataFrame(rows).pivot(index="bank", columns="kebijakan", values="kehabisan %"), stack=False, height=330)
    st.info("**Bank Zeta** adalah bank baru (histori ATM 12 minggu). Tanpa kolaborasi, modelnya lemah; dengan federated "
            "learning, bank ini memperoleh manfaat terbesar tanpa perlu membuka data siapa pun.")
    vc = pd.Series(res.extras["val_crps"]).rename(index=label).sort_values()
    st.markdown("**Seleksi model (CRPS validasi)**")
    st.dataframe(vc.to_frame("CRPS validasi").style.format("{:.3f}"), width="stretch")

# ---------------------------------------------------------------- tentang
with tab_about:
    st.markdown(f"""
### Masalah
Bank mengisi ATM tiap Senin. Isi terlalu sedikit → ATM kosong (nasabah kecewa, kunjungan darurat, reputasi turun,
terutama menjelang hari besar seperti **Lebaran**). Isi terlalu banyak → uang menganggur. Praktik umum
"rata-rata × 1,2" tidak mengukur risiko. Data transaksi tiap bank rahasia, dan bank/ATM baru belum punya histori.

### Solusi: tiga lapis
1. **Forecast probabilistik** total penarikan 1–14 hari (QuantileMLP, 9 kuantil). Isi untuk service level s =
   kuantil-s total 7 hari.
2. **Federated learning** antar 6 bank: model dipilih dari CRPS validasi → **{label(res.extras['selected'])}**.
3. **Conformal prediction adaptif (ACI) per bank**: menjaga janji service level saat pola berubah (Natal, Paskah).

### Data
NN5 forecasting competition: penarikan tunai harian 111 ATM di Inggris, Mar 1996 – Mei 1998.
Sumber Zenodo (Monash Forecasting Archive), lisensi CC BY 4.0. Pembagian ATM ke 6 bank bersifat fiktif.

### Keterbatasan
Data Inggris 1990-an (bukan Indonesia); kolaborasi federated disimulasikan; satuan nilai tidak diketahui;
siklus pengisian tetap mingguan; backtest bukan uji coba operasional.
""")
