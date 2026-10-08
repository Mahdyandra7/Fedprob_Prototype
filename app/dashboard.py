"""Dashboard demo: Federated Probabilistic Forecasting (data simulasi).

Jalankan dari root project:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedprob.data import SCENARIO_DESCRIPTIONS, SCENARIOS, get_scenario, oracle_paths
from fedprob.experiment import METHOD_LABELS, run_experiment_cached, method_label
from fedprob.federated import FedConfig
from fedprob.metrics import independent_paths, path_min_metrics, reliability
from fedprob.plots import plotly_fan, plotly_history, plotly_paths, plotly_series
from fedprob.training import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results"

ALL_METHODS = list(METHOD_LABELS) + ["fedavg+cqr", "fedavg+aci", "fedavg_ft+aci", "local+aci", "diffusion_fed+aci"]
DEFAULT = ["oracle", "ets", "local", "fedavg", "fedavg_ft", "clustered", "central", "fedavg+aci"]

st.set_page_config(page_title="FedProb Demo", layout="wide")
st.title("Federated Probabilistic Forecasting")
st.caption(
    "Beberapa bank tidak bisa berbagi data. Apakah mereka tetap bisa membangun model "
    "forecast yang lebih akurat **dan** jujur soal ketidakpastiannya, secara bersama-sama?"
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Pengaturan")
    scenario = st.selectbox("Skenario", list(SCENARIOS), format_func=lambda s: s.replace("_", " "))
    st.caption(SCENARIO_DESCRIPTIONS[scenario])
    default_alpha = float(SCENARIOS[scenario].get("alpha", 0.3))
    alpha = st.slider("Heterogenitas antar bank (alpha)", 0.0, 1.0, default_alpha, 0.05)
    n_banks = st.slider("Jumlah bank", 3, 8, 8)
    seed = st.number_input("Seed simulasi", 0, 9999, 42)

    st.subheader("Federated")
    rounds = st.slider("Jumlah ronde", 5, 60, 30, 5)
    local_epochs = st.slider("Epoch lokal per ronde", 1, 5, 2)
    mu = st.select_slider("FedProx mu", [0.001, 0.01, 0.1, 1.0], value=0.1)

    methods = st.multiselect("Metode", ALL_METHODS, default=DEFAULT, format_func=method_label)
    st.caption("Metode *Diffusion* membangkitkan jalur skenario (tab 5), tetapi training-nya lebih lama (~4 menit).")
    run = st.button("Jalankan eksperimen", type="primary", width="stretch")

cfg = get_scenario(scenario, alpha=alpha, n_banks=n_banks, seed=int(seed))
train_cfg = TrainConfig()


@st.cache_resource(show_spinner=False)
def _run(cfg, methods, rounds, local_epochs, mu):
    status = st.status("Menjalankan eksperimen...", expanded=True)
    res = run_experiment_cached(
        cfg, tuple(methods), train_cfg, FedConfig(rounds=rounds, local_epochs=local_epochs), mu,
        cache_dir=CACHE, verbose=False, progress=status.write,
    )
    status.update(label="Selesai", state="complete", expanded=False)
    return res


if "params" not in st.session_state or run:
    st.session_state.params = (cfg, tuple(methods), rounds, local_epochs, mu)

if not st.session_state.params[1]:
    st.warning("Pilih minimal satu metode.")
    st.stop()

res = _run(*st.session_state.params)
sim = res.sim
methods_run = list(res.preds)
path_methods = [m for m in methods_run if f"{m}_paths" in res.extras]

tab_data, tab_fc, tab_metric, tab_fed, tab_paths = st.tabs(
    ["1. Data simulasi", "2. Forecast per bank", "3. Perbandingan metode", "4. Proses federated",
     "5. Skenario jalur (diffusion)"]
)

# ---------------------------------------------------------------- tab 1
with tab_data:
    c1, c2 = st.columns([3, 1])
    with c1:
        norm = st.toggle("Tampilkan ternormalisasi (per bank)", value=False)
        st.plotly_chart(plotly_series(sim, normalize=norm), width="stretch")
    with c2:
        st.dataframe(
            pd.DataFrame(
                {
                    "bank": [b.name for b in sim.banks],
                    "ukuran": [b.size for b in sim.banks],
                    "kelompok": [getattr(b, "group", 0) for b in sim.banks],
                    "histori train (hari)": [sim.val_start - s for s in sim.start_idx],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    st.info(
        "Data ini **buatan** (simulasi), sehingga distribusi sebenarnya diketahui. "
        "Metode *Oracle* memakai distribusi sebenarnya itu; ia adalah batas terbaik yang mungkin dicapai."
    )

# ---------------------------------------------------------------- tab 2
with tab_fc:
    c1, c2 = st.columns(2)
    bank = c1.selectbox("Bank", range(len(sim.banks)), format_func=lambda k: f"{sim.banks[k].name} ({sim.banks[k].size})")
    shown = c2.multiselect("Metode", methods_run, default=[m for m in ("local", "fedavg") if m in methods_run] or methods_run[:2],
                           format_func=method_label)
    client = res.clients[bank]
    for m in shown:
        fig = plotly_fan(sim, bank, res.pred_original_scale(m, bank), client.test.origins, title=method_label(m))
        st.plotly_chart(fig, width="stretch")
        row = res.metrics.query("method == @m and bank == @client.name").iloc[0]
        st.caption(
            f"CRPS {row.CRPS:.3f} · MAE {row.MAE:.3f} · coverage 80% = {row.Coverage80:.0%} "
            f"· coverage 90% = {row.Coverage90:.0%} (skala ternormalisasi: 1 unit = 5% saldo rata-rata)"
        )

# ---------------------------------------------------------------- tab 3
with tab_metric:
    st.markdown(
        "**CRPS** menilai akurasi dan kalibrasi sekaligus (lebih kecil lebih baik). "
        "**Coverage 80%** idealnya mendekati 80%: terlalu rendah berarti model *terlalu percaya diri*."
    )
    summary = res.summary().rename(index=method_label)
    st.dataframe(
        summary.style.format({c: "{:.3f}" for c in summary.columns if not c.startswith("Coverage")})
        .format({c: "{:.0%}" for c in summary.columns if c.startswith("Coverage")}),
        width="stretch",
    )
    metric = st.radio("Metrik per bank", ["CRPS", "MAE", "Coverage80", "Lebar80"], horizontal=True)
    pivot = res.metrics.pivot(index="bank", columns="method", values=metric)[methods_run].rename(columns=method_label)
    st.bar_chart(pivot, stack=False, height=350)

    st.subheader("Kalibrasi (reliability diagram)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="ideal"))
    y = np.concatenate([c.test.y for c in res.clients])
    for m in methods_run:
        rel = reliability(y, np.concatenate(res.preds[m]))
        fig.add_trace(go.Scatter(x=rel.nominal, y=rel.observed, mode="lines+markers", name=method_label(m)))
    fig.update_layout(xaxis_title="kuantil nominal", yaxis_title="proporsi aktual <= prediksi", height=420)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tab 4
with tab_fed:
    fed_hist = {k: v for k, v in res.histories.items()
                if k in ("fedavg", "fedprox", "diffusion_fed") or k.startswith("clustered_klaster")}
    if not fed_hist:
        st.info("Jalankan metode FedAvg/FedProx/Clustered untuk melihat proses federated.")
    else:
        st.markdown(
            "Setiap ronde: server mengirim model -> tiap bank melatih dengan datanya sendiri -> "
            "bank mengirim **bobot** (bukan data) -> server merata-ratakan. Kurva di bawah adalah loss "
            "validasi model global per ronde."
        )
        st.plotly_chart(plotly_history(fed_hist), width="stretch")
        which = st.selectbox("Loss validasi per bank untuk", list(fed_hist))
        per_bank = pd.DataFrame(fed_hist[which]).set_index("round").filter(like="val_Bank")
        st.line_chart(per_bank.rename(columns=lambda c: c.removeprefix("val_")), height=350)

    if "cluster_labels" in res.extras:
        st.subheader("Clustered FL: kelompok yang ditemukan")
        st.markdown(
            "Server mengelompokkan bank dari **kemiripan arah update bobot** (bukan dari data). "
            "Bandingkan dengan kelompok sebenarnya di simulasi."
        )
        lab = res.extras["cluster_labels"]
        c1, c2 = st.columns([1, 2])
        c1.dataframe(pd.DataFrame({"bank": [b.name for b in sim.banks], "klaster ditemukan": lab,
                                   "kelompok sebenarnya": [getattr(b, "group", 0) for b in sim.banks]}),
                     hide_index=True, width="stretch")
        sim_mat = res.extras["cluster_similarity"]
        c2.plotly_chart(go.Figure(go.Heatmap(z=sim_mat, x=[b.name for b in sim.banks], y=[b.name for b in sim.banks],
                                             colorscale="Blues", zmin=-1, zmax=1))
                        .update_layout(title="Cosine similarity update antar bank", height=380),
                        width="stretch")

# ---------------------------------------------------------------- tab 5
with tab_paths:
    if not path_methods:
        st.info("Pilih metode **Diffusion (federated)** atau **Diffusion (centralized)** lalu jalankan eksperimen.")
    else:
        st.markdown(
            "Model diffusion membangkitkan **jalur lengkap** masa depan, bukan hanya rentang per hari. "
            "Dengan ratusan jalur kita bisa menjawab pertanyaan risiko seperti *\"berapa peluang saldo turun "
            "lebih dari 5% kapan pun dalam 14 hari?\"*. Jalur merah = 5% skenario terburuk (untuk stress testing)."
        )
        c1, c2, c3 = st.columns(3)
        pm = c1.selectbox("Model", path_methods, format_func=method_label)
        pb = c2.selectbox("Bank ", range(len(sim.banks)), format_func=lambda k: sim.banks[k].name)
        cl = res.clients[pb]
        oi = c3.slider("Origin (hari ke- dalam periode uji)", 0, len(cl.test.origins) - 1, 7)
        origin = int(cl.test.origins[oi])
        paths = cl.denorm(res.extras[f"{pm}_paths"][pb][oi])
        st.plotly_chart(plotly_paths(sim, pb, origin, paths), width="stretch")

        today = sim.y[origin, pb]
        thr = today - 0.05 * cl.center
        orc = oracle_paths(sim, pb, origin, paths.shape[1], 2000, seed=origin)
        m1, m2, m3 = st.columns(3)
        m1.metric("P(turun > 5%) menurut model", f"{(paths.min(axis=1) < thr).mean():.0%}")
        m2.metric("P(turun > 5%) sebenarnya (Oracle)", f"{(orc.min(axis=1) < thr).mean():.0%}")
        m3.metric("Titik terendah, skenario 5% terburuk", f"{np.quantile(paths.min(axis=1), 0.05):,.1f}",
                  delta=f"{np.quantile(paths.min(axis=1), 0.05) / today - 1:.1%} dari hari ini")

        st.subheader("Evaluasi tingkat-jalur (semua bank & origin uji)")
        rows = []
        quant_ref = next((m for m in ("fedavg", "central", "local") if m in res.preds), None)
        for c in res.clients:
            last = c.test.x_hist[:, -1]
            rows.append({"metode": method_label(pm), **path_min_metrics(res.extras[f"{pm}_paths"][c.bank], c.test.y, last)})
            if quant_ref:
                rows.append({"metode": f"{method_label(quant_ref)} (hari independen)",
                             **path_min_metrics(independent_paths(res.preds[quant_ref][c.bank]), c.test.y, last)})
        st.dataframe(pd.DataFrame(rows).groupby("metode", sort=False).mean().round(3), width="stretch")
        st.caption(
            "CRPS_min & Coverage80_min: kualitas prediksi **titik terendah** 14 hari. Brier_turun: akurasi peluang "
            "'turun > 5%' (lebih kecil lebih baik). Model kuantil hanya tahu distribusi per hari, sehingga jalurnya "
            "harus dibuat dengan menganggap tiap hari independen; akibatnya titik terendah jadi salah."
        )
