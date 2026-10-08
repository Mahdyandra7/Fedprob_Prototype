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

from fedprob.data import SCENARIO_DESCRIPTIONS, SCENARIOS, get_scenario
from fedprob.experiment import METHOD_LABELS, run_experiment_cached
from fedprob.federated import FedConfig
from fedprob.metrics import reliability
from fedprob.plots import plotly_fan, plotly_history, plotly_series
from fedprob.training import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results"

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
    alpha = st.slider("Heterogenitas antar bank (alpha)", 0.0, 1.0, default_alpha, 0.1)
    n_banks = st.slider("Jumlah bank", 3, 8, 8)
    seed = st.number_input("Seed simulasi", 0, 9999, 42)

    st.subheader("Federated")
    rounds = st.slider("Jumlah ronde", 5, 60, 30, 5)
    local_epochs = st.slider("Epoch lokal per ronde", 1, 5, 2)
    mu = st.select_slider("FedProx mu", [0.001, 0.01, 0.1, 1.0], value=0.01)

    methods = st.multiselect(
        "Metode",
        list(METHOD_LABELS),
        default=["oracle", "ets", "local", "fedavg", "fedprox", "fedavg_ft", "central"],
        format_func=lambda m: METHOD_LABELS[m],
    )
    run = st.button("Jalankan eksperimen", type="primary", width="stretch")

cfg = get_scenario(scenario, alpha=alpha, n_banks=n_banks, seed=int(seed))
fed_cfg = FedConfig(rounds=rounds, local_epochs=local_epochs)
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

tab_data, tab_fc, tab_metric, tab_fed = st.tabs(
    ["1. Data simulasi", "2. Forecast per bank", "3. Perbandingan metode", "4. Proses federated"]
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
                           format_func=lambda m: METHOD_LABELS[m])
    client = res.clients[bank]
    for m in shown:
        fig = plotly_fan(sim, bank, res.pred_original_scale(m, bank), client.test.origins, title=METHOD_LABELS[m])
        st.plotly_chart(fig, width="stretch")
        row = res.metrics.query("method == @m and bank == @client.name").iloc[0]
        st.caption(
            f"CRPS {row.CRPS:.3f} · MAE {row.MAE:.3f} · coverage 80% = {row.Coverage80:.0%} "
            f"· coverage 90% = {row.Coverage90:.0%} (skala ternormalisasi)"
        )

# ---------------------------------------------------------------- tab 3
with tab_metric:
    st.markdown(
        "**CRPS** menilai akurasi dan kalibrasi sekaligus (lebih kecil lebih baik). "
        "**Coverage 80%** idealnya mendekati 80%: terlalu rendah berarti model *terlalu percaya diri*."
    )
    summary = res.summary().rename(index=METHOD_LABELS)
    st.dataframe(
        summary.style.format({c: "{:.3f}" for c in summary.columns if not c.startswith("Coverage")})
        .format({c: "{:.0%}" for c in summary.columns if c.startswith("Coverage")}),
        width="stretch",
    )
    metric = st.radio("Metrik per bank", ["CRPS", "MAE", "Coverage80", "Lebar80"], horizontal=True)
    pivot = res.metrics.pivot(index="bank", columns="method", values=metric)[methods_run].rename(columns=METHOD_LABELS)
    st.bar_chart(pivot, stack=False, height=350)

    st.subheader("Kalibrasi (reliability diagram)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="ideal"))
    y = np.concatenate([c.test.y for c in res.clients])
    for m in methods_run:
        q = np.concatenate(res.preds[m])
        rel = reliability(y, q)
        fig.add_trace(go.Scatter(x=rel.nominal, y=rel.observed, mode="lines+markers", name=METHOD_LABELS[m]))
    fig.update_layout(xaxis_title="kuantil nominal", yaxis_title="proporsi aktual <= prediksi", height=420)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tab 4
with tab_fed:
    fed_hist = {k: v for k, v in res.histories.items() if k in ("fedavg", "fedprox")}
    if not fed_hist:
        st.info("Jalankan metode FedAvg/FedProx untuk melihat proses federated.")
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
