"""Membangkitkan notebook tahapan di folder notebooks/.

Notebook ditulis sebagai kode di sini agar mudah di-review di git. Jalankan:
    python scripts/build_notebooks.py
lalu eksekusi notebook dengan jupyter nbconvert (lihat CLAUDE.md).
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """\
%matplotlib inline
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.precision", 3)
plt.rcParams["figure.dpi"] = 110"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {}

# ---------------------------------------------------------------------------
NOTEBOOKS["01_simulasi_data"] = [
    md("""
# 01 · Simulasi Data

**Tujuan:** membangkitkan data saldo dana harian beberapa bank fiktif dan memahami
skenario yang akan diuji.

Mengapa simulasi?
1. Data antar bank sungguhan bersifat rahasia (inilah masalah yang ingin dipecahkan
   federated learning).
2. Pada data simulasi, **distribusi sebenarnya diketahui**, sehingga kita bisa menguji
   dengan tepat apakah interval prediksi model memang "jujur".

Model setiap bank *k* pada hari *t*:

$$y_k(t) = S_k\\,[\\,D_k(t) + \\beta_k M(t) + I_k(t) + \\sigma_k(t)\\,\\varepsilon_k(t)\\,]$$

| Komponen | Arti |
|---|---|
| $S_k$ | ukuran bank (besar/menengah/kecil) |
| $D_k(t)$ | tren + pola mingguan + efek gajian + efek Lebaran (+ shock) |
| $M(t)$ | faktor pasar bersama (AR(1)), dirasakan semua bank |
| $I_k(t)$ | faktor khusus bank (AR(1)) |
| $\\sigma_k(t)\\varepsilon_k(t)$ | noise, volatilitas bisa naik saat shock |

Parameter **alpha** mengatur heterogenitas: 0 = semua bank berpola sama, 1 = sangat berbeda (non-IID).
"""),
    code(SETUP),
    code("""
from fedprob.data import SCENARIOS, SCENARIO_DESCRIPTIONS, get_scenario, simulate
from fedprob.plots import plot_series

sim = simulate(get_scenario("normal"))
sim.to_frame().head()
"""),
    code("""
pd.DataFrame([{ "bank": b.name, "ukuran": b.size, "skala": b.scale, "tren/thn": b.trend,
                "beta_pasar": b.beta, "noise_sd": b.noise_sd } for b in sim.banks])
"""),
    md("Series asli: perbedaan skala antar bank sangat besar. Inilah alasan setiap bank dinormalisasi sendiri-sendiri."),
    code("""
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
plot_series(sim, ax=axes[0], title="Skala asli")
plot_series(sim, normalize=True, banks=[0, 3, 6], ax=axes[1], title="Ternormalisasi (3 bank)")
plt.tight_layout()
"""),
    md("Zoom 4 bulan: terlihat pola mingguan, lonjakan sekitar tanggal gajian, dan penurunan menjelang Lebaran."),
    code("""
s = slice(sim.dates.get_loc("2024-02-15"), sim.dates.get_loc("2024-06-15"))
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(sim.dates[s], sim.y[s, 0], lw=1)
ax.axvline(pd.Timestamp("2024-04-10"), color="red", ls=":", label="Lebaran 2024")
ax.set_title(f"{sim.banks[0].name}: detail pola harian"); ax.legend()
"""),
    md("## Skenario yang disiapkan"),
    code("""
for name, desc in SCENARIO_DESCRIPTIONS.items():
    print(f"{name:12s} : {desc}")
"""),
    code("""
fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(12, 3 * len(SCENARIOS)))
for ax, name in zip(axes, SCENARIOS):
    plot_series(simulate(get_scenario(name)), normalize=True, ax=ax, title=f"Skenario: {name}")
plt.tight_layout()
"""),
    md("""
## Pembagian data (tanpa kebocoran waktu)
- **train**: semua sebelum 180 hari terakhir
- **validasi**: 90 hari (untuk memilih epoch/ronde terbaik)
- **uji**: 90 hari terakhir (forecast 14 hari ke depan, origin setiap hari → 77 origin per bank)

Normalisasi: setiap bank diskalakan **relatif terhadap saldo rata-ratanya** (1 unit = 5% saldo
rata-rata), sehingga bank besar dan kecil berada di skala yang sama. Normalisasi mean/std biasa
ternyata gagal untuk bank berhistori pendek (lihat `PROGRESS.md`).

Input model: 56 hari terakhir + fitur kalender hari-hari target (hari, gajian, Lebaran, musim).
"""),
    code("""
from fedprob.data import build_all_clients
clients = build_all_clients(sim)
pd.DataFrame([{ "bank": c.name, "n_train": len(c.train), "n_val": len(c.val), "n_test": len(c.test)} for c in clients])
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["02_baseline"] = [
    md("""
# 02 · Baseline Klasik & Metrik Probabilistik

Sebelum memakai deep learning, kita perlu pembanding:
- **Seasonal Naive**: "minggu depan sama dengan minggu lalu", dengan interval dari residual historis.
- **ETS (Holt-Winters)**: model statistik klasik dengan interval Gaussian.
- **Oracle**: distribusi *sebenarnya* (hanya mungkin pada data simulasi), yaitu batas terbaik.

### Bagaimana menilai forecast probabilistik?
| Metrik | Arti | Ideal |
|---|---|---|
| **CRPS** | akurasi + kalibrasi sekaligus | sekecil mungkin |
| **Coverage 80%** | % nilai aktual yang masuk interval 80% | ≈ 80% |
| **Lebar 80%** | lebar interval (ketajaman) | sesempit mungkin *asal* coverage terpenuhi |

Model yang intervalnya sempit tapi coverage-nya 50% berarti **terlalu percaya diri**, dan itu berbahaya untuk manajemen risiko.
"""),
    code(SETUP),
    code("""
from fedprob import QUANTILES
from fedprob.data import get_scenario, simulate, build_all_clients
from fedprob.experiment import _oracle_preds
from fedprob.models import seasonal_naive_quantiles, ets_quantiles
from fedprob.metrics import evaluate, reliability
from fedprob.plots import plot_fan, plot_reliability

sim = simulate(get_scenario("normal"))
clients = build_all_clients(sim)
"""),
    code("""
preds = {"seasonal_naive": [], "ets": [], "oracle": []}
for c in clients:
    preds["seasonal_naive"].append(seasonal_naive_quantiles(c))
    preds["ets"].append(ets_quantiles(sim, c))
    preds["oracle"].append(_oracle_preds(sim, c))

rows = [{"method": m, "bank": c.name, **evaluate(c.test.y, p)}
        for m, plist in preds.items() for c, p in zip(clients, plist)]
metrics = pd.DataFrame(rows)
metrics.groupby("method").mean(numeric_only=True)
"""),
    code("""
k = 0; c = clients[k]
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for ax, m in zip(axes, preds):
    plot_fan(sim, k, c.denorm(preds[m][k]), c.test.origins, ax=ax, title=m)
plt.tight_layout()
"""),
    code("""
y = np.concatenate([c.test.y for c in clients])
plot_reliability({m: reliability(y, np.concatenate(p)) for m, p in preds.items()})
"""),
    md("""
**Catatan penting tentang Oracle.** Oracle memakai distribusi *sebenarnya*, tetapi coverage 80%-nya
di periode uji ini hanya sekitar 72%. Mengapa? Periode uji hanya 90 hari, dan faktor pasar bersama
(AR(1), sangat persisten) membuat error semua bank dan semua horizon **saling berkorelasi**. Jadi
efektifnya kita hanya mengamati sedikit "kejadian" independen. Di unit test (`tests/test_core.py`)
yang memakai banyak periode acak, coverage Oracle tepat ~80%.

Pelajarannya: coverage yang diukur di periode pendek itu berisik. Karena itu kita membandingkan
model **terhadap Oracle**, bukan hanya terhadap angka nominal. Jarak antara baseline dan Oracle
adalah "ruang perbaikan" yang ingin kita kejar.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["03_probabilistic"] = [
    md("""
# 03 · Model Probabilistik (Quantile Neural Network)

Model memprediksi **7 kuantil sekaligus** (5%, 10%, 25%, 50%, 75%, 90%, 95%) untuk 14 hari ke depan.

- **Loss: pinball loss.** Untuk kuantil τ, kesalahan *di bawah* prediksi diberi bobot τ dan di atas
  diberi bobot (1−τ). Akibatnya model "terdorong" ke kuantil yang benar.
- **Kuantil tidak saling silang.** Model memprediksi median, lalu menambah/mengurangi jarak positif
  (softplus), sehingga q05 ≤ q10 ≤ … ≤ q95 selalu terjamin.
- **Arsitektur:** MLP kecil (2 hidden layer × 128). Inputnya 56 hari histori + fitur kalender.

Di notebook ini model dilatih untuk **satu bank saja** (Local-only), sebagai fondasi sebelum federated.
"""),
    code(SETUP),
    code("""
import torch
from fedprob.data import get_scenario, simulate, build_all_clients
from fedprob.experiment import _oracle_preds
from fedprob.models import QuantileMLP, ets_quantiles
from fedprob.training import TrainConfig, train_with_early_stopping, predict
from fedprob.metrics import evaluate, reliability
from fedprob.plots import plot_fan, plot_history, plot_reliability

sim = simulate(get_scenario("normal"))
clients = build_all_clients(sim)
c = clients[0]
model = QuantileMLP()
print(model)
print("jumlah parameter:", sum(p.numel() for p in model.parameters()))
"""),
    code("""
res = train_with_early_stopping([c], TrainConfig(epochs=80, patience=15))
print("epoch terbaik:", res.best_epoch)
plot_history({"train": [{"epoch": h["epoch"], "val_loss": h["train_loss"]} for h in res.history],
              "validasi": res.history}, title="Pinball loss")
"""),
    code("""
p_nn = predict(res.model, c.test)
p_ets = ets_quantiles(sim, c)
p_or = _oracle_preds(sim, c)
pd.DataFrame({"QuantileMLP": evaluate(c.test.y, p_nn), "ETS": evaluate(c.test.y, p_ets),
              "Oracle": evaluate(c.test.y, p_or)}).T
"""),
    code("""
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
plot_fan(sim, 0, c.denorm(p_nn), c.test.origins, ax=axes[0], title="QuantileMLP (Local-only)")
plot_fan(sim, 0, c.denorm(p_or), c.test.origins, ax=axes[1], title="Oracle")
plt.tight_layout()
"""),
    md("""
## Bagaimana jika data sebuah bank sedikit?
Bank kecil dalam skenario `data_langka` hanya punya ~4 bulan histori. Model lokal sulit belajar dari
data sesedikit itu. **Inilah motivasi utama federated learning.**
"""),
    code("""
sim_l = simulate(get_scenario("data_langka"))
cl = build_all_clients(sim_l)
rows = []
for k in [0, 5]:
    r = train_with_early_stopping([cl[k]], TrainConfig())
    rows.append({"bank": cl[k].name, "n_train": len(cl[k].train), **evaluate(cl[k].test.y, predict(r.model, cl[k].test))})
pd.DataFrame(rows)
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["04_federated"] = [
    md("""
# 04 · Federated Learning: Local vs FedAvg vs Centralized

**Masalah:** setiap bank punya data sendiri yang tidak boleh dibagikan.

| Pendekatan | Data dibagikan? | Keterangan |
|---|---|---|
| **Local-only** | tidak | tiap bank melatih modelnya sendiri |
| **Federated (FedAvg)** | **tidak**, hanya bobot model | bank berkolaborasi tanpa membuka data |
| **FedProx** | tidak | FedAvg + penalti agar model lokal tidak menjauh dari global (untuk data non-IID) |
| **FedAvg + fine-tune** | tidak | model global disesuaikan sedikit ke tiap bank (personalisasi) |
| **Centralized** | **ya**, semua data digabung | tidak realistis (melanggar privasi), dipakai sebagai batas atas |

**Alur FedAvg per ronde:**
1. Server mengirim model global ke semua bank
2. Tiap bank melatih 2 epoch dengan datanya sendiri
3. Bank mengirim bobot ke server
4. Server merata-ratakan bobot (ditimbang jumlah data) → model global baru
"""),
    code(SETUP),
    code("""
from fedprob.experiment import run_experiment_cached, METHOD_LABELS
from fedprob.plots import plot_fan, plot_history, plot_metric_bars, plot_reliability
from fedprob.metrics import reliability

res = run_experiment_cached("normal", cache_dir="../results")
res.summary()
"""),
    code("""
plot_history({k: v for k, v in res.histories.items() if k in ("fedavg", "fedprox", "central")},
             title="Loss validasi: model global per ronde (central: per epoch)")
"""),
    code("""
plot_metric_bars(res.metrics, "CRPS", methods=["ets", "local", "fedavg", "fedavg_ft", "central", "oracle"])
"""),
    code("""
k = 6; c = res.clients[k]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, m in zip(axes, ["local", "fedavg"]):
    plot_fan(res.sim, k, res.pred_original_scale(m, k), c.test.origins, ax=ax, title=f"{c.name}: {METHOD_LABELS[m]}")
plt.tight_layout()
"""),
    code("""
y = np.concatenate([c.test.y for c in res.clients])
plot_reliability({METHOD_LABELS[m]: reliability(y, np.concatenate(res.preds[m]))
                  for m in ["ets", "local", "fedavg", "central"]})
"""),
    md("""
### Cara membaca hasil
- Jika **FedAvg ≈ Centralized**, kolaborasi tanpa berbagi data hampir sebaik menggabungkan data.
- Jika **FedAvg < Local** (CRPS lebih kecil), setiap bank diuntungkan dengan berkolaborasi.
- Bandingkan coverage: apakah interval tetap "jujur" (≈ nominal)?
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["05_eksperimen_skenario"] = [
    md("""
# 05 · Eksperimen Skenario: Kapan Federated Learning Menolong?

Kita jalankan semua metode pada setiap skenario, lalu pada beberapa tingkat heterogenitas.
Hasil di-cache ke `results/`, jadi eksekusi kedua berjalan cepat.
"""),
    code(SETUP),
    code("""
from fedprob.data import SCENARIOS, get_scenario
from fedprob.experiment import run_experiment_cached, METHOD_LABELS
from fedprob.plots import plot_fan

MAIN = ["ets", "local", "fedavg", "fedprox", "fedavg_ft", "central", "oracle"]
results = {s: run_experiment_cached(s, cache_dir="../results", verbose=False) for s in SCENARIOS}
"""),
    md("## Ringkasan CRPS (rata-rata semua bank, lebih kecil lebih baik)"),
    code("""
crps = pd.DataFrame({s: r.summary()["CRPS"] for s, r in results.items()}).loc[MAIN]
# Sorot metode realistis terbaik (tanpa Oracle & Centralized, yang tidak bisa dicapai di dunia nyata)
realistic = [METHOD_LABELS[m] for m in MAIN if m not in ("oracle", "central")]
crps.rename(index=METHOD_LABELS).style.format("{:.3f}").highlight_min(axis=0, subset=pd.IndexSlice[realistic, :], color="#cde7cd")
"""),
    md("## Coverage interval 80% (ideal ≈ 0.80)"),
    code("""
cov = pd.DataFrame({s: r.summary()["Coverage80"] for s, r in results.items()}).loc[MAIN]
cov.rename(index=METHOD_LABELS).style.format("{:.0%}")
"""),
    md("""
## Fokus: bank kecil pada skenario `data_langka`
Pertanyaan kunci: apakah bank dengan data sedikit paling diuntungkan oleh federasi?
"""),
    code("""
m = results["data_langka"].metrics
m[m.method.isin(["local", "fedavg", "fedavg_ft", "central"])].pivot(index="bank", columns="method", values="CRPS")[["local", "fedavg", "fedavg_ft", "central"]]
"""),
    code("""
r = results["data_langka"]; k = 7; c = r.clients[k]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, meth in zip(axes, ["local", "fedavg_ft"]):
    plot_fan(r.sim, k, r.pred_original_scale(meth, k), c.test.origins, ax=ax, title=f"{c.name}: {METHOD_LABELS[meth]}")
plt.tight_layout()
"""),
    md("## Skenario shock: apakah model tetap jujur saat terjadi guncangan?"),
    code("""
r = results["shock"]; k = 5; c = r.clients[k]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, meth in zip(axes, ["fedavg", "oracle"]):
    plot_fan(r.sim, k, r.pred_original_scale(meth, k), c.test.origins, ax=ax, title=f"{c.name}: {METHOD_LABELS[meth]}")
plt.tight_layout()
"""),
    md("""
## Apakah hasilnya stabil? Ulangi `data_langka` dengan beberapa seed
Satu periode uji itu berisik (lihat catatan Oracle di notebook 02). Karena itu kita ulangi
simulasi dengan seed berbeda, lalu fokus pada bank kecil yang datanya langka.
"""),
    code("""
SEEDS = [42, 7, 123]
FOCUS = ["oracle", "local", "fedavg", "fedavg_ft", "central"]
rows = []
for sd in SEEDS:
    r = run_experiment_cached(get_scenario("data_langka", seed=sd), methods=tuple(FOCUS), cache_dir="../results", verbose=False)
    mm = r.metrics.copy(); mm["seed"] = sd
    rows.append(mm)
seeds_df = pd.concat(rows)
seeds_df["kelompok"] = np.where(seeds_df["bank"].isin(["Bank F", "Bank G", "Bank H"]), "bank kecil (data langka)", "bank lain")
seeds_df.groupby(["kelompok", "method"])[["CRPS", "Coverage80"]].agg(["mean", "std"]).loc[:, :].round(3)
"""),
    code("""
pivot = seeds_df.groupby(["kelompok", "method"])["CRPS"].mean().unstack()[FOCUS]
ax = pivot.rename(columns=METHOD_LABELS).plot.bar(figsize=(9, 4), rot=0)
ax.set_ylabel("CRPS rata-rata (3 seed)"); ax.set_title("Siapa yang paling diuntungkan federated learning?")
"""),
    md("## Sweep heterogenitas (alpha)"),
    code("""
alphas = [0.0, 0.3, 0.6, 0.9]
sweep = []
for a in alphas:
    r = run_experiment_cached(get_scenario("normal", alpha=a), methods=("local", "fedavg", "fedprox", "fedavg_ft", "central"),
                              cache_dir="../results", verbose=False)
    sweep.append(r.summary()["CRPS"].rename(a))
sweep = pd.DataFrame(sweep)
ax = sweep.rename(columns=METHOD_LABELS).plot(marker="o", figsize=(8, 4))
ax.set_xlabel("alpha (heterogenitas)"); ax.set_ylabel("CRPS rata-rata"); ax.set_title("Pengaruh heterogenitas data antar bank")
sweep
"""),
    md("""
## Kesimpulan sementara
*(Berdasarkan eksekusi 2026-10-08; angka pasti lihat tabel di atas dan `PROGRESS.md`.)*

1. **Federated learning paling menolong bank dengan data sedikit.** Pada `data_langka` (rata-rata 3 seed),
   CRPS bank kecil turun dari ~0.215 (Local) ke ~0.156 (FedAvg), **sekitar 27% lebih baik**, dan setara
   Centralized (~0.154) **tanpa satu pun data dibagikan**. Interval model lokal terlalu lebar (coverage 96%);
   federasi membuatnya lebih tajam dan tetap terkalibrasi (~84%).
2. **Bank dengan data banyak hampir tidak diuntungkan.** Pada skenario `normal`, Local ≈ FedAvg ≈ Centralized.
   Data yang cukup sudah memadai untuk proses sesederhana ini.
3. **Heterogenitas (non-IID) melemahkan FedAvg.** Saat alpha naik, CRPS FedAvg memburuk, dan
   **personalisasi (fine-tune lokal)** memulihkan sebagian besar kerugian. FedProx belum memberi
   perbaikan berarti di sini.
4. **Semua model gagal saat shock.** Coverage 80% anjlok ke ~40%: model yang belajar dari data
   "normal" menjadi **terlalu percaya diri** ketika dunia berubah. Ini menjawab langsung pertanyaan
   riset tentang model yang *jujur soal ketidakpastian* dan menjadi motivasi tahap berikutnya.

## Langkah riset berikutnya
1. **Diffusion model** untuk membangkitkan skenario masa depan (arah riset ke-2 di essay).
2. Model global yang lebih kuat (Transformer/TFT, DeepAR) dan forecast konformal untuk menjamin coverage.
3. Privasi formal: *differential privacy* dan *secure aggregation* pada FedAvg.
4. Validasi pada data riil publik, misalnya data perbankan agregat per provinsi.
"""),
]


def main():
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, OUT / f"{name}.ipynb")
        print("ditulis:", OUT / f"{name}.ipynb")


if __name__ == "__main__":
    main()
