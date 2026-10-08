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
from fedprob.experiment import run_experiment_cached, method_label
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
    plot_fan(res.sim, k, res.pred_original_scale(m, k), c.test.origins, ax=ax, title=f"{c.name}: {method_label(m)}")
plt.tight_layout()
"""),
    code("""
y = np.concatenate([c.test.y for c in res.clients])
plot_reliability({method_label(m): reliability(y, np.concatenate(res.preds[m]))
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
from fedprob.experiment import run_experiment_cached, method_label
from fedprob.plots import plot_fan

MAIN = ["ets", "local", "fedavg", "fedprox", "fedavg_ft", "clustered", "fedavg+aci", "central", "oracle"]
results = {s: run_experiment_cached(s, cache_dir="../results", verbose=False) for s in SCENARIOS}
"""),
    md("## Ringkasan CRPS (rata-rata semua bank, lebih kecil lebih baik)"),
    code("""
crps = pd.DataFrame({s: r.summary()["CRPS"] for s, r in results.items()}).loc[MAIN]
# Sorot metode realistis terbaik (tanpa Oracle & Centralized, yang tidak bisa dicapai di dunia nyata)
realistic = [method_label(m) for m in MAIN if m not in ("oracle", "central")]
crps.rename(index=method_label).style.format("{:.3f}").highlight_min(axis=0, subset=pd.IndexSlice[realistic, :], color="#cde7cd")
"""),
    md("## Coverage interval 80% (ideal ≈ 0.80)"),
    code("""
cov = pd.DataFrame({s: r.summary()["Coverage80"] for s, r in results.items()}).loc[MAIN]
cov.rename(index=method_label).style.format("{:.0%}")
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
    plot_fan(r.sim, k, r.pred_original_scale(meth, k), c.test.origins, ax=ax, title=f"{c.name}: {method_label(meth)}")
plt.tight_layout()
"""),
    md("## Skenario shock: apakah model tetap jujur saat terjadi guncangan?"),
    code("""
r = results["shock"]; k = 5; c = r.clients[k]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, meth in zip(axes, ["fedavg", "oracle"]):
    plot_fan(r.sim, k, r.pred_original_scale(meth, k), c.test.origins, ax=ax, title=f"{c.name}: {method_label(meth)}")
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
ax = pivot.rename(columns=method_label).plot.bar(figsize=(9, 4), rot=0)
ax.set_ylabel("CRPS rata-rata (3 seed)"); ax.set_title("Siapa yang paling diuntungkan federated learning?")
"""),
    md("## Sweep heterogenitas (alpha)"),
    code("""
alphas = [0.0, 0.3, 0.6, 0.9]
sweep = []
for a in alphas:
    r = run_experiment_cached(get_scenario("normal", alpha=a), methods=("local", "fedavg", "fedprox", "fedavg_ft", "clustered", "central"),
                              cache_dir="../results", verbose=False)
    sweep.append(r.summary()["CRPS"].rename(a))
sweep = pd.DataFrame(sweep)
ax = sweep.rename(columns=method_label).plot(marker="o", figsize=(8, 4))
ax.set_xlabel("alpha (heterogenitas)"); ax.set_ylabel("CRPS rata-rata"); ax.set_title("Pengaruh heterogenitas data antar bank")
sweep
"""),
    md("""
## Kesimpulan
*(Berdasarkan eksekusi 2026-10-08; angka lengkap lihat tabel di atas dan `PROGRESS.md`.)*

1. **Federated learning paling menolong bank dengan data sedikit.** Pada `data_langka` (rata-rata 3 seed),
   CRPS bank kecil turun dari ~0.215 (Local) ke ~0.156 (FedAvg), **sekitar 27% lebih baik**, dan setara
   Centralized (~0.154) **tanpa satu pun data dibagikan**. Interval model lokal terlalu lebar (coverage 96%);
   federasi membuatnya lebih tajam dan tetap terkalibrasi (~84%).
2. **Bank dengan data banyak hampir tidak diuntungkan** dari FedAvg biasa: pada `normal`, Local ≈ FedAvg ≈ Centralized.
3. **FedProx (mu = 0.1) adalah metode realistis terbaik** di `normal`, `data_langka`, dan `bank_baru`, dengan coverage
   mendekati 80%. Penalti proksimal berfungsi sebagai regularisasi.
4. **Heterogenitas melemahkan FedAvg.** Saat alpha naik, CRPS FedAvg memburuk; **fine-tune lokal** paling
   konsisten memulihkannya. Bila heterogenitasnya berbentuk **kelompok** (`klaster`), **Clustered FL** terbaik
   (lihat notebook 07).
5. **Semua model gagal saat shock** (coverage 80% ≈ 40%): model menjadi *terlalu percaya diri* ketika dunia berubah.
   **ACI** (notebook 06) mengembalikan coverage ke ~81% di keenam skenario, termasuk shock.

## Pendalaman
- Notebook 06: conformal prediction (CQR & ACI)
- Notebook 07: FedProx & Clustered FL untuk data non-IID
- Notebook 08: diffusion model untuk skenario jalur & stress testing
"""),
]


# ---------------------------------------------------------------------------
NOTEBOOKS["06_conformal"] = [
    md("""
# 06 · Conformal Prediction: Membuat Interval Tetap Jujur

Notebook 05 menunjukkan bahwa saat **shock**, coverage interval 80% semua model anjlok ke ~40%.
Model menjadi *terlalu percaya diri* tepat ketika ketidakpastian paling penting.

**Conformal prediction** memperbaiki interval *setelah* model dilatih, hanya dengan melihat seberapa
sering interval meleset:

| Metode | Cara kerja | Kalibrasi pada |
|---|---|---|
| **CQR** (Romano dkk., 2019) | lebarkan/sempitkan interval sebesar kuantil skor meleset | data validasi (statik) |
| **ACI** (Gibbs & Candès, 2021) | setiap forecast yang terbukti meleset → interval berikutnya melebar | jendela 30 forecast terbaru (online) |

Skor kesesuaian untuk interval (lo, hi): $s = \\max(lo - y,\\; y - hi)$. Nilai positif berarti meleset.

**Privasi:** kalibrasi dilakukan **lokal di tiap bank**, sehingga skor tidak pernah dikirim ke server.
Conformal bisa langsung dipasang di atas model federated mana pun.
"""),
    code(SETUP),
    code("""
from fedprob.conformal import aci, cqr, online_bias_shift
from fedprob.experiment import run_experiment_cached, method_label
from fedprob.metrics import evaluate
from fedprob.plots import plot_fan

SC = ["normal", "data_langka", "shock"]
results = {s: run_experiment_cached(s, cache_dir="../results", verbose=False) for s in SC}
M = ["fedavg", "fedavg+cqr", "fedavg+aci", "oracle"]
tab = pd.concat({s: r.summary().loc[M, ["CRPS", "Coverage80", "Coverage90", "Lebar80"]] for s, r in results.items()}, axis=1)
tab.rename(index=method_label)
"""),
    md("""
**Cara membaca:**
- **ACI** membawa coverage 80% ke ~81% di **semua** skenario. CRPS hanya sedikit berubah pada kondisi normal.
- Pada `shock`, **ACI memulihkan coverage 80% dari ~40% ke ~81%**. Interval menjadi jauh lebih lebar, dan
  itulah sikap yang *jujur*: model mengakui bahwa ia sedang tidak yakin. CRPS sedikit memburuk karena pusat
  prediksinya tetap salah.
- **CQR statik hampir tidak membantu**, bahkan menurunkan coverage pada `data_langka` (73% → 69%). Kalibrasinya
  memakai periode validasi, yang perilakunya berbeda dari periode uji: perubahan periode inilah yang tidak bisa
  ditangani metode statik.
"""),
    code("""
r = results["shock"]; k = 5; c = r.clients[k]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, m in zip(axes, ["fedavg", "fedavg+aci"]):
    plot_fan(r.sim, k, r.pred_original_scale(m, k), c.test.origins, ax=ax, title=f"{c.name} (shock): {method_label(m)}")
plt.tight_layout()
"""),
    md("""
## Coverage dari waktu ke waktu
Coverage bergulir (rata-rata 7 origin) untuk interval 80%, semua bank. Garis putus-putus = target 80%.
"""),
    code("""
def rolling_cov(r, m, w=7):
    inside = []
    for c, p in zip(r.clients, r.preds[m]):
        inside.append(((c.test.y >= p[..., 1]) & (c.test.y <= p[..., 5])).mean(axis=1))
    return pd.Series(np.mean(inside, axis=0), index=r.sim.dates[r.clients[0].test.origins]).rolling(w, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(11, 3.5))
for m in ["fedavg", "fedavg+cqr", "fedavg+aci"]:
    rolling_cov(results["shock"], m).plot(ax=ax, label=method_label(m))
ax.axhline(0.8, color="grey", ls="--"); ax.set_ylim(0, 1.05); ax.set_ylabel("coverage 80%")
ax.set_title("Skenario shock: ACI bereaksi setelah interval mulai meleset"); ax.legend(fontsize=8)
"""),
    md("""
## Sensitivitas ACI terhadap gamma & window
`gamma` = seberapa cepat alpha bereaksi; `window` = berapa banyak skor terbaru yang dipakai.
"""),
    code("""
rows = []
for s in ["normal", "shock"]:
    r = results[s]
    for g in [0.005, 0.01, 0.02, 0.05]:
        for w in [30, 60]:
            ev = [evaluate(c.test.y, aci(c.val.y, pv, c.val.origins, c.test.y, pt, c.test.origins, gamma=g, window=w))
                  for c, pv, pt in zip(r.clients, r.preds_val["fedavg"], r.preds["fedavg"])]
            rows.append({"skenario": s, "gamma": g, "window": w, **pd.DataFrame(ev).mean()[["CRPS", "Coverage80", "Lebar80"]]})
pd.DataFrame(rows).pivot_table(index=["gamma", "window"], columns="skenario", values=["CRPS", "Coverage80"])
"""),
    md("""
## Hasil negatif: koreksi bias online
Ide: selain melebarkan interval, geser juga *pusat* prediksi sebesar median residual terbaru.
Hasilnya ternyata **lebih buruk** di semua skenario. Shock berbentuk "V", sehingga residual lama justru
mendorong prediksi ke arah yang salah saat data sudah pulih. Pada kondisi normal, model sendiri sudah
memakai data terbaru, jadi koreksi tambahan hanya menambah noise. Hasil negatif ini tetap dicatat.
"""),
    code("""
rows = []
for s in ["normal", "shock"]:
    r = results[s]
    for bw in [None, 14, 28]:
        ev = [evaluate(c.test.y, aci(c.val.y, pv, c.val.origins, c.test.y, pt, c.test.origins, bias_window=bw))
              for c, pv, pt in zip(r.clients, r.preds_val["fedavg"], r.preds["fedavg"])]
        rows.append({"skenario": s, "koreksi bias": bw or "tanpa", **pd.DataFrame(ev).mean()[["MAE", "CRPS", "Coverage80"]]})
pd.DataFrame(rows).set_index(["skenario", "koreksi bias"])
"""),
    md("""
## Kesimpulan
- Conformal prediction adalah lapisan "kejujuran" yang murah, bisa dipasang di atas model apa pun, dan
  **tetap menjaga privasi** karena kalibrasinya lokal.
- **ACI** adalah satu-satunya metode di prototipe ini yang tetap terkalibrasi saat shock.
- Batasannya: conformal hanya memperbaiki **lebar** interval, bukan **akurasi** pusatnya. Untuk forecast yang
  akurat saat rezim berubah, dibutuhkan model yang bisa beradaptasi atau membangkitkan skenario (notebook 08).
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["07_heterogenitas"] = [
    md("""
# 07 · Mengatasi Heterogenitas: FedProx & Clustered FL

Notebook 05 menunjukkan bahwa saat pola antar bank berbeda (non-IID), satu model global (FedAvg) memburuk.
Tiga strategi yang dibandingkan di sini, semuanya **tanpa berbagi data**:

1. **FedProx**: penalti $\\frac{\\mu}{2}\\lVert w - w_{global}\\rVert^2$ agar model lokal tidak menjauh dari model global.
2. **Personalisasi (fine-tune)**: model global disesuaikan sedikit ke tiap bank.
3. **Clustered FL** (Sattler dkk., 2020): server mengelompokkan bank berdasarkan **kemiripan arah update bobot**,
   lalu melatih satu model per kelompok.

Skenario baru **`klaster`** berisi dua kelompok bank (misalnya ritel vs korporat). Kelompok 0 punya efek
gajian kuat, sedangkan kelompok 1 tidak punya efek gajian tetapi efek Lebarannya dua kali lebih kuat.
"""),
    code(SETUP),
    code("""
from fedprob.data import get_scenario, simulate
from fedprob.experiment import run_experiment_cached, method_label
from fedprob.plots import plot_series

sim = simulate(get_scenario("klaster"))
pd.DataFrame([{ "bank": b.name, "kelompok": b.group, "gajian": b.payday_amp, "lebaran": b.lebaran_amp,
                "pola mingguan": np.round(b.weekly, 1)} for b in sim.banks])
"""),
    code("""
fig, ax = plt.subplots(figsize=(12, 3.5))
s = slice(sim.dates.get_loc("2024-03-01"), sim.dates.get_loc("2024-05-15"))
for k in [0, 1]:
    y = sim.y[:, k]; ax.plot(sim.dates[s], (y[s] - y[s].mean()) / y[s].std(), label=f"{sim.banks[k].name} (kelompok {sim.banks[k].group})")
ax.axvline(pd.Timestamp("2024-04-10"), color="red", ls=":", label="Lebaran")
ax.set_title("Dua kelompok bank berpola berbeda"); ax.legend(fontsize=8)
"""),
    md("## Apakah server bisa menemukan kelompok tanpa melihat data?"),
    code("""
res = {s: run_experiment_cached(s, cache_dir="../results", verbose=False) for s in ["normal", "heterogen", "klaster"]}
for s, r in res.items():
    print(f"{s:10s} klaster ditemukan: {r.extras['cluster_labels']}   kelompok sebenarnya: {[b.group for b in r.sim.banks]}")
"""),
    code("""
r = res["klaster"]
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(r.extras["cluster_similarity"], cmap="Blues", vmin=-1, vmax=1)
names = [b.name for b in r.sim.banks]
ax.set_xticks(range(8), names, rotation=45); ax.set_yticks(range(8), names)
ax.set_title("Cosine similarity update bobot antar bank"); plt.colorbar(im)
"""),
    md("""
Pada `klaster`, bank genap (A, C, E, G) dan ganjil (B, D, F, H) jelas membentuk dua blok. Server menemukan
kelompok yang **tepat** hanya dari update bobot. Pada `normal` dan `heterogen` tidak ada struktur kelompok
(silhouette < 0.3), sehingga Clustered FL dengan benar kembali menjadi FedAvg biasa.
"""),
    md("## Perbandingan CRPS"),
    code("""
M = ["local", "fedavg", "fedprox", "fedavg_ft", "clustered", "central", "oracle"]
pd.DataFrame({s: r.summary().loc[M, "CRPS"] for s, r in res.items()}).rename(index=method_label).style.format("{:.3f}").highlight_min(axis=0, subset=pd.IndexSlice[[method_label(m) for m in M[:-2]], :], color="#cde7cd")
"""),
    md("## Sensitivitas FedProx terhadap mu"),
    code("""
rows = []
for s in ["normal", "heterogen", "klaster"]:
    for mu in [0.01, 0.1, 1.0]:
        r = run_experiment_cached(s, methods=("fedprox",), fedprox_mu=mu, cache_dir="../results", verbose=False)
        rows.append({"skenario": s, "mu": mu, **r.summary().loc["fedprox", ["CRPS", "Coverage80", "Lebar80"]]})
pd.DataFrame(rows).pivot_table(index="mu", columns="skenario", values=["CRPS", "Coverage80"])
"""),
    md("""
## Kesimpulan
- **Clustered FL** menemukan struktur kelompok yang sebenarnya hanya dari update bobot, dan pada `klaster`
  mengalahkan FedAvg maupun Local. Bila tidak ada struktur, ia otomatis kembali menjadi FedAvg.
- **Fine-tune lokal** adalah cara paling sederhana dan konsisten untuk menangani heterogenitas acak (`heterogen`).
- **FedProx** sensitif terhadap mu: mu=0.1 sedikit membantu, sedangkan mu=1.0 terlalu mengekang (CRPS buruk,
  interval terlalu lebar).
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["08_diffusion"] = [
    md("""
# 08 · Diffusion Model: Membangkitkan Skenario Masa Depan

Model kuantil memberi rentang **per hari**. Banyak pertanyaan risiko justru menyangkut **seluruh jalur**:

> *"Berapa peluang saldo turun lebih dari 5% **kapan pun** dalam 14 hari ke depan?"*
> *"Seberapa rendah titik terendah saldo pada 5% skenario terburuk?"*

Untuk itu dibutuhkan model yang membangkitkan **jalur lengkap**, yaitu model generatif. Di sini dipakai
**DDPM bersyarat** (Ho dkk., 2020; mirip TimeGrad/CSDI):

1. **Training:** jalur masa depan asli diberi noise sebanyak *k* langkah, lalu jaringan belajar menebak noise
   itu dengan syarat histori dan kalender.
2. **Sampling:** mulai dari noise murni, lalu dibersihkan 50 langkah, dan diulang 200 kali sehingga
   menghasilkan 200 skenario.

Model ini juga dilatih secara **federated**: arah riset ke-2 dan ke-3 di essay digabung menjadi satu.

Dua pelajaran teknis yang ditemukan saat membangun model ini (detail di `PROGRESS.md`):
- Tanpa **x0-clipping** saat sampling, galat menumpuk dan jalur "melayang" (CRPS memburuk 0.22 → 0.35).
- *Loss* menebak noise **tidak selaras** dengan kualitas forecast, sehingga epoch/ronde terbaik dipilih
  dengan pinball loss dari sampel.
"""),
    code(SETUP),
    code("""
from fedprob.data import oracle_paths
from fedprob.experiment import run_experiment_cached, method_label
from fedprob.metrics import independent_paths, path_min_metrics
from fedprob.plots import plot_fan, plot_history, plot_paths

DM = ("oracle", "fedavg", "local", "diffusion_fed", "diffusion_central", "diffusion_fed+aci")
res = {s: run_experiment_cached(s, methods=DM, cache_dir="../results", verbose=False) for s in ["normal", "data_langka"]}
pd.concat({s: r.summary()[["MAE", "CRPS", "Coverage80", "Lebar80"]] for s, r in res.items()}, axis=1).rename(index=method_label)
"""),
    code("""
plot_history({k: v for k, v in res["normal"].histories.items() if k in ("diffusion_fed",)},
             title="Diffusion federated: pinball loss validasi (dari sampel) per ronde")
"""),
    md("## Skenario masa depan dari satu titik waktu"),
    code("""
r = res["normal"]; k = 2; c = r.clients[k]
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, oi in zip(axes, [7, 40]):
    o = int(c.test.origins[oi])
    plot_paths(r.sim, k, o, c.denorm(r.extras["diffusion_fed_paths"][k][oi]), ax=ax,
               title=f"{c.name}, origin {r.sim.dates[o].date()}")
plt.tight_layout()
"""),
    md("""
Setiap garis biru adalah satu skenario yang mungkin. Garis merah adalah **5% skenario terburuk** menurut
titik terendahnya, bahan untuk *stress testing*. Pola mingguan tetap terlihat di setiap jalur karena model
mempelajari struktur jalur, bukan hanya rentang per hari.
"""),
    md("""
## Mengapa jalur bersama penting?
Pembanding: membangkitkan jalur dari model kuantil (FedAvg) dengan **menarik tiap hari secara independen**.
Distribusi per harinya sama, tetapi keterkaitan antar hari hilang.
"""),
    code("""
rows = []
for s, r in res.items():
    for c in r.clients:
        last = c.test.x_hist[:, -1]
        orc = np.stack([(oracle_paths(r.sim, c.bank, int(o), 14, 500, seed=int(o)) - c.center) / c.scale for o in c.test.origins])
        for name, S in [("Oracle", orc),
                        ("Diffusion (federated)", r.extras["diffusion_fed_paths"][c.bank]),
                        ("Diffusion (centralized)", r.extras["diffusion_central_paths"][c.bank]),
                        ("FedAvg kuantil, hari independen", independent_paths(r.preds["fedavg"][c.bank]))]:
            rows.append({"skenario": s, "metode": name, **path_min_metrics(S, c.test.y, last)})
path_tab = pd.DataFrame(rows).groupby(["skenario", "metode"], sort=False).mean()
path_tab
"""),
    md("""
**Cara membaca:**
- **Coverage80_min**: seberapa sering titik terendah aktual jatuh dalam interval 80% prediksi (ideal ≈ Oracle).
  Pendekatan "hari independen" jauh di bawah target karena mengabaikan bahwa hari-hari yang berdekatan saling
  terkait.
- **Brier_turun** & **P_turun_pred** vs **Frek_turun_aktual**: kualitas peluang "turun > 5%". Bandingkan
  peluang rata-rata yang diprediksi dengan frekuensi aktual: apakah model meremehkan risiko?
"""),
    md("## Skenario stres pada bank dengan data langka"),
    code("""
r = res["data_langka"]; k = 6; c = r.clients[k]; oi = 20; o = int(c.test.origins[oi])
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
plot_paths(r.sim, k, o, c.denorm(r.extras["diffusion_fed_paths"][k][oi]), ax=axes[0], title=f"{c.name}: Diffusion federated")
orc = oracle_paths(r.sim, k, o, 14, 100, seed=o)
plot_paths(r.sim, k, o, orc, ax=axes[1], title=f"{c.name}: Oracle (proses sebenarnya)")
plt.tight_layout()
"""),
    md("""
## Kesimpulan
- Forecast per hari: diffusion federated mencapai CRPS **0.190** pada `normal` (FedAvg 0.203, Local 0.195), dan
  diffusion centralized **0.181** pada `data_langka`. Diffusion sekaligus memberi **jalur skenario lengkap**
  yang tidak bisa diberikan model kuantil.
- Pertanyaan tingkat jalur (titik terendah 14 hari): coverage 80% jalur diffusion **69–79%**, sedangkan jalur
  "hari independen" hanya **48–49%** (Oracle 76%).
- **Batasan jujur:** diffusion **federated** meremehkan peluang "turun > 5%" (memprediksi 9–10%, aktual 18–20%),
  sedangkan versi centralized lebih baik (13–14%). Ekor distribusi masih menjadi titik lemah, dan federasi
  memperberatnya. Model juga tidak pernah melihat shock, sehingga skenario terburuknya belum mencakup
  krisis yang belum pernah terjadi. Arah lanjutannya adalah *conditional generation* dengan variabel stres
  eksplisit dan data historis krisis.
"""),
]


# ---------------------------------------------------------------------------
# STUDI KASUS DATA RIIL: KasPintar (pengisian kas ATM)
# ---------------------------------------------------------------------------
NOTEBOOKS["09_kasus_atm_data"] = [
    md("""
# 09 · Studi Kasus Nyata: KasPintar, Sistem Rekomendasi Pengisian Kas ATM

## Masalah
Bank mengisi ATM secara berkala lewat jasa pengangkutan uang (*cash-in-transit*), misalnya **setiap Senin**.
Jumlah yang diisi harus cukup sampai Senin berikutnya.

| Jika diisi… | Akibatnya |
|---|---|
| **terlalu sedikit** | ATM kosong → nasabah kecewa, kunjungan darurat mahal, reputasi turun (di Indonesia paling terasa menjelang **Lebaran**) |
| **terlalu banyak** | uang menganggur di mesin → biaya dana, asuransi, risiko keamanan |

Praktik umum: **"rata-rata 4 minggu terakhir × 1,2"**. Ini satu angka tanpa ukuran risiko.
Selain itu, data transaksi tiap bank **rahasia**, dan **ATM/bank baru** belum punya histori.

> **Pertanyaan bisnis:** berapa uang yang harus diisi ke tiap ATM minggu ini agar peluang kehabisan
> ≤ 5%, dengan uang menganggur seminimal mungkin?

Pertanyaan ini membutuhkan ketiga hal dalam pertanyaan riset: **akurat**, **jujur soal ketidakpastian**,
dan **belajar tanpa berbagi data**.

## Data riil: NN5
Penarikan tunai harian dari **111 ATM di Inggris** (18 Mar 1996 – 17 Mei 1998), dari kompetisi forecasting
NN5. Sumber: Zenodo (Monash Forecasting Archive), lisensi **CC BY 4.0**. Nilai sudah diskalakan oleh
penyelenggara, jadi di sini disebut "unit kas".
"""),
    code(SETUP),
    code("""
from fedprob.realdata.nn5 import load_nn5, holiday_effect, UK_HOLIDAYS
d = load_nn5()
df = d.to_frame()
print(f"{d.y.shape[1]} ATM x {d.y.shape[0]} hari: {d.dates[0].date()} s/d {d.dates[-1].date()}")
print(f"nilai hilang: {np.isnan(d.raw).mean():.1%}, nilai nol: {(d.raw == 0).mean():.1%} -> diimputasi (median hari sama 4 minggu terakhir)")
df.iloc[:5, :6]
"""),
    md("## Pola: mingguan, hari libur, dan variasi antar ATM"),
    code("""
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for j in [0, 10, 50]:
    axes[0, 0].plot(df.index, df.iloc[:, j].rolling(7).mean(), lw=1, label=d.names[j])
axes[0, 0].set_title("Rata-rata 7 hari, 3 ATM"); axes[0, 0].legend(fontsize=8)
dow = df.groupby(df.index.dayofweek).mean().mean(axis=1)
axes[0, 1].bar(["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"], dow.values, color="#2f6fdf")
axes[0, 1].set_title("Rata-rata penarikan per hari (semua ATM)")
s = df.loc["1997-12-01":"1998-01-10"].mean(axis=1)
axes[1, 0].plot(s.index, s.values, marker="o", ms=3)
axes[1, 0].axhline(df.values.mean(), color="grey", ls="--", label="rata-rata harian")
axes[1, 0].set_title("Natal & Tahun Baru 1997: lonjakan sebelum libur"); axes[1, 0].legend(fontsize=8)
axes[1, 0].tick_params(axis="x", rotation=45)
wk = df.resample("W-SUN").sum().iloc[1:-1]
axes[1, 1].hist([(df.std() / df.mean()).values, (wk.std() / wk.mean()).values], bins=20, label=["harian", "mingguan"])
axes[1, 1].set_title("Variasi (CV) per ATM: total mingguan jauh lebih stabil"); axes[1, 1].legend()
plt.tight_layout()
"""),
    md("""
**Pengamatan:**
- Penarikan memuncak **Kamis–Jumat** (persiapan akhir pekan). Senin dan Sabtu paling rendah.
- **Menjelang Natal penarikan melonjak** lebih dari 2× rata-rata. Efek hari besar ini analog dengan Lebaran di Indonesia.
- Variasi harian tinggi (CV ~40%), tetapi **total mingguan jauh lebih stabil** (CV ~15%). Karena itu sistem
  memprediksi **total kumulatif**, bukan hanya nilai harian.
"""),
    md("""
## Dari forecast ke keputusan: target kumulatif
Karena penarikan selalu ≥ 0, total kumulatif hanya bisa naik, sehingga:

$$\\text{ATM kehabisan dalam siklus} \\iff \\sum_{i=1}^{7} y_{t+i} > \\text{isi}$$

Maka **isi untuk service level 95% = kuantil-95% dari total penarikan 7 hari**. Model memprediksi kuantil
total kumulatif 1–14 hari secara langsung, dalam satuan "hari permintaan rata-rata" ATM tersebut.
"""),
    md("""
## Setting federated: 6 bank fiktif
NN5 tidak menyebut pemilik ATM, sehingga 111 ATM dibagi **acak tetapi deterministik** ke 6 bank. Setiap bank
adalah satu klien federated, dan data ATM-nya tidak keluar dari bank. **Bank Zeta** diperlakukan sebagai
**bank baru**: ATM-nya hanya punya histori 12 minggu sebelum periode validasi.
"""),
    code("""
from fedprob.cash.pipeline import ATMConfig, build_atm_clients
cfg = ATMConfig()
atms, vs, ts = build_atm_clients(d, cfg)
pd.DataFrame([{"bank": n, "jumlah ATM": s, "ukuran": u, "histori train (hari)": vs - (vs - cfg.cold_history if b == cfg.cold_bank else 0)}
              for b, (n, s, u) in enumerate(d.banks)])
"""),
    code("""
print("train   :", d.dates[0].date(), "s/d", d.dates[vs - 1].date())
print("validasi:", d.dates[vs].date(), "s/d", d.dates[ts - 1].date(), f"({cfg.val_days} hari)")
print("uji     :", d.dates[ts].date(), "s/d", d.dates[-1].date(), f"({cfg.test_days} hari: Natal, Tahun Baru, Paskah 1998)")
"""),
]

NOTEBOOKS["10_kasus_atm_model"] = [
    md("""
# 10 · KasPintar: Memilih & Mengkalibrasi Model

Kandidat model diambil dari temuan tahap simulasi, lalu **diputuskan oleh data riil**:

| Komponen | Kandidat | Cara memilih |
|---|---|---|
| Forecaster | QuantileMLP (9 kuantil, 5%–99%) untuk total kumulatif 1–14 hari | – |
| Kolaborasi | Local, **FedAvg**, FedProx, FedProx+fine-tune, Clustered FL, (Centralized = batas atas) | **CRPS validasi** |
| Kejujuran | ACI satu sisi per ATM vs **per bank** | coverage di periode uji |

Diffusion **tidak dipakai**: keputusan isi cukup dijawab kuantil total kumulatif, dan di simulasi diffusion
federated meremehkan risiko ekor.
"""),
    code(SETUP),
    code("""
from fedprob.cash.pipeline import run_atm_case_cached, label, Q_ATM, SERVICE_LEVELS
from fedprob.cash.report import plot_cum_fan
from fedprob.conformal import aci_upper
res = run_atm_case_cached(cache_dir="../results", verbose=False)
print("Sistem KasPintar memakai:", label(res.system))
"""),
    md("## Seleksi model: CRPS validasi vs uji (total 7 hari, lebih kecil lebih baik)"),
    code("""
test_crps = res.metrics.groupby("method", sort=False)["CRPS"].mean()
tab = pd.DataFrame({"CRPS validasi": pd.Series(res.extras["val_crps"]), "CRPS uji": test_crps})
tab.rename(index=label).style.format("{:.3f}").highlight_min(axis=0, color="#cde7cd")
"""),
    md("""
**Pelajaran:** di simulasi FedProx yang terbaik, tetapi di data riil **FedAvg** terbaik di antara metode federated
(validasi maupun uji), dan hampir menyamai Centralized. Simulasi memberi hipotesis, data riil yang memutuskan.
Clustered FL tidak menemukan struktur kelompok (silhouette < 0.3), sehingga perilakunya sama dengan FedAvg.
"""),
    code("""
print("silhouette Clustered FL:", {k: round(v, 2) for k, v in res.extras.get("cluster_silhouette", {}).items()})
"""),
    md("## Siapa yang paling diuntungkan kolaborasi? CRPS per bank"),
    code("""
m = res.metrics[res.metrics.method.isin(["local", "fedavg", "central"])]
piv = m.groupby(["bank", "method"]).CRPS.mean().unstack()[["local", "fedavg", "central"]]
piv["perbaikan FedAvg vs Local"] = 1 - piv["fedavg"] / piv["local"]
piv.style.format({"local": "{:.3f}", "fedavg": "{:.3f}", "central": "{:.3f}", "perbaikan FedAvg vs Local": "{:.0%}"})
"""),
    md("## Kalibrasi: apakah kuantil atas bisa dipercaya?"),
    md("""
Proporsi total 7 hari aktual ≤ kuantil-s (semua origin harian periode uji). Idealnya sama dengan s.
"""),
    code("""
sel = res.extras["selected"]
cal = res.metrics[res.metrics.method == sel][[c for c in res.metrics.columns if c.startswith("atas")]].mean()
pd.DataFrame({"tanpa ACI": [cal[f"atas{int(l*100)}"] for l in SERVICE_LEVELS],
              "ACI per bank": [cal[f"atas{int(l*100)}_aci"] for l in SERVICE_LEVELS]},
             index=[f"{l:.0%}" for l in SERVICE_LEVELS]).style.format("{:.1%}")
"""),
    md("""
### Hasil negatif: ACI per ATM gagal
ACI per ATM hanya punya ~30 skor terbaru yang saling tumpang tindih (≈ 4 minggu independen). Kuantil 98% tidak
mungkin diestimasi dari data sesedikit itu. Menggabungkan skor **semua ATM dalam satu bank** (data tetap di
bank) menyelesaikan masalah ini.
"""),
    code("""
from fedprob.cash.policy import CYCLE
rows = []
for lv in (0.95, 0.98):
    qi = Q_ATM.index(lv)
    per_atm = [aci_upper(a.val.y, pv[..., qi], a.val.origins, a.test.y, pt[..., qi], a.test.origins, lv)
               for a, pv, pt in zip(res.atms, res.preds_val[sel], res.preds[sel])]
    cov = lambda adj: np.mean([(a.test.y[:, CYCLE - 1] <= q[:, CYCLE - 1]).mean() for a, q in zip(res.atms, adj)])
    rows.append({"target": f"{lv:.0%}", "tanpa ACI": cov([p[..., qi] for p in res.preds[sel]]),
                 "ACI per ATM": cov(per_atm), "ACI per bank": cov(res.aci[sel][lv])})
pd.DataFrame(rows).set_index("target").style.format("{:.1%}")
"""),
    md("## Contoh forecast: total penarikan kumulatif 14 hari"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
plot_cum_fan(res, 0, 2, ax=axes[0])     # minggu biasa
plot_cum_fan(res, 0, 3, ax=axes[1])     # minggu menjelang Natal
plt.tight_layout()
"""),
    md("""
Pita biru adalah rentang kemungkinan total penarikan. Garis putus-putus biru adalah **isi rekomendasi**
(kuantil 95% total 7 hari setelah ACI), garis titik oranye adalah aturan praktis, dan garis hitam adalah
penarikan aktual. Pada minggu menjelang Natal, rentangnya otomatis melebar dan rekomendasi isi naik.
"""),
]

NOTEBOOKS["11_kasus_atm_keputusan"] = [
    md("""
# 11 · KasPintar: Dari Forecast ke Keputusan (Backtest 23 Minggu)

Simulasi operasional: setiap Minggu malam sistem membuat rekomendasi, Senin pagi ATM diisi, lalu dicek
apakah ATM kehabisan sebelum Senin berikutnya. Periode uji: **Des 1997 – Mei 1998**, termasuk Natal, Tahun Baru,
dan Paskah.

| Kebijakan | Keterangan |
|---|---|
| Aturan praktis | rata-rata 4 minggu terakhir × faktor (1,2 = praktik umum) |
| Seasonal naive | "minggu lalu" + kuantil residual |
| Local | QuantileMLP per bank, tanpa kolaborasi |
| FedAvg | federated, tanpa kalibrasi |
| **KasPintar** | FedAvg + ACI per bank |
| Centralized + ACI | batas atas (data digabung, tidak realistis) |
"""),
    code(SETUP),
    code("""
from fedprob.cash.pipeline import run_atm_case_cached, label
from fedprob.cash.report import (idle_at_same_stockout, plot_cum_fan, plot_tradeoff, tradeoff_curve,
                                 weekly_report, weekly_stockout)
res = run_atm_case_cached(cache_dir="../results", verbose=False)
SYS = res.system
P = ["aturan_praktis", "seasonal_naive", "local", "fedavg", SYS, "central+aci"]
"""),
    md("## Hasil backtest pada target service level 95%"),
    code("""
bt = pd.DataFrame({label(m): res.backtest(m, level=0.95, factor=1.2) for m in P}).T
bt[["kehabisan_%", "menganggur_%", "isi_rata2", "kurang_rata2"]].style.format("{:.1f}")
"""),
    md("""
- **kehabisan_%**: persentase siklus (ATM × minggu) di mana ATM kosong sebelum pengisian berikutnya. Target ≤ 5%.
- **menganggur_%**: total uang yang tersisa di mesin, relatif terhadap total penarikan.
"""),
    md("## Trade-off: kehabisan vs uang menganggur"),
    code("""
fig, ax = plt.subplots(figsize=(8, 5))
plot_tradeoff(res, ["aturan_praktis", "local", "fedavg", SYS], ax=ax)
ax.axvline(5, color="grey", ls=":", lw=1)
"""),
    code("""
cmp = idle_at_same_stockout(res, SYS, 0.95)
print(f"Pada tingkat kehabisan yang sama ({cmp['kehabisan_%']:.1f}%):")
print(f"  aturan praktis butuh faktor ~x{cmp['faktor_aturan_setara']:.2f} dan uang menganggur {cmp['menganggur_aturan_%']:.1f}% dari permintaan")
print(f"  KasPintar: uang menganggur {cmp['menganggur_sistem_%']:.1f}%  ->  hemat {cmp['penghematan_relatif_%']:.0f}% uang menganggur")
"""),
    md("## Minggu per minggu: apa yang terjadi saat Natal & Paskah?"),
    code("""
ws = pd.DataFrame({label(m): weekly_stockout(res, m) for m in ["aturan_praktis", "fedavg", SYS]})
ax = ws.plot(marker="o", figsize=(12, 4))
for h in ["1997-12-25", "1998-04-10"]:
    ax.axvline(pd.Timestamp(h), color="red", ls=":", lw=1)
ax.axhline(5, color="grey", ls="--", lw=1); ax.set_ylabel("% ATM kehabisan"); ax.set_title("ATM kehabisan per minggu (garis merah: Natal, Jumat Agung)")
"""),
    md("## Bank baru (cold-start): manfaat kolaborasi"),
    code("""
rows = []
for b in range(len(res.data.banks)):
    atms = res.data.atms_of(b)
    for m in ["aturan_praktis", "local+aci", SYS]:
        r = res.backtest(m, level=0.95, atms=atms)
        rows.append({"bank": res.bank_name(b), "kebijakan": label(m), "kehabisan_%": r["kehabisan_%"], "menganggur_%": r["menganggur_%"]})
pd.DataFrame(rows).pivot(index="bank", columns="kebijakan").round(1)
"""),
    md("## Laporan Senin untuk operator kas (contoh)"),
    code("""
rep = weekly_report(res, bank=5, week=3)
print("Bank Zeta, minggu", rep.attrs["minggu"])
rep.style.format({c: "{:.1f}" for c in rep.columns if rep[c].dtype.kind == "f" and "P(" not in c}).format({"P(habis) jika aturan praktis": "{:.0%}"})
"""),
    md("""
Kolom **P(habis) jika aturan praktis** menunjukkan ATM mana yang berisiko bila bank tetap memakai cara lama.
Kolom *aktual* hanya ada di backtest; dalam operasi nyata, operator hanya melihat rekomendasi dan peluangnya.
"""),
    md("## Ilustrasi biaya (asumsi bisa diubah)"),
    code("""
from fedprob.cash.policy import total_cost
for cof, pen in [(0.15, 30), (0.15, 60), (0.30, 30)]:
    c = {label(m): total_cost(res.backtest(m, 0.95), cof, pen) for m in ["aturan_praktis", "local", SYS]}
    print(f"biaya simpan {cof:.0%}/thn, penalti kehabisan {pen} unit:", {k: round(v, 2) for k, v in c.items()})
"""),
    md("""
## Kesimpulan studi kasus
*(Angka pasti lihat tabel di atas dan `PROGRESS.md`.)*

1. **KasPintar menepati janjinya.** Pada target 95%, tingkat kehabisan mendekati 5%, termasuk melewati Natal dan
   Paskah, berkat ACI per bank. Model tanpa kalibrasi dan aturan praktis kehabisan ~2× lebih sering.
2. **Lebih efisien.** Pada tingkat kehabisan yang sama, uang menganggur jauh lebih kecil dibanding aturan praktis.
3. **Kolaborasi paling menolong bank baru.** ATM bank Zeta (histori 12 minggu) jauh lebih jarang kehabisan
   dibanding bila bank itu memodelkan sendiri.
4. **Pelajaran metodologis:** (a) model terbaik di simulasi (FedProx) bukan yang terbaik di data riil (FedAvg), sehingga
   seleksi harus berbasis validasi; (b) kalibrasi per ATM gagal karena sampel kecil, sedangkan per bank berhasil.

**Keterbatasan:** data ATM Inggris 1996–1998 (bukan Indonesia), pembagian ke bank bersifat fiktif, satuan nilai
tidak diketahui, satu siklus pengisian tetap (mingguan), dan backtest bukan uji coba operasional.
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
