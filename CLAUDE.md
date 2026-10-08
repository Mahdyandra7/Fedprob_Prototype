# CLAUDE.md — Konteks Project (baca ini dulu di setiap sesi)

## Tentang project
Prototipe awal riset **Time Series AI untuk sektor keuangan**, turunan dari essay LPDP
pemilik project (`LPDP_26.docx`, dokumen pribadi, **tidak** di-commit). Pertanyaan riset:

> "How can we build models that are accurate and honest about their own uncertainty and
> capable of learning from data that can never be fully shared?"

Tiga arah riset di essay: (1) probabilistic forecasting, (2) diffusion models,
(3) federated learning. **Ketiganya sudah ada di prototipe** (tahap 0–9 selesai, 2026-10-08).

Keputusan yang sudah diambil pemilik (jangan ditanyakan ulang):
- Data = **simulasi dengan skenario**, project baru (tidak terkait skripsi ARDL/DPK/makro).
- Output = notebook bertahap (bahasa Indonesia) + dashboard Streamlit untuk demo presentasi.
- Tenggat presentasi belum pasti, jadi pengembangan dilakukan bertahap.
- Progres harus selalu dicatat di `PROGRESS.md`, di-commit, dan di-push ke GitHub.
- **Studi kasus data riil (tahap 10, selesai):** "KasPintar", rekomendasi pengisian kas ATM pada data NN5
  (111 ATM Inggris, CC BY 4.0) → 6 bank fiktif, Zeta = bank baru. Ini **inti presentasi wawancara**;
  simulasi diposisikan sebagai laboratorium pemilihan model. Ringkasan: `docs/kasus_atm.md`.

## Struktur
```
src/fedprob/
  data/simulator.py   simulasi saldo harian K bank: tren+mingguan+gajian+Lebaran+faktor pasar AR(1)
                      + idiosinkratik AR(1) + noise; kelompok bank (n_groups, group_alpha);
                      oracle_quantiles() / oracle_paths()
  data/scenarios.py   preset: normal, heterogen, data_langka, shock, bank_baru, klaster
  data/windows.py     (lookback=56, horizon=14) + fitur kalender; split waktu train/val/test;
                      normalisasi relatif level: z = (y - rata2) / (5% x rata2)
  models/quantile_net.py  QuantileMLP (kuantil tidak saling silang) + pinball loss
  models/diffusion.py     DiffusionForecaster (DDPM bersyarat, x0-clipping, seleksi via pinball dari sampel)
  models/baselines.py     seasonal naive & ETS dengan interval
  training.py         training lokal/terpusat + early stopping; generik terhadap model
                      (model harus punya training_loss / eval_loss / predict_quantiles)
  federated/fedavg.py     FedAvg / FedProx (mu) + fine_tune; model_fn & init_state opsional
  federated/clustered.py  Clustered FL (cosine similarity update, silhouette >= 0.3)
  conformal.py        CQR statik, ACI online (gamma=0.01, window=30), online_bias_shift (hasil negatif)
  metrics.py          CRPS, pinball, coverage, reliability + metrik jalur (crps_samples,
                      path_min_metrics, independent_paths)
  experiment.py       run_experiment(scenario, methods) -> ExperimentResult; versi cached
  plots.py            matplotlib (notebook) & plotly (dashboard), termasuk plot_paths
  realdata/nn5.py     unduh NN5 (Zenodo, cek MD5) -> data/raw/, parser .tsf, imputasi, libur UK, ATM -> 6 bank
  cash/policy.py      aturan praktis, backtest siklus mingguan (CYCLE=7), biaya ilustratif
  cash/pipeline.py    run_atm_case_cached: target kumulatif, QuantileMLP(residual=False, 9 kuantil),
                      kandidat federated dipilih via CRPS validasi, ACI satu sisi digabung per bank
                      (conformal.aci_upper_pooled). Cache results/atm_<hash>.pkl (CACHE_VERSION)
  cash/report.py      laporan Senin, trade-off, plot fan kumulatif, kehabisan per minggu
notebooks/01..11      dibangkitkan dari scripts/build_notebooks.py (09-11 = kasus ATM)
scripts/export_figures.py  gambar presentasi -> docs/figures/ (--kasus untuk kasus_*.png)
docs/alur_presentasi.md    alur slide (berpusat pada kasus ATM) + persiapan tanya-jawab
docs/kasus_atm.md          studi kasus: masalah -> solusi -> hasil -> keterbatasan
app/dashboard.py      demo Streamlit simulasi (5 tab)
app/kaspintar.py      alat operator kas ATM (4 tab)
tests/                test_core.py, test_extensions.py, test_atm.py (25 test)
```

## Konvensi
- Metrik dihitung pada **skala ternormalisasi** per bank (1 unit = 5% saldo rata-rata).
- Kuantil tetap: `fedprob.QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)`; pita di `fedprob.BANDS`.
- Nama metode: `oracle, seasonal_naive, ets, local, fedavg, fedprox, fedavg_ft, clustered, central,
  diffusion_fed, diffusion_central`; conformal sebagai sufiks `+cqr` / `+aci` (mis. `fedavg+aci`).
  Label tampilan: `experiment.method_label(m)`.
- Default FedProx mu = 0.1 (hasil sweep notebook 07).
- Cache hasil di `results/exp_<hash>.pkl`. Hash mencakup config, methods, dan semua config training.
  Mengubah default akan memaksa hitung ulang.
- Skenario lama harus tetap identik: efek kelompok memakai RNG terpisah (`seed + 1000`).
- Komentar & teks notebook berbahasa Indonesia, nama kode berbahasa Inggris.
- Notebook: edit `scripts/build_notebooks.py`, bangkitkan ulang, lalu eksekusi.
  **Perhatian:** skrip menulis ulang SEMUA notebook (output hilang). Kembalikan yang tidak diubah dengan
  `git checkout -- notebooks/<nama>.ipynb`.
- Kasus ATM: metrik pada skala kumulatif ternormalisasi per ATM (1 unit = permintaan harian rata-rata).
  Nama kebijakan tambahan: `aturan_praktis`; label via `cash.pipeline.label(m)`.

## Perintah
```
.venv\Scripts\activate
python -m pytest -q
python scripts/build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/*.ipynb
python scripts/export_figures.py
python scripts/export_figures.py --kasus
streamlit run app/dashboard.py
streamlit run app/kaspintar.py
```
Waktu: eksperimen default ~3 menit/skenario; diffusion ~4 menit; seluruh notebook dari nol ~1 jam;
pipeline kasus ATM ~14 menit. Jalankan training berat **sendirian**: bila berbarengan dengan proses lain
(notebook/streamlit), torch berebut CPU dan bisa >4x lebih lambat.

## Lingkungan
Windows, Python 3.14 (venv `.venv`, torch 2.14 CPU), GPU GTX 1650 (tidak dipakai; model kecil).
GitHub CLI (`gh`) tidak terpasang. Push memakai git biasa ke remote `origin` =
https://github.com/Mahdyandra7/Fedprob_Prototype (branch `main`). Setelah tiap tahap: update PROGRESS.md, commit, `git push`.
