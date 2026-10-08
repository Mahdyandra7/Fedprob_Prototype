# CLAUDE.md — Konteks Project (baca ini dulu di setiap sesi)

## Tentang project
Prototipe awal riset **Time Series AI untuk sektor keuangan**, turunan dari essay LPDP
pemilik project (`LPDP_26.docx`, dokumen pribadi, **tidak** di-commit). Pertanyaan riset:

> "How can we build models that are accurate and honest about their own uncertainty and
> capable of learning from data that can never be fully shared?"

Tiga arah riset di essay: (1) probabilistic forecasting, (2) diffusion models,
(3) federated learning. **Prototipe ini hanya mencakup (1) + (3).** Diffusion ditunda.

Keputusan yang sudah diambil pemilik (jangan ditanyakan ulang):
- Data = **simulasi dengan skenario**, project baru (tidak terkait skripsi ARDL/DPK/makro).
- Output = notebook bertahap (bahasa Indonesia) + dashboard Streamlit untuk demo presentasi.
- Tenggat presentasi belum pasti, jadi pengembangan dilakukan bertahap.
- Progres harus selalu dicatat di `PROGRESS.md` dan di-commit ke GitHub.

## Struktur
```
src/fedprob/
  data/simulator.py   simulasi saldo harian K bank: tren+mingguan+gajian+Lebaran
                      + faktor pasar AR(1) + idiosinkratik AR(1) + noise; oracle_quantiles()
  data/scenarios.py   preset: normal, heterogen, data_langka, shock, bank_baru
  data/windows.py     (lookback=56, horizon=14) + fitur kalender; split waktu train/val/test
  models/quantile_net.py  QuantileMLP (kuantil tidak saling silang) + pinball loss
  models/baselines.py     seasonal naive & ETS dengan interval
  training.py         training lokal/terpusat + early stopping
  federated/fedavg.py FedAvg / FedProx (mu) manual + fine_tune (personalisasi)
  metrics.py          CRPS (aprox), pinball, coverage, lebar interval, reliability
  experiment.py       run_experiment(scenario) -> semua metode + tabel metrik; versi cached
  plots.py            matplotlib (notebook) & plotly (dashboard)
notebooks/01..05      tahapan belajar/eksperimen
app/dashboard.py      demo Streamlit
tests/test_core.py    sanity check
```

## Konvensi
- Metrik dihitung pada **skala ternormalisasi** per bank (dibagi std train) agar adil antar bank.
- Kuantil tetap: `fedprob.QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)`.
- Metode: `oracle, seasonal_naive, ets, local, fedavg, fedprox, fedavg_ft, central`.
- Komentar & teks notebook berbahasa Indonesia, nama kode berbahasa Inggris.
- Notebook dibangkitkan oleh `scripts/build_notebooks.py`. Edit script itu, lalu bangkitkan ulang.

## Perintah
```
.venv\Scripts\activate
python -m pytest -q
python scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
streamlit run app/dashboard.py
```

## Lingkungan
Windows, Python 3.14 (venv `.venv`), GPU GTX 1650 (tidak wajib; model kecil, CPU cukup).
GitHub CLI (`gh`) tidak terpasang. Push memakai git biasa ke remote `origin` =
https://github.com/Mahdyandra7/Fedprob_Prototype (branch `main`). Setelah tiap tahap: update PROGRESS.md, commit, `git push`.
