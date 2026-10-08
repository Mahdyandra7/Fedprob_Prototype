# FedProb: Federated Probabilistic Forecasting (Prototipe)

> *How can we build models that are accurate and honest about their own uncertainty,
> and capable of learning from data that can never be fully shared?*

Prototipe awal riset **Time Series AI untuk sektor keuangan** yang menggabungkan tiga arah riset:

1. **Probabilistic forecasting**: model memberi rentang kemungkinan beserta peluangnya (interval 50/80/90%),
   dan **conformal prediction** menjaga interval tetap jujur.
2. **Federated learning**: beberapa bank melatih model bersama **tanpa membagikan data**; yang dikirim hanya
   bobot model (FedAvg, FedProx, Clustered FL).
3. **Diffusion model**: membangkitkan ratusan **skenario jalur** masa depan untuk analisis risiko dan stress
   testing, juga dilatih secara federated.

Karena data antar bank bersifat rahasia, prototipe ini memakai **data simulasi**: saldo dana harian 8 bank
fiktif (tren, pola mingguan, gajian, Lebaran, faktor pasar bersama, shock, kelompok bank). Keuntungannya,
distribusi sebenarnya diketahui (*Oracle*), sehingga kejujuran interval prediksi bisa diuji dengan tepat.

## Temuan utama
| # | Temuan | Angka kunci |
|---|---|---|
| 1 | Federasi paling menolong bank dengan data sedikit | CRPS bank kecil 0.215 → 0.156 (−27%), setara data digabung (0.154) |
| 2 | Clustered FL menemukan kelompok bank hanya dari update bobot | kelompok tepat 100%; CRPS 0.277 → 0.252 |
| 3 | Saat shock semua model terlalu percaya diri; ACI memperbaikinya | coverage 80%: 40% → 81% |
| 4 | Diffusion memberi skenario jalur yang lebih terkalibrasi | coverage titik terendah 14 hari: 69–79% vs 48% |

Rincian, hasil negatif, dan keterbatasan ada di [`PROGRESS.md`](PROGRESS.md). Alur presentasi ada di
[`docs/alur_presentasi.md`](docs/alur_presentasi.md).

## Notebook
| Notebook | Isi |
|---|---|
| `01_simulasi_data` | data simulasi & 6 skenario |
| `02_baseline` | baseline klasik, metrik probabilistik (CRPS, coverage), Oracle |
| `03_probabilistic` | QuantileMLP & pinball loss |
| `04_federated` | Local vs FedAvg vs Centralized |
| `05_eksperimen_skenario` | semua metode × semua skenario, 3 seed, sweep heterogenitas |
| `06_conformal` | CQR & ACI, termasuk hasil negatif koreksi bias |
| `07_heterogenitas` | FedProx (sweep mu) & Clustered FL |
| `08_diffusion` | diffusion federated, skenario jalur, metrik risiko tingkat-jalur |

## Metode
| Metode | Berbagi data? | Keterangan |
|---|---|---|
| Seasonal Naive, ETS | tidak | baseline klasik dengan interval |
| Local-only | tidak | tiap bank melatih QuantileMLP sendiri |
| FedAvg / FedProx | **tidak** | kolaborasi lewat rata-rata bobot |
| FedAvg + fine-tune | tidak | model global dipersonalisasi ke tiap bank |
| Clustered FL | tidak | satu model per kelompok bank yang ditemukan otomatis |
| `+cqr` / `+aci` | tidak | conformal prediction, dikalibrasi lokal di tiap bank |
| Diffusion (federated) | tidak | DDPM bersyarat, membangkitkan jalur skenario |
| Centralized | ya | batas atas (tidak realistis secara privasi) |
| Oracle | n/a | distribusi sebenarnya (batas terbaik) |

## Menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # sekaligus memasang paket fedprob (editable)

python -m pytest -q                       # sanity check (16 test)
streamlit run app/dashboard.py            # dashboard demo
jupyter lab notebooks/                    # notebook tahapan
python scripts/export_figures.py          # gambar presentasi -> docs/figures/
```
Eksekusi seluruh notebook dari nol membutuhkan ~1 jam di CPU (hasil di-cache di `results/`).

## Struktur
```
src/fedprob/   simulator, skenario, model (kuantil, diffusion), federated (FedAvg/FedProx/clustered),
               conformal, metrik, eksperimen, plot
notebooks/     01-08 (dibangkitkan dari scripts/build_notebooks.py)
app/           dashboard Streamlit
docs/          alur presentasi & gambar
tests/         unit test
results/       cache hasil eksperimen (tidak di-commit)
```
