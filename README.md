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

Komponen terbaik dari simulasi lalu diterapkan pada **data riil** sebagai alat end-to-end:
**KasPintar**, rekomendasi pengisian kas ATM (data NN5, 111 ATM). Lihat [`docs/kasus_atm.md`](docs/kasus_atm.md).

## Studi kasus data riil: KasPintar
| Hasil (uji 23 minggu × 111 ATM, target 95%) | Angka kunci |
|---|---|
| Kehabisan kas: aturan praktis ×1.2 vs KasPintar (FedAvg + ACI) | 11.4% → **5.4%** |
| Uang menganggur pada tingkat kehabisan yang sama | 29.3% → **20.0%** (−32%) |
| Minggu belanja Natal (15 Des 1997), ATM kehabisan | 75% → **6%** |
| Bank baru (histori 12 minggu): CRPS Local → federated | 1.592 → **0.319** (−80%) |
| Coverage target 95%: tanpa ACI → ACI per bank | 88.9% → **94.3%** |

## Temuan utama (simulasi)
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
| `09_kasus_atm_data` | studi kasus: masalah ATM, eksplorasi data NN5, pembagian bank |
| `10_kasus_atm_model` | forecaster kumulatif federated, seleksi model, ACI per bank |
| `11_kasus_atm_keputusan` | backtest kebijakan isi, trade-off, minggu Natal, bank baru, biaya |

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

python -m pytest -q                       # sanity check (25 test)
streamlit run app/dashboard.py            # dashboard demo (simulasi)
streamlit run app/kaspintar.py            # alat operator kas ATM (data riil NN5)
jupyter lab notebooks/                    # notebook tahapan
python scripts/export_figures.py          # gambar presentasi -> docs/figures/
python scripts/export_figures.py --kasus  # gambar studi kasus ATM
```
Data NN5 diunduh otomatis ke `data/raw/` saat pertama kali dipakai. Pipeline kasus ATM ~14 menit di CPU.
Eksekusi seluruh notebook dari nol membutuhkan ~1 jam di CPU (hasil di-cache di `results/`).

## Struktur
```
src/fedprob/   simulator, skenario, model (kuantil, diffusion), federated (FedAvg/FedProx/clustered),
               conformal, metrik, eksperimen, plot
  realdata/    pemuat data NN5
  cash/        kebijakan isi ATM, backtest, pipeline & laporan KasPintar
notebooks/     01-11 (dibangkitkan dari scripts/build_notebooks.py)
app/           dashboard.py (simulasi), kaspintar.py (kasus ATM)
docs/          alur presentasi, studi kasus ATM, gambar
tests/         unit test
results/       cache hasil eksperimen (tidak di-commit)
```

## Atribusi data
Studi kasus memakai **NN5 Daily Dataset** (kompetisi forecasting NN5) dari Monash Time Series Forecasting
Archive: Godahewa dkk. (2021), Zenodo record [4656110](https://zenodo.org/records/4656110), lisensi
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data diimputasi (nilai hilang/nol) dan ATM dibagi
ke bank fiktif. Data mentah tidak disertakan di repo.
