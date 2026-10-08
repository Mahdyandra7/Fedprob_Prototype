# FedProb: Federated Probabilistic Forecasting (Prototipe)

> *How can we build models that are accurate and honest about their own uncertainty,
> and capable of learning from data that can never be fully shared?*

Prototipe awal riset **Time Series AI untuk sektor keuangan** yang menggabungkan dua arah riset:

1. **Probabilistic forecasting**: model tidak hanya memberi satu angka, tetapi juga rentang
   kemungkinan beserta peluangnya (interval 50/80/90%).
2. **Federated learning**: beberapa bank melatih model bersama **tanpa membagikan data**;
   yang dikirim hanya bobot model.

Karena data antar bank bersifat rahasia, prototipe ini memakai **data simulasi**: saldo dana harian
8 bank fiktif (tren, pola mingguan, gajian, Lebaran, faktor pasar bersama, shock). Keuntungannya,
distribusi sebenarnya diketahui (*Oracle*), sehingga kejujuran interval prediksi bisa diuji dengan tepat.

## Pertanyaan yang dijawab
| Pertanyaan | Di mana |
|---|---|
| Seperti apa datanya & skenarionya? | `notebooks/01_simulasi_data.ipynb` |
| Bagaimana menilai forecast probabilistik? | `notebooks/02_baseline.ipynb` |
| Bagaimana model kuantil bekerja? | `notebooks/03_probabilistic.ipynb` |
| Apakah federasi menyamai penggabungan data? | `notebooks/04_federated.ipynb` |
| Kapan federasi paling menolong? | `notebooks/05_eksperimen_skenario.ipynb` |

## Metode yang dibandingkan
| Metode | Berbagi data? | Keterangan |
|---|---|---|
| Seasonal Naive, ETS | tidak | baseline klasik dengan interval |
| Local-only | tidak | tiap bank melatih QuantileMLP sendiri |
| FedAvg / FedProx | **tidak** | kolaborasi lewat rata-rata bobot |
| FedAvg + fine-tune | tidak | model global dipersonalisasi ke tiap bank |
| Centralized | ya | batas atas (tidak realistis secara privasi) |
| Oracle | n/a | distribusi sebenarnya (batas terbaik) |

## Menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # sekaligus memasang paket fedprob (editable)

python -m pytest -q                       # sanity check
streamlit run app/dashboard.py            # dashboard demo
jupyter lab notebooks/                    # notebook tahapan
```

## Struktur
```
src/fedprob/   simulator, skenario, model, federated, metrik, eksperimen, plot
notebooks/     01-05 (dibangkitkan dari scripts/build_notebooks.py)
app/           dashboard Streamlit
tests/         unit test
results/       cache hasil eksperimen (tidak di-commit)
```

Progres & temuan tiap tahap dicatat di [`PROGRESS.md`](PROGRESS.md).

## Rencana lanjutan
- Conformal prediction untuk menjamin coverage interval
- **Diffusion model** untuk membangkitkan skenario masa depan / stress testing
- Differential privacy & secure aggregation pada federated learning
- Validasi dengan data riil publik
