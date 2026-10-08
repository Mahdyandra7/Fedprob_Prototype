# Alur Presentasi: Federated Probabilistic Forecasting

Bahan presentasi ide riset (±10–12 menit). Setiap slide berisi **pesan utama**, **isi**, dan
**catatan pembicara**. Gambar ada di `docs/figures/` (dibangkitkan oleh `scripts/export_figures.py`).
Angka berasal dari eksekusi 2026-10-08 (lihat `PROGRESS.md`).

---

## Slide 1: Judul
**Accurate, Honest, and Private: Probabilistic Forecasting for Financial Time Series with Federated Learning**

*Catatan:* Buka dengan pertanyaan riset dari essay. Sebutkan bahwa yang ditampilkan adalah prototipe awal
berbasis simulasi yang menjadi pijakan riset S2.

---

## Slide 2: Masalahnya
**Pesan:** Lembaga keuangan butuh forecast yang akurat *dan* jujur soal ketidakpastian, tetapi datanya tidak
boleh dibagikan.

- Forecast satu angka ("saldo minggu depan Rp X") menyembunyikan risiko.
- Data tiap bank bersifat rahasia (regulasi, privasi nasabah), sehingga model hanya bisa belajar dari data sendiri.
- Bank kecil atau baru memiliki data paling sedikit, padahal paling membutuhkan model yang baik.

> *"How can we build models that are accurate and honest about their own uncertainty and capable of
> learning from data that can never be fully shared?"*

---

## Slide 3: Tiga arah riset → satu sistem
| Arah | Menjawab | Di prototipe |
|---|---|---|
| Probabilistic forecasting | *seberapa yakin?* | QuantileMLP + conformal prediction |
| Federated learning | *belajar tanpa berbagi data* | FedAvg, FedProx, Clustered FL |
| Diffusion model | *skenario apa saja yang mungkin?* | DDPM bersyarat, dilatih federated |

*Catatan:* Tekankan bahwa ketiganya saling melengkapi, bukan proyek terpisah.

---

## Slide 4: Mengapa simulasi?
**Gambar:** `01_data_simulasi.png`

- 8 bank fiktif: tren, pola mingguan, gajian, Lebaran, faktor pasar bersama, dan guncangan.
- **Distribusi sebenarnya diketahui (Oracle)**, sehingga kejujuran interval bisa diuji dengan tepat. Hal ini
  mustahil dilakukan dengan data riil.
- 6 skenario terkontrol: normal, heterogen, data langka, shock, bank baru, klaster.

*Catatan:* Simulasi bukan pengganti data riil, melainkan **laboratorium** untuk memahami *kapan* sebuah
metode bekerja. Validasi dengan data riil adalah tahap berikutnya.

---

## Slide 5: Cara menilai "jujur"
- **CRPS**: akurasi + kalibrasi dalam satu angka (lebih kecil lebih baik).
- **Coverage 80%**: seberapa sering nilai aktual jatuh di interval 80% (ideal ≈ 80%).
- Interval sempit tapi coverage 50% → **terlalu percaya diri** → berbahaya bagi manajemen risiko.

---

## Slide 6: Temuan 1 — Federasi menolong bank dengan data sedikit
**Gambar:** `02_data_langka_local_vs_fedavg.png`, `03_siapa_diuntungkan.png`

- Bank kecil berhistori 4 bulan: CRPS **0.215 → 0.156 (−27%)** dengan FedAvg (rata-rata 3 seed).
- Setara Centralized (0.154), yaitu menggabungkan semua data, **tanpa satu pun data berpindah**.
- Bank dengan data banyak hanya sedikit diuntungkan (0.178 → 0.172).

*Catatan:* Inilah argumen utama untuk regulator/industri: kolaborasi tanpa membuka data paling bermanfaat
bagi pemain kecil, yang relevan untuk BPR atau bank daerah di Indonesia.

---

## Slide 7: Temuan 2 — Tidak semua bank sama (non-IID)
**Gambar:** `06_sweep_heterogenitas.png`, `05_clustered_similarity.png`

- Makin berbeda pola antar bank, satu model global (FedAvg) makin buruk.
- **Clustered FL**: server menemukan kelompok bank **hanya dari arah update bobot**, tanpa melihat data.
  Pada skenario 2 kelompok: tepat 100%, CRPS 0.277 → 0.252 (lebih baik daripada Local 0.261).
- **FedProx (μ = 0.1)** menjadi metode realistis terbaik pada sebagian besar skenario (mis. data langka 0.188).

---

## Slide 8: Temuan 3 — Saat krisis, semua model terlalu percaya diri
**Gambar:** `04_shock_conformal.png`

- Shock (penarikan dana mendadak): coverage 80% semua model anjlok ke **~40%**.
- **Adaptive Conformal Inference (ACI)**: interval melebar otomatis setelah mulai meleset → coverage kembali **~81%**.
- ACI konsisten ~81% di **keenam** skenario, dan kalibrasinya **lokal di tiap bank** (tetap privat).

*Catatan:* Ini inti kata *honest* di pertanyaan riset. Model yang tahu kapan ia tidak tahu.

---

## Slide 9: Temuan 4 — Diffusion: dari rentang ke skenario
**Gambar:** `07_diffusion_skenario.png`

- Model kuantil hanya tahu rentang **per hari**. Pertanyaan risiko sering menyangkut **seluruh jalur**
  ("pernahkah saldo turun > 5% dalam 14 hari?").
- Diffusion federated: CRPS harian **0.190** (FedAvg 0.203) + ratusan skenario lengkap.
- Titik terendah 14 hari: coverage 80% **69–79%** vs **48%** bila hari dianggap independen.

---

## Slide 10: Keterbatasan (jujur)
- Data **simulasi**; validasi riil belum dilakukan.
- Diffusion federated **meremehkan risiko ekor** (memprediksi peluang turun ~10%, aktual ~20%).
- Model tidak pernah melihat krisis, sehingga skenario terburuknya terbatas pada yang pernah terjadi.
- Belum ada privasi formal (differential privacy / secure aggregation): bobot model masih bisa membocorkan informasi.

*Catatan:* Menyebutkan keterbatasan menunjukkan kematangan riset. Pewawancara menghargai ini.

---

## Slide 11: Rencana riset (S2) & kontribusi
1. **Validasi data riil** (open-source time series multi-entitas).
2. **Diffusion bersyarat stres**: skenario krisis yang dikendalikan (stress testing ala regulator).
3. **Privasi formal**: DP-FedAvg + secure aggregation; ukur *trade-off* privasi vs akurasi.
4. **Conformal federated**: jaminan coverage lintas bank.

Kontribusi ke Indonesia: kerangka kolaborasi data keuangan yang patuh regulasi (OJK/UU PDP), riset & bahan
ajar di Program Data Science, serta pembinaan komunitas IRIS.

---

## Slide 12: Penutup
> Akurat. Jujur soal ketidakpastian. Belajar tanpa membuka data.

Kode, notebook, dan dashboard: `github.com/Mahdyandra7/Fedprob_Prototype`

---

# Persiapan Tanya-Jawab

**"Kenapa pakai data simulasi, bukan data riil?"**
Data antar bank memang tidak tersedia publik, dan itulah masalah yang ingin dipecahkan. Simulasi memberi
*ground truth* sehingga saya bisa membuktikan sebuah metode benar-benar terkalibrasi. Langkah berikutnya adalah
validasi pada data riil open-source.

**"Apa bedanya federated learning dengan sekadar berbagi model?"**
Model dilatih *bersama* secara iteratif: setiap ronde bank melatih lokal, server merata-ratakan. Hasilnya
setara menggabungkan data (CRPS 0.156 vs 0.154) tanpa memindahkan data.

**"Apakah federated learning benar-benar aman?"**
Belum sepenuhnya. Bobot model masih bisa membocorkan informasi (*model inversion*). Karena itu differential
privacy dan secure aggregation masuk rencana riset.

**"Kenapa tidak cukup dengan model statistik klasik (ARIMA/ETS)?"**
ETS adalah pembanding yang kuat dan bahkan paling tahan saat shock karena terus dilatih ulang. Namun ETS tidak
bisa belajar lintas bank dan tidak memberi skenario jalur bersama. Pendekatan yang ditawarkan menggabungkan
keduanya.

**"Apa itu diffusion model, dan kenapa cocok untuk keuangan?"**
Model generatif yang belajar "membersihkan" noise menjadi jalur realistis. Ia bisa membangkitkan ratusan skenario
masa depan sehingga cocok untuk stress testing, yang selama ini dilakukan dengan skenario yang dirancang manual.

**"Hasil mana yang paling tidak sesuai harapan?"**
(1) Koreksi bias online justru memperburuk forecast. (2) CQR statik tidak membantu saat periode berubah.
(3) Diffusion federated meremehkan risiko ekor. Ketiganya dicatat sebagai hasil negatif dan memberi arah riset.

**"Bagaimana relevansinya dengan Indonesia?"**
Ada ribuan BPR/BPD dengan data terbatas. Temuan 1 menunjukkan merekalah yang paling diuntungkan oleh kolaborasi
tanpa berbagi data. Ini sejalan dengan UU Pelindungan Data Pribadi dan agenda AI nasional.
