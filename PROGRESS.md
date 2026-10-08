# Log Progres

Catatan kronologis tiap tahap pengembangan. Entri terbaru di atas.
Setiap sesi baru: baca `CLAUDE.md` lalu entri teratas di sini.

---

## 2026-10-08 · Tahap 6–9: Conformal, heterogenitas, diffusion, bahan presentasi (selesai)

**Dikerjakan**
- **Tahap 6, conformal** (`conformal.py`, notebook 06): CQR statik & ACI online, dikalibrasi lokal per bank.
- **Tahap 7, heterogenitas** (`federated/clustered.py`, notebook 07): Clustered FL, skenario baru `klaster`
  (2 kelompok bank, RNG terpisah agar skenario lama identik), sweep FedProx mu. Default mu diubah 0.01 → 0.1.
- **Tahap 8, diffusion** (`models/diffusion.py`, notebook 08): DDPM bersyarat (MLP, 50 langkah, cosine schedule),
  dilatih federated & terpusat; metrik risiko tingkat-jalur (`path_min_metrics`, `independent_paths`).
- **Tahap 9, presentasi**: `docs/alur_presentasi.md` (12 slide + persiapan tanya-jawab), `scripts/export_figures.py`
  → 7 gambar di `docs/figures/`.
- Refactor: loop training/federated generik terhadap model; runner eksperimen dengan dispatcher +
  metode conformal sebagai sufiks. Dashboard: tab 5 (skenario jalur) & info clustered. 16 unit test lolos.

**Hasil** (CRPS, seed 42; coverage80 dalam kurung)

| Metode | normal | heterogen | data_langka | shock | bank_baru | klaster |
|---|---|---|---|---|---|---|
| Local | 0.195 | **0.201** | 0.210 | 0.510 | 0.193 | 0.261 |
| FedAvg | 0.203 | 0.223 | 0.194 | 0.504 (40%) | 0.195 | 0.277 |
| FedProx mu=0.1 | **0.193** | 0.213 | **0.188** | 0.463 (50%) | **0.189** | 0.275 |
| FedAvg+FT | 0.201 | 0.205 | 0.192 | 0.509 | 0.194 | 0.254 |
| Clustered FL | 0.205 | 0.224 | 0.192 | 0.507 | 0.202 | **0.252** |
| FedAvg+ACI | 0.205 (81%) | 0.222 (82%) | 0.197 (81%) | 0.528 (**81%**) | 0.199 (82%) | 0.268 (82%) |
| Diffusion fed | **0.190** | – | 0.190 | – | – | – |
| Oracle | 0.162 | 0.173 | 0.157 | 0.185 | 0.160 | 0.224 |

Temuan:
1. **ACI** adalah satu-satunya metode yang tetap terkalibrasi saat shock (coverage80 40% → 81%), dan konsisten
   ~81% di keenam skenario. Harganya: interval melebar ~4x dan CRPS sedikit memburuk saat shock.
2. **CQR statik hampir tidak membantu**: kalibrasi dari periode validasi tidak berlaku saat periode uji berubah
   (bahkan menurunkan coverage pada `data_langka` 73% → 69%).
3. **Clustered FL** menemukan 2 kelompok dengan tepat pada `klaster` (silhouette 0.70), dan pada skenario lain
   kembali menjadi satu kelompok. Ambang silhouette 0.3 dikalibrasi: pada update acak, silhouette bisa mencapai
   ~0.22 (p99).
4. **FedProx mu=0.1** adalah metode realistis terbaik pada normal/data_langka/bank_baru, dengan coverage ~80%.
   mu=1.0 terlalu mengekang (CRPS 0.31–0.39).
5. **Diffusion federated**: CRPS harian 0.190 (lebih baik dari FedAvg 0.203). Untuk titik terendah 14 hari,
   coverage80 69–79% vs 48% jika hari dianggap independen. **Namun** peluang "turun > 5%" diremehkan
   (prediksi ~9–10%, aktual 18–20%); versi centralized lebih baik (13–14%).

**Hasil negatif & pelajaran teknis**
- Koreksi bias online (geser median dengan residual terbaru) **memperburuk** CRPS di semua skenario
  (shock berbentuk V; model sudah melihat data terbaru).
- DDPM tanpa **x0-clipping** saat sampling: galat menumpuk, CRPS 0.22 → 0.35 saat training lebih lama.
- Loss tebak-noise DDPM **tidak selaras** dengan kualitas forecast, sehingga seleksi epoch/ronde memakai pinball
  loss dari 32 sampel (sepertiga origin validasi, agar 3x lebih cepat).

**Berikutnya**
- [ ] **Data riil open-source** (permintaan pemilik): time series multi-entitas yang mirip (tidak harus bank).
      Kandidat: konsumsi listrik per pelanggan (UCI ElectricityLoadDiagrams), penjualan per toko (M5/Walmart,
      Rossmann), lalu lintas/sepeda per stasiun. Perlu adaptor `windows.py` untuk data eksternal.
- [ ] Diffusion bersyarat stres (skenario krisis terkendali)
- [ ] Privasi formal: DP-FedAvg + secure aggregation
- [ ] Opsional: slide deck dari `docs/alur_presentasi.md`

---

## 2026-10-08 · Tahap 0–5: Fondasi prototipe (selesai)

**Dikerjakan**
- Rencana disepakati: Probabilistic forecasting + Federated learning pada data simulasi (diffusion ditunda).
- Paket `src/fedprob`: simulator + 5 skenario, windowing, baseline (Seasonal Naive, ETS), QuantileMLP,
  FedAvg/FedProx/fine-tune, metrik (CRPS, coverage), runner eksperimen + cache, plot.
- Notebook 01–05 dieksekusi tanpa error (output tersimpan), dashboard Streamlit lolos smoke test
  (`streamlit.testing.AppTest`), dan 9 unit test lolos.

**Keputusan teknis penting (beserta alasannya)**
- **Normalisasi relatif terhadap level** (`z = (y - rata2) / (5% x rata2)`), bukan mean/std.
  Percobaan pertama memakai mean/std, dan hasilnya FedAvg *kalah* dari Local untuk bank kecil. Penyebabnya,
  std dari histori 120 hari jauh lebih kecil, sehingga data bank kecil tampak "lebih bergejolak" dan model
  bersama tidak cocok lintas bank. Setelah diganti, federasi justru paling menolong bank kecil.
- **Origin uji setiap hari** (77 per bank), bukan tiap 7 hari, karena metrik dengan 11 origin terlalu berisik.
- Model: MLP kecil (2×128), cukup dijalankan di CPU; satu eksperimen penuh memakan waktu ~1,5–2 menit.

**Hasil** (CRPS skala ternormalisasi, lebih kecil lebih baik; seed 42 kecuali disebut lain)

| Metode | normal | heterogen | data_langka | shock | bank_baru |
|---|---|---|---|---|---|
| ETS | 0.203 | 0.209 | 0.197 | **0.442** | 0.201 |
| Local | **0.195** | **0.201** | 0.210 | 0.510 | **0.193** |
| FedAvg | 0.203 | 0.223 | 0.194 | 0.504 | 0.195 |
| FedProx | 0.198 | 0.223 | 0.202 | 0.507 | 0.200 |
| FedAvg+FT | 0.201 | 0.205 | **0.192** | 0.509 | 0.194 |
| Centralized | 0.199 | 0.212 | 0.195 | 0.470 | 0.204 |
| Oracle | 0.162 | 0.173 | 0.157 | 0.185 | 0.160 |

Temuan:
1. **Data langka → federasi sangat menolong.** Rata-rata 3 seed, untuk bank kecil F/G/H: CRPS Local 0.215,
   FedAvg 0.156 (−27%), dan Centralized 0.154. Interval Local terlalu lebar (coverage80 = 96%); FedAvg tetap
   terkalibrasi (84%).
2. **Bank dengan data cukup:** Local ≈ FedAvg ≈ Central, sehingga manfaat federasi kecil.
3. **Non-IID (alpha tinggi):** FedAvg memburuk (0.197 → 0.223 dari alpha 0 ke 0.9); fine-tune memulihkan
   sebagian besar (0.205). FedProx (mu = 0.01) belum membantu.
4. **Shock:** coverage80 semua model yang dilatih anjlok ke ~40%, artinya model terlalu percaya diri. ETS
   paling tahan karena terus dilatih ulang. Catatan: Oracle "tahu" shock-nya, jadi bukan pembanding adil
   di skenario ini.
5. Coverage Oracle di satu periode uji hanya ~72% (berisik karena error berkorelasi lewat faktor pasar).
   Rata-rata 3 seed mendekati 77–80%. Bandingkan model terhadap Oracle, bukan hanya terhadap nominal.

**Berikutnya (kandidat, urut prioritas)**
- [x] Repo terhubung ke GitHub: https://github.com/Mahdyandra7/Fedprob_Prototype
- [x] Conformal prediction (mis. adaptive conformal) untuk memperbaiki coverage saat shock (tahap 6)
- [x] Coba FedProx dengan mu lebih besar / clustered FL untuk skenario heterogen (tahap 7)
- [x] Diffusion model untuk generasi skenario / stress testing (arah riset ke-2) (tahap 8)
- [x] Bahan presentasi: alur cerita "masalah → simulasi → temuan 1–4 → arah riset" (tahap 9)
