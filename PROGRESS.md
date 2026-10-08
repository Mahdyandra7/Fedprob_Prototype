# Log Progres

Catatan kronologis tiap tahap pengembangan. Entri terbaru di atas.
Setiap sesi baru: baca `CLAUDE.md` lalu entri teratas di sini.

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
- [ ] Conformal prediction (mis. adaptive conformal) untuk memperbaiki coverage saat shock
- [ ] Coba FedProx dengan mu lebih besar / clustered FL untuk skenario heterogen
- [ ] Diffusion model untuk generasi skenario / stress testing (arah riset ke-2)
- [ ] Bahan presentasi: alur cerita "masalah → simulasi → temuan 1–4 → arah riset"
