# Ringkasan End-to-End: Bekal Presentasi

Panduan belajar untuk memahami project ini dari awal sampai akhir sebelum presentasi. Pasangannya:
slide deck (artifact "KasPintar: Accurate, Honest, Private", 15 slide, catatan pembicara di tiap slide),
[`alur_presentasi.md`](alur_presentasi.md) (alur + tanya-jawab), dan [`kasus_atm.md`](kasus_atm.md) (rincian kasus).

---

## 1. Cerita dalam satu paragraf
Essay LPDP bertanya: *bagaimana membangun model yang akurat, jujur soal ketidakpastiannya, dan bisa belajar
dari data yang tidak boleh dibagikan?* Saya menjawabnya dalam dua tahap. **Pertama, laboratorium simulasi**
(8 bank fiktif, 6 skenario). Karena distribusi sebenarnya diketahui, saya bisa menguji metode mana yang benar-benar
jujur. **Kedua, kasus nyata "KasPintar"**: rekomendasi pengisian kas ATM pada data terbuka NN5 (111 ATM Inggris),
dibagi ke 6 bank fiktif. Komponen terbaik dari simulasi (prediksi kuantil + federated learning + kalibrasi ACI)
dirakit menjadi alat. Hasilnya, bank baru paling diuntungkan (CRPS −80%), angka "95% aman" benar-benar
tercapai (94.3%), dan ATM kosong turun dari 11.4% ke 5.4% dengan uang menganggur 32% lebih sedikit dibanding
aturan praktis.

## 2. Alur end-to-end (dari data ke keputusan)

```
Data harian per ATM ──► Target kumulatif ──► Model kuantil ──► Federated ──► ACI per bank ──► Laporan Senin
(NN5, 111 ATM)          (total 1–14 hari)    (QuantileMLP)    (6 bank)      (kalibrasi)       (isi per ATM)
```

| Langkah | Apa yang terjadi | Kenapa begitu | Kode |
|---|---|---|---|
| 1. Data | Unduh NN5, isi nilai hilang (median hari sama 4 minggu terakhir), bagi 111 ATM ke 6 bank. Zeta hanya diberi 12 minggu histori. | Meniru banyak bank dengan data rahasia, termasuk bank baru. | `realdata/nn5.py` |
| 2. Target | Prediksi **penarikan kumulatif** sejak Senin, 1–14 hari ke depan. | ATM kosong ⇔ kumulatif > isi. Jadi isi aman 95% = **kuantil 95%** kumulatif 7 hari. | `data/windows.py` (`target="cumsum"`) |
| 3. Model | QuantileMLP memberi 9 kuantil (5%–99%) yang tidak saling silang, dilatih dengan pinball loss. | Langsung menjawab pertanyaan peluang, tidak hanya satu angka. | `models/quantile_net.py` |
| 4. Federated | Tiap bank melatih di datanya sendiri; server merata-ratakan bobot. Kandidat: FedAvg, FedProx, FedProx+FT, Clustered. Dipilih via CRPS validasi → **FedAvg**. | Data tidak keluar dari bank; bank kecil ikut belajar dari pola bank lain. | `federated/`, `cash/pipeline.py` |
| 5. Kalibrasi | ACI satu sisi: tiap minggu, jika terlalu sering kehabisan, buffer dinaikkan; jika terlalu jarang, diturunkan. Skor digabung **per bank**. | Model mentah terlalu percaya diri (95% → 88.9%). Per ATM datanya terlalu sedikit. | `conformal.py` (`aci_upper_pooled`) |
| 6. Keputusan | Isi = kuantil terkalibrasi pada service level pilihan (90/95/98/99%). Backtest 23 minggu. | Operator memilih trade-off risiko vs uang menganggur secara eksplisit. | `cash/policy.py`, `cash/report.py`, `app/kaspintar.py` |

## 3. Istilah yang wajib bisa dijelaskan dengan bahasa sederhana

| Istilah | Penjelasan singkat |
|---|---|
| **Probabilistic forecast** | Prediksi berupa rentang/peluang, bukan satu angka. |
| **Kuantil 95%** | Nilai yang menurut model hanya terlampaui 5% dari waktu. |
| **Pinball loss** | Fungsi loss untuk melatih kuantil; menghukum asimetris (untuk kuantil 95%, prediksi kerendahan dihukum jauh lebih berat). |
| **CRPS** | Skor kualitas seluruh distribusi prediksi (gabungan akurasi dan ketajaman). Makin kecil makin baik. Satuannya = satuan data (di kasus ATM: 1 unit = permintaan harian rata-rata ATM tersebut). |
| **Coverage** | Seberapa sering nilai aktual berada di dalam/di bawah prediksi. Jujur = coverage ≈ target. |
| **Federated learning (FedAvg)** | Model dilatih di tiap bank, hanya bobot dikirim ke server lalu dirata-rata. Data tidak berpindah. |
| **FedProx** | FedAvg + penalti (mu) agar model lokal tidak menjauh dari model global; berguna saat data antar bank berbeda. |
| **Clustered FL** | Mengelompokkan bank yang mirip (dari kemiripan update), satu model per kelompok. |
| **Centralized** | Batas atas: seolah semua data digabung. Tidak realistis karena data rahasia. |
| **Conformal / ACI** | Kalibrasi tanpa asumsi distribusi. ACI (Adaptive Conformal Inference) menyesuaikan buffer secara online setelah tiap kesalahan, sehingga tahan perubahan pola. |
| **CQR** | Conformal statik (kalibrasi sekali dari periode validasi). Di simulasi tidak membantu saat pola berubah. |
| **Diffusion model** | Model generatif yang membangkitkan banyak jalur masa depan. Di simulasi akurat secara harian tetapi meremehkan risiko ekor, jadi tidak dipakai untuk keputusan ATM. |
| **Backtest** | Menjalankan kebijakan pada periode lalu yang tidak dilihat model, lalu membandingkan dengan kejadian aktual. |

## 4. Temuan simulasi yang dibawa ke kasus nyata

1. **Federasi menolong bank berdata sedikit**: CRPS −27% untuk bank kecil, hampir setara data digabung.
   Bank dengan data cukup hampir tidak terbantu.
2. **Saat shock, semua model terlalu percaya diri** (coverage 80% → ~40%). **ACI** satu-satunya yang
   mengembalikannya (~81%), dibayar dengan interval lebih lebar.
3. **CQR statik dan koreksi bias online tidak membantu** (hasil negatif yang dilaporkan).
4. **Clustered FL** menemukan kelompok bank dengan tepat bila memang ada (silhouette 0.70), dan kembali ke satu
   kelompok bila tidak ada.
5. **Diffusion**: CRPS harian bagus (0.190), tetapi peluang "turun > 5%" diremehkan (prediksi ~9–10%, aktual 18–20%).

Pesan: simulasi **memandu pilihan kandidat**, tetapi pemenang di data riil tetap dipilih lewat validasi
(terbukti: di data riil FedAvg mengalahkan FedProx).

## 5. Angka kunci (hafalkan)

| Klaim | Angka |
|---|---|
| FedAvg vs Centralized vs Local (CRPS uji) | 0.361 vs 0.366 vs 0.471 |
| Bank baru Zeta (CRPS Local → FedAvg) | 1.592 → 0.319 (−80%) |
| Zeta tanpa kolaborasi, uang menganggur | 56% (vs 21% dengan federated) |
| Bank besar Alfa | 5% lebih buruk dengan FedAvg |
| Coverage target 95%: tanpa ACI / ACI per ATM / ACI per bank | 88.9% / 90.6% / **94.3%** |
| Target 98%: ACI per ATM gagal | 88.3% (per bank 97.6%) |
| ATM kehabisan: aturan praktis / KasPintar | 11.4% / **5.4%** |
| Uang menganggur pada kehabisan sama | 29.3% vs 20.0% (hemat 32%) |
| Biaya ilustratif per ATM-minggu | 3.49 → 1.70 (−51%) |
| Minggu Natal 15 Des 1997, ATM kosong | 75% vs 6.3% |
| Minggu dengan >10% ATM kosong | 7 (aturan) / 8 (FedAvg tanpa ACI) / 3 (KasPintar) |
| Minggu terburuk KasPintar | 16 Mar 1998, 21.6% |

## 6. Pembagian waktu (±13 menit)

| Menit | Slide | Fokus |
|---|---|---|
| 0–1 | 1–2 | Judul, pertanyaan riset, tiga kebutuhan |
| 1–3 | 3–4 | Masalah ATM; simulasi untuk memilih model |
| 3–6 | 5–7 | Data NN5, empat langkah KasPintar, cara membaca fan chart |
| 6–9 | 8–11 | Hasil: kolaborasi, kejujuran, dampak bisnis, minggu sulit |
| 9–11 | 12 | Demo (maksimal 2 menit) |
| 11–13 | 13–15 | Keterbatasan, rencana S2, penutup |

Jika waktu dipotong menjadi ±8 menit: lewati slide 4 (sebut satu kalimat), 7, dan 11; demo diganti screenshot.

## 7. Checklist sebelum presentasi
- [ ] `streamlit run app/kaspintar.py` dibuka di tab browser **sebelum** mulai (jangan training apa pun saat itu).
- [ ] Jangan hapus folder `results/` (cache; tanpa itu dashboard butuh ~14 menit training ulang).
- [ ] Siapkan screenshot/rekaman demo: Bank Zeta, minggu 22 Des 1997, service level 95% → 98%.
- [ ] Ganti `[Nama Anda]` di slide judul.
- [ ] Latihan sekali dengan stopwatch, lalu baca bagian "Persiapan Tanya-Jawab" di `alur_presentasi.md`.
