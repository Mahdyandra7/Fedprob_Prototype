# Studi Kasus Data Riil: "KasPintar", Rekomendasi Pengisian Kas ATM

Penerapan end-to-end ide riset (probabilistic forecasting + federated learning + conformal) pada satu
masalah perbankan nyata dengan data terbuka. Angka berasal dari eksekusi 2026-10-08 (notebook 09–11,
cache `results/atm_*.pkl`). Gambar ada di `docs/figures/kasus_*.png` (`python scripts/export_figures.py --kasus`).

## 1. Masalah
Bank mengisi ATM secara berkala lewat jasa pengangkutan uang, di kasus ini **setiap Senin untuk 7 hari**.

- **Isi terlalu sedikit**: ATM kosong sebelum jadwal berikutnya. Akibatnya nasabah kecewa, perlu kunjungan
  darurat yang mahal, dan reputasi bank turun. Di Indonesia hal ini paling terasa menjelang Lebaran.
- **Isi terlalu banyak**: uang menganggur di mesin dan menimbulkan biaya dana, asuransi, serta risiko.
- Praktik umum masih berupa aturan praktis "rata-rata 4 minggu terakhir × 1,2". Aturan ini hanya memberi satu
  angka tanpa ukuran risiko, dan faktor 1,2 tidak berarti apa-apa secara peluang.
- Data transaksi tiap bank **rahasia**. Bank kecil atau baru tidak punya cukup histori untuk membangun model
  yang baik.

**Pertanyaan bisnis:** *berapa uang yang harus diisi ke tiap ATM minggu ini agar peluang kehabisan ≤ 5%,
dengan uang menganggur seminimal mungkin?*

## 2. Data
**NN5** (kompetisi forecasting NN5, 2008): penarikan tunai harian **111 ATM di Inggris**, 18 Mar 1996 – 17 Mei
1998 (791 hari). Sumber: Zenodo record 4656110 (Monash Forecasting Archive), lisensi **CC BY 4.0**.
Nilai hilang 1,9% dan nol 0,5% diimputasi dengan median hari yang sama pada 4 minggu terakhir.

Untuk meniru setting multi-bank, 111 ATM dibagi secara deterministik ke **6 bank fiktif**: Alfa & Beta (30 ATM),
Gamma & Delta (17), Epsilon (9), dan **Zeta (8 ATM, bank baru dengan histori hanya 12 minggu)**.

| Periode | Tanggal | Keterangan |
|---|---|---|
| Latih | 1996-03-18 s/d 1997-09-07 | Zeta hanya 84 hari terakhir |
| Validasi | 1997-09-08 s/d 1997-11-30 | memilih model, kalibrasi awal ACI |
| Uji | 1997-12-01 s/d 1998-05-17 | 23 siklus mingguan × 111 ATM; Natal, Tahun Baru, Paskah |

## 3. Solusi: model dipilih dari temuan simulasi
| Komponen | Pilihan | Alasan |
|---|---|---|
| Target | **penarikan kumulatif** 1–14 hari | ATM kosong sebelum hari ke-h ⇔ kumulatif penarikan > isi. Jadi **isi untuk service level 95% = kuantil 95% kumulatif 7 hari**. |
| Model | QuantileMLP (9 kuantil, 5%–99%) | Kuantil tidak saling silang dan langsung menjawab pertanyaan keputusan. |
| Kolaborasi | **federated antar bank** (kandidat FedAvg, FedProx, FedProx+fine-tune, Clustered FL) | Data ATM tidak keluar dari banknya. Yang terbaik dipilih **otomatis lewat CRPS validasi**. |
| Kejujuran | **ACI satu sisi, skor digabung per bank** | Satu-satunya metode yang tetap terkalibrasi saat shock di simulasi. Kalibrasi tetap di dalam bank (privat). |
| Keputusan | isi = kuantil terkalibrasi pada service level pilihan (90/95/98/99%) | Operator memilih trade-off kehabisan vs uang menganggur secara eksplisit. |

**Tidak dipakai:** diffusion model. Keputusan ini cukup dijawab kuantil kumulatif, dan di simulasi diffusion
federated meremehkan risiko ekor. Diffusion disimpan untuk riset lanjutan (stress testing).

## 4. Hasil

### 4a. Akurasi forecast (CRPS total 7 hari, makin kecil makin baik)
| Metode | CRPS validasi | CRPS uji |
|---|---|---|
| Seasonal naive + kuantil residual | 0.520 | 0.556 |
| Local (tiap bank sendiri) | 0.426 | 0.471 |
| **FedAvg (terpilih)** | **0.379** | **0.361** |
| FedProx (mu = 0.1) | 0.402 | 0.416 |
| FedProx + fine-tune | 0.391 | 0.399 |
| Clustered FL | 0.384 | 0.371 |
| Centralized (data digabung, batas atas) | 0.375 | 0.366 |

- FedAvg **setara Centralized** (0.361 vs 0.366) tanpa memindahkan data.
- Berbeda dengan simulasi, di data riil **FedAvg mengalahkan FedProx**. Karena itu pemilihan model lewat
  validasi menjadi bagian dari sistem dan tidak ditetapkan dari awal.
- Clustered FL tidak menemukan kelompok bank (silhouette 0.14 < 0.3), sehingga perilakunya kembali ke FedAvg.
  Ini wajar karena pembagian ATM ke bank dilakukan secara acak.

### 4b. Kejujuran (coverage kuantil kumulatif 7 hari)
| Target | Tanpa ACI | ACI per ATM | **ACI per bank** |
|---|---|---|---|
| 90% | 81.5% | – | **89.2%** |
| 95% | 88.9% | 90.6% | **94.3%** |
| 98% | 94.9% | 88.3% | **97.6%** |
| 99% | 97.1% | – | **98.8%** |

Tanpa ACI, model **terlalu percaya diri** (target 95% hanya tercapai 88.9%). ACI per ATM **gagal** di level
tinggi karena tiap ATM hanya punya sedikit pengamatan. Menggabungkan skor kalibrasi semua ATM dalam satu
bank menyelesaikan masalah itu, dan data tetap berada di dalam bank.

### 4c. Dampak bisnis (backtest 23 minggu × 111 ATM, target 95%)
| Kebijakan | ATM kehabisan | Uang menganggur (% permintaan) |
|---|---|---|
| Aturan praktis ×1.2 | 11.4% | 21.3% |
| Seasonal naive | 10.2% | 22.7% |
| Local | 11.0% | 17.5% |
| FedAvg tanpa ACI | 11.0% | 15.2% |
| **KasPintar (FedAvg + ACI)** | **5.4%** | 20.0% |
| Centralized + ACI (batas atas) | 4.8% | 20.9% |

- **Pada tingkat kehabisan yang sama (5.4%)**, aturan praktis butuh faktor ×1.29 dan uang menganggur 29.3%.
  KasPintar hanya 20.0%, atau **32% lebih sedikit uang menganggur**.
- Total biaya ilustratif (uang menganggur × biaya dana 15%/tahun + peluang kehabisan × penalti 30 unit)
  **turun ~51%**: 3.49 → 1.70 unit per ATM-minggu. Penurunan terutama datang dari kehabisan yang lebih jarang.
  Urutan ini tetap sama pada asumsi biaya lain (penalti ×2, biaya dana 30%).
- Kurva trade-off (`kasus_03_tradeoff.png`): kurva model berada di kiri-bawah aturan praktis di semua
  service level. Peran ACI adalah membuat angka "95%" **benar-benar berarti 95%**, bukan menggeser kurva.

### 4d. Minggu sulit (`kasus_04_kehabisan_mingguan.png`)
- Minggu belanja Natal (15 Des 1997): aturan praktis membuat **75% ATM kehabisan**, KasPintar 6.3%.
- Minggu dengan > 10% ATM kehabisan: aturan praktis 7 minggu, FedAvg tanpa ACI 8 minggu, **KasPintar 3 minggu**.
- KasPintar tetap punya minggu buruk (16 Mar 1998: 21.6%). ACI hanya bereaksi **setelah** meleset, sehingga
  lonjakan yang belum pernah terlihat tetap lolos. Hal ini dilaporkan apa adanya.

### 4e. Siapa yang paling diuntungkan (`kasus_05_per_bank.png`)
| Bank | CRPS Local | CRPS FedAvg | Perubahan |
|---|---|---|---|
| Alfa (besar) | 0.357 | 0.376 | −5% (sedikit lebih buruk) |
| Beta / Gamma / Delta | 0.363–0.401 | 0.337–0.364 | +7–9% |
| Epsilon (kecil) | 0.443 | 0.385 | +13% |
| **Zeta (baru, 12 minggu)** | **1.592** | **0.319** | **+80%** |

Bank Zeta dengan model Local + ACI memang bisa mencapai target kehabisan (3.8%), tetapi harus menimbun
**56% uang menganggur**. Dengan federated + ACI, Zeta mencapai 5.4% kehabisan dan 21% uang menganggur, setara
bank besar. Temuan simulasi berulang di data riil: **kolaborasi paling menolong pemain dengan data paling sedikit**,
sedangkan bank besar hampir tidak membutuhkannya.

## 5. Alat: dashboard operator `app/kaspintar.py`
`streamlit run app/kaspintar.py`, dengan empat tab:
1. **Laporan Senin**: pilih bank, minggu, dan service level. Dashboard menampilkan rekomendasi isi per ATM,
   peluang kehabisan bila memakai aturan praktis, fan chart kumulatif, dan hasil aktual (backtest).
2. **Kinerja**: backtest semua kebijakan, kurva trade-off, dan kehabisan per minggu.
3. **Kolaborasi federated**: manfaat per bank dan bank baru.
4. **Tentang**: masalah, data, metode, keterbatasan.

## 6. Keterbatasan
- Data Inggris 1996–1998, bukan Indonesia. Pembagian bank **fiktif** (acak), sehingga heterogenitas antar bank
  lebih kecil daripada kenyataan.
- Satuan uang NN5 tidak diketahui, sehingga biaya dihitung dalam unit relatif.
- Backtest mengasumsikan isi selalu tersedia dan permintaan tidak berubah saat ATM kosong. Kunjungan darurat
  dan sisa uang minggu lalu tidak dimodelkan.
- ACI reaktif: lonjakan yang belum pernah terjadi tetap lolos di minggu pertamanya.
- Belum ada privasi formal (differential privacy / secure aggregation).

## 7. Arah lanjutan
Data ATM Indonesia (efek Lebaran, gajian), kalender acara sebagai input, diffusion untuk skenario stres
(misalnya bencana atau gangguan jaringan), conformal federated lintas bank, dan optimasi rute pengisian.
