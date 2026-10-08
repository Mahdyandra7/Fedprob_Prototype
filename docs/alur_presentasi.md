# Alur Presentasi: Dari Ide Riset ke Alat Nyata

Bahan presentasi wawancara (±12–15 menit). Alurnya: **masalah nyata → ide riset → laboratorium simulasi →
alat pada data riil → hasil → rencana S2**. Setiap slide berisi **pesan utama**, **isi**, dan **catatan pembicara**.
Gambar ada di `docs/figures/` (`python scripts/export_figures.py` dan `--kasus`). Angka berasal dari
eksekusi 2026-10-08. Rincian kasus ada di [`kasus_atm.md`](kasus_atm.md).

---

## Slide 1: Judul
**Accurate, Honest, and Private: Time Series AI untuk Keputusan Keuangan**
*Studi kasus: KasPintar, rekomendasi pengisian kas ATM dengan federated probabilistic forecasting*

*Catatan:* Buka dengan pertanyaan riset dari essay. Yang ditampilkan adalah prototipe kerja: ide → simulasi →
alat pada data riil.

---

## Slide 2: Pertanyaan riset
> *"How can we build models that are accurate and honest about their own uncertainty and capable of
> learning from data that can never be fully shared?"*

| Kebutuhan | Arah riset |
|---|---|
| Tahu **seberapa yakin** | Probabilistic forecasting + conformal prediction |
| Belajar **tanpa berbagi data** | Federated learning |
| Membayangkan **skenario** masa depan | Diffusion model |

---

## Slide 3: Masalah nyata, ATM kosong atau uang menganggur
**Gambar:** `kasus_01_data_nn5.png`

- Bank mengisi ATM tiap Senin. Isi kurang → ATM kosong, nasabah kecewa, kunjungan darurat. Isi lebih → uang
  menganggur dan biaya dana.
- Praktik umum: "rata-rata 4 minggu × 1,2". Aturan ini hanya memberi satu angka tanpa ukuran risiko.
- Data transaksi tiap bank rahasia, dan bank kecil/baru tidak punya cukup histori.
- **Pertanyaan bisnis:** isi berapa agar peluang kehabisan ≤ 5% dengan uang menganggur minimal?

*Catatan:* Masalah ini menuntut ketiga hal di slide 2: peluang (bukan satu angka), kejujuran saat hari
libur/Lebaran, dan kolaborasi tanpa membuka data. Di Indonesia paling terasa menjelang Lebaran.

---

## Slide 4: Laboratorium dulu, simulasi untuk memilih model
**Gambar:** `01_data_simulasi.png`, `04_shock_conformal.png`

Di simulasi (8 bank fiktif, 6 skenario) distribusi sebenarnya diketahui, sehingga kejujuran bisa diuji tepat:
- Federasi menolong bank berdata sedikit: CRPS **−27%**, setara data digabung.
- Saat shock, coverage semua model anjlok ke ~40%. **ACI** mengembalikannya ke ~81%.
- Diffusion memberi skenario jalur, tetapi **meremehkan risiko ekor**.

*Catatan:* Simulasi bukan tujuan akhir, melainkan cara **memilih secara bijak** komponen yang dibawa ke
kasus nyata: kuantil + federated + ACI. Diffusion tidak dipakai untuk keputusan ini.

---

## Slide 5: Data riil dan setting
- **NN5**: penarikan tunai harian 111 ATM di Inggris, 1996–1998 (open data, CC BY 4.0).
- Dibagi ke 6 bank fiktif (2 besar, 2 menengah, 2 kecil). **Bank Zeta = bank baru, histori 12 minggu.**
- Uji 23 minggu (Des 1997 – Mei 1998), termasuk Natal, Tahun Baru, dan Paskah. Model dilatih tanpa melihat
  periode ini.

---

## Slide 6: Solusi KasPintar
**Gambar:** `kasus_02_fan_kumulatif.png`

1. **Target cerdas:** prediksi *kumulatif* penarikan. ATM habis ⇔ kumulatif > isi, sehingga
   **isi untuk 95% aman = kuantil 95%**.
2. **Federated:** model dilatih bersama 6 bank. Hanya bobot yang dikirim, data ATM tidak keluar dari bank.
   Varian terbaik (FedAvg/FedProx/Clustered) dipilih otomatis lewat validasi.
3. **ACI per bank:** kalibrasi online agar "95%" benar-benar 95%.
4. **Keputusan:** tiap Senin, rekomendasi isi per ATM + peluang kehabisan.

*Catatan:* Tunjuk gambar kiri. Aturan praktis (garis oranye) kehabisan di tengah minggu belanja Natal, sedangkan
rekomendasi (garis biru putus) cukup.

---

## Slide 7: Hasil 1, kolaborasi tanpa berbagi data
**Gambar:** `kasus_05_per_bank.png`

- FedAvg: CRPS **0.361**, setara Centralized 0.366, dan jauh di atas Local 0.471.
- **Bank baru Zeta: CRPS 1.592 → 0.319 (−80%).** Tanpa kolaborasi, Zeta harus menimbun 56% uang menganggur
  untuk mencapai target. Dengan federated cukup 21%, setara bank besar.
- Bank besar (Alfa) hampir tidak butuh kolaborasi (−5%). **Temuan simulasi berulang di data riil.**

---

## Slide 8: Hasil 2, jujur soal ketidakpastian
| Target | Tanpa ACI | ACI per ATM | ACI per bank |
|---|---|---|---|
| 95% | 88.9% | 90.6% | **94.3%** |
| 98% | 94.9% | 88.3% | **97.6%** |

- Model tanpa ACI **terlalu percaya diri**.
- ACI per ATM gagal karena datanya terlalu sedikit. Skor kalibrasi digabung per bank tetap privat dan tepat sasaran.

*Catatan:* Ini contoh riset yang jujur: ide pertama (per ATM) gagal, kegagalannya dianalisis, lalu diperbaiki.

---

## Slide 9: Hasil 3, dampak bisnis
**Gambar:** `kasus_03_tradeoff.png`, `kasus_04_kehabisan_mingguan.png`

| Kebijakan (target 95%) | ATM kehabisan | Uang menganggur |
|---|---|---|
| Aturan praktis ×1.2 | 11.4% | 21.3% |
| **KasPintar** | **5.4%** | 20.0% |

- Pada kehabisan yang sama (5.4%), aturan praktis butuh **32% lebih banyak** uang menganggur.
- Minggu belanja Natal: aturan praktis **75%** ATM kosong, KasPintar **6%**.
- Biaya total ilustratif turun ~51%.

---

## Slide 10: Demo alat
`streamlit run app/kaspintar.py`: Laporan Senin untuk operator kas (pilih bank, minggu, service level →
rekomendasi isi per ATM, peluang kehabisan, fan chart, dan hasil aktual).

*Catatan:* Demo singkat: pilih Bank Zeta, minggu 22 Des, lalu geser service level 95% → 98% dan tunjukkan
total isi naik.

---

## Slide 11: Keterbatasan (jujur)
- Data Inggris 1990-an dan pembagian bank fiktif. Belum data Indonesia.
- ACI reaktif: lonjakan yang belum pernah terjadi tetap lolos (16 Mar 1998: 22% ATM kosong).
- Backtest sederhana: tanpa kunjungan darurat dan sisa kas.
- Belum ada privasi formal (differential privacy / secure aggregation).
- Di data riil FedAvg mengalahkan FedProx, berbeda dengan simulasi. Karena itu model dipilih lewat validasi.

---

## Slide 12: Rencana riset S2 & kontribusi
1. **Data Indonesia**: ATM/likuiditas dengan efek Lebaran & gajian; kalender acara sebagai input.
2. **Diffusion bersyarat stres**: skenario krisis/bencana untuk stress testing.
3. **Privasi formal**: DP-FedAvg + secure aggregation; *trade-off* privasi vs akurasi.
4. **Conformal federated**: jaminan coverage lintas bank.

Kontribusi ke Indonesia: kolaborasi data keuangan yang patuh regulasi (OJK, UU PDP), terutama untuk BPR/BPD
dengan data terbatas; riset & bahan ajar di Program Data Science; komunitas IRIS.

---

## Slide 13: Penutup
> Akurat. Jujur soal ketidakpastian. Belajar tanpa membuka data. **Dan berguna untuk keputusan nyata.**

Kode, notebook, dan dashboard: `github.com/Mahdyandra7/Fedprob_Prototype`

---

# Persiapan Tanya-Jawab

**"Kenapa data Inggris, bukan Indonesia?"**
Data penarikan ATM per mesin di Indonesia tidak tersedia publik. NN5 adalah benchmark terbuka yang dipakai
komunitas forecasting, sehingga hasilnya bisa direproduksi. Polanya (mingguan, hari libur, lonjakan musiman)
analog dengan Indonesia. Validasi pada data Indonesia adalah rencana S2.

**"Bank-banknya fiktif, apakah federated-nya bermakna?"**
Pembagian ATM ke bank memang buatan, tetapi data tiap ATM riil. Yang diuji adalah: bisakah bank berdata sedikit
belajar dari bank lain tanpa melihat datanya? Jawabannya ya (Zeta −80% CRPS). Karena pembagiannya acak,
heterogenitas antar bank kecil. Itu sebabnya Clustered FL tidak menemukan kelompok, dan hal ini konsisten.

**"Kenapa tidak cukup aturan praktis dengan faktor lebih besar?"**
Bisa, tetapi mahal: untuk kehabisan 5.4% faktornya harus ×1.29 dengan uang menganggur 29% (vs 20%).
Faktor tetap tidak tahu minggu mana yang berisiko. Model memberi buffer besar hanya ketika perlu.

**"Kenapa FedAvg, padahal di simulasi FedProx yang terbaik?"**
Sistem memilih otomatis lewat CRPS validasi, dan di data riil FedAvg menang (0.379 vs 0.402). Ini justru
pelajaran: hasil simulasi memandu kandidat, bukan memastikan pemenang.

**"Apakah federated learning benar-benar aman?"**
Belum sepenuhnya. Bobot model masih bisa membocorkan informasi (*model inversion*). Karena itu differential
privacy dan secure aggregation masuk rencana riset.

**"Apa itu conformal / ACI dengan bahasa sederhana?"**
Setiap minggu sistem mengecek: apakah ATM yang diisi pada level "95%" benar-benar aman 95% dari waktu? Jika
terlalu sering kehabisan, buffer dinaikkan sedikit; jika terlalu jarang, diturunkan. Pengecekan dilakukan di
dalam bank sendiri.

**"Kenapa diffusion tidak dipakai di alat ini?"**
Keputusan isi ATM cukup dijawab kuantil kumulatif. Di simulasi, diffusion federated meremehkan risiko ekor,
dan itu berbahaya untuk keputusan yang menyangkut kehabisan. Diffusion tetap menjadi arah riset untuk skenario stres.

**"Hasil mana yang tidak sesuai harapan?"**
(1) ACI per ATM gagal di level tinggi. (2) FedProx kalah dari FedAvg di data riil. (3) Masih ada minggu dengan
lonjakan yang tidak tertangkap. (4) Di simulasi: koreksi bias dan CQR statik tidak membantu.

**"Bagaimana relevansinya dengan Indonesia?"**
Jaringan ATM besar dengan lonjakan Lebaran, dan ribuan BPR/BPD dengan data terbatas. Hasil Bank Zeta
menunjukkan pemain kecil/baru paling diuntungkan oleh kolaborasi tanpa berbagi data, sejalan dengan UU PDP.
