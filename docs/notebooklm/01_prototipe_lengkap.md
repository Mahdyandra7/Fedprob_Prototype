# Prototipe Riset "FedProb" dan Studi Kasus KasPintar: Dokumen Lengkap

Dokumen ini menjelaskan seluruh prototipe riset Time Series AI untuk sektor keuangan, dari pertanyaan riset
sampai hasil dan keterbatasannya. Ditulis sebagai satu sumber utuh dalam bentuk prosa agar mudah dipahami tanpa
membaca kode. Kode, notebook, dan dashboard tersedia di github.com/Mahdyandra7/Fedprob_Prototype. Semua angka
berasal dari eksekusi prototipe pada Oktober 2026.

---

## 1. Asal-usul: dari skripsi ke pertanyaan riset

Skripsi S1 pemilik project memakai model ARDL untuk memprediksi pertumbuhan Dana Pihak Ketiga (DPK) perbankan
Indonesia. Pengalaman itu meninggalkan satu pertanyaan yang belum terjawab, yang kemudian menjadi pertanyaan
riset di essay beasiswa:

> "How can we build models that are accurate and honest about their own uncertainty and capable of learning
> from data that can never be fully shared?"

Pertanyaan itu dipecah menjadi tiga arah riset:

1. **Probabilistic forecasting**: model tidak hanya memberi satu angka, tetapi rentang kemungkinan beserta
   peluangnya. Pengambil keputusan di perbankan dan manajemen risiko mendapat gambaran lengkap.
2. **Diffusion model untuk time series keuangan**: membangkitkan banyak jalur masa depan yang mungkin di bawah
   kondisi ekonomi atau shock yang berbeda, berguna untuk stress testing dan simulasi kebijakan.
3. **Federated learning**: melatih model di banyak sumber data tanpa data itu meninggalkan asalnya. Ini menjawab
   ketegangan antara kebutuhan data besar dan aturan kerahasiaan/regulasi di sektor keuangan.

Prototipe ini dibangun untuk menunjukkan bahwa ketiga arah itu bukan sekadar wacana: semuanya sudah diimplementasikan,
diuji, dan satu kombinasinya diterapkan pada masalah nyata.

## 2. Strategi: laboratorium dulu, lalu kasus nyata

Prototipe dikerjakan dalam dua tahap.

- **Tahap laboratorium (simulasi).** Data saldo harian bank dibangkitkan dengan simulator, sehingga distribusi
  sebenarnya diketahui. Ini penting karena "kejujuran" model (apakah interval 80% benar-benar mengenai 80% kejadian)
  hanya bisa diuji tepat bila kebenarannya diketahui. Simulasi dipakai untuk **memilih komponen secara bijak**.
- **Tahap kasus nyata (KasPintar).** Komponen terbaik dirakit menjadi alat rekomendasi pengisian kas ATM pada data
  terbuka NN5. Di sini pertanyaannya bergeser dari "metode mana yang paling bagus?" menjadi "apakah ini membantu
  keputusan nyata?".

## 3. Laboratorium simulasi

### 3.1 Simulator
Saldo harian 8 bank fiktif dibangkitkan dari komponen: tren, pola mingguan, efek gajian, efek Lebaran, faktor
pasar bersama (AR(1), membuat bank saling berkorelasi), komponen idiosinkratik per bank (AR(1)), dan noise.
Bank dibuat berbeda ukuran dan panjang histori. Simulator juga menghasilkan "oracle", yaitu kuantil sebenarnya,
sebagai batas atas yang tidak mungkin dikalahkan.

Enam skenario diuji: **normal**, **heterogen** (pola antar bank berbeda), **data langka** (beberapa bank hanya
punya sedikit histori), **shock** (penurunan tiba-tiba di periode uji), **bank baru**, dan **klaster** (dua
kelompok bank dengan pola berbeda).

### 3.2 Setting forecasting
Model melihat 56 hari terakhir (lookback) dan memprediksi 14 hari ke depan (horizon), dengan fitur kalender.
Data dinormalisasi relatif terhadap level tiap bank: 1 unit = 5% saldo rata-rata bank tersebut. Keputusan ini
penting: percobaan pertama memakai normalisasi mean/std dan hasilnya federated learning justru kalah untuk bank
kecil, karena histori pendek membuat data bank kecil tampak "lebih bergejolak". Setelah normalisasi diganti,
federasi justru paling menolong bank kecil.

### 3.3 Metode yang dibandingkan
- **Baseline**: Seasonal Naive dan ETS (exponential smoothing) dengan interval.
- **QuantileMLP**: jaringan saraf kecil (2 lapis × 128) yang mengeluarkan 7 kuantil (5%, 10%, 25%, 50%, 75%,
  90%, 95%) sekaligus, dilatih dengan pinball loss, dengan struktur yang menjamin kuantil tidak saling silang.
- **Local**: tiap bank melatih modelnya sendiri.
- **Centralized**: semua data digabung (batas atas yang tidak realistis karena data rahasia).
- **FedAvg**: tiap bank melatih secara lokal, server merata-ratakan bobot model; data tidak berpindah.
- **FedProx**: FedAvg dengan penalti (mu) agar model lokal tidak menjauh dari model global. Nilai mu = 0.1 dipilih
  lewat sweep; mu = 1.0 terlalu mengekang.
- **FedAvg + fine-tune**: model global disesuaikan lagi sedikit di tiap bank.
- **Clustered FL**: bank dikelompokkan berdasarkan kemiripan update modelnya (cosine similarity), satu model per
  kelompok; kelompok hanya dibentuk bila silhouette ≥ 0.3 (ambang dikalibrasi: pada update acak silhouette bisa
  mencapai ~0.22).
- **Conformal prediction**: CQR statik (kalibrasi sekali dari periode validasi) dan **ACI** (Adaptive Conformal
  Inference) yang menyesuaikan lebar interval secara online setiap kali melihat kesalahan.
- **Diffusion model**: DDPM bersyarat (50 langkah, cosine schedule) yang membangkitkan jalur 14 hari, dilatih
  federated dan terpusat.

### 3.4 Metrik
- **CRPS** (Continuous Ranked Probability Score): skor kualitas seluruh distribusi prediksi, menggabungkan
  ketepatan dan ketajaman. Makin kecil makin baik.
- **Coverage**: seberapa sering nilai aktual jatuh di dalam interval. Model jujur bila coverage ≈ target.
- **Pinball loss** untuk tiap kuantil, serta metrik tingkat jalur untuk diffusion (misalnya titik terendah 14 hari).

### 3.5 Hasil simulasi (CRPS, seed 42; makin kecil makin baik)

| Metode | normal | heterogen | data langka | shock | bank baru | klaster |
|---|---|---|---|---|---|---|
| Local | 0.195 | 0.201 | 0.210 | 0.510 | 0.193 | 0.261 |
| FedAvg | 0.203 | 0.223 | 0.194 | 0.504 | 0.195 | 0.277 |
| FedProx mu=0.1 | 0.193 | 0.213 | 0.188 | 0.463 | 0.189 | 0.275 |
| FedAvg + fine-tune | 0.201 | 0.205 | 0.192 | 0.509 | 0.194 | 0.254 |
| Clustered FL | 0.205 | 0.224 | 0.192 | 0.507 | 0.202 | 0.252 |
| FedAvg + ACI | 0.205 | 0.222 | 0.197 | 0.528 | 0.199 | 0.268 |
| Diffusion federated | 0.190 | – | 0.190 | – | – | – |
| Oracle (batas atas) | 0.162 | 0.173 | 0.157 | 0.185 | 0.160 | 0.224 |

### 3.6 Temuan simulasi
1. **Federasi paling menolong bank berdata sedikit.** Rata-rata 3 seed untuk bank kecil pada skenario data langka:
   CRPS Local 0.215, FedAvg 0.156 (−27%), Centralized 0.154. Interval Local terlalu lebar (coverage 80% = 96%),
   sedangkan FedAvg tetap terkalibrasi (84%). Bank dengan data cukup hampir tidak terbantu.
2. **Saat shock, semua model yang dilatih terlalu percaya diri**: coverage 80% anjlok ke ~40%. **ACI satu-satunya
   metode yang tetap terkalibrasi** (kembali ke ~81%), dan konsisten ~81% di keenam skenario. Harganya: interval
   melebar sekitar 4 kali dan CRPS sedikit memburuk saat shock.
3. **CQR statik hampir tidak membantu**, karena kalibrasi dari masa lalu tidak berlaku saat pola berubah (pada data
   langka bahkan menurunkan coverage 73% → 69%).
4. **Clustered FL** menemukan dua kelompok dengan tepat pada skenario klaster (silhouette 0.70) dan kembali menjadi
   satu kelompok pada skenario lain.
5. **FedProx mu = 0.1** adalah metode realistis terbaik pada skenario normal, data langka, dan bank baru.
6. **Diffusion federated** memberi CRPS harian terbaik (0.190) dan menangkap ketergantungan antar hari (coverage
   titik terendah 69–79%, dibanding 48% bila hari dianggap independen). **Namun** ia meremehkan risiko ekor:
   peluang "turun lebih dari 5%" diprediksi ~9–10% padahal aktualnya 18–20%. Versi terpusat lebih baik (13–14%).
   Karena itu diffusion tidak dipakai untuk keputusan yang menyangkut risiko kehabisan.

### 3.7 Hasil negatif dan pelajaran teknis
- Koreksi bias online (menggeser median dengan residual terbaru) **memperburuk** CRPS di semua skenario.
- Diffusion tanpa x0-clipping saat sampling menumpuk galat (CRPS 0.22 → 0.35).
- Loss tebak-noise pada diffusion tidak selaras dengan kualitas forecast, sehingga pemilihan model memakai pinball
  loss dari sampel.

## 4. Studi kasus data riil: KasPintar

### 4.1 Masalah
Bank mengisi ATM secara berkala, di kasus ini setiap Senin untuk 7 hari. Isi terlalu sedikit membuat ATM kosong
(nasabah kecewa, kunjungan darurat mahal, reputasi turun); isi terlalu banyak membuat uang menganggur (biaya dana,
asuransi, risiko). Praktik umum memakai aturan praktis "rata-rata 4 minggu terakhir × 1,2", yang hanya memberi satu
angka tanpa ukuran risiko. Data transaksi tiap bank rahasia, dan bank kecil atau baru tidak punya cukup histori.

**Pertanyaan bisnis:** berapa uang yang harus diisi ke tiap ATM minggu ini agar peluang kehabisan ≤ 5% dengan uang
menganggur seminimal mungkin?

Masalah ini menuntut ketiga hal dari pertanyaan riset sekaligus: peluang (bukan satu angka), kejujuran saat musim
libur, dan kolaborasi tanpa membuka data. Di Indonesia masalah ini paling terasa menjelang Lebaran.

### 4.2 Data
NN5: penarikan tunai harian 111 ATM di Inggris, 18 Maret 1996 – 17 Mei 1998 (791 hari), benchmark terbuka dari
kompetisi forecasting NN5 (lisensi CC BY 4.0, Zenodo). Nilai hilang 1,9% diisi median hari yang sama pada 4 minggu
terakhir. Karena tidak ada data multi-bank publik, 111 ATM dibagi secara deterministik ke 6 bank fiktif: Alfa dan
Beta (30 ATM), Gamma dan Delta (17), Epsilon (9), dan **Zeta (8 ATM, bank baru dengan histori hanya 12 minggu)**.
Periode latih sampai 7 September 1997, validasi 8 September – 30 November 1997, dan uji 1 Desember 1997 – 17 Mei
1998 (23 siklus mingguan, mencakup Natal, Tahun Baru, dan Paskah).

### 4.3 Solusi dalam empat langkah
1. **Target kumulatif.** Yang diprediksi adalah total penarikan sejak Senin untuk 1–14 hari ke depan. ATM kosong
   sebelum hari ke-h tepat ketika kumulatif penarikan melebihi isi, sehingga **isi untuk service level 95% =
   kuantil 95% dari kumulatif 7 hari**. Forecast langsung menjadi keputusan.
2. **Model federated.** QuantileMLP dengan 9 kuantil (5%–99%) dilatih bersama oleh 6 bank tanpa data keluar dari
   bank. Kandidat FedAvg, FedProx, FedProx + fine-tune, dan Clustered FL; yang terbaik dipilih otomatis lewat CRPS
   validasi. Terpilih: **FedAvg**.
3. **Kalibrasi ACI per bank.** Setiap minggu sistem mengecek apakah ATM yang diisi pada level 95% benar-benar aman
   95% dari waktu; jika terlalu sering kehabisan, buffer dinaikkan, jika terlalu jarang, diturunkan. Skor kesalahan
   semua ATM dalam satu bank digabung (bukan per ATM), dan tetap berada di dalam bank.
4. **Laporan Senin.** Rekomendasi isi per ATM beserta peluang kehabisan, pada service level yang dipilih operator
   (90/95/98/99%). Alat ini tersedia sebagai dashboard Streamlit dengan empat tab: Laporan Senin, Kinerja,
   Kolaborasi federated, dan Tentang.

Diffusion tidak dipakai di alat ini karena kuantil kumulatif sudah menjawab keputusan dan diffusion meremehkan
risiko ekor di simulasi.

### 4.4 Hasil
**Akurasi (CRPS total 7 hari, periode uji):** Seasonal naive 0.556, Local 0.471, FedProx 0.416, FedProx + fine-tune
0.399, Clustered FL 0.371, **FedAvg 0.361**, Centralized 0.366. FedAvg setara dengan menggabungkan semua data,
tanpa memindahkan data. Berbeda dari simulasi, di data riil FedAvg mengalahkan FedProx, sehingga pemilihan model
lewat validasi menjadi bagian dari sistem. Clustered FL tidak menemukan kelompok (silhouette 0.14), wajar karena
pembagian ATM ke bank dilakukan acak.

**Siapa yang diuntungkan:** Bank Zeta (baru) CRPS Local 1.592 → FedAvg 0.319 (turun 80%). Tanpa kolaborasi, Zeta
bisa mencapai target kehabisan hanya dengan menimbun 56% uang menganggur; dengan federated cukup 21%, setara bank
besar. Bank menengah membaik 7–9%, Epsilon (kecil) 13%, sedangkan bank besar Alfa justru 5% lebih buruk.
Temuan simulasi berulang di data riil: kolaborasi paling menolong pemain dengan data paling sedikit.

**Kejujuran (coverage kuantil kumulatif 7 hari):** untuk target 90/95/98/99%, model tanpa ACI mencapai
81.5/88.9/94.9/97.1% (terlalu percaya diri). ACI per ATM gagal di level tinggi (target 98% malah 88.3%) karena tiap
ATM punya sedikit pengamatan. **ACI per bank** mencapai 89.2/94.3/97.6/98.8%. Ini contoh proses riset yang jujur:
ide pertama gagal, dianalisis, lalu diperbaiki.

**Dampak bisnis (backtest 23 minggu × 111 ATM, target 95%):** aturan praktis ×1.2 membuat 11.4% siklus ATM
kehabisan dengan 21.3% uang menganggur; FedAvg tanpa ACI 11.0% / 15.2%; **KasPintar (FedAvg + ACI) 5.4% / 20.0%**;
Centralized + ACI 4.8% / 20.9%. Agar aturan praktis mencapai kehabisan yang sama (5.4%), faktornya harus 1.29 dengan
uang menganggur 29.3%, jadi KasPintar menghemat 32% uang menganggur. Biaya ilustratif turun dari 3.49 ke 1.70 unit
per ATM-minggu (−51%), dan urutannya tetap sama pada asumsi biaya lain.

**Minggu sulit:** pada minggu belanja Natal (15 Desember 1997), aturan praktis membuat 75% ATM kosong, KasPintar
6.3%. Jumlah minggu dengan lebih dari 10% ATM kosong: aturan praktis 7, FedAvg tanpa ACI 8, KasPintar 3. Minggu
terburuk KasPintar adalah 16 Maret 1998 (21.6%), karena ACI hanya bereaksi setelah meleset.

### 4.5 Keterbatasan
- Data Inggris 1996–1998, bukan Indonesia; pembagian bank fiktif sehingga perbedaan antar bank lebih kecil dari
  kenyataan.
- Satuan uang NN5 tidak diketahui, sehingga biaya dihitung dalam unit relatif.
- Backtest tidak memodelkan kunjungan darurat dan sisa kas minggu lalu.
- ACI reaktif: lonjakan yang belum pernah terjadi tetap lolos di minggu pertamanya.
- Federated learning belum dilengkapi privasi formal: bobot model masih bisa membocorkan informasi (model
  inversion). Differential privacy dan secure aggregation belum diterapkan.

## 5. Rencana riset lanjutan (S2)
1. **Data Indonesia**: ATM dan likuiditas dengan efek Lebaran dan gajian; kalender acara sebagai input agar lonjakan
   bisa diantisipasi, bukan hanya direaksi.
2. **Diffusion bersyarat stres**: skenario krisis atau bencana untuk stress testing, dengan risiko ekor yang
   terkalibrasi.
3. **Privasi formal**: DP-FedAvg dan secure aggregation, serta pengukuran trade-off privasi vs akurasi.
4. **Conformal federated**: jaminan coverage lintas bank tanpa berbagi skor kalibrasi.

## 6. Penerapan lain di Indonesia
Metode ini cocok untuk masalah dengan empat ciri: keputusan berbasis risiko, data tersebar dan rahasia, banyak
pemain kecil dengan data sedikit, serta lonjakan musiman atau shock. Contoh konkret:
- **Likuiditas BPR/BPRS dan koperasi**: cadangan kas tiap kantor menjelang Lebaran dan gajian. Ribuan BPR kecil
  dengan data sedikit dan rahasia, persis seperti Bank Zeta; kolaborasi bisa difasilitasi asosiasi atau regulator.
- **Distribusi uang tunai per daerah** (Bank Indonesia ke kantor perwakilan dan bank) saat Ramadan dan Lebaran.
- **Plafon kredit UMKM dari arus kas transaksi QRIS/e-wallet** yang tersebar di banyak penyedia jasa pembayaran.
- **Stok obat dan darah di puskesmas/rumah sakit** untuk musim DBD atau wabah; data pasien dilindungi UU PDP.
- **Harga dan stok pangan daerah** (Tim Pengendalian Inflasi Daerah): kapan operasi pasar dan berapa cadangan.
- **Stress testing pascabencana** dengan diffusion: skenario lonjakan penarikan tunai atau logistik di daerah
  terdampak.

Tantangan penerapan nyata: kerja sama kelembagaan (siapa yang menjadi server federated) dan privasi formal.

## 7. Glosarium singkat
- **Kuantil 95%**: nilai yang menurut model hanya terlampaui 5% dari waktu.
- **Pinball loss**: loss untuk melatih kuantil; menghukum secara asimetris.
- **CRPS**: skor kualitas seluruh distribusi prediksi; makin kecil makin baik.
- **Coverage**: proporsi nilai aktual yang jatuh di dalam/di bawah prediksi; jujur bila ≈ target.
- **FedAvg / FedProx / Clustered FL**: varian federated learning (rata-rata bobot; dengan penalti kedekatan ke model
  global; dengan pengelompokan bank yang mirip).
- **ACI**: kalibrasi conformal online yang menyesuaikan buffer setelah tiap kesalahan, tahan terhadap perubahan pola.
- **Backtest**: menjalankan kebijakan pada periode lalu yang tidak dilihat model, lalu membandingkan dengan kejadian
  aktual.
