# Panduan NotebookLM: Essay LPDP + Prototipe FedProb

File ini untuk dibaca sendiri, **tidak perlu diunggah**. Isinya: sumber yang diunggah, pengaturan, dan prompt siap pakai.

## 1. Sumber yang diunggah (satu notebook)

| # | File | Isi | Catatan |
|---|---|---|---|
| 1 | `LPDP_26.docx` (folder project) | Essay LPDP: Personal Statement + Komitmen & Kontribusi | Unggah langsung. Jika .docx ditolak, buka di Word → Save As PDF |
| 2 | `docs/notebooklm/01_prototipe_lengkap.md` | Seluruh prototipe dalam prosa: asal-usul, simulasi, kasus ATM, hasil, keterbatasan, rencana S2, penerapan di Indonesia | Sumber utama project |
| 3 | `docs/kasus_atm.md` | Rincian studi kasus KasPintar | |
| 4 | `docs/alur_presentasi.md` | Alur 13 slide + **Persiapan Tanya-Jawab** | Penting untuk simulasi wawancara |
| 5 | `docs/ringkasan_end_to_end.md` | Alur data → keputusan, glosarium, angka kunci, pembagian waktu | |
| 6 | (opsional) `PROGRESS.md` | Log kronologis pengembangan + keputusan teknis | Berguna bila ditanya "bagaimana prosesnya" |
| 7 | (opsional) Slide deck | Unduh sebagai PDF dari menu Share › Export | Agar NotebookLM tahu isi slide yang ditampilkan |

Jika file `.md` ditolak, ganti ekstensinya menjadi `.txt` atau tempel isinya lewat "Copied text".
Sumber tambahan yang berguna (opsional, cari sendiri): halaman program S2 tujuan, panduan wawancara LPDP, dokumen RPJPN 2025–2045 / strategi nasional AI.

## 2. Pengaturan chat (Configure chat → Custom)

```
Kamu adalah mentor riset dan pewawancara beasiswa LPDP yang kritis tetapi suportif. Aku adalah
penulis essay dan pembuat prototipe dalam sumber. Jawab dalam bahasa Indonesia kecuali aku menulis
dalam bahasa Inggris. Selalu dasarkan jawaban pada sumber dan sebutkan bagian yang dirujuk. Jika
sesuatu tidak ada di sumber, katakan terus terang. Jangan mengarang angka. Utamakan penjelasan
sederhana, lalu tantang logikaku bila ada yang lemah atau tidak konsisten.
```

## 3. Prompt siap pakai

### A. Memahami gambaran besar
1. `Jelaskan hubungan antara essay LPDP dan prototipe FedProb dalam 5 poin: dari pengalaman pribadi, skripsi, pertanyaan riset, prototipe, sampai rencana kontribusi.`
2. `Jelaskan alur KasPintar dari data mentah sampai rekomendasi isi ATM seolah kepada orang awam, lalu ulangi untuk dosen statistika.`
3. `Apa satu kalimat yang paling tepat untuk merangkum kontribusi prototipe ini terhadap pertanyaan riset di essay?`

### B. Menjernihkan ide riset
4. `Pertanyaan riset di essay punya tiga bagian: accurate, honest, dan learning from data that can never be fully shared. Untuk tiap bagian, bukti apa yang sudah ada di prototipe dan apa yang belum terjawab?`
5. `Dari keterbatasan prototipe, susun 3 kemungkinan rumusan pertanyaan riset tesis S2 yang spesifik dan bisa dikerjakan dalam 1–1,5 tahun. Beri kelebihan dan risiko masing-masing.`
6. `Mengapa diffusion tidak dipakai di KasPintar padahal disebut di essay? Bagaimana menjelaskannya agar tidak terlihat seperti arah riset yang ditinggalkan?`
7. `Apa perbedaan temuan simulasi dan data riil (misalnya FedAvg vs FedProx)? Apa pelajaran metodologisnya?`

### C. Konsistensi essay ↔ prototipe ↔ presentasi
8. `Periksa konsistensi antara essay, prototipe, dan alur presentasi. Adakah klaim di essay yang tidak didukung prototipe, atau hasil prototipe yang bertentangan dengan narasi essay?`
9. `Essay menyebut rencana kontribusi jangka pendek, menengah, dan panjang. Hubungkan masing-masing dengan prototipe ini secara konkret (misalnya bahan ajar, proyek IRIS, kolaborasi industri).`
10. `Bagian mana dari essay yang paling mungkin digali pewawancara, dan bagaimana prototipe ini bisa dipakai sebagai bukti?`

### D. Simulasi wawancara LPDP
11. `Lakukan simulasi wawancara LPDP: ajukan satu pertanyaan sekali jalan, tunggu jawabanku, lalu beri umpan balik (isi, struktur, kejelasan) dan skor 1–10 sebelum pertanyaan berikutnya. Campur pertanyaan tentang motivasi, rencana studi, riset, komitmen kembali ke Indonesia, dan prototipe.`
12. `Same as above, but conduct the interview in English and also correct my grammar and word choice briefly.`
13. `Ajukan 10 pertanyaan paling sulit/menjebak tentang prototipe ini (data Inggris, bank fiktif, privasi, kenapa tidak deep learning lebih besar, dll.) beserta poin jawaban yang didukung sumber.`
14. `Cari pengalaman kegagalan atau kelemahan diri yang disebut di essay. Bantu aku menyiapkan jawaban jujur dan kuat jika pewawancara menggalinya, termasuk apa yang sudah berubah, dengan prototipe ini sebagai salah satu bukti.`
15. `Pewawancara bertanya: "Kenapa negara dan universitas tujuan yang disebut di essay, dan bagaimana programnya mendukung riset ini?" Susun jawaban 1 menit berdasarkan essay dan prototipe.`

### E. Dampak untuk Indonesia
16. `Berdasarkan sumber, susun jawaban 45 detik untuk pertanyaan: "Apa penerapan konkret metode ini untuk Indonesia?" Fokus pada dua contoh terkuat.`
17. `Bagaimana prototipe ini mendukung agenda yang disebut di essay (Indonesia Emas 2045, strategi AI nasional, Asta Cita)? Hindari klaim berlebihan.`

### F. Latihan & hafalan
18. `Buat 20 flashcard berisi istilah dan angka kunci dari prototipe (CRPS, coverage, ACI, FedAvg, hasil Zeta, dampak bisnis).`
19. `Uji aku dengan 10 soal pilihan ganda tentang hasil KasPintar, lalu jelaskan jawaban yang salah.`

## 4. Fitur Studio yang disarankan
- **Audio Overview** (podcast), dengan instruksi kustom:
  `Bahas dalam bahasa Indonesia. Fokus pada perjalanan dari essay LPDP ke prototipe KasPintar: masalah, solusi, hasil, keterbatasan, dan rencana S2. Jelaskan istilah teknis dengan analogi sederhana. Akhiri dengan 3 pertanyaan yang mungkin diajukan pewawancara.`
  Bisa didengarkan di perjalanan sebagai latihan.
- **Mind Map**: melihat hubungan essay → pertanyaan riset → tiga arah → prototipe → kontribusi.
- **Flashcards / Quiz**: hafalan angka kunci.
- **Briefing doc / Study guide**: ringkasan cetak untuk dibaca sebelum wawancara.

## 5. Catatan
- NotebookLM bisa salah mengutip angka. Cocokkan angka penting dengan `docs/ringkasan_end_to_end.md` bagian 5.
- Essay adalah dokumen pribadi; notebook NotebookLM bersifat privat kecuali kamu membagikannya.
