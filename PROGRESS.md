# Log Progres

Catatan kronologis tiap tahap pengembangan. Entri terbaru di atas.
Setiap sesi baru: baca `CLAUDE.md` lalu entri teratas di sini.

---

## 2026-10-08 · Tahap 0–5: Fondasi prototipe

**Dikerjakan**
- Rencana awal disepakati: Probabilistic forecasting + Federated learning pada data simulasi
  (diffusion ditunda).
- Paket `src/fedprob`: simulator + 5 skenario, windowing, baseline (Seasonal Naive, ETS),
  QuantileMLP, FedAvg/FedProx/fine-tune, metrik (CRPS, coverage), runner eksperimen + cache, plot.
- Notebook 01–05 (dibangkitkan dari `scripts/build_notebooks.py`), dashboard Streamlit, unit test.

**Hasil**
- _(diisi setelah notebook dieksekusi)_

**Berikutnya (kandidat)**
- [ ] Hubungkan repo ke GitHub (menunggu URL repo dari pemilik)
- [ ] Analisis hasil & tulis kesimpulan di notebook 05
- [ ] Conformal prediction untuk menjamin coverage
- [ ] Diffusion model untuk generasi skenario (arah riset ke-2)
