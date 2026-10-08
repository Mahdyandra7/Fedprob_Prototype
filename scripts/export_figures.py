"""Ekspor gambar kunci untuk presentasi ke docs/figures/.

Memakai hasil eksperimen yang sudah di-cache oleh notebook (jalankan notebook dulu).
    python scripts/export_figures.py            # gambar simulasi (01-07)
    python scripts/export_figures.py --kasus    # gambar studi kasus ATM (kasus_*)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedprob.data import get_scenario, simulate
from fedprob.experiment import method_label, run_experiment_cached
from fedprob.plots import plot_fan, plot_paths, plot_series

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
CACHE = ROOT / "results"
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("disimpan:", OUT / name)


def run(cfg, **kw):
    return run_experiment_cached(cfg, cache_dir=CACHE, verbose=False, **kw)


def main():
    # 1. Data simulasi
    sim = simulate(get_scenario("normal"))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    plot_series(sim, ax=axes[0], title="Saldo harian 8 bank fiktif (skala asli)")
    s = slice(sim.dates.get_loc("2024-02-15"), sim.dates.get_loc("2024-06-15"))
    axes[1].plot(sim.dates[s], sim.y[s, 0], lw=1)
    axes[1].axvline(pd.Timestamp("2024-04-10"), color="red", ls=":", label="Lebaran 2024")
    axes[1].set_title("Detail: pola mingguan, gajian, dan Lebaran (Bank A)")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "01_data_simulasi.png")

    # 2. Data langka: Local vs FedAvg pada bank kecil
    r = run("data_langka")
    k, c = 6, r.clients[6]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, m in zip(axes, ["local", "fedavg"]):
        row = r.metrics.query("method == @m and bank == @c.name").iloc[0]
        plot_fan(r.sim, k, r.pred_original_scale(m, k), c.test.origins, ax=ax,
                 title=f"{c.name} (histori 4 bulan): {method_label(m)}  |  CRPS {row.CRPS:.3f}")
    fig.tight_layout()
    save(fig, "02_data_langka_local_vs_fedavg.png")

    # 3. Siapa yang diuntungkan (3 seed)
    focus = ("oracle", "local", "fedavg", "fedavg_ft", "central")
    rows = []
    for sd in (42, 7, 123):
        m = run(get_scenario("data_langka", seed=sd), methods=focus).metrics.copy()
        rows.append(m)
    df = pd.concat(rows)
    df["kelompok"] = np.where(df.bank.isin(["Bank F", "Bank G", "Bank H"]), "bank kecil\n(data langka)", "bank lain")
    piv = df.groupby(["kelompok", "method"]).CRPS.mean().unstack()[list(focus)].rename(columns=method_label)
    ax = piv.plot.bar(figsize=(9, 4), rot=0)
    ax.set_ylabel("CRPS (lebih kecil lebih baik)")
    ax.set_title("Federated learning paling menolong bank dengan data sedikit (rata-rata 3 seed)")
    ax.legend(fontsize=8)
    save(ax.figure, "03_siapa_diuntungkan.png")

    # 4. Shock: FedAvg vs FedAvg + ACI
    r = run("shock")
    k, c = 5, r.clients[5]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, m in zip(axes, ["fedavg", "fedavg+aci"]):
        row = r.metrics.query("method == @m and bank == @c.name").iloc[0]
        plot_fan(r.sim, k, r.pred_original_scale(m, k), c.test.origins, ax=ax,
                 title=f"Shock, {c.name}: {method_label(m)}  |  coverage 80% = {row.Coverage80:.0%}")
    fig.tight_layout()
    save(fig, "04_shock_conformal.png")

    # 5. Clustered FL: kelompok ditemukan dari update bobot
    r = run("klaster")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(r.extras["cluster_similarity"], cmap="Blues", vmin=-1, vmax=1)
    names = [b.name for b in r.sim.banks]
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Kemiripan update bobot antar bank\n(server menemukan 2 kelompok tanpa melihat data)")
    fig.colorbar(im)
    save(fig, "05_clustered_similarity.png")

    # 6. Sweep heterogenitas
    sweep = []
    for a in (0.0, 0.3, 0.6, 0.9):
        rr = run(get_scenario("normal", alpha=a), methods=("local", "fedavg", "fedprox", "fedavg_ft", "clustered", "central"))
        sweep.append(rr.summary()["CRPS"].rename(a))
    ax = pd.DataFrame(sweep).rename(columns=method_label).plot(marker="o", figsize=(8, 4))
    ax.set_xlabel("alpha (heterogenitas antar bank)")
    ax.set_ylabel("CRPS rata-rata")
    ax.set_title("Makin berbeda pola antar bank, makin perlu personalisasi")
    ax.legend(fontsize=8)
    save(ax.figure, "06_sweep_heterogenitas.png")

    # 7. Diffusion: skenario jalur
    dm = ("oracle", "fedavg", "local", "diffusion_fed", "diffusion_central", "diffusion_fed+aci")
    r = run("normal", methods=dm)
    k, c, oi = 2, r.clients[2], 7
    o = int(c.test.origins[oi])
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_paths(r.sim, k, o, c.denorm(r.extras["diffusion_fed_paths"][k][oi]), ax=ax,
               title=f"Diffusion federated: 100 skenario 14 hari ke depan ({c.name})")
    save(fig, "07_diffusion_skenario.png")


def kasus():
    """Gambar studi kasus KasPintar (data riil NN5)."""
    from fedprob.cash.pipeline import label, run_atm_case_cached
    from fedprob.cash.report import plot_cum_fan, plot_tradeoff, weekly_stockout
    from fedprob.realdata.nn5 import load_nn5

    res = run_atm_case_cached(cache_dir=CACHE, verbose=False)
    sys_ = res.system

    # 1. Data: penarikan harian beberapa ATM
    d = load_nn5()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for j in (0, 40, 90):
        ax.plot(d.dates, d.y[:, j], lw=0.7, label=d.names[j])
    for h in ("1996-12-25", "1997-12-25"):
        ax.axvline(pd.Timestamp(h), color="red", ls=":", lw=1)
    ax.set_title("Penarikan tunai harian 3 ATM di Inggris (NN5); garis merah = Natal")
    ax.legend(fontsize=8)
    save(fig, "kasus_01_data_nn5.png")

    # 2. Fan chart kumulatif: minggu biasa vs menjelang Natal
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    plot_cum_fan(res, 0, 2, ax=axes[0])
    plot_cum_fan(res, 0, 3, ax=axes[1])
    fig.tight_layout()
    save(fig, "kasus_02_fan_kumulatif.png")

    # 3. Trade-off kehabisan vs uang menganggur
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_tradeoff(res, ["aturan_praktis", "local", res.extras["selected"], sys_], ax=ax)
    ax.axvline(5, color="grey", ls=":", lw=1)
    save(fig, "kasus_03_tradeoff.png")

    # 4. Kehabisan per minggu (Natal, Paskah)
    ws = pd.DataFrame({label(m): weekly_stockout(res, m) for m in ["aturan_praktis", res.extras["selected"], sys_]})
    ax = ws.plot(marker="o", figsize=(12, 4))
    for h in ("1997-12-25", "1998-04-10"):
        ax.axvline(pd.Timestamp(h), color="red", ls=":", lw=1)
    ax.axhline(5, color="grey", ls="--", lw=1)
    ax.set_ylabel("% ATM kehabisan")
    ax.set_title("ATM kehabisan per minggu (garis merah: Natal, Jumat Agung; target 5%)")
    save(ax.figure, "kasus_04_kehabisan_mingguan.png")

    # 5. Per bank: aturan praktis vs local vs sistem (kehabisan & uang menganggur)
    rows = []
    for b in range(len(res.data.banks)):
        for m in ["aturan_praktis", "local+aci", sys_]:
            r = res.backtest(m, level=0.95, atms=res.data.atms_of(b))
            rows.append({"bank": res.bank_name(b), "kebijakan": label(m),
                         "kehabisan": r["kehabisan_%"], "menganggur": r["menganggur_%"]})
    df = pd.DataFrame(rows)
    order = [res.bank_name(b) for b in range(len(res.data.banks))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, col, title in [(axes[0], "kehabisan", "% siklus ATM kehabisan (target 5%)"),
                           (axes[1], "menganggur", "uang menganggur (% dari permintaan)")]:
        df.pivot(index="bank", columns="kebijakan", values=col).loc[order].plot.bar(ax=ax, rot=0, legend=ax is axes[0])
        ax.set_title(title)
        ax.set_xlabel("")
    axes[0].axhline(5, color="grey", ls="--", lw=1)
    axes[0].set_ylim(0, 17)
    axes[0].legend(fontsize=8, ncol=2, loc="upper center")
    fig.suptitle("Per bank, target 95% (Bank Zeta = bank baru, histori 12 minggu)")
    fig.tight_layout()
    save(fig, "kasus_05_per_bank.png")

if __name__ == "__main__":
    import sys

    kasus() if "--kasus" in sys.argv else main()
