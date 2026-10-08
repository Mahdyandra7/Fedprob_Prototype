"""Uji kewarasan untuk tahap 6-8: conformal, clustered FL, diffusion, metrik jalur."""

import numpy as np
import torch

from fedprob import QUANTILES
from fedprob.conformal import aci, cqr
from fedprob.data import get_scenario, simulate
from fedprob.federated import cluster_clients
from fedprob.metrics import coverage, crps_samples, independent_paths, path_min_metrics
from fedprob.models import DiffusionForecaster

Z = np.array([-1.645, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.645])


def _overconfident(n, H=14, scale=0.5, seed=0):
    """y ~ N(0,1) tetapi prediksi kuantil hanya selebar N(0, scale^2)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, H)).astype(np.float32)
    q = np.broadcast_to(scale * Z, (n, H, 7)).astype(np.float32).copy()
    return y, q


def test_cqr_fixes_overconfidence():
    yv, qv = _overconfident(500, seed=1)
    yt, qt = _overconfident(500, seed=2)
    assert coverage(yt, qt, 0.8) < 0.6
    adj = cqr(yv, qv, qt)
    assert abs(coverage(yt, adj, 0.8) - 0.8) < 0.05
    assert (np.diff(adj, axis=-1) >= 0).all()


def test_aci_adapts_and_has_no_leakage():
    yv, qv = _overconfident(80, seed=3)
    yt, qt = _overconfident(80, seed=4)
    vo, to = np.arange(100, 180), np.arange(200, 280)
    adj = aci(yv, qv, vo, yt, qt, to)
    assert coverage(yt, adj, 0.8) > coverage(yt, qt, 0.8) + 0.15
    # mengubah data di masa depan tidak boleh mengubah penyesuaian origin pertama
    yt2 = yt.copy()
    yt2[40:] += 100
    adj2 = aci(yv, qv, vo, yt2, qt, to)
    np.testing.assert_allclose(adj[0], adj2[0])


def test_groups_do_not_change_old_scenarios():
    a = simulate(get_scenario("normal"))
    b = simulate(get_scenario("normal", n_groups=1, group_alpha=0.7))
    np.testing.assert_allclose(a.y, b.y)


def test_cluster_clients_recovers_groups():
    rng = np.random.default_rng(0)
    d1, d2 = rng.standard_normal(50), rng.standard_normal(50)
    ups = np.stack([d1 + 0.2 * rng.standard_normal(50) for _ in range(4)]
                   + [d2 + 0.2 * rng.standard_normal(50) for _ in range(4)])
    labels, _, _ = cluster_clients(ups)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1 and labels[0] != labels[4]
    labels1, _, _ = cluster_clients(rng.standard_normal((8, 50)))
    assert len(set(labels1)) == 1  # tanpa struktur -> satu kelompok


def test_diffusion_shapes_and_training_step():
    m = DiffusionForecaster(n_samples=16, eval_samples=8)
    xh, xc, y = torch.randn(6, 56), torch.randn(6, 14, 11), torch.randn(6, 14)
    loss = m.training_loss(xh, xc, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.sample(xh, xc).shape == (6, 16, 14)
    q = m.predict_quantiles(xh, xc)
    assert q.shape == (6, 14, len(QUANTILES))
    assert (q.diff(dim=-1) >= 0).all()
    assert torch.isfinite(m.eval_loss(xh, xc, y))


def test_crps_samples_matches_theory():
    rng = np.random.default_rng(0)
    val = crps_samples(rng.standard_normal((3000, 400)), rng.standard_normal(3000))
    assert abs(val - 1 / np.sqrt(np.pi)) < 0.02  # CRPS N(0,1) vs y~N(0,1) = 1/sqrt(pi)


def test_path_metrics_prefer_joint_paths():
    """Jalur random-walk: sampel bersama yang benar harus mengalahkan sampel hari-independen."""
    rng = np.random.default_rng(0)
    n, S, H = 300, 300, 14
    true_paths = np.cumsum(rng.standard_normal((n, S, H)), axis=2)
    y = np.cumsum(rng.standard_normal((n, H)), axis=1)
    q = np.quantile(true_paths, QUANTILES, axis=1).transpose(1, 2, 0)
    indep = independent_paths(q, S)
    assert indep.shape == (n, S, H)
    good = path_min_metrics(true_paths, y, np.zeros(n))
    bad = path_min_metrics(indep, y, np.zeros(n))
    assert good["CRPS_min"] < bad["CRPS_min"]
    assert abs(good["Coverage80_min"] - 0.8) < abs(bad["Coverage80_min"] - 0.8)
