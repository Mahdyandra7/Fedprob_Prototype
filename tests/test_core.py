"""Uji kewarasan (sanity check) komponen inti."""

import numpy as np
import pytest
import torch

from fedprob import QUANTILES
from fedprob.data import build_all_clients, get_scenario, oracle_quantiles, simulate
from fedprob.federated import FedConfig, average_weights, run_federated
from fedprob.metrics import coverage, crps_approx, evaluate, pinball
from fedprob.models import QuantileMLP, seasonal_naive_quantiles
from fedprob.training import TrainConfig, predict, train_with_early_stopping


@pytest.fixture(scope="module")
def sim():
    return simulate(get_scenario("normal"))


# --- simulator --------------------------------------------------------------

def test_shape_and_reproducible(sim):
    assert sim.y.shape == (1096, 8)
    again = simulate(get_scenario("normal"))
    np.testing.assert_allclose(sim.y, again.y)


def test_alpha_zero_gives_same_process():
    s = simulate(get_scenario("normal", alpha=0.0))
    # parameter proses (tanpa skala) identik untuk semua bank
    weekly = np.stack([b.weekly for b in s.banks])
    assert np.allclose(weekly, weekly[0])
    assert len({round(b.trend, 9) for b in s.banks}) == 1
    assert len({round(b.noise_sd, 9) for b in s.banks}) == 1


def test_limited_history_masks_data():
    s = simulate(get_scenario("data_langka"))
    for k in (5, 6, 7):
        assert np.isnan(s.y[: s.start_idx[k], k]).all()
        assert not np.isnan(s.y[s.start_idx[k]:, k]).any()
        assert s.val_start - s.start_idx[k] == 120


def test_windows_no_leakage(sim):
    c = build_all_clients(sim)[0]
    H = c.train.y.shape[1]
    assert c.train.origins.max() + H < sim.val_start
    assert c.val.origins.min() >= sim.val_start - 1
    assert c.val.origins.max() + H < sim.test_start
    assert c.test.origins.min() >= sim.test_start - 1


# --- metrik -----------------------------------------------------------------

def test_oracle_is_calibrated(sim):
    """Kuantil oracle harus terkalibrasi: coverage ~ nominal & pinball terkecil."""
    rng = np.random.default_rng(1)
    ys, qs = [], []
    for k in range(sim.config.n_banks):
        for t in rng.integers(100, sim.config.n_days - 15, size=25):
            qs.append(oracle_quantiles(sim, k, int(t), 14, QUANTILES, n_samples=1000, seed=int(t)) / sim.banks[k].scale)
            ys.append(sim.y[t + 1: t + 15, k] / sim.banks[k].scale)
    y, q = np.stack(ys), np.stack(qs)
    assert abs(coverage(y, q, 0.8) - 0.8) < 0.05
    assert abs(coverage(y, q, 0.5) - 0.5) < 0.06
    # kuantil yang digeser harus lebih buruk
    assert crps_approx(y, q) < crps_approx(y, q + 0.5)
    assert crps_approx(y, q) < crps_approx(y, (q - q[..., 3:4]) * 0.5 + q[..., 3:4])


def test_pinball_known_value():
    y = np.array([[1.0]])
    q = np.array([[[0.0, 1.0, 2.0]]])
    np.testing.assert_allclose(pinball(y, q, (0.1, 0.5, 0.9)), [0.1, 0.0, 0.1])


# --- model & federated -------------------------------------------------------

def test_quantiles_do_not_cross():
    m = QuantileMLP()
    out = m(torch.randn(32, 56), torch.randn(32, 14, 11))
    assert out.shape == (32, 14, len(QUANTILES))
    assert (out.diff(dim=-1) >= 0).all()


def test_average_weights():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([3.0, 6.0])}
    np.testing.assert_allclose(average_weights([a, b], [2, 1])["w"].numpy(), [1.0, 2.0])


def test_fedavg_single_client_matches_local_quality(sim):
    """Dengan 1 klien & 1 epoch lokal per ronde, FedAvg = training lokal biasa."""
    c = build_all_clients(sim)[:1]
    fed = run_federated(c, FedConfig(rounds=15, local_epochs=1))
    loc = train_with_early_stopping(c, TrainConfig(epochs=15, patience=100))
    f = evaluate(c[0].test.y, predict(fed.model, c[0].test))["CRPS"]
    l_ = evaluate(c[0].test.y, predict(loc.model, c[0].test))["CRPS"]
    sn = evaluate(c[0].test.y, seasonal_naive_quantiles(c[0]))["CRPS"]
    assert abs(f - l_) / l_ < 0.25
    assert f < sn * 1.2  # model belajar sesuatu yang masuk akal
