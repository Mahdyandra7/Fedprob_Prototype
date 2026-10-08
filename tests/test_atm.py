"""Uji kewarasan studi kasus ATM (data NN5, target kumulatif, kebijakan isi kas)."""

import numpy as np
import pandas as pd
import pytest

from fedprob.cash.policy import backtest, rule_of_thumb_loads, weekly_origins
from fedprob.conformal import aci_upper
from fedprob.data.windows import build_series_data, calendar_features, merge_clients
from fedprob.realdata.nn5 import BANKS, assign_banks, parse_tsf

TSF = """# contoh
@relation X
@attribute series_name string
@attribute start_timestamp date
@frequency daily
@data
T1:1996-03-18 00-00-00:1,2,?,4
T2:1996-03-18 00-00-00:0,5,6,7
"""


def test_parse_tsf():
    names, start, values = parse_tsf(TSF)
    assert names == ["T1", "T2"]
    assert start == pd.Timestamp("1996-03-18")
    assert values.shape == (4, 2) and np.isnan(values[2, 0]) and values[3, 1] == 7


def test_real_nn5_shape():
    from fedprob.realdata.nn5 import load_nn5

    try:
        d = load_nn5()
    except Exception as e:  # tanpa internet & tanpa file lokal
        pytest.skip(f"NN5 tidak tersedia: {e}")
    assert d.y.shape[1] == 111 and d.y.shape[0] >= 735
    assert d.dates[0] == pd.Timestamp("1996-03-18")
    assert not np.isnan(d.y).any() and (d.y > 0).all()
    assert d.mask.mean() < 0.05


def test_bank_assignment():
    b = assign_banks(111)
    assert np.bincount(b).tolist() == [size for _, size, _ in BANKS]
    np.testing.assert_array_equal(b, assign_banks(111))


def test_cumsum_target():
    y = np.arange(1, 201, dtype=float)
    dates = pd.date_range("2000-01-03", periods=200)
    c = build_series_data(y, calendar_features(dates), 0, 120, 160, "x", target="cumsum", horizon=7)
    t, tgt = c.train.origins[0], c.train.y[0]
    expected = np.cumsum(y[t + 1: t + 8]) / c.center
    np.testing.assert_allclose(tgt, expected, rtol=1e-5)
    assert (np.diff(c.test.y, axis=1) > 0).all()
    np.testing.assert_allclose(c.to_amount(tgt), np.cumsum(y[t + 1: t + 8]), rtol=1e-5)
    m = merge_clients([c, c], "bank")
    assert len(m.train) == 2 * len(c.train)


def test_backtest_known_cases():
    d = np.array([100.0, 120, 80])
    assert backtest(d, d)["kehabisan_%"] == 0 and backtest(d, d)["menganggur_%"] == 0
    assert backtest(d * 0.5, d)["kehabisan_%"] == 100
    r = backtest(d * 1.5, d)
    assert r["kehabisan_%"] == 0 and abs(r["menganggur_%"] - 50) < 1e-9


def test_rule_of_thumb_constant():
    y = np.full(100, 10.0)
    np.testing.assert_allclose(rule_of_thumb_loads(y, np.array([50, 60]), factor=1.2), 84.0)


def test_weekly_origins_sunday():
    dates = pd.date_range("1997-12-01", periods=30)  # Senin
    idx = weekly_origins(dates, np.arange(30))
    assert (dates[idx].dayofweek == 6).all() and len(idx) == 4


def test_aci_upper_reaches_level():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((400, 7))
    q = np.full((400, 7), 0.5)
    adj = aci_upper(y[:100], q[:100], np.arange(100), y[100:], q[100:], np.arange(100, 400), 0.95)
    assert abs((y[100:] <= adj).mean() - 0.95) < 0.03


def test_aci_upper_pooled_reaches_high_level():
    """Dengan skor dari banyak seri, kuantil 98% pun bisa dikalibrasi."""
    from fedprob.conformal import aci_upper_pooled

    rng = np.random.default_rng(1)
    S, n_val, n_test, H = 20, 60, 120, 7
    y = rng.standard_normal((S, n_val + n_test, H))
    q = np.full_like(y, 1.0)  # nominal ~84%, target 98%
    adj = aci_upper_pooled(y[:, :n_val], q[:, :n_val], np.arange(n_val),
                           y[:, n_val:], q[:, n_val:], np.arange(n_val, n_val + n_test), 0.98)
    assert adj.shape == (S, n_test, H)
    assert abs((y[:, n_val:] <= adj).mean() - 0.98) < 0.015
