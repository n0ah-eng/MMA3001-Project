"""Tests for the ML gap filler (synthetic data with a known daily cycle)."""

import numpy as np
import pandas as pd
import pytest

from sensorgap.interpolation import fill_linear
from sensorgap.ml import DAY, FEATURES, MLGapFiller, gap_features, make_training_set
from sensorgap.validation import insert_gaps, score


def daily_series(days=40, noise=0.05, seed=0):
    """Temperature-like series: 22 C + 2 C daily cycle + small noise."""
    n = days * DAY
    idx = pd.date_range("2024-03-04", periods=n, freq="10min", tz="Australia/Melbourne")
    t = np.arange(n) / 6  # hours
    rng = np.random.default_rng(seed)
    return pd.Series(22 + 2 * np.sin(2 * np.pi * (t - 9) / 24) + noise * rng.standard_normal(n), index=idx)


def test_gap_features_shapes_and_linear_estimate():
    s = daily_series(days=3, noise=0)
    masked = s.copy()
    masked.iloc[300:310] = np.nan
    X, y_lin, pos = gap_features(masked)
    assert list(X.columns) == FEATURES
    assert np.array_equal(pos, np.arange(300, 310))
    assert np.allclose(y_lin, fill_linear(masked).iloc[pos])
    assert ((X["pos_frac"] > 0) & (X["pos_frac"] < 1)).all()
    assert np.allclose(X["gap_hours"], 11 * 10 / 60)  # 10 missing -> 11 intervals


def test_analog_feature_is_exact_for_a_perfectly_periodic_signal():
    # With an exactly repeating day, "yesterday's shape" reconstructs the gap perfectly.
    s = daily_series(days=3, noise=0)
    masked = s.copy()
    masked.iloc[200:236] = np.nan  # 6-hour gap on day 2
    X, y_lin, pos = gap_features(masked)
    assert (X["analog_missing"] == 0).all()
    assert np.allclose(y_lin + X["analog_minus_lin"], s.iloc[pos])


def test_analog_missing_flag_on_first_day():
    s = daily_series(days=2, noise=0)
    masked = s.copy()
    masked.iloc[20:30] = np.nan  # no data 24 h earlier
    X, _, _ = gap_features(masked)
    assert (X["analog_missing"] == 1).all() and (X["analog_minus_lin"] == 0).all()


def test_training_targets_are_residuals_from_linear():
    s = daily_series(days=10)
    X, r, groups, gap_len = make_training_set(s, [6, 36], n_gaps=5, seeds=(0,))
    assert len(X) == len(r) == len(groups) == len(gap_len) > 0
    assert set(np.unique(groups)) <= set(range(5))
    assert set(np.unique(gap_len)) == {6, 36}


@pytest.fixture(scope="module")
def fitted_ridge():
    s = daily_series()
    split = int(0.7 * len(s))
    return s, split, MLGapFiller("ridge").fit(s.iloc[:split], [1, 6, 36, 144])


def test_ml_beats_linear_on_long_gaps(fitted_ridge):
    s, split, filler = fitted_ridge
    start = split + 2 * DAY
    masked, hidden = insert_gaps(s, np.array([start]), 144)  # 24 h gap in the test period
    ml = score(s, filler(masked), masked, hidden)["rmse"]
    lin = score(s, fill_linear(masked), masked, hidden)["rmse"]
    assert ml < 0.5 * lin  # the daily cycle is learnable; linear misses it entirely


def test_gating_falls_back_to_linear_on_short_gaps(fitted_ridge):
    s, split, filler = fitted_ridge
    assert filler.min_gap_len_ > 1  # CV found no benefit on 10-min gaps
    masked = s.copy()
    masked.iloc[[split + 50, split + 500]] = np.nan
    assert np.allclose(filler(masked), fill_linear(masked))


def test_cv_table_and_attributes(fitted_ridge):
    _, _, filler = fitted_ridge
    assert set(filler.cv_by_gap_.columns) == {"rmse_linear", "rmse_ml"}
    assert filler.cv_rmse_ > 0 and filler.n_train_ > 0 and filler.fit_time_s_ > 0


def test_unknown_model_and_unfitted_use():
    with pytest.raises(ValueError):
        MLGapFiller("neural_net")
    with pytest.raises(RuntimeError):
        MLGapFiller("ridge")(daily_series(days=1))
