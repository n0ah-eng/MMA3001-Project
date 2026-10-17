"""Tests for the mask-and-reconstruct validation harness."""

import numpy as np
import pandas as pd
import pytest

from sensorgap.interpolation import INTERPOLATION_METHODS
from sensorgap.validation import insert_gaps, run_gap_experiment, sample_gap_starts, score


@pytest.fixture
def series():
    idx = pd.date_range("2024-03-01", periods=2000, freq="10min", tz="Australia/Melbourne")
    t = np.arange(2000) / 6
    s = pd.Series(22 + np.sin(2 * np.pi * t / 24), index=idx)
    s.iloc[::11] = np.nan  # natural single-sample dropouts, like the real data
    return s


def test_gap_starts_respect_spacing_and_bounds(series):
    obs = series.notna().to_numpy()
    gap_len = 18
    starts = sample_gap_starts(obs, gap_len, 20, np.random.default_rng(0))
    assert len(starts) == 20
    assert starts.min() >= gap_len and starts.max() + 2 * gap_len <= len(series)
    # Consecutive gaps separated by at least one gap length of untouched data.
    assert np.all(np.diff(starts) >= 2 * gap_len)
    # Every gap has an observed sample immediately either side.
    assert obs[starts - 1].all() and obs[starts + gap_len].all()


def test_gap_starts_reproducible(series):
    obs = series.notna().to_numpy()
    a = sample_gap_starts(obs, 6, 10, np.random.default_rng(42))
    b = sample_gap_starts(obs, 6, 10, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_gap_starts_too_short_returns_empty():
    assert sample_gap_starts(np.ones(10, bool), 6, 5, np.random.default_rng(0)).size == 0


def test_gap_starts_invalid_length():
    with pytest.raises(ValueError):
        sample_gap_starts(np.ones(100, bool), 0, 5, np.random.default_rng(0))


def test_insert_gaps_scores_only_true_values(series):
    masked, hidden = insert_gaps(series, np.array([100]), 30)
    assert masked.iloc[100:130].isna().all()
    # Natural NaNs inside the gap are hidden but not scored (no truth exists).
    assert hidden.sum() == series.iloc[100:130].notna().sum()
    assert hidden.sum() < 30


def test_score_perfect_fill_is_zero(series):
    masked, hidden = insert_gaps(series, np.array([200]), 10)
    m = score(series, series, masked, hidden)
    assert m["rmse"] == 0 and m["max_abs"] == 0 and m["bias"] == 0


def test_score_known_error():
    idx = pd.date_range("2024-01-01", periods=5, freq="10min")
    truth = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=idx)
    filled = truth + np.array([0, 1.0, -1.0, 0, 0])
    masked = truth.copy()
    masked.iloc[1:3] = np.nan
    m = score(truth, filled, masked, masked.isna().to_numpy())
    assert m["rmse"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(0.0)


def test_score_rejects_unfilled():
    idx = pd.date_range("2024-01-01", periods=3, freq="10min")
    truth = pd.Series([1.0, 2.0, 3.0], index=idx)
    masked = truth.copy()
    masked.iloc[1] = np.nan
    with pytest.raises(ValueError):
        score(truth, masked, masked, masked.isna().to_numpy())


def test_run_gap_experiment_is_paired(series):
    res = run_gap_experiment(series, INTERPOLATION_METHODS, [1, 6], n_gaps=10, seeds=(0, 1))
    assert len(res) == 2 * 2 * len(INTERPOLATION_METHODS)
    # All methods scored on exactly the same samples for a given (seed, gap_len).
    assert (res.groupby(["seed", "gap_len"])["n_scored"].nunique() == 1).all()
    # Smooth daily cycle: every method should be very accurate on 10-min gaps.
    assert res.loc[res.gap_len == 1, "rmse"].max() < 0.01


def test_gap_starts_respect_allowed_region(series):
    obs = series.notna().to_numpy()
    allowed = np.arange(len(series)) >= 1400  # e.g. test period only
    starts = sample_gap_starts(obs, 6, 10, np.random.default_rng(0), allowed=allowed)
    assert len(starts) > 0
    assert (starts - 6 >= 1400).all()  # gap *and* its buffer inside the region


def test_add_skill_and_summarise_pass_rule():
    from sensorgap.validation import add_skill, summarise
    rows = []
    for method, rmse in [("mean", 1.0), ("good", 0.2), ("lucky", 0.45), ("bad", 0.6)]:
        rows.append({"sensor_id": "S", "location": "L", "variable": "temperature_C", "seed": 0,
                     "gap_len": 6, "method": method, "rmse": rmse, "max_abs": rmse, "bias": 0.0,
                     "overshoot_frac": 0.0, "runtime_s": 0.001, "n_gaps": 5})
    res = add_skill(pd.DataFrame(rows))
    assert res.set_index("method").loc["good", "skill"] == pytest.approx(0.8)
    summ = summarise(res, {"temperature_C": 0.5}, min_skill=0.6).set_index("method")
    # "lucky" meets the RMSE threshold but not the skill rule; "bad" fails both.
    assert summ["passes"].to_dict() == {"bad": False, "good": True, "lucky": False, "mean": False}
