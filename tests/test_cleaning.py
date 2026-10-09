"""Tests for sensorgap.cleaning."""

import numpy as np
import pandas as pd
import pytest

from sensorgap.cleaning import (
    apply_physical_bounds,
    audit_sensors,
    drop_stale_duplicates,
    find_gaps,
    to_regular_grid,
)
from sensorgap.io import load_env_csv


@pytest.fixture
def clean_df(raw_env_csv):
    return drop_stale_duplicates(load_env_csv(raw_env_csv))


def test_drop_stale_duplicates_removes_reupload(clean_df):
    assert clean_df.attrs["n_duplicates"] == 1
    assert not clean_df.duplicated(["sensor_id", "time"]).any()


def test_drop_stale_duplicates_keeps_earliest_upload(clean_df):
    a = clean_df[clean_df.sensor_id == "A"]
    assert a["upload_time"].max().month == 3  # the July re-upload was discarded


def test_apply_physical_bounds():
    df = pd.DataFrame({"co2_ppm": [300.0, 500.0, 9000.0], "humidity_RH": [50.0, 101.0, 40.0]})
    out = apply_physical_bounds(df)
    assert out["co2_ppm"].isna().tolist() == [True, False, True]
    assert out.attrs["n_out_of_bounds"] == {"co2_ppm": 2, "humidity_RH": 1}
    assert df["co2_ppm"].notna().all()  # input not modified


def test_audit_sensors(clean_df):
    s = audit_sensors(clean_df)
    assert s.loc["A", "n_readings"] == 2
    assert s.loc["A", "median_interval_min"] == pytest.approx(10.0)


def test_to_regular_grid_inserts_nan_for_missing_bins():
    t = pd.date_range("2024-03-01", periods=6, freq="10min", tz="Australia/Melbourne")
    df = pd.DataFrame({"sensor_id": "X", "time": t.delete([2, 3]), "temperature_C": [1.0, 2, 5, 6]})
    grid = to_regular_grid(df, "X", ["temperature_C"])
    assert len(grid) == 6
    assert grid["temperature_C"].isna().tolist() == [False, False, True, True, False, False]


def test_to_regular_grid_across_daylight_saving_end():
    # Melbourne clocks go back 03:00 -> 02:00 on 2024-04-07; the repeated local hour
    # must not merge or drop samples. 3 h of 10-min data = 18 samples.
    t = pd.date_range("2024-04-06 15:00", periods=18, freq="10min", tz="UTC").tz_convert("Australia/Melbourne")
    df = pd.DataFrame({"sensor_id": "X", "time": t, "v": np.arange(18.0)})
    grid = to_regular_grid(df, "X", ["v"])
    assert len(grid) == 18
    assert grid["v"].notna().all()
    assert grid.index.is_unique


def test_to_regular_grid_unknown_sensor():
    df = pd.DataFrame({"sensor_id": ["X"], "time": [pd.Timestamp("2024", tz="UTC")], "v": [1.0]})
    with pytest.raises(KeyError):
        to_regular_grid(df, "nope", ["v"])


def test_find_gaps():
    idx = pd.date_range("2024-01-01", periods=8, freq="10min")
    s = pd.Series([1, np.nan, np.nan, 4, 5, np.nan, 7, np.nan], index=idx)
    gaps = find_gaps(s)
    assert gaps["n_missing"].tolist() == [2, 1, 1]
    assert gaps["start"].iloc[0] == idx[1]


def test_find_gaps_none():
    assert find_gaps(pd.Series([1.0, 2.0])).empty
