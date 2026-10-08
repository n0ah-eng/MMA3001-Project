"""Tests for sensorgap.io (payload parsing and CSV loading)."""

import math

import pandas as pd
import pytest
from conftest import make_payload

from sensorgap.io import LOCAL_TZ, VARIABLE_MAP, load_env_csv, parse_payload


def test_parse_payload_maps_names_and_values():
    row = parse_payload(make_payload(100, 22.5, 610))
    assert row == {"time_unix": 100, "temperature_C": 22.5, "co2_ppm": 610.0}


def test_parse_payload_missing_value_is_nan_not_zero():
    row = parse_payload(make_payload(100, None, 610))
    assert math.isnan(row["temperature_C"])


def test_parse_payload_rejects_unit_change():
    # A temperature reported in °F must not be silently mixed with °C data.
    with pytest.raises(ValueError, match="Unexpected unit"):
        parse_payload(make_payload(100, 70.0, 500, unit_temp="°F"))


@pytest.mark.parametrize("bad", ["not json", "[]", None])
def test_parse_payload_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_payload(bad)


def test_load_env_csv_structure(raw_env_csv):
    df = load_env_csv(raw_env_csv)
    assert len(df) == 4                        # corrupt row dropped
    assert df.attrs["n_bad_payloads"] == 1
    assert set(VARIABLE_MAP.values()) <= set(df.columns)  # all columns present
    assert str(df["time"].dt.tz) == LOCAL_TZ
    # IDs must stay strings: 13-digit IDs as floats would lose precision.
    assert pd.api.types.is_string_dtype(df["sensor_id"])
