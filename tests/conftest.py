"""Shared pytest fixtures: tiny synthetic files mimicking the raw exports."""

import json

import pandas as pd
import pytest


def make_payload(t: int, temp: float | None = 21.0, co2: float = 500.0, unit_temp: str = "°C") -> str:
    """Build a JSON payload in the same structure as the Monash export."""
    temp_value = {"time": t} if temp is None else {"time": t, "value": temp}
    return json.dumps(
        [
            {"values": [temp_value], "variable": {"name": "Temperature", "unit": unit_temp}},
            {"values": [{"time": t, "value": co2}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}},
        ]
    )


@pytest.fixture
def raw_env_csv(tmp_path):
    """Raw CSV with 3 unique readings, 1 stale re-upload and 1 corrupt row."""
    t0 = 1_710_000_000  # 2024-03-10 (UTC)
    rows = [
        ("A", make_payload(t0, 21.0, 450), "2024-03-10 03:00:00+11"),
        ("A", make_payload(t0 + 600, 21.5, 480), "2024-03-10 03:10:00+11"),
        ("A", make_payload(t0 + 600, 21.5, 480), "2024-07-01 12:00:00+10"),  # stale re-upload
        ("B", make_payload(t0, None, 600), "2024-03-10 03:00:00+11"),          # missing temp value
        ("B", "not json", "2024-03-10 03:00:00+11"),                            # corrupt payload
    ]
    df = pd.DataFrame(rows, columns=["sensorid", "jsondata", "createdate"])
    path = tmp_path / "env.csv"
    df.to_csv(path, index=False)
    return path
