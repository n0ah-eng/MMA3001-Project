"""Loading raw Monash Smart Infrastructure sensor exports into tidy tables.

The environmental-sensor export stores every upload as one CSV row whose
``jsondata`` column holds a JSON list of measurements, e.g.::

    [{"values": [{"time": 1702273965, "value": 23.8}],
      "variable": {"name": "Temperature", "unit": "°C", ...}}, ...]

This module flattens that into a *wide* table with one row per upload and one
column per measured variable, using consistent snake_case names that carry
their units.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Timezone of the sensors (Clayton campus, Victoria).
LOCAL_TZ = "Australia/Melbourne"

#: Raw variable name in the JSON payload -> tidy column name (unit in suffix).
VARIABLE_MAP: dict[str, str] = {
    "Temperature": "temperature_C",
    "Carbon dioxide": "co2_ppm",
    "Humidity": "humidity_RH",
    "Pressure": "pressure_mb",
    "Formaldehyde": "formaldehyde_ugm3",
    "Light Volatile Organic Compounds": "voc_ppb",
    "Particulate matter 1": "pm1_ugm3",
    "Particulate matter 2.5": "pm2_5_ugm3",
    "Particulate matter 4": "pm4_ugm3",
    "Particulate matter 10": "pm10_ugm3",
    "Battery": "battery_V",
}

#: Expected unit for each raw variable; used to detect unexpected unit changes.
EXPECTED_UNITS: dict[str, str] = {
    "Temperature": "°C",
    "Carbon dioxide": "ppm",
    "Humidity": "HR%",
    "Pressure": "mb",
    "Formaldehyde": "µg/m3",
    "Light Volatile Organic Compounds": "ppb",
    "Particulate matter 1": "µg/m3",
    "Particulate matter 2.5": "µg/m3",
    "Particulate matter 4": "µg/m3",
    "Particulate matter 10": "µg/m3",
    "Battery": "V",
}


def parse_payload(payload: str) -> dict[str, float]:
    """Flatten one ``jsondata`` payload into ``{column_name: value}``.

    Args:
        payload: JSON string from the ``jsondata`` column.

    Returns:
        Dictionary with key ``"time_unix"`` (int seconds, UTC) and one key per
        recognised variable (see :data:`VARIABLE_MAP`). Variables whose value is
        absent from the payload are set to ``NaN`` rather than dropped, so a
        missing reading is visible downstream.

    Raises:
        ValueError: If the payload is not valid JSON, contains no timestamp,
            or reports a unit that differs from :data:`EXPECTED_UNITS`
            (a silent unit change would corrupt every later calculation).

    Example:
        >>> p = '[{"values": [{"time": 10, "value": 21.5}], "variable": {"name": "Temperature", "unit": "°C"}}]'
        >>> parse_payload(p)
        {'time_unix': 10, 'temperature_C': 21.5}
    """
    try:
        items = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    row: dict[str, float] = {}
    for item in items:
        var = item.get("variable", {})
        name = var.get("name")
        if name not in VARIABLE_MAP:
            continue  # unknown variable: ignore rather than fail the whole file
        unit = var.get("unit")
        if unit != EXPECTED_UNITS[name]:
            raise ValueError(f"Unexpected unit {unit!r} for {name!r}")
        for tv in item.get("values", []):
            if "time" in tv and "time_unix" not in row:
                row["time_unix"] = int(tv["time"])
            # Some payloads omit 'value' -> record as missing, not zero.
            row[VARIABLE_MAP[name]] = float(tv["value"]) if tv.get("value") is not None else np.nan

    if "time_unix" not in row:
        raise ValueError("Payload contains no timestamp")
    return row


def load_env_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw environmental-sensor CSV export into a wide table.

    Args:
        path: Path to the raw CSV (columns include ``sensorid``, ``jsondata``,
            ``createdate``).
        nrows: Optional limit on rows read (useful for quick tests).

    Returns:
        DataFrame with columns ``sensor_id`` (str), ``time`` (tz-aware,
        Australia/Melbourne; the time the *sensor* took the reading),
        ``upload_time`` (tz-aware; when the server received it), and one float
        column per variable in :data:`VARIABLE_MAP`. Rows whose payload cannot
        be parsed are dropped and counted in ``df.attrs["n_bad_payloads"]``.

    Note:
        ``time`` (measurement time) and ``upload_time`` differ: the export
        contains many re-uploads of old readings. Always analyse on ``time``.
    """
    raw = pd.read_csv(
        path,
        usecols=["sensorid", "jsondata", "createdate"],
        dtype={"sensorid": str},
        nrows=nrows,
    )

    records, keep, n_bad = [], [], 0
    for i, payload in enumerate(raw["jsondata"]):
        try:
            records.append(parse_payload(payload))
            keep.append(i)
        except ValueError:
            n_bad += 1

    wide = pd.DataFrame.from_records(records)
    wide.insert(0, "sensor_id", raw["sensorid"].iloc[keep].to_numpy())
    wide.insert(1, "time", pd.to_datetime(wide.pop("time_unix"), unit="s", utc=True).dt.tz_convert(LOCAL_TZ))
    wide.insert(
        2,
        "upload_time",
        pd.to_datetime(raw["createdate"].iloc[keep].to_numpy(), format="ISO8601", utc=True).tz_convert(LOCAL_TZ),
    )
    # Guarantee every expected column exists, even if absent from this extract.
    for col in VARIABLE_MAP.values():
        if col not in wide:
            wide[col] = np.nan
    wide.attrs["n_bad_payloads"] = n_bad
    return wide


def load_locations(path: str | Path, sheet: str = "ENV") -> pd.DataFrame:
    """Load the sensor ID → location lookup table.

    Args:
        path: Path to ``Sensor_ID_and_Locations.xlsx``.
        sheet: Sheet name; ``"ENV"`` holds the environmental sensors.

    Returns:
        DataFrame indexed by ``sensor_id`` (str) with columns ``building``,
        ``room``, ``area`` and ``label``.
    """
    loc = pd.read_excel(path, sheet_name=sheet, dtype={"ID": str})
    loc = loc.rename(
        columns={"ID": "sensor_id", "Building": "building", "Floor/Room": "room",
                 "Area": "area", "Physical Label": "label"}
    )
    return loc.set_index("sensor_id")[["building", "room", "area", "label"]]
