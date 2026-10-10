"""Build the cleaned, regularly gridded dataset from the raw export.

Usage (from the repository root):
    python scripts/01_build_clean_dataset.py

Inputs:  data/raw/5EnvSensor_MayToDec2024_180kRows.csv
         data/raw/Sensor_ID_and_Locations.xlsx
Outputs: data/processed/env_clean.parquet         de-duplicated, bound-checked readings
         data/processed/grid_<sensor_id>.parquet  10-min grid per study sensor
         results/data_audit.csv                   per-sensor coverage summary
         results/natural_gaps.csv                 gaps already present in the data
"""

from pathlib import Path

from sensorgap.cleaning import (
    apply_physical_bounds,
    audit_sensors,
    drop_stale_duplicates,
    find_gaps,
    to_regular_grid,
)
from sensorgap.io import load_env_csv, load_locations

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"

#: Sensors with enough continuous data to study, with their deployment window.
#: Windows exclude isolated test readings far outside the main deployment.
STUDY_SENSORS = {
    "6012002000869": ("2024-03-01", "2024-06-13"),  # Bldg 60, G.38 (Compactus_upper)
    "6012002000326": ("2025-09-01", "2026-04-22"),  # location not in lookup table
}
VARIABLES = ["temperature_C", "co2_ppm", "humidity_RH", "pressure_mb"]


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    print("Loading raw export (takes ~1 min)...")
    raw = load_env_csv(RAW / "5EnvSensor_MayToDec2024_180kRows.csv")
    print(f"  rows: {len(raw):,}   unparseable payloads: {raw.attrs['n_bad_payloads']}")

    df = drop_stale_duplicates(raw)
    pct = df.attrs["n_duplicates"] / len(raw)
    print(f"  stale duplicates removed: {df.attrs['n_duplicates']:,} ({pct:.1%})")

    df = apply_physical_bounds(df)
    print(f"  out-of-bounds values set to NaN: {df.attrs['n_out_of_bounds']}")
    df.to_parquet(PROC / "env_clean.parquet")

    locations = load_locations(RAW / "Sensor_ID_and_Locations.xlsx")
    audit = audit_sensors(df, locations)
    audit.to_csv(RES / "data_audit.csv")
    print("\nSensor audit:\n", audit.to_string())

    gap_tables = []
    for sid, (start, end) in STUDY_SENSORS.items():
        grid = to_regular_grid(df, sid, VARIABLES, start=start, end=end)
        grid.to_parquet(PROC / f"grid_{sid}.parquet")
        gaps = find_gaps(grid["temperature_C"]).assign(sensor_id=sid)
        gap_tables.append(gaps)
        print(f"\n{sid}: {len(grid):,} grid points, "
              f"{grid['temperature_C'].isna().mean():.1%} missing, {len(gaps)} gaps")
        if len(gaps):
            print("  gap length (samples) quantiles:",
                  gaps["n_missing"].quantile([0.5, 0.9, 0.99, 1.0]).to_dict())

    import pandas as pd
    pd.concat(gap_tables).to_csv(RES / "natural_gaps.csv", index=False)


if __name__ == "__main__":
    main()
