from __future__ import annotations

import pandas as pd

from src.data.generator import BUILDING_PROFILES, generate_synthetic_data

EXPECTED_BUILDING_NAMES = [
    "Engineering Block",
    "Library",
    "Hostel A",
    "Hostel B",
    "Administration",
    "Academic Block",
]

REQUIRED_COLUMNS = {
    "timestamp",
    "building_id",
    "building_name",
    "water_consumption_litres",
    "water_consumption_liters",
    "occupancy",
    "occupancy_estimate",
    "temperature",
    "temperature_c",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_anomaly",
    "anomaly_label",
    "anomaly_type",
    "event_id",
    "anomaly_event_id",
    "is_synthetic",
}


def test_generator_shape_schema_defaults() -> None:
    df = generate_synthetic_data(days=90, buildings=6, seed=42)
    # Expected: 90 * 24 * 6 = 12,960
    assert len(df) == 12960
    assert REQUIRED_COLUMNS.issubset(set(df.columns))
    assert df["building_id"].nunique() == 6
    assert sorted(df["building_name"].unique()) == sorted(EXPECTED_BUILDING_NAMES)


def test_generator_deterministic_seed() -> None:
    a = generate_synthetic_data(seed=42)
    b = generate_synthetic_data(seed=42)
    assert a.equals(b)


def test_generator_different_seed_changes_output() -> None:
    a = generate_synthetic_data(seed=42)
    b = generate_synthetic_data(seed=7)
    assert not a.equals(b)


def test_hourly_continuity_per_building() -> None:
    df = generate_synthetic_data(seed=42)
    for _, bdf in df.groupby("building_id"):
        ts = pd.to_datetime(bdf["timestamp"], utc=True).sort_values()
        assert (ts.diff().dropna() == pd.Timedelta(hours=1)).all()


def test_anomaly_contiguity_and_no_overlap() -> None:
    df = generate_synthetic_data(seed=42)
    anomaly = df[df["is_anomaly"] == 1].copy()
    for _, event_df in anomaly.groupby("event_id"):
        ts = pd.to_datetime(event_df["timestamp"], utc=True).sort_values()
        assert (ts.diff().dropna() == pd.Timedelta(hours=1)).all()

    for _, bdf in df.sort_values("timestamp").groupby("building_id"):
        active = bdf[bdf["is_anomaly"] == 1]
        if active.empty:
            continue
        change = active["event_id"].ne(active["event_id"].shift())
        blocks = change.cumsum()
        assert all(group["event_id"].nunique() == 1 for _, group in active.groupby(blocks))
