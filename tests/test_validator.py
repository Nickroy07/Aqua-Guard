from __future__ import annotations

import pandas as pd
import pytest

from src.data.generator import generate_synthetic_data
from src.data.validator import (
    validate_anomaly_labels,
    validate_duplicates,
    validate_event_continuity,
    validate_full_dataset,
    validate_missing_values,
    validate_ranges,
    validate_schema,
)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return generate_synthetic_data(days=90, buildings=6, seed=42)


def test_validation_clean_data(clean_df: pd.DataFrame) -> None:
    results = validate_full_dataset(clean_df)
    assert results["passed"] is True
    assert results["checks_passed"] == results["total_checks"]


def test_validation_schema_missing_column(clean_df: pd.DataFrame) -> None:
    corrupted = clean_df.drop(columns=["water_consumption_litres", "water_consumption_liters"])
    res = validate_schema(corrupted)
    assert res["passed"] is False
    assert "water_consumption_litres" in res["missing_columns"]


def test_validation_negative_consumption(clean_df: pd.DataFrame) -> None:
    corrupted = clean_df.copy()
    corrupted.loc[0, "water_consumption_litres"] = -50.0
    res = validate_ranges(corrupted)
    assert res["passed"] is False
    assert res["negative_water_count"] == 1


def test_validation_duplicate_building_timestamp(clean_df: pd.DataFrame) -> None:
    corrupted = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    res = validate_duplicates(corrupted)
    assert res["passed"] is False
    assert res["duplicate_building_timestamp_count"] >= 1


def test_validation_corrupted_anomaly_labels(clean_df: pd.DataFrame) -> None:
    corrupted = clean_df.copy()
    # Anomaly marked as 1 but type set to 'none'
    idx = (corrupted["is_anomaly"] == 1).idxmax()
    corrupted.loc[idx, "anomaly_type"] = "none"
    res = validate_anomaly_labels(corrupted)
    assert res["passed"] is False


def test_validation_discontinuous_events(clean_df: pd.DataFrame) -> None:
    corrupted = clean_df.copy()
    event_ids = corrupted.loc[corrupted["is_anomaly"] == 1, "event_id"].dropna().unique()
    if len(event_ids) > 0:
        target_event = event_ids[0]
        # Shift one timestamp in the event to introduce a 2-hour gap
        event_indices = corrupted[corrupted["event_id"] == target_event].index
        if len(event_indices) > 2:
            corrupted.loc[event_indices[-1], "timestamp"] = corrupted.loc[event_indices[-1], "timestamp"] + pd.Timedelta(hours=2)
            res = validate_event_continuity(corrupted)
            assert res["passed"] is False
            assert res["discontinuous_events"] >= 1
