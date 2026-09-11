from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "building_id",
    "building_name",
    "water_consumption_litres",
    "occupancy",
    "temperature",
    "is_anomaly",
    "anomaly_type",
    "event_id",
    "is_synthetic",
]


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    # Check if required columns (or their aliases) are present
    missing_cols = []
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            continue
        # Check alias
        if col == "water_consumption_litres" and "water_consumption_liters" in df.columns:
            continue
        if col == "occupancy" and "occupancy_estimate" in df.columns:
            continue
        if col == "temperature" and "temperature_c" in df.columns:
            continue
        if col == "is_anomaly" and "anomaly_label" in df.columns:
            continue
        if col == "event_id" and "anomaly_event_id" in df.columns:
            continue
        missing_cols.append(col)

    passed = len(missing_cols) == 0
    return {
        "passed": passed,
        "missing_columns": missing_cols,
        "total_columns": int(len(df.columns)),
    }


def validate_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    water_col = "water_consumption_litres" if "water_consumption_litres" in df.columns else "water_consumption_liters"
    label_col = "is_anomaly" if "is_anomaly" in df.columns else "anomaly_label"
    
    critical_cols = ["timestamp", "building_id", water_col, label_col]
    critical_missing = int(df[critical_cols].isna().sum().sum())
    total_missing = int(df.isna().sum().sum())
    
    return {
        "passed": critical_missing == 0,
        "critical_missing_count": critical_missing,
        "total_missing_count": total_missing,
    }


def validate_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    dup_building_ts = int(df.duplicated(subset=["building_id", "timestamp"]).sum())
    return {
        "passed": dup_building_ts == 0,
        "duplicate_building_timestamp_count": dup_building_ts,
    }


def validate_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    water_col = "water_consumption_litres" if "water_consumption_litres" in df.columns else "water_consumption_liters"
    occ_col = "occupancy" if "occupancy" in df.columns else "occupancy_estimate"
    temp_col = "temperature" if "temperature" in df.columns else "temperature_c"

    negative_water = int((df[water_col] < 0).sum())
    negative_occ = int((df[occ_col] < 0).sum()) if occ_col in df.columns else 0
    invalid_temp = int((~df[temp_col].between(-20, 60)).sum()) if temp_col in df.columns else 0

    passed = (negative_water == 0) and (negative_occ == 0) and (invalid_temp == 0)
    return {
        "passed": passed,
        "negative_water_count": negative_water,
        "negative_occupancy_count": negative_occ,
        "invalid_temperature_count": invalid_temp,
        "min_water": float(df[water_col].min()),
        "max_water": float(df[water_col].max()),
    }


def validate_anomaly_labels(df: pd.DataFrame) -> Dict[str, Any]:
    label_col = "is_anomaly" if "is_anomaly" in df.columns else "anomaly_label"
    event_col = "event_id" if "event_id" in df.columns else "anomaly_event_id"

    # anomaly == 0 should imply anomaly_type == 'none'
    normal_consistent = bool((df[label_col] == 0).eq(df["anomaly_type"].eq("none")).all())
    # anomaly == 1 should imply event_id is not null
    event_id_consistent = bool((df[label_col] == 1).eq(df[event_col].notna()).all())

    passed = normal_consistent and event_id_consistent
    return {
        "passed": passed,
        "normal_type_consistent": normal_consistent,
        "event_id_consistent": event_id_consistent,
        "anomaly_row_count": int((df[label_col] == 1).sum()),
        "anomaly_rate": float(df[label_col].mean()),
    }


def validate_event_continuity(df: pd.DataFrame) -> Dict[str, Any]:
    label_col = "is_anomaly" if "is_anomaly" in df.columns else "anomaly_label"
    event_col = "event_id" if "event_id" in df.columns else "anomaly_event_id"

    events = df[df[label_col] == 1].dropna(subset=[event_col])
    if events.empty:
        return {"passed": True, "event_count": 0, "discontinuous_events": 0}

    discontinuous = 0
    for _, event_df in events.groupby(event_col):
        sorted_ts = pd.to_datetime(event_df["timestamp"], utc=True).sort_values()
        diffs = sorted_ts.diff().dropna()
        if not (diffs == pd.Timedelta(hours=1)).all():
            discontinuous += 1

    return {
        "passed": discontinuous == 0,
        "event_count": int(events[event_col].nunique()),
        "discontinuous_events": discontinuous,
    }


def validate_full_dataset(df: pd.DataFrame, expected_days: int = 90, expected_buildings: int = 6) -> Dict[str, Any]:
    expected_rows = expected_days * 24 * expected_buildings
    actual_rows = int(len(df))
    actual_buildings = int(df["building_id"].nunique())

    results = {
        "row_count": {
            "passed": actual_rows == expected_rows,
            "actual": actual_rows,
            "expected": expected_rows,
        },
        "building_count": {
            "passed": actual_buildings == expected_buildings,
            "actual": actual_buildings,
            "expected": expected_buildings,
        },
        "schema": validate_schema(df),
        "missing_values": validate_missing_values(df),
        "duplicates": validate_duplicates(df),
        "ranges": validate_ranges(df),
        "anomaly_labels": validate_anomaly_labels(df),
        "event_continuity": validate_event_continuity(df),
    }

    all_passed = all(check["passed"] for check in results.values())
    return {
        "passed": all_passed,
        "checks_passed": sum(1 for check in results.values() if check["passed"]),
        "total_checks": len(results),
        "checks": results,
    }
