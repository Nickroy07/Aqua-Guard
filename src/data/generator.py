from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BuildingProfile:
    building_id: str
    building_name: str
    base_consumption_lph: float
    peak_occupancy: int
    occupancy_variability: float
    night_baseline_factor: float
    building_type: str = "academic"  # "academic", "residential", "office", "library"


BUILDING_PROFILES: List[BuildingProfile] = [
    BuildingProfile("B01", "Engineering Block", 180.0, 500, 0.22, 0.30, "academic"),
    BuildingProfile("B02", "Library", 110.0, 350, 0.18, 0.22, "library"),
    BuildingProfile("B03", "Hostel A", 240.0, 450, 0.15, 0.55, "residential"),
    BuildingProfile("B04", "Hostel B", 250.0, 480, 0.16, 0.52, "residential"),
    BuildingProfile("B05", "Administration", 90.0, 220, 0.14, 0.20, "office"),
    BuildingProfile("B06", "Academic Block", 160.0, 550, 0.25, 0.28, "academic"),
]

ANOMALY_TYPES = ["spike", "sustained_overnight", "gradual_drift"]


def _hour_factor(hour: int, building_type: str = "academic") -> float:
    if building_type == "residential":
        # Residential patterns: morning rise (6-9), daytime lull (10-16), evening peak (17-23), overnight lull (0-5)
        if 0 <= hour <= 5:
            return 0.35
        if 6 <= hour <= 8:
            return 1.15
        if 9 <= hour <= 16:
            return 0.65
        if 17 <= hour <= 22:
            return 1.10
        return 0.60
    elif building_type == "office":
        # Office patterns: 8-17 weekdays peak, minimal evening/night
        if 0 <= hour <= 6:
            return 0.15
        if 7 <= hour <= 8:
            return 0.50
        if 9 <= hour <= 16:
            return 1.05
        if 17 <= hour <= 19:
            return 0.40
        return 0.20
    elif building_type == "library":
        # Library patterns: opens 8, peaks afternoon/evening (13-21)
        if 0 <= hour <= 7:
            return 0.18
        if 8 <= hour <= 12:
            return 0.70
        if 13 <= hour <= 21:
            return 1.05
        return 0.35
    else:
        # Standard academic: lecture & lab hours
        if 0 <= hour <= 5:
            return 0.22
        if 6 <= hour <= 7:
            return 0.45
        if 8 <= hour <= 12:
            return 1.05
        if 13 <= hour <= 17:
            return 0.95
        if 18 <= hour <= 21:
            return 0.45
        return 0.25


def _weekend_factor(is_weekend: bool, building_type: str = "academic") -> float:
    if not is_weekend:
        return 1.0
    if building_type == "residential":
        return 1.15  # Students spend more time in hostels on weekends
    elif building_type == "library":
        return 0.85  # Study during weekends
    elif building_type == "office":
        return 0.25  # Administration largely closed
    else:
        return 0.35  # Academic classrooms mostly closed


def _temperature_from_time(index: int, total_steps: int, rng: np.random.Generator) -> float:
    seasonal_wave = 4.5 * np.sin((2 * np.pi * index) / max(total_steps, 1))
    diurnal_wave = 3.0 * np.sin((2 * np.pi * (index % 24)) / 24)
    return float(24.0 + seasonal_wave + diurnal_wave + rng.normal(0.0, 1.0))


def _sample_event_window(
    rng: np.random.Generator,
    available: np.ndarray,
    duration: int,
    prefer_overnight: bool,
    hours: np.ndarray,
) -> Tuple[int, int] | None:
    max_start = len(available) - duration
    if max_start <= 0:
        return None
    candidates = []
    for start in range(max_start + 1):
        end = start + duration
        if not np.all(available[start:end]):
            continue
        if prefer_overnight and not np.all((hours[start:end] >= 0) & (hours[start:end] <= 5)):
            continue
        candidates.append(start)
    if not candidates:
        return None
    start = int(rng.choice(candidates))
    return start, start + duration


def _inject_anomalies(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    anomaly_event_counter = 0

    out_frames: List[pd.DataFrame] = []
    for building_id, bdf in df.groupby("building_id", sort=False):
        bdf = bdf.sort_values("timestamp").copy()
        n = len(bdf)
        available = np.ones(n, dtype=bool)
        hours = bdf["hour"].to_numpy()

        events_to_create = int(rng.integers(4, 7))

        bdf["anomaly_label"] = 0
        bdf["anomaly_type"] = "none"
        bdf["anomaly_event_id"] = pd.NA

        for _ in range(events_to_create):
            anomaly_type = str(rng.choice(ANOMALY_TYPES, p=[0.45, 0.30, 0.25]))
            if anomaly_type == "spike":
                duration = int(rng.integers(1, 4))
                window = _sample_event_window(rng, available, duration, False, hours)
                if window is None:
                    continue
                start, end = window
                factors = rng.uniform(1.6, 2.4, duration)
            elif anomaly_type == "sustained_overnight":
                duration = int(rng.integers(4, 9))
                window = _sample_event_window(rng, available, duration, True, hours)
                if window is None:
                    continue
                start, end = window
                factors = rng.uniform(1.35, 1.85, duration)
            else:  # gradual_drift
                duration = int(rng.integers(8, 25))
                window = _sample_event_window(rng, available, duration, False, hours)
                if window is None:
                    continue
                start, end = window
                factors = np.linspace(rng.uniform(1.10, 1.25), rng.uniform(1.45, 1.80), duration)

            anomaly_event_counter += 1
            event_id = f"{building_id}-E{anomaly_event_counter:03d}"
            idx = bdf.index[start:end]

            bdf.loc[idx, "water_consumption_litres"] = bdf.loc[idx, "water_consumption_litres"].to_numpy() * factors
            bdf.loc[idx, "anomaly_label"] = 1
            bdf.loc[idx, "anomaly_type"] = anomaly_type
            bdf.loc[idx, "anomaly_event_id"] = event_id

            available[start:end] = False

        out_frames.append(bdf)

    combined = pd.concat(out_frames, ignore_index=True)
    combined["water_consumption_litres"] = combined["water_consumption_litres"].clip(lower=1.0).round(2)
    # Maintain backwards compatibility aliases
    combined["water_consumption_liters"] = combined["water_consumption_litres"]
    combined["is_anomaly"] = combined["anomaly_label"]
    combined["event_id"] = combined["anomaly_event_id"]
    combined["occupancy"] = combined["occupancy_estimate"]
    combined["temperature"] = combined["temperature_c"]
    return combined


def generate_synthetic_data(
    start_date: str = "2026-01-01",
    days: int = 90,
    freq: str = "hourly",
    buildings: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    if freq != "hourly":
        raise ValueError("Only hourly frequency is supported in this prototype.")
    if buildings > len(BUILDING_PROFILES):
        raise ValueError(f"Maximum supported buildings: {len(BUILDING_PROFILES)}")

    rng = np.random.default_rng(seed)
    periods = days * 24
    timestamps = pd.date_range(start=start_date, periods=periods, freq="h", tz="UTC")

    frames: List[pd.DataFrame] = []
    for profile in BUILDING_PROFILES[:buildings]:
        hours = timestamps.hour
        day_of_week = timestamps.dayofweek
        is_weekend = (day_of_week >= 5).astype(int)

        occupancy = []
        temperature = []
        water = []

        for idx, ts in enumerate(timestamps):
            hour_fac = _hour_factor(ts.hour, profile.building_type)
            weekend_fac = _weekend_factor(ts.dayofweek >= 5, profile.building_type)

            occ_mean = profile.peak_occupancy * hour_fac * weekend_fac
            occ_noise = rng.normal(0.0, profile.peak_occupancy * profile.occupancy_variability * 0.08)
            occ_value = max(0.0, occ_mean + occ_noise)

            temp_value = _temperature_from_time(idx, periods, rng)

            base = profile.base_consumption_lph * (profile.night_baseline_factor if ts.hour <= 5 else 1.0)
            occ_component = 0.65 * occ_value
            temp_component = max(0.0, temp_value - 20.0) * 1.8
            noise = rng.normal(0.0, profile.base_consumption_lph * 0.05)
            consumption = max(1.0, base + occ_component + temp_component + noise)

            occupancy.append(round(occ_value, 2))
            temperature.append(round(temp_value, 2))
            water.append(consumption)

        bdf = pd.DataFrame(
            {
                "timestamp": timestamps,
                "building_id": profile.building_id,
                "building_name": profile.building_name,
                "water_consumption_litres": np.round(water, 2),
                "occupancy_estimate": occupancy,
                "temperature_c": temperature,
                "hour": hours,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_synthetic": True,
            }
        )
        frames.append(bdf)

    data = pd.concat(frames, ignore_index=True)
    data = _inject_anomalies(data, seed)
    data = data.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    columns = [
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
    ]

    return data[columns]


def dataset_metadata(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "rows": int(len(df)),
        "buildings": int(df["building_id"].nunique()),
        "anomaly_rate": float(df["is_anomaly"].mean() if "is_anomaly" in df.columns else df["anomaly_label"].mean()),
    }
