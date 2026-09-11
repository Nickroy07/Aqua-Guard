from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BaselineModel:
    group_medians: pd.DataFrame
    occupancy_reference: pd.DataFrame
    water_col: str
    occ_col: str


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str]:
    water_col = "water_consumption_litres" if "water_consumption_litres" in df.columns else "water_consumption_liters"
    occ_col = "occupancy" if "occupancy" in df.columns else "occupancy_estimate"
    return water_col, occ_col


def fit_expected_baseline(train_df: pd.DataFrame) -> BaselineModel:
    water_col, occ_col = _resolve_columns(train_df)
    key_cols = ["building_id", "hour", "is_weekend"]

    medians = (
        train_df.groupby(key_cols, as_index=False)[water_col]
        .median()
        .rename(columns={water_col: "group_median"})
    )
    occ_ref = (
        train_df.groupby(key_cols, as_index=False)[occ_col]
        .median()
        .rename(columns={occ_col: "occ_median"})
    )
    return BaselineModel(group_medians=medians, occupancy_reference=occ_ref, water_col=water_col, occ_col=occ_col)


def apply_expected_baseline(df: pd.DataFrame, model: BaselineModel) -> pd.DataFrame:
    key_cols = ["building_id", "hour", "is_weekend"]
    out = df.copy()

    water_col = model.water_col if model.water_col in out.columns else ("water_consumption_litres" if "water_consumption_litres" in out.columns else "water_consumption_liters")
    occ_col = model.occ_col if model.occ_col in out.columns else ("occupancy" if "occupancy" in out.columns else "occupancy_estimate")

    out = out.merge(model.group_medians, on=key_cols, how="left")
    out = out.merge(model.occupancy_reference, on=key_cols, how="left")

    global_occ_median = out[occ_col].median() if occ_col in out.columns else 100.0
    out["occ_median"] = out["occ_median"].replace(0, np.nan).fillna(global_occ_median)
    occ_ratio = (out[occ_col] / out["occ_median"]).clip(lower=0.5, upper=1.7)

    out["expected_consumption"] = out["group_median"] * (0.75 + 0.25 * occ_ratio)
    out["expected_consumption"] = out["expected_consumption"].clip(lower=1.0).round(2)

    out["absolute_deviation"] = (out[water_col] - out["expected_consumption"]).round(2)
    out["percentage_deviation"] = ((out["absolute_deviation"] / out["expected_consumption"]) * 100.0).round(2)
    out["potential_excess_consumption"] = np.maximum(0.0, out["absolute_deviation"]).round(2)

    return out.drop(columns=["group_median", "occ_median"])


def baseline_summary(df: pd.DataFrame) -> Dict[str, float]:
    water_col = "water_consumption_litres" if "water_consumption_litres" in df.columns else "water_consumption_liters"
    return {
        "total_observed_liters": float(df[water_col].sum()),
        "total_expected_liters": float(df["expected_consumption"].sum()),
        "potential_excess_liters": float(df["potential_excess_consumption"].sum()),
    }
