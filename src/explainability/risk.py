from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def add_event_persistence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_persistence_hours"] = 0
    event_col = "event_id" if "event_id" in out.columns else "anomaly_event_id"

    for _, event_df in out.dropna(subset=[event_col]).groupby(event_col):
        hours = len(event_df)
        out.loc[event_df.index, "event_persistence_hours"] = hours
    return out


def _tier(score: float) -> str:
    """Classify risk score into transparent tiers: LOW, MEDIUM, HIGH."""
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def compute_risk_score(df: pd.DataFrame, confidence_col: str = "hybrid_score") -> pd.DataFrame:
    """
    Transparent anomaly risk scoring based on:
    1. Magnitude of deviation (percentage deviation from baseline)
    2. Duration / persistence (consecutive hours of anomalous pattern)
    3. Operational timing severity (overnight 0-5 AM low-occupancy periods)
    4. Model detection confidence
    """
    out = add_event_persistence(df)
    
    # 1. Magnitude: 0% -> 0.0, 100%+ -> 1.0
    pct_component = np.clip(out["percentage_deviation"].abs() / 100.0, 0.0, 1.0)
    
    # 2. Duration / Persistence: 1 hour -> ~0.1, 12+ hours -> 1.0
    persistence_component = np.clip(out["event_persistence_hours"] / 12.0, 0.0, 1.0)
    
    # 3. Overnight timing (elevated risk if unexpected water runs when buildings are empty)
    overnight_component = out["hour"].apply(lambda h: 0.25 if 0 <= int(h) <= 5 else 0.0)

    # 4. Detector confidence
    confidence = out[confidence_col] if confidence_col in out.columns else 0.5
    confidence = np.clip(confidence, 0.0, 1.0)

    # Weighted risk formulation
    raw_risk = (
        0.40 * pct_component
        + 0.25 * persistence_component
        + 0.20 * confidence
        + 0.15 * overnight_component
    )
    out["risk_score"] = np.clip(raw_risk, 0.0, 1.0).round(4)
    out["risk_tier"] = out["risk_score"].apply(_tier)

    # For non-anomalies, zero out risk
    label_col = "is_anomaly" if "is_anomaly" in out.columns else "anomaly_label"
    out.loc[out[label_col] == 0, ["risk_score", "risk_tier", "event_persistence_hours"]] = [0.0, "LOW", 0]
    return out


def explain_anomaly(row: pd.Series) -> Dict[str, str]:
    """
    Provides explainability adhering to Responsible AI standards:
    - Observed vs expected consumption
    - Plain-English rationale
    - Multiple possible operational causes
    - Recommended human action
    - Explicit scientific limitation / uncertainty disclaimer (NEVER 'leak confirmed')
    """
    label_col = "is_anomaly" if "is_anomaly" in row else "anomaly_label"
    water_col = "water_consumption_litres" if "water_consumption_litres" in row else "water_consumption_liters"

    is_anom = int(row.get(label_col, 0))
    if is_anom == 0:
        return {
            "explanation": "Normal consumption pattern consistent with expected baseline.",
            "possible_causes": "Standard baseline facility operations.",
            "recommended_action": "No intervention required. Continue routine monitoring.",
            "uncertainty_statement": "Normal operational range.",
        }

    observed = float(row.get(water_col, 0.0))
    expected = float(row.get("expected_consumption", 0.0))
    deviation_pct = float(row.get("percentage_deviation", 0.0))
    persistence = int(row.get("event_persistence_hours", 1))
    hour = int(row.get("hour", 12))
    anomaly_type = str(row.get("anomaly_type", "unusual pattern"))
    building_name = str(row.get("building_name", "Facility"))

    # Contextual explanation
    if anomaly_type == "spike":
        reason = (
            f"Consumption was abruptly elevated (+{deviation_pct:.1f}% above expected) for {persistence} hour(s). "
            f"Observed: {observed:,.0f} L vs Expected: {expected:,.0f} L in {building_name}."
        )
        causes = "Fixture issue, one-time maintenance flushing, lab discharge, irrigation valve burst, or meter surge."
        action = "Verify meter reading stability, inspect restrooms and immediate utility lines for active leaks or open taps."
    elif anomaly_type == "sustained_overnight":
        reason = (
            f"Sustained abnormal usage (+{deviation_pct:.1f}% above expected) during overnight low-occupancy hours (Hour {hour:02d}:00). "
            f"Observed: {observed:,.0f} L vs Expected: {expected:,.0f} L in {building_name}."
        )
        causes = "Continuous toilet flapper run, stuck solenoid valve, cooling tower makeup overflow, or irrigation timer overrun."
        action = "Inspect building washrooms, verify cooling tower float valves, check automated irrigation schedules, and conduct physical walk-through."
    else:  # gradual_drift
        reason = (
            f"Gradual upward drift (+{deviation_pct:.1f}% above expected) persisting over {persistence} consecutive hours. "
            f"Observed: {observed:,.0f} L vs Expected: {expected:,.0f} L in {building_name}."
        )
        causes = "Slow fixture wear, creeping valve leakage, underground pipe weeping, HVAC loop inefficiency, or unrecorded schedule change."
        action = "Review sub-meter integrity, check distribution isolation valves, compare with recent occupancy shifts, and dispatch facility inspection."

    uncertainty = (
        "Potential abnormal usage identified from aggregate meter data. "
        "This is NOT a confirmed physical leak and requires human on-site verification before any maintenance action."
    )

    return {
        "explanation": reason,
        "possible_causes": causes,
        "recommended_action": action,
        "uncertainty_statement": uncertainty,
    }


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    explanations = [explain_anomaly(row) for _, row in out.iterrows()]
    out["explanation"] = [e["explanation"] for e in explanations]
    out["possible_causes"] = [e["possible_causes"] for e in explanations]
    out["recommended_action"] = [e["recommended_action"] for e in explanations]
    out["possible_interpretation"] = [e["possible_causes"] for e in explanations]  # backwards compatibility
    out["uncertainty_statement"] = [e["uncertainty_statement"] for e in explanations]
    return out
