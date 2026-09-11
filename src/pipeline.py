from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.baselines.expected_consumption import apply_expected_baseline, fit_expected_baseline
from src.data.generator import generate_synthetic_data
from src.evaluation.metrics import compare_models, time_aware_split
from src.explainability.risk import add_explanations, compute_risk_score
from src.models.detectors import (
    hybrid_detector,
    isolation_forest_detector,
    rolling_statistical_detector,
    static_threshold_detector,
)
from src.rag.retrieval import build_recommendation
from src.utils.io_utils import save_dataframe


DEFAULT_DATASET_NAME = "synthetic_water_consumption_2026_90d_6b_seed42.csv"


def generate_dataset(output_path: Path, start_date: str = "2026-01-01", days: int = 90, buildings: int = 6, seed: int = 42) -> pd.DataFrame:
    df = generate_synthetic_data(start_date=start_date, days=days, buildings=buildings, seed=seed)
    save_dataframe(df, output_path)
    return df


def run_model_pipeline(df: pd.DataFrame, kb_path: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, any]], str]:
    split = time_aware_split(df, train_ratio=0.7)

    baseline_model = fit_expected_baseline(split.train)
    train_scored = apply_expected_baseline(split.train, baseline_model)
    test_scored = apply_expected_baseline(split.test, baseline_model)

    static = static_threshold_detector(test_scored, pct_threshold=35.0)
    statistical = rolling_statistical_detector(test_scored, window=24, z_threshold=2.5)
    ml = isolation_forest_detector(train_scored, test_scored, contamination=0.03)
    hybrid = hybrid_detector(
        statistical.predictions,
        ml.predictions,
        statistical.scores,
        ml.scores,
        percentage_dev=test_scored["percentage_deviation"],
    )

    test_scored = test_scored.copy()
    test_scored["pred_static"] = static.predictions
    test_scored["pred_statistical"] = statistical.predictions
    test_scored["pred_ml"] = ml.predictions
    test_scored["pred_hybrid"] = hybrid.predictions

    test_scored["static_score"] = static.scores
    test_scored["statistical_score"] = statistical.scores
    test_scored["ml_score"] = ml.scores
    test_scored["hybrid_score"] = hybrid.scores

    results = compare_models(test_scored, ["pred_static", "pred_statistical", "pred_ml", "pred_hybrid"])

    # Choose model with strongest balanced F1 score
    final_method = max(results.items(), key=lambda x: x[1]["f1"])[0]
    final_score_col = final_method.replace("pred_", "") + "_score"

    test_scored = compute_risk_score(test_scored, confidence_col=final_score_col)
    test_scored = add_explanations(test_scored)

    label_col = "is_anomaly" if "is_anomaly" in test_scored.columns else "anomaly_label"
    recs = []
    for _, row in test_scored.iterrows():
        if row[label_col] == 1:
            query = f"{row['anomaly_type']} anomaly in {row['building_name']} during hour {row['hour']}"
            recs.append(build_recommendation(query, kb_path))
        else:
            recs.append("No recommendation needed. System operating within baseline parameters.")
    test_scored["recommendation"] = recs

    return test_scored, results, final_method


def run_end_to_end(output_dir: Path, kb_path: Path, seed: int = 42) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / DEFAULT_DATASET_NAME

    df = generate_dataset(dataset_path, seed=seed)
    scored, comparison, final_method = run_model_pipeline(df, kb_path)

    results_path = output_dir / "anomaly_results.csv"
    comparison_path = output_dir / "model_comparison.json"
    summary_path = output_dir / "run_summary.json"

    save_dataframe(scored, results_path)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    
    water_col = "water_consumption_litres" if "water_consumption_litres" in scored.columns else "water_consumption_liters"
    daily_potential = scored["potential_excess_consumption"].sum() / max(scored["timestamp"].dt.date.nunique(), 1)
    label_col = "is_anomaly" if "is_anomaly" in scored.columns else "anomaly_label"
    
    summary = {
        "final_method": final_method,
        "rows_total": int(len(df)),
        "rows_test": int(len(scored)),
        "total_observed_litres_test": float(scored[water_col].sum()),
        "total_expected_litres_test": float(scored["expected_consumption"].sum()),
        "potential_excess_litres_test": float(scored["potential_excess_consumption"].sum()),
        "anomalies_detected_test": int((scored[final_method] == 1).sum()),
        "ground_truth_anomalies_test": int((scored[label_col] == 1).sum()),
        "estimated_annualized_potential_if_pattern_persists_litres": float(daily_potential * 365),
        "dataset_path": str(dataset_path),
        "results_path": str(results_path),
        "comparison_path": str(comparison_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "dataset": str(dataset_path),
        "results": str(results_path),
        "comparison": str(comparison_path),
        "summary": str(summary_path),
    }
