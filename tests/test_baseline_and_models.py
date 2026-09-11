from __future__ import annotations

from pathlib import Path

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


def _scored_test_frame():
    df = generate_synthetic_data(seed=42)
    split = time_aware_split(df)
    model = fit_expected_baseline(split.train)
    train_scored = apply_expected_baseline(split.train, model)
    test_scored = apply_expected_baseline(split.test, model)
    return train_scored, test_scored


def test_baseline_columns_and_values() -> None:
    _, test_scored = _scored_test_frame()
    for col in [
        "expected_consumption",
        "absolute_deviation",
        "percentage_deviation",
        "potential_excess_consumption",
    ]:
        assert col in test_scored.columns
    assert (test_scored["expected_consumption"] > 0).all()
    assert (test_scored["potential_excess_consumption"] >= 0).all()


def test_detector_outputs_binary_predictions() -> None:
    train_scored, test_scored = _scored_test_frame()

    static = static_threshold_detector(test_scored)
    statistical = rolling_statistical_detector(test_scored)
    ml = isolation_forest_detector(train_scored, test_scored)
    hybrid = hybrid_detector(
        statistical.predictions,
        ml.predictions,
        statistical.scores,
        ml.scores,
        percentage_dev=test_scored["percentage_deviation"],
    )

    for pred in [static.predictions, statistical.predictions, ml.predictions, hybrid.predictions]:
        assert set(pred.unique()).issubset({0, 1})


def test_risk_scoring_and_explanations() -> None:
    train_scored, test_scored = _scored_test_frame()
    statistical = rolling_statistical_detector(test_scored)
    ml = isolation_forest_detector(train_scored, test_scored)
    hybrid = hybrid_detector(
        statistical.predictions,
        ml.predictions,
        statistical.scores,
        ml.scores,
        percentage_dev=test_scored["percentage_deviation"],
    )

    test_scored = test_scored.copy()
    test_scored["pred_hybrid"] = hybrid.predictions
    test_scored["hybrid_score"] = hybrid.scores
    scored = compute_risk_score(test_scored, confidence_col="hybrid_score")
    explained = add_explanations(scored)

    assert explained["risk_score"].between(0, 1).all()
    assert set(explained["risk_tier"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})
    assert explained["explanation"].notna().all()
    assert explained["possible_causes"].notna().all()
    assert explained["recommended_action"].notna().all()
    assert explained["uncertainty_statement"].notna().all()
    # Responsible AI safeguard: verify that "Leak confirmed" is NEVER present
    for statement in explained["uncertainty_statement"]:
        assert "leak confirmed" not in statement.lower()


def test_model_comparison_metrics_and_confusion_matrix() -> None:
    train_scored, test_scored = _scored_test_frame()
    static = static_threshold_detector(test_scored)
    statistical = rolling_statistical_detector(test_scored)
    ml = isolation_forest_detector(train_scored, test_scored)
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

    results = compare_models(test_scored, ["pred_static", "pred_statistical", "pred_ml", "pred_hybrid"])
    assert set(results.keys()) == {"pred_static", "pred_statistical", "pred_ml", "pred_hybrid"}
    for metrics in results.values():
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert all(k in cm for k in ["true_positive", "false_positive", "true_negative", "false_negative"])
        assert "anomalies_detected" in metrics
        assert "potential_excess_litres_detected" in metrics


def test_recommendation_grounding() -> None:
    recommendation = build_recommendation(
        "sustained overnight anomaly in Engineering Block",
        Path("docs/knowledge_base/water_ops_guidance.md"),
    )
    assert "Evidence" in recommendation
    assert "water_ops_guidance.md" in recommendation
    assert "Recommended Operational Action:" in recommendation
