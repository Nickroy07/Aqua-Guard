from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class DetectionResults:
    predictions: pd.Series
    scores: pd.Series


def static_threshold_detector(df: pd.DataFrame, pct_threshold: float = 35.0) -> DetectionResults:
    preds = (df["percentage_deviation"] > pct_threshold).astype(int)
    scores = (df["percentage_deviation"] / pct_threshold).clip(lower=0.0)
    return DetectionResults(predictions=preds, scores=scores)


def rolling_statistical_detector(
    df: pd.DataFrame,
    window: int = 24,
    z_threshold: float = 2.5,
) -> DetectionResults:
    out = df.sort_values(["building_id", "timestamp"]).copy()
    grouped = out.groupby("building_id")["absolute_deviation"]
    rolling_mean = grouped.transform(lambda s: s.rolling(window=window, min_periods=12).mean())
    rolling_std = grouped.transform(lambda s: s.rolling(window=window, min_periods=12).std().replace(0, np.nan))
    
    # Avoid zero division
    z = (out["absolute_deviation"] - rolling_mean) / rolling_std
    z = z.fillna(0.0)
    preds = (z > z_threshold).astype(int)
    scores = (z / z_threshold).clip(lower=0.0)
    return DetectionResults(predictions=preds, scores=scores)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["expected_consumption"] = df["expected_consumption"]
    features["absolute_deviation"] = df["absolute_deviation"]
    features["percentage_deviation"] = df["percentage_deviation"]
    
    occ_col = "occupancy" if "occupancy" in df.columns else "occupancy_estimate"
    temp_col = "temperature" if "temperature" in df.columns else "temperature_c"
    
    features["occupancy"] = df[occ_col]
    features["temperature"] = df[temp_col]
    features["hour"] = df["hour"]
    features["is_weekend"] = df["is_weekend"]
    
    # Cyclical hour encoding
    features["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    
    # Overnight flag (0-5 AM)
    features["is_overnight"] = (df["hour"] <= 5).astype(float)
    
    # Building one-hot dummies
    dummies = pd.get_dummies(df["building_id"], prefix="b", dtype=float)
    features = pd.concat([features, dummies], axis=1)
    
    return features.fillna(0.0)


def isolation_forest_detector(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    contamination: float = 0.025,
    random_state: int = 42,
) -> DetectionResults:
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    x_train = _build_features(train_df)
    x_test = _build_features(test_df)

    all_columns = sorted(set(x_train.columns).union(set(x_test.columns)))
    x_train = x_train.reindex(columns=all_columns, fill_value=0.0)
    x_test = x_test.reindex(columns=all_columns, fill_value=0.0)

    model.fit(x_train)
    raw_pred = model.predict(x_test)
    raw_scores = -model.score_samples(x_test)

    # Normalize anomaly scores to [0, 1]
    min_s, max_s = raw_scores.min(), raw_scores.max()
    normalized_scores = (raw_scores - min_s) / (max_s - min_s + 1e-6)

    # Flag as anomaly if model flags it (-1) AND percentage deviation is positive
    is_anomaly = ((raw_pred == -1) & (test_df["percentage_deviation"] > 15.0)).astype(int)
    preds = pd.Series(is_anomaly, index=test_df.index)
    scores = pd.Series(normalized_scores, index=test_df.index)

    return DetectionResults(predictions=preds, scores=scores)


def hybrid_detector(
    stat_preds: pd.Series,
    ml_preds: pd.Series,
    stat_scores: pd.Series,
    ml_scores: pd.Series,
    percentage_dev: pd.Series | None = None,
) -> DetectionResults:
    # Ensemble: Flag if ML confirms statistical flag, or if ML is highly confident with positive deviation
    combined_score = (0.50 * stat_scores.clip(0, 1) + 0.50 * ml_scores.clip(0, 1))
    if percentage_dev is not None:
        preds = (((stat_preds == 1) | (ml_preds == 1)) & (percentage_dev > 15.0)).astype(int)
    else:
        preds = ((stat_preds == 1) | (ml_preds == 1)).astype(int)

    return DetectionResults(predictions=preds, scores=combined_score)
