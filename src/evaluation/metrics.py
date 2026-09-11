from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def time_aware_split(df: pd.DataFrame, train_ratio: float = 0.7) -> TimeSplit:
    data = df.sort_values("timestamp").copy()
    cutoff_idx = int(len(data) * train_ratio)
    train = data.iloc[:cutoff_idx].copy()
    test = data.iloc[cutoff_idx:].copy()
    return TimeSplit(train=train, test=test)


def false_positive_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    negatives = (y_true == 0)
    if negatives.sum() == 0:
        return 0.0
    fp = ((y_pred == 1) & negatives).sum()
    return float(fp / negatives.sum())


def event_level_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    label_col = "is_anomaly" if "is_anomaly" in df.columns else "anomaly_label"
    event_col = "event_id" if "event_id" in df.columns else "anomaly_event_id"

    events = df[df[label_col] == 1].dropna(subset=[event_col])
    if events.empty:
        return {"event_recall": 0.0, "mean_detection_delay_hours": float("nan")}

    detected_flags: List[int] = []
    delays: List[float] = []

    for _, event_df in events.groupby(event_col):
        event_df = event_df.sort_values("timestamp")
        first_time = pd.to_datetime(event_df["timestamp"].iloc[0], utc=True)
        detected_rows = event_df[event_df[pred_col] == 1]
        if detected_rows.empty:
            detected_flags.append(0)
            continue
        detected_flags.append(1)
        det_time = pd.to_datetime(detected_rows["timestamp"].iloc[0], utc=True)
        delay = (det_time - first_time).total_seconds() / 3600.0
        delays.append(float(delay))

    return {
        "event_recall": float(np.mean(detected_flags)) if detected_flags else 0.0,
        "mean_detection_delay_hours": float(np.mean(delays)) if delays else float("nan"),
    }


def classification_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, Any]:
    label_col = "is_anomaly" if "is_anomaly" in df.columns else "anomaly_label"
    y_true = df[label_col].astype(int)
    y_pred = df[pred_col].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Operational metrics
    anomalies_detected = int(y_pred.sum())
    detection_rate = float(anomalies_detected / len(y_pred)) if len(y_pred) > 0 else 0.0
    
    excess_col = "potential_excess_consumption"
    excess_flagged = float(df.loc[y_pred == 1, excess_col].sum()) if excess_col in df.columns else 0.0

    return {
        # Model performance metrics (evaluation against synthetic ground truth)
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_positive_rate": fpr,
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        # Operational impact metrics (simulated/potential)
        "anomalies_detected": anomalies_detected,
        "anomaly_detection_rate": detection_rate,
        "potential_excess_litres_detected": round(excess_flagged, 2),
    }


def compare_models(df: pd.DataFrame, prediction_columns: List[str]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for col in prediction_columns:
        cls = classification_metrics(df, col)
        event = event_level_metrics(df, col)
        results[col] = {**cls, **event}
    return results
