# Model Evaluation & Methodology

AquaGuard AI evaluates anomaly detection approaches using a **chronological time-aware split** (70% train / 30% held-out test). This ensures strictly zero future data leakage.

## Evaluated Methods

1. **Static Threshold (>35% deviation)**: Fixed baseline rule flagging consumption exceeding 35% above expected baseline.
2. **Rolling Statistical Z-Score (>2.5)**: 24-hour moving window calculating standard deviation of deviation.
3. **Isolation Forest (Unsupervised ML)**: Multi-variate tree ensemble trained on expected baseline, deviation, cyclical hour encoding, weekend flags, and building profiles with calibrated anomaly scoring.
4. **Hybrid Ensemble**: Combines statistical indicators with Isolation Forest confidence scores.

## Evaluation Metrics

For each model, the pipeline calculates both machine learning classification metrics and operational facility metrics:
- **Precision**: Proportion of flagged anomalies that were true operational anomalies.
- **Recall**: Proportion of true anomaly hours successfully identified.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
- **False Positive Rate (FPR)**: Rate of false alarms across normal operating hours.
- **Event Recall**: Proportion of multi-hour anomalous incidents detected at least once.
- **Mean Detection Delay**: Average delay (in hours) between anomaly initiation and the first AI alert.
- **Potential Excess Volume Detected (L)**: Litres of water consumption flagged for inspection.

## Separation of Model Performance vs Real-World Impact

- **Model Performance**: Evaluated against known synthetic ground-truth labels for objective algorithmic comparison.
- **Real-World Impact**: Reported exclusively as *potential*, *simulated*, and *projected* figures. AquaGuard AI explicitly avoids claiming historical water savings from a software prototype.
