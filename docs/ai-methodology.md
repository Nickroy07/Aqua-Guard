# AI Methodology

## 1. Expected-Consumption Baseline
Before applying machine learning, AquaGuard AI establishes an occupancy-adjusted expected consumption baseline:
$$\text{Expected Consumption} = \text{Median}_{\text{bldg, hour, weekend}} \times \left(0.75 + 0.25 \times \frac{\text{Occupancy}}{\text{Median Occupancy}}\right)$$

### Computed Baseline Metrics:
- $\text{Expected Consumption (Litres)}$
- $\text{Absolute Deviation} = \text{Observed} - \text{Expected}$
- $\text{Percentage Deviation} = \frac{\text{Observed} - \text{Expected}}{\text{Expected}} \times 100$
- $\text{Potential Excess Consumption} = \max(0, \text{Observed} - \text{Expected})$

## 2. Machine Learning Anomaly Detection
To capture complex patterns beyond simple univariate thresholding, AquaGuard AI trains an **Isolation Forest** on engineered features:
- Expected baseline consumption
- Absolute and percentage deviation
- Cyclical time features ($\sin(2\pi \cdot \text{hour} / 24)$, $\cos(2\pi \cdot \text{hour} / 24)$)
- Low-occupancy overnight indicator ($\text{hour} \in [0, 5]$)
- Weekend indicator
- Building one-hot identifiers

## 3. Temporal Integrity (Zero Data Leakage)
A strict chronological split (first 70% train / last 30% test) is maintained across all 6 buildings. Baseline medians and ML parameters are fit solely on the training partition and evaluated on the future test partition.

## 4. Multi-Method Evaluation
AquaGuard AI demonstrates why ML adds value over simpler rules by benchmarking:
- Static threshold
- Rolling statistical z-score
- Isolation Forest
- Hybrid ensemble
Model selection is driven by balanced F1 performance and operational detection latency.
