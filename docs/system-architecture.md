# System Architecture

1. Synthetic data generation (`src/data/generator.py`)
2. Data audit (`tools/run_data_audit.py`)
3. Baseline modeling (`src/baselines/expected_consumption.py`)
4. Detectors (`src/models/detectors.py`)
5. Evaluation (`src/evaluation/metrics.py`)
6. Explainability and risk (`src/explainability/risk.py`)
7. Local retrieval recommendations (`src/rag/retrieval.py`)
8. Streamlit app (`app/dashboard.py`)

Outputs:
- `data/processed/synthetic_water_consumption_2026_90d_6b_seed42.csv`
- `data/processed/anomaly_results.csv`
- `data/processed/model_comparison.json`
- `data/processed/audit_run/audit_report.json`
