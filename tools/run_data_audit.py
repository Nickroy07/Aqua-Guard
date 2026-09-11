#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generator import generate_synthetic_data
from src.data.validator import validate_full_dataset
from src.utils.io_utils import load_dataframe


def run_audit(dataset_path: Path, report_path: Path, run_pytest: bool) -> Dict[str, Any]:
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_data().to_csv(dataset_path, index=False)
    
    df = load_dataframe(dataset_path)
    validation_results = validate_full_dataset(df, expected_days=90, expected_buildings=6)

    report: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "audit_checks": validation_results["checks"],
        "summary": {
            "checks_passed": validation_results["checks_passed"],
            "checks_total": validation_results["total_checks"],
            "all_passed": validation_results["passed"],
        },
        "diagnostics": {
            "total_rows": int(len(df)),
            "buildings": sorted(df["building_name"].unique().tolist()),
            "anomaly_rate": float(df["is_anomaly"].mean() if "is_anomaly" in df.columns else df["anomaly_label"].mean()),
            "Post-hoc diagnostic, NOT a production model evaluation.": {
                "synthetic_ground_truth_available": True,
                "scope": "Diagnostic telemetry validation",
            },
        },
    }

    if run_pytest:
        # Run pytest via the current python environment
        pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
        completed = subprocess.run(pytest_cmd, capture_output=True, text=True, check=False)
        report["pytest"] = {
            "command": " ".join(pytest_cmd),
            "return_code": completed.returncode,
            "stdout": completed.stdout[-5000:],
            "stderr": completed.stderr[-2000:],
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AquaGuard dataset audit")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/synthetic_water_consumption_2026_90d_6b_seed42.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/audit_run/audit_report.json"),
    )
    parser.add_argument("--run-pytest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(args.dataset, args.report, run_pytest=args.run_pytest)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
