#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import DEFAULT_DATASET_NAME, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AquaGuard dataset.")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--buildings", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed") / DEFAULT_DATASET_NAME,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_dataset(
        output_path=args.output,
        start_date=args.start_date,
        days=args.days,
        buildings=args.buildings,
        seed=args.seed,
    )
    print(f"Generated {len(df)} rows at {args.output}")


if __name__ == "__main__":
    main()
