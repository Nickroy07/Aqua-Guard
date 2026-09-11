#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_end_to_end


def main() -> None:
    outputs = run_end_to_end(
        output_dir=Path("data/processed"),
        kb_path=Path("docs/knowledge_base/water_ops_guidance.md"),
        seed=42,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
