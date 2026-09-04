"""Run the full Financial Operations Analytics pipeline end to end.

    python scripts/run_pipeline.py

Steps: build processed tables -> EDA -> revenue forecast -> churn -> profitability.
All figures land in reports/figures/; processed parquet in data/processed/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (python scripts/run_pipeline.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (
    churn_analysis,
    data_cleaning,
    eda,
    geo_analysis,
    profitability,
    revenue_forecast,
)


def main() -> None:
    steps = [
        ("Build processed tables", data_cleaning.run),
        ("Exploratory data analysis", eda.run),
        ("Revenue forecasting", revenue_forecast.run),
        ("Churn analysis", churn_analysis.run),
        ("Profitability analysis", profitability.run),
        ("Shipping distance analysis", geo_analysis.run),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n{'#' * 70}\n# STEP {i}/{len(steps)}: {label}\n{'#' * 70}")
        fn()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
