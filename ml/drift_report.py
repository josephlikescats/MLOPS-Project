"""
Standalone PSI report for two feature CSVs (kept from Part 1).

    python -m ml.drift_report <training.csv> <scoring.csv>

The PSI implementation now lives in ml/monitoring.py, which this delegates to
so there is only one definition of the metric. For the full Part 2 monitoring
run - data quality, drift, performance, latency and the retraining decision -
use ``python -m ml.monitoring`` instead; this command is the narrow
two-files-in, PSI-out case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.monitoring import population_stability_index, psi_status

LEGACY_FEATURES = ["recency_days", "frequency", "monetary", "average_order_value"]


def main() -> None:
    parser = argparse.ArgumentParser(description="PSI between two feature files")
    parser.add_argument("training_csv")
    parser.add_argument("scoring_csv")
    parser.add_argument("--output", default="artifacts/drift_report.json")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="columns to compare (default: the four Part 1 RFM features)",
    )
    args = parser.parse_args()

    training = pd.read_csv(args.training_csv)
    scoring = pd.read_csv(args.scoring_csv)
    columns = args.features or [
        column for column in LEGACY_FEATURES if column in training.columns
    ]

    report = {
        column: {
            "psi": round(population_stability_index(training[column], scoring[column]), 4),
            "status": psi_status(population_stability_index(training[column], scoring[column])),
        }
        for column in columns
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
