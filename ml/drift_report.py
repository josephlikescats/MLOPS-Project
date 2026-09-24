"""Produce a simple PSI report comparing training and scoring feature distributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges = expected.quantile([i / bins for i in range(bins + 1)]).drop_duplicates().values
    if len(edges) < 2:
        return 0.0
    expected_counts = pd.cut(expected, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    actual_counts = pd.cut(actual, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    expected_counts = expected_counts.clip(lower=0.0001)
    actual_counts = actual_counts.clip(lower=0.0001)
    return float(((actual_counts - expected_counts) * (actual_counts / expected_counts).map(lambda value: __import__("math").log(value))).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_csv")
    parser.add_argument("scoring_csv")
    parser.add_argument("--output", default="artifacts/drift_report.json")
    args = parser.parse_args()
    training = pd.read_csv(args.training_csv)
    scoring = pd.read_csv(args.scoring_csv)
    columns = ["recency_days", "frequency", "monetary", "average_order_value"]
    report = {column: population_stability_index(training[column], scoring[column]) for column in columns}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()