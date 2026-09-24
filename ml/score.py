"""Batch-score customer feature rows with the trained repeat-purchase model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def score_customers(input_csv: str, model_path: str, output_csv: str) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    data = pd.read_csv(input_csv)
    probabilities = bundle["model"].predict_proba(data[bundle["features"]])[:, 1]
    result = data.copy()
    result["repeat_purchase_probability"] = probabilities
    result["repeat_purchase_prediction"] = (probabilities >= 0.5).astype(int)
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--model", default="artifacts/repeat_purchase_model.joblib")
    parser.add_argument("--output", default="artifacts/customer_scores.csv")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Scored {len(score_customers(args.input_csv, args.model, args.output))} customers")


if __name__ == "__main__":
    main()