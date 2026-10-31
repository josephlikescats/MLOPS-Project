"""
Batch churn scoring.

    python -m ml.score_churn                      # build the snapshot and score it
    python -m ml.score_churn --input features.csv # score a prepared feature file

With no ``--input`` the scorer builds the current scoring snapshot itself from
the warehouse, which is the form a scheduled job wants: one command from
transactions to a ranked retention list. The output carries the model version
that produced each score, so a list on someone's desk can always be traced
back to a registered model version.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ml.config import PATHS
from ml.data import load_transactions
from ml.features import build_scoring_snapshot
from ml.registry import ModelBundle, load_model_bundle


def score_frame(bundle: ModelBundle, features: pd.DataFrame) -> pd.DataFrame:
    """Attach probability, decision and risk band to a feature frame."""
    predictions = bundle.predict(features)
    scored = pd.concat([features.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    scored["model_version"] = bundle.model_version
    scored["decision_threshold"] = bundle.threshold
    return scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def run_scoring(
    source: str = "auto",
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    model_uri: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Score either a prepared CSV or a freshly built scoring snapshot."""
    bundle = load_model_bundle(model_uri)
    output = output_csv or PATHS.path("scores")

    if input_csv:
        features = pd.read_csv(input_csv)
        origin = str(input_csv)
    else:
        transactions = load_transactions(source)
        features = build_scoring_snapshot(transactions)
        origin = f"{source} snapshot"
        # Keep the exact feature rows that were scored, so the monitoring job
        # can diff this population against the training baseline later.
        features.to_csv(PATHS.path("scoring_snapshot"), index=False)

    scored = score_frame(bundle, features)
    if top_n:
        scored = scored.head(top_n)

    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)

    flagged = int(scored["churn_prediction"].sum())
    print(
        f"score: {len(scored):,} customers from {origin} using {bundle.model_version} "
        f"(threshold {bundle.threshold:.2f}) -> {flagged:,} flagged at risk"
    )
    print(scored["risk_band"].value_counts().to_string())
    print(f"score: written to {output}")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score customers for churn risk")
    parser.add_argument("--source", default="auto", choices=["auto", "postgres", "landing"])
    parser.add_argument("--input", type=Path, default=None, help="prepared feature CSV to score")
    parser.add_argument("--output", type=Path, default=PATHS.path("scores"))
    parser.add_argument("--model-uri", default=None, help="e.g. models:/retail-churn-classifier/2")
    parser.add_argument("--top", type=int, default=None, help="keep only the N highest-risk rows")
    args = parser.parse_args()

    try:
        run_scoring(args.source, args.input, args.output, args.model_uri, args.top)
    except Exception as exc:  # noqa: BLE001 - clean operator-facing failure
        print(f"score: FAILED - {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
