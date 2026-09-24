"""Train and evaluate a repeat-purchase model from the validated warehouse data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.db import get_engine

FEATURES = ["recency_days", "frequency", "monetary", "average_order_value"]


def build_training_data(transactions: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Create customer features at a temporal cutoff and a future-purchase label."""
    transactions = transactions.copy()
    transactions["invoice_date"] = pd.to_datetime(transactions["invoice_date"])
    transactions["revenue"] = pd.to_numeric(transactions["revenue"])
    cutoff = transactions["invoice_date"].max() - pd.to_timedelta(int(horizon_days), unit="D")
    history = transactions[transactions["invoice_date"] <= cutoff]
    future = transactions[transactions["invoice_date"] > cutoff]

    features = history.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
    )
    features["recency_days"] = (cutoff - features["last_purchase"]).dt.days
    features["average_order_value"] = features["monetary"] / features["frequency"].clip(lower=1)
    future_customers = set(future["customer_id"].unique())
    features["repeat_purchase"] = features.index.to_series().isin(future_customers).astype(int)
    return features.reset_index()[["customer_id", *FEATURES, "repeat_purchase"]]


def train_model(data: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Use a chronological customer split and return a fitted pipeline plus metrics."""
    data = data.sort_values("customer_id").reset_index(drop=True)
    split = max(1, int(len(data) * 0.8))
    train, test = data.iloc[:split], data.iloc[split:]
    if test.empty or train["repeat_purchase"].nunique() < 2:
        raise ValueError("Training data needs both classes and a non-empty test split")

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), FEATURES),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(train[FEATURES], train["repeat_purchase"])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "roc_auc": float(roc_auc_score(test["repeat_purchase"], probabilities)),
        "accuracy": float(accuracy_score(test["repeat_purchase"], predictions)),
        "precision": float(precision_score(test["repeat_purchase"], predictions, zero_division=0)),
        "recall": float(recall_score(test["repeat_purchase"], predictions, zero_division=0)),
        "f1": float(f1_score(test["repeat_purchase"], predictions, zero_division=0)),
    }
    return model, metrics


def load_transactions() -> pd.DataFrame:
    query = """
        SELECT customer_id, invoice_no, invoice_date, revenue
        FROM staging.sales_valid
        WHERE customer_id IS NOT NULL
    """
    return pd.read_sql(query, get_engine())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_training_data(load_transactions())
    model, metrics = train_model(data)
    joblib.dump({"model": model, "features": FEATURES, "label": "repeat_purchase"}, output_dir / "repeat_purchase_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()