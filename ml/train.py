"""
Churn model training pipeline.

One command rebuilds the whole model lineage:

    python -m ml.train --source auto

  1. load validated transactions (Postgres, or the landing CSVs offline)
  2. build the snapshot feature table and the chronological train/val/test split
  3. fit every candidate in ml/models.py on the training block
  4. select on the *validation* block, and tune the decision threshold there
  5. evaluate the winner once on the untouched test block
  6. log every candidate to MLflow (params, metrics, artifacts) and register
     the winner as a new version of the registered model
  7. write the joblib artifact, metrics and model card to artifacts/

Step 4 and 5 are kept apart on purpose: selecting a model and choosing a
threshold are both decisions fitted to data, so they happen on validation. The
test block is read once, at the end, and never influences a choice.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    # mlflow prints emoji status lines (e.g. the "View run" banner) straight to
    # stdout; Windows consoles default to cp1252, which can't encode them and
    # crashes the run silently, so nothing gets logged. Force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import pandas as pd
import sklearn

from ml.config import (
    CHURN_CONFIG,
    FEATURES,
    LABEL,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PATHS,
    REGISTERED_MODEL_NAME,
    SELECTION_METRIC,
    THRESHOLDS,
)
from ml.data import load_transactions
from ml.features import (
    build_feature_table,
    chronological_split,
    describe_split,
)
from ml.metrics import classification_metrics, scalar_metrics, tune_threshold
from ml.models import build_candidates, feature_importances, hyperparameters


def _class_weight_ratio(labels: pd.Series) -> float:
    """negatives / positives, for XGBoost's scale_pos_weight."""
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    return float(negatives / positives) if positives else 1.0


def evaluate_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    min_precision: float | None = None,
) -> tuple[str, dict[str, dict]]:
    """Fit and score every candidate; return the winner's name and all results.

    Each candidate's validation probabilities decide both selection and its own
    threshold, so a model is never judged at a threshold that suits a different
    model's calibration.
    """
    candidates = build_candidates(_class_weight_ratio(train[LABEL]))
    results: dict[str, dict] = {}

    for name, pipeline in candidates.items():
        print(f"train: fitting {name} on {len(train):,} rows")
        pipeline.fit(train[FEATURES], train[LABEL])

        validation_probabilities = pipeline.predict_proba(validation[FEATURES])[:, 1]
        threshold, validation_metrics = tune_threshold(
            validation[LABEL], validation_probabilities, min_precision=min_precision
        )
        test_probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
        test_metrics = classification_metrics(test[LABEL], test_probabilities, threshold)

        results[name] = {
            "pipeline": pipeline,
            "threshold": threshold,
            "validation": validation_metrics,
            "test": test_metrics,
            "params": hyperparameters(pipeline),
            "feature_importances": feature_importances(pipeline),
        }
        print(
            f"  {name}: validation {SELECTION_METRIC}="
            f"{validation_metrics.get(SELECTION_METRIC):.4f} "
            f"threshold={threshold:.2f} | test f1={test_metrics['f1']:.4f}"
        )

    def selection_score(name: str) -> float:
        value = results[name]["validation"].get(SELECTION_METRIC)
        return float(value) if value is not None else -1.0

    best = max(results, key=selection_score)
    print(f"train: selected {best} on validation {SELECTION_METRIC}")
    return best, results


def _log_to_mlflow(
    results: dict[str, dict],
    best_name: str,
    split_summary: dict,
    source: str,
    feature_table: pd.DataFrame,
) -> dict:
    """Log one nested run per candidate and register the winner.

    Returns the tracking details to embed in the artifact so a saved model can
    always be traced back to the run that produced it. MLflow failures are
    non-fatal: the joblib artifact is the fallback the API can always load.
    """
    tracking: dict = {"tracking_uri": MLFLOW_TRACKING_URI, "experiment": MLFLOW_EXPERIMENT}
    try:
        import mlflow
        from mlflow.models import infer_signature
    except ImportError:
        print("train: mlflow not installed, skipping experiment tracking")
        return {**tracking, "logged": False, "reason": "mlflow not installed"}

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"churn-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as parent:
            mlflow.set_tags(
                {
                    "project": "ecommerce-sales-churn-mlops",
                    "stage": "part2-training",
                    "data_source": source,
                    "selected_model": best_name,
                }
            )
            mlflow.log_params(
                {f"churn_{key}": value for key, value in CHURN_CONFIG.as_dict().items() if key != "features"}
            )
            mlflow.log_param("n_features", len(FEATURES))
            mlflow.log_param("selection_metric", SELECTION_METRIC)
            mlflow.log_dict(CHURN_CONFIG.as_dict(), "churn_config.json")
            mlflow.log_dict(split_summary, "split_summary.json")

            best_run_id = None
            for name, result in results.items():
                with mlflow.start_run(run_name=name, nested=True) as child:
                    mlflow.set_tags({"model_family": name, "selected": str(name == best_name)})
                    mlflow.log_params(result["params"])
                    mlflow.log_param("decision_threshold", result["threshold"])
                    mlflow.log_metrics(
                        {f"validation_{k}": v for k, v in scalar_metrics(result["validation"]).items()}
                    )
                    mlflow.log_metrics(
                        {f"test_{k}": v for k, v in scalar_metrics(result["test"]).items()}
                    )
                    if result["feature_importances"]:
                        mlflow.log_dict(result["feature_importances"], "feature_importances.json")

                    sample = feature_table[FEATURES].head(100)
                    signature = infer_signature(
                        sample, result["pipeline"].predict_proba(sample)[:, 1]
                    )
                    mlflow.sklearn.log_model(
                        result["pipeline"],
                        artifact_path="model",
                        signature=signature,
                        input_example=sample.head(5),
                    )
                    if name == best_name:
                        best_run_id = child.info.run_id

            tracking.update(
                {
                    "logged": True,
                    "parent_run_id": parent.info.run_id,
                    "run_id": best_run_id,
                    "experiment_id": parent.info.experiment_id,
                }
            )

            # Register the winner. The registry needs a database-backed store,
            # so this is attempted separately and a failure only downgrades the
            # run record rather than losing the trained model.
            try:
                version = mlflow.register_model(
                    model_uri=f"runs:/{best_run_id}/model",
                    name=REGISTERED_MODEL_NAME,
                    tags={"selected_on": SELECTION_METRIC, "model_family": best_name},
                )
                tracking["registered_model"] = REGISTERED_MODEL_NAME
                tracking["model_version"] = str(version.version)
                print(f"train: registered {REGISTERED_MODEL_NAME} version {version.version}")
            except Exception as exc:  # noqa: BLE001 - registry is optional infrastructure
                tracking["registered_model"] = None
                tracking["registry_error"] = f"{exc.__class__.__name__}: {exc}"
                print(f"train: model registry unavailable ({exc.__class__.__name__}), artifact still written")

    except Exception as exc:  # noqa: BLE001 - tracking must not break training
        print(f"train: mlflow logging failed ({exc.__class__.__name__}: {exc})")
        return {**tracking, "logged": False, "reason": f"{exc.__class__.__name__}: {exc}"}

    return tracking


def _write_artifacts(
    best_name: str,
    result: dict,
    split_summary: dict,
    tracking: dict,
    source: str,
    feature_table: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Write the model bundle, metrics and model card.

    The bundle carries its own feature list, threshold and label contract, so
    the API never has to guess any of them, and a reference feature sample so
    the drift job has a baseline to compare against without re-reading the
    training source.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()

    bundle = {
        "model": result["pipeline"],
        "features": FEATURES,
        "label": LABEL,
        "threshold": result["threshold"],
        "model_family": best_name,
        "config": CHURN_CONFIG.as_dict(),
        "metrics": {"validation": result["validation"], "test": result["test"]},
        "trained_at": trained_at,
        "data_source": source,
        "tracking": tracking,
    }
    joblib.dump(bundle, output_dir / PATHS.model)

    metrics_payload = {
        "selected_model": best_name,
        "selection_metric": SELECTION_METRIC,
        "threshold": result["threshold"],
        "trained_at": trained_at,
        "data_source": source,
        "split": split_summary,
        "validation": result["validation"],
        "test": result["test"],
        "tracking": tracking,
    }
    (output_dir / PATHS.metrics).write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    model_card = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_family": best_name,
        "trained_at": trained_at,
        "intended_use": (
            "Rank active customers by the probability of placing no order in the next "
            f"{CHURN_CONFIG.inactivity_days} days, to prioritise retention outreach."
        ),
        "label_definition": (
            f"churned = 1 if the customer places no order within {CHURN_CONFIG.inactivity_days} "
            "days after the snapshot cutoff"
        ),
        "eligible_population": (
            f"customers with at least one order in the {CHURN_CONFIG.min_history_days} days "
            "before the cutoff"
        ),
        "features": FEATURES,
        "split_strategy": "chronological by snapshot cutoff (train < validation < test)",
        "split": split_summary,
        "metrics": {"validation": result["validation"], "test": result["test"]},
        "feature_importances": result["feature_importances"],
        "decision_threshold": result["threshold"],
        "monitoring_thresholds": {
            "psi_warn": THRESHOLDS.psi_warn,
            "psi_alert": THRESHOLDS.psi_alert,
            "min_precision": THRESHOLDS.min_precision,
            "min_recall": THRESHOLDS.min_recall,
            "min_roc_auc": THRESHOLDS.min_roc_auc,
            "max_model_age_days": THRESHOLDS.max_model_age_days,
        },
        "limitations": [
            "Observational transaction data only; no demographic or marketing-contact features.",
            "A customer buying through a different account or channel reads as churn.",
            "Trained on 2009-2011 UK retail data; seasonality is learned from two cycles only.",
            "Probabilities are not calibrated for monetary expected-value decisions.",
        ],
        "ethical_considerations": [
            "No personally identifying attributes are used; customer_id is a pseudonymous key.",
            "Scores prioritise outreach and must not be used to vary pricing per customer.",
            "The decision threshold should be reviewed with the retention team before any campaign.",
        ],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "tracking": tracking,
    }
    (output_dir / PATHS.model_card).write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )

    # The training feature table doubles as the drift baseline.
    feature_table.to_csv(output_dir / PATHS.training_snapshot, index=False)

    return metrics_payload


def run_training(
    source: str = "auto",
    output_dir: Path | None = None,
    min_precision: float | None = None,
) -> dict:
    """Execute the full training pipeline and return the metrics payload."""
    output_dir = output_dir or PATHS.directory

    transactions = load_transactions(source)
    print(
        f"train: {len(transactions):,} transactions, "
        f"{transactions['customer_id'].nunique():,} customers, "
        f"{transactions['invoice_date'].min():%Y-%m-%d} to {transactions['invoice_date'].max():%Y-%m-%d}"
    )

    feature_table = build_feature_table(transactions)
    train, validation, test = chronological_split(feature_table)
    split_summary = describe_split(train, validation, test)
    print(f"train: {len(feature_table):,} snapshot rows across {feature_table['snapshot_date'].nunique()} cutoffs")
    print(json.dumps(split_summary, indent=2))

    if train[LABEL].nunique() < 2 or validation.empty or test.empty:
        raise ValueError("chronological split produced an unusable block; check the date coverage")

    best_name, results = evaluate_candidates(train, validation, test, min_precision=min_precision)
    tracking = _log_to_mlflow(results, best_name, split_summary, source, feature_table)
    payload = _write_artifacts(
        best_name, results[best_name], split_summary, tracking, source, feature_table, output_dir
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "split"}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the customer-churn model")
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "postgres", "landing"],
        help="where to read validated transactions from (default: auto)",
    )
    parser.add_argument("--output-dir", default=str(PATHS.directory))
    parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="constrain threshold tuning to thresholds meeting this precision",
    )
    args = parser.parse_args()
    try:
        run_training(args.source, Path(args.output_dir), args.min_precision)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the operator
        print(f"train: FAILED - {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
