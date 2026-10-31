"""
Central configuration for the Part 2 MLOps layer.

Everything tunable about the churn problem lives here so that the label
definition, the feature list, the split boundaries and the monitoring
thresholds are stated once and imported everywhere else (training, scoring,
the API, the dashboard, the monitoring job). Credentials are never stored
here - they come from environment variables via scripts/db.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

# ---------------------------------------------------------------- churn label
# A customer is "churned" at a cutoff date if they place no order at all in the
# INACTIVITY_DAYS that follow it. This is the "clear inactivity window" the
# assignment asks for: it is a property of the customer's future behaviour, not
# an opinion, and it is reproducible from the transaction table alone.
INACTIVITY_DAYS = int(os.environ.get("CHURN_INACTIVITY_DAYS", 90))

# A customer must have bought at least once in the MIN_HISTORY_DAYS before the
# cutoff to be scoreable - otherwise "recency" is meaningless and the row is
# really a cold-start case, not a churn case.
MIN_HISTORY_DAYS = int(os.environ.get("CHURN_MIN_HISTORY_DAYS", 365))

# Separately: how much history the dataset must hold *before the first cutoff*,
# so that tenure and the 90d-vs-prior-90d trend features are computable. This
# is deliberately shorter than the eligibility lookback above - requiring a
# full year here would discard a year of otherwise usable snapshots.
MIN_SNAPSHOT_HISTORY_DAYS = int(os.environ.get("CHURN_MIN_SNAPSHOT_HISTORY_DAYS", 180))

# ------------------------------------------------------------------ snapshots
# Features are built at several month-end cutoffs rather than a single one.
# Each cutoff needs INACTIVITY_DAYS of data after it to be labellable, so the
# last few months of the dataset can only ever be scored, never trained on.
SNAPSHOT_FREQUENCY = "ME"  # pandas month-end

# Chronological split: the snapshot cutoffs are ordered in time and cut into
# three contiguous blocks. No customer-month appears in more than one block and
# no future information reaches the training set.
TRAIN_FRACTION = 0.6
VALIDATION_FRACTION = 0.2  # test gets the remainder (0.2)

# ------------------------------------------------------------------- features
# Feature order is part of the model contract: the artifact stores this list and
# the API builds its request frame from it, so a reordering can never silently
# feed the model the wrong column.
FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary",
    "average_order_value",
    "tenure_days",
    "distinct_products",
    "avg_items_per_order",
    "avg_interpurchase_days",
    "orders_last_90d",
    "revenue_last_90d",
    "revenue_trend_ratio",
    "distinct_active_months",
]
LABEL = "churned"

# --------------------------------------------------------------------- MLflow
# Default to a local store so the pipeline is runnable with no server at all.
# docker-compose publishes an MLflow server on 5000 backed by Postgres; set
# MLFLOW_TRACKING_URI=http://localhost:5000 to use that instead.
# A sqlite backend is used rather than a bare ./mlruns directory because the
# MLflow *model registry* is unavailable on the plain-file store. This keeps
# `mlflow.register_model` working with no server running.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "retail-churn")
REGISTERED_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "retail-churn-classifier")

# The metric the model-selection step maximises on the validation block.
SELECTION_METRIC = "roc_auc"

# ----------------------------------------------------------------- monitoring


@dataclass(frozen=True)
class MonitoringThresholds:
    """Alert thresholds for the monitoring job.

    PSI convention: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant
    shift. Performance floors are set just under the values the selected model
    achieved on the held-out test block, so a real regression trips them but
    ordinary month-to-month noise does not.
    """

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    min_precision: float = 0.55
    min_recall: float = 0.55
    min_roc_auc: float = 0.65
    max_null_rate: float = 0.05
    max_p95_latency_ms: float = 250.0

    # Retraining is triggered by any of: a significant drift alert, a
    # performance metric below its floor, or simply age.
    max_model_age_days: int = 30


THRESHOLDS = MonitoringThresholds()


@dataclass(frozen=True)
class ArtifactPaths:
    """Filenames written by the training and monitoring jobs."""

    directory: Path = ARTIFACTS_DIR
    model: str = "churn_model.joblib"
    metrics: str = "churn_metrics.json"
    model_card: str = "churn_model_card.json"
    training_snapshot: str = "churn_training_features.csv"
    scoring_snapshot: str = "churn_scoring_features.csv"
    scores: str = "churn_scores.csv"
    monitoring_report: str = "monitoring_report.json"

    def path(self, name: str) -> Path:
        return self.directory / getattr(self, name)


PATHS = ArtifactPaths()


@dataclass(frozen=True)
class ChurnConfig:
    """The full label/split contract, recorded alongside every model run."""

    inactivity_days: int = INACTIVITY_DAYS
    min_history_days: int = MIN_HISTORY_DAYS
    min_snapshot_history_days: int = MIN_SNAPSHOT_HISTORY_DAYS
    snapshot_frequency: str = SNAPSHOT_FREQUENCY
    train_fraction: float = TRAIN_FRACTION
    validation_fraction: float = VALIDATION_FRACTION
    features: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURES))
    label: str = LABEL

    def as_dict(self) -> dict:
        return {
            "inactivity_days": self.inactivity_days,
            "min_history_days": self.min_history_days,
            "min_snapshot_history_days": self.min_snapshot_history_days,
            "snapshot_frequency": self.snapshot_frequency,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "features": list(self.features),
            "label": self.label,
        }


CHURN_CONFIG = ChurnConfig()
