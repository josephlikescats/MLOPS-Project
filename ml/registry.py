"""
Model resolution for everything that serves predictions.

The API, the batch scorer, the monitoring job and the dashboard all need "the
current churn model" and must agree on what that means. They all call
``load_model_bundle`` here, which resolves in a fixed order:

  1. ``CHURN_MODEL_URI`` if set - an explicit MLflow URI
     (``models:/retail-churn-classifier/3``, ``models:/<name>@champion``, or
     ``runs:/<run_id>/model``). This is how a specific registered version gets
     pinned in a deployment.
  2. the local joblib artifact, which carries the full serving contract -
     model, feature order, decision threshold, metrics and the MLflow run and
     version it came from.
  3. the latest version in the MLflow registry, if one is reachable.

The local artifact is preferred over the registry by default so the container
starts with no network dependency, while still reporting the registry version
it corresponds to. Loading never silently guesses a threshold or a feature
order: both are part of the bundle, and a registry-only load reads the
threshold back from the run that produced it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.config import (
    FEATURES,
    MLFLOW_TRACKING_URI,
    PATHS,
    REGISTERED_MODEL_NAME,
    THRESHOLDS,
)
from ml.metrics import DEFAULT_THRESHOLD

MODEL_URI_ENV = "CHURN_MODEL_URI"
# Points at a specific artifact file, for pinning a deployment to one
# bundle without rebuilding the image or touching the registry.
MODEL_PATH_ENV = "CHURN_MODEL_PATH"


@dataclass
class ModelBundle:
    """A loaded model plus the contract needed to use it correctly."""

    model: Any
    features: list[str]
    threshold: float
    label: str = "churned"
    model_family: str = "unknown"
    version: str | None = None
    registered_model: str | None = None
    run_id: str | None = None
    trained_at: str | None = None
    origin: str = "artifact"
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        """A single human-readable version string for API responses and logs."""
        if self.registered_model and self.version:
            return f"{self.registered_model}/v{self.version}"
        if self.run_id:
            return f"run:{self.run_id[:8]}"
        return f"{self.model_family}@{self.trained_at or 'unknown'}"

    @property
    def age_days(self) -> float | None:
        """Days since training, used by the retraining-criteria check."""
        if not self.trained_at:
            return None
        try:
            trained = datetime.fromisoformat(self.trained_at)
        except ValueError:
            return None
        if trained.tzinfo is None:
            trained = trained.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - trained).total_seconds() / 86400

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        """Churn probability for each row, with the bundle's feature order."""
        missing = [column for column in self.features if column not in frame.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        probabilities = self.model.predict_proba(frame[self.features])[:, 1]
        return pd.Series(probabilities, index=frame.index, name="churn_probability")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Probability plus the thresholded decision and a risk band."""
        probabilities = self.predict_proba(frame)
        return pd.DataFrame(
            {
                "churn_probability": probabilities,
                "churn_prediction": (probabilities >= self.threshold).astype(int),
                "risk_band": risk_band(probabilities),
            },
            index=frame.index,
        )


def risk_band(probabilities: pd.Series) -> pd.Series:
    """Bucket probabilities into bands the dashboard and outreach lists use."""
    return pd.cut(
        probabilities,
        bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)


def _bundle_from_artifact(path: Path) -> ModelBundle:
    raw = joblib.load(path)
    tracking = raw.get("tracking") or {}
    return ModelBundle(
        model=raw["model"],
        features=list(raw.get("features") or FEATURES),
        threshold=float(raw.get("threshold", DEFAULT_THRESHOLD)),
        label=raw.get("label", "churned"),
        model_family=raw.get("model_family", "unknown"),
        version=tracking.get("model_version"),
        registered_model=tracking.get("registered_model"),
        run_id=tracking.get("run_id"),
        trained_at=raw.get("trained_at"),
        origin="artifact",
        metrics=raw.get("metrics") or {},
        config=raw.get("config") or {},
    )


def _bundle_from_mlflow(model_uri: str) -> ModelBundle:
    """Load from an MLflow URI and recover the threshold from its source run."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)

    client = MlflowClient()
    run_id: str | None = None
    version: str | None = None
    registered_model: str | None = None

    if model_uri.startswith("runs:/"):
        run_id = model_uri.split("/")[1]
    elif model_uri.startswith("models:/"):
        reference = model_uri.removeprefix("models:/")
        registered_model, _, qualifier = reference.partition("/")
        if "@" in registered_model:
            registered_model, _, alias = registered_model.partition("@")
            details = client.get_model_version_by_alias(registered_model, alias)
        elif qualifier in ("", "latest"):
            versions = client.search_model_versions(f"name='{registered_model}'")
            if not versions:
                raise ValueError(f"no versions registered for {registered_model!r}")
            details = max(versions, key=lambda v: int(v.version))
        else:
            details = client.get_model_version(registered_model, qualifier)
        version, run_id = str(details.version), details.run_id

    threshold, trained_at, family = DEFAULT_THRESHOLD, None, "unknown"
    if run_id:
        run = client.get_run(run_id)
        threshold = float(run.data.params.get("decision_threshold", DEFAULT_THRESHOLD))
        family = run.data.tags.get("model_family", "unknown")
        trained_at = datetime.fromtimestamp(
            run.info.start_time / 1000, tz=timezone.utc
        ).isoformat()

    return ModelBundle(
        model=model,
        features=list(FEATURES),
        threshold=threshold,
        model_family=family,
        version=version,
        registered_model=registered_model,
        run_id=run_id,
        trained_at=trained_at,
        origin="mlflow",
    )


def load_model_bundle(model_uri: str | None = None, artifact_path: Path | None = None) -> ModelBundle:
    """Resolve the current churn model. See the module docstring for the order."""
    uri = model_uri or os.environ.get(MODEL_URI_ENV)
    if uri:
        return _bundle_from_mlflow(uri)

    override = os.environ.get(MODEL_PATH_ENV)
    path = artifact_path or (Path(override) if override else PATHS.path("model"))
    if path.exists():
        return _bundle_from_artifact(path)

    try:
        return _bundle_from_mlflow(f"models:/{REGISTERED_MODEL_NAME}/latest")
    except Exception as exc:  # noqa: BLE001 - report both failures together
        raise FileNotFoundError(
            f"no churn model available: {path} is missing and the MLflow registry "
            f"could not be read ({exc.__class__.__name__}: {exc}). "
            "Run `python -m ml.train` first."
        ) from exc


def retraining_signals(bundle: ModelBundle) -> dict:
    """Evaluate the age-based retraining criterion for this bundle."""
    age = bundle.age_days
    return {
        "model_version": bundle.model_version,
        "trained_at": bundle.trained_at,
        "age_days": round(age, 2) if age is not None else None,
        "max_model_age_days": THRESHOLDS.max_model_age_days,
        "stale": bool(age is not None and age > THRESHOLDS.max_model_age_days),
    }
