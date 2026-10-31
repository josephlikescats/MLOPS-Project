"""
Model and data monitoring, and the retraining decision.

    python -m ml.monitoring --source auto

Four things are checked, because a churn model can fail in four different ways
and only one of them is visible in the model's own metrics:

  * **input data quality** - nulls, non-finite values, impossible values
    (negative recency, zero frequency), and whether the batch is the size we
    expected. A feature pipeline that breaks upstream shows up here first.
  * **feature drift** - per-feature PSI of the live scoring population against
    the training baseline. PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25
    significant, which is the usual banding.
  * **model performance** - precision, recall and ROC-AUC on the most recent
    snapshot whose inactivity window has actually closed. This is the only
    honest performance read available: a customer whose 90-day window is still
    open has no label yet, by construction.
  * **scoring latency** - wall-clock per-row cost of a batch score, against a
    p95 budget. The API tracks its own live latency separately on /metrics.

The report ends with ``retrain_recommended`` and the list of reasons behind
it, so the decision is auditable rather than a judgement call.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import CHURN_CONFIG, FEATURES, LABEL, PATHS, THRESHOLDS
from ml.data import load_transactions
from ml.features import build_scoring_snapshot, build_snapshot, snapshot_cutoffs
from ml.metrics import classification_metrics
from ml.registry import ModelBundle, load_model_bundle, retraining_signals

PSI_EPSILON = 1e-4


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> float:
    """PSI of ``actual`` against ``expected``, using quantile bins.

    Bin edges come from the reference (training) distribution, so the number
    reads as "how far has the live population moved from what the model was
    trained on". Duplicate quantile edges are dropped, which happens on
    features with a heavy point mass such as orders_last_90d.
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0

    quantiles = [index / bins for index in range(bins + 1)]
    edges = np.unique(expected.quantile(quantiles).values)
    if len(edges) < 2:
        return 0.0
    # Open the outer edges so live values beyond the training range still bin.
    edges[0], edges[-1] = -np.inf, np.inf

    expected_share = (
        pd.cut(expected, bins=edges).value_counts(normalize=True, sort=False).clip(lower=PSI_EPSILON)
    )
    actual_share = (
        pd.cut(actual, bins=edges).value_counts(normalize=True, sort=False).clip(lower=PSI_EPSILON)
    )
    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def psi_status(value: float) -> str:
    if value >= THRESHOLDS.psi_alert:
        return "significant"
    if value >= THRESHOLDS.psi_warn:
        return "moderate"
    return "stable"


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None
) -> dict:
    """Per-feature PSI plus the worst offender and an overall status."""
    features = features or FEATURES
    per_feature = {}
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            per_feature[feature] = {"psi": None, "status": "unavailable"}
            continue
        value = population_stability_index(reference[feature], current[feature])
        per_feature[feature] = {"psi": round(value, 4), "status": psi_status(value)}

    scored = {name: info for name, info in per_feature.items() if info["psi"] is not None}
    worst = max(scored, key=lambda name: scored[name]["psi"]) if scored else None
    statuses = [info["status"] for info in scored.values()]
    overall = (
        "significant" if "significant" in statuses
        else "moderate" if "moderate" in statuses
        else "stable"
    )
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": per_feature,
        "worst_feature": worst,
        "worst_psi": scored[worst]["psi"] if worst else None,
        "overall_status": overall,
        "thresholds": {"warn": THRESHOLDS.psi_warn, "alert": THRESHOLDS.psi_alert},
    }


def data_quality_report(frame: pd.DataFrame, features: list[str] | None = None) -> dict:
    """Null rates, non-finite values and domain violations in a scoring batch."""
    features = features or FEATURES
    checks: dict[str, dict] = {}
    # Values that cannot be true if the feature pipeline ran correctly.
    domain_rules = {
        "recency_days": lambda s: s < 0,
        "tenure_days": lambda s: s < 0,
        "frequency": lambda s: s < 1,
        "monetary": lambda s: s < 0,
        "average_order_value": lambda s: s < 0,
        "distinct_products": lambda s: s < 0,
        "orders_last_90d": lambda s: s < 0,
        "revenue_last_90d": lambda s: s < 0,
        "distinct_active_months": lambda s: s < 1,
    }

    for feature in features:
        if feature not in frame.columns:
            checks[feature] = {"status": "missing_column"}
            continue
        series = pd.to_numeric(frame[feature], errors="coerce")
        null_rate = float(series.isna().mean()) if len(series) else 0.0
        non_finite = int(np.isinf(series.dropna()).sum())
        violations = int(domain_rules[feature](series.dropna()).sum()) if feature in domain_rules else 0
        checks[feature] = {
            "null_rate": round(null_rate, 4),
            "non_finite": non_finite,
            "domain_violations": violations,
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
            "mean": float(series.mean()) if series.notna().any() else None,
            # avg_interpurchase_days is NaN by design for single-order
            # customers, so its null rate is informational rather than a fault.
            "status": _quality_status(feature, null_rate, non_finite, violations),
        }

    failing = [name for name, check in checks.items() if check.get("status") == "fail"]
    return {
        "rows": int(len(frame)),
        "duplicate_customers": int(frame["customer_id"].duplicated().sum())
        if "customer_id" in frame.columns
        else None,
        "checks": checks,
        "failing_features": failing,
        "status": "fail" if failing else "pass",
        "max_null_rate": THRESHOLDS.max_null_rate,
    }


def _quality_status(feature: str, null_rate: float, non_finite: int, violations: int) -> str:
    if non_finite or violations:
        return "fail"
    if feature == "avg_interpurchase_days":
        return "pass"  # structurally missing for one-order customers
    return "fail" if null_rate > THRESHOLDS.max_null_rate else "pass"


def performance_report(
    bundle: ModelBundle, labelled: pd.DataFrame, cutoff: pd.Timestamp
) -> dict:
    """Score the model on a snapshot whose inactivity window has closed."""
    if labelled.empty or LABEL not in labelled.columns:
        return {"status": "unavailable", "reason": "no labelled snapshot available"}

    probabilities = bundle.predict_proba(labelled).to_numpy()
    metrics = classification_metrics(labelled[LABEL], probabilities, bundle.threshold)

    breaches = []
    if metrics["precision"] < THRESHOLDS.min_precision:
        breaches.append(f"precision {metrics['precision']:.3f} < {THRESHOLDS.min_precision}")
    if metrics["recall"] < THRESHOLDS.min_recall:
        breaches.append(f"recall {metrics['recall']:.3f} < {THRESHOLDS.min_recall}")
    if metrics["roc_auc"] is not None and metrics["roc_auc"] < THRESHOLDS.min_roc_auc:
        breaches.append(f"roc_auc {metrics['roc_auc']:.3f} < {THRESHOLDS.min_roc_auc}")

    return {
        "status": "fail" if breaches else "pass",
        "evaluation_cutoff": str(pd.Timestamp(cutoff).date()),
        "metrics": metrics,
        "breaches": breaches,
        "floors": {
            "min_precision": THRESHOLDS.min_precision,
            "min_recall": THRESHOLDS.min_recall,
            "min_roc_auc": THRESHOLDS.min_roc_auc,
        },
    }


def latency_report(bundle: ModelBundle, frame: pd.DataFrame, repeats: int = 5) -> dict:
    """Measure per-row scoring latency over a few repeats of the batch."""
    if frame.empty:
        return {"status": "unavailable", "reason": "empty scoring batch"}

    durations_ms = []
    for _ in range(repeats):
        started = time.perf_counter()
        bundle.predict_proba(frame)
        durations_ms.append((time.perf_counter() - started) * 1000)

    per_batch = np.array(durations_ms)
    per_row = per_batch / len(frame)
    p95 = float(np.percentile(per_batch, 95))
    return {
        "status": "fail" if p95 > THRESHOLDS.max_p95_latency_ms else "pass",
        "rows_scored": int(len(frame)),
        "repeats": repeats,
        "batch_ms_mean": round(float(per_batch.mean()), 2),
        "batch_ms_p95": round(p95, 2),
        "per_row_ms_mean": round(float(per_row.mean()), 4),
        "max_p95_latency_ms": THRESHOLDS.max_p95_latency_ms,
    }


def _baseline_cutoff(reference: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Reduce a multi-cutoff training table to its most recent single cutoff.

    Drift has to be measured like-for-like: one snapshot against one snapshot.
    Comparing a single live cutoff against the whole pooled training table
    instead reports drift on every calendar-dependent feature - tenure_days
    and recency_days both grow mechanically as the dataset advances - which
    swamps the genuine signal. Taking the last training cutoff makes the two
    populations the same shape, measured months apart.
    """
    if "snapshot_date" not in reference.columns or reference.empty:
        return reference, None
    cutoffs = pd.to_datetime(reference["snapshot_date"])
    latest = cutoffs.max()
    return reference[cutoffs == latest].reset_index(drop=True), str(pd.Timestamp(latest).date())


def _latest_labelled_snapshot(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """The most recent cutoff whose full inactivity window is inside the data."""
    cutoffs = snapshot_cutoffs(
        transactions,
        inactivity_days=CHURN_CONFIG.inactivity_days,
        frequency=CHURN_CONFIG.snapshot_frequency,
        min_snapshot_history_days=CHURN_CONFIG.min_snapshot_history_days,
    )
    if not cutoffs:
        return pd.DataFrame(), None
    cutoff = cutoffs[-1]
    return build_snapshot(
        transactions,
        cutoff,
        inactivity_days=CHURN_CONFIG.inactivity_days,
        min_history_days=CHURN_CONFIG.min_history_days,
    ), cutoff


def run_monitoring(
    source: str = "auto",
    output_path: Path | None = None,
    reference_path: Path | None = None,
) -> dict:
    """Run every check and write the combined report."""
    output = output_path or PATHS.path("monitoring_report")
    reference_file = reference_path or PATHS.path("training_snapshot")

    bundle = load_model_bundle()
    transactions = load_transactions(source)
    current = build_scoring_snapshot(transactions)

    if reference_file.exists():
        reference = pd.read_csv(reference_file)
        reference, reference_cutoff = _baseline_cutoff(reference)
        reference_origin = (
            f"{reference_file.name} @ {reference_cutoff}" if reference_cutoff else reference_file.name
        )
    else:
        # Without the saved baseline, fall back to the earliest labellable
        # snapshot so drift is still measured against *something* historical.
        reference, recomputed_cutoff = _latest_labelled_snapshot(transactions)
        reference_origin = f"recomputed-snapshot @ {recomputed_cutoff}"

    labelled, evaluation_cutoff = _latest_labelled_snapshot(transactions)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "model": {
            "version": bundle.model_version,
            "family": bundle.model_family,
            "origin": bundle.origin,
            "threshold": bundle.threshold,
            "trained_at": bundle.trained_at,
        },
        "scoring_population": {
            "rows": int(len(current)),
            "cutoff": str(pd.Timestamp(current["snapshot_date"].iloc[0]).date())
            if not current.empty
            else None,
        },
        "data_quality": data_quality_report(current),
        "drift": {**drift_report(reference, current), "reference": reference_origin},
        "performance": performance_report(bundle, labelled, evaluation_cutoff)
        if evaluation_cutoff is not None
        else {"status": "unavailable", "reason": "no labellable snapshot"},
        "latency": latency_report(bundle, current),
        "model_age": retraining_signals(bundle),
    }

    report["alerts"] = _collect_alerts(report)
    report["retrain_recommended"] = bool(report["alerts"])
    report["status"] = "action_required" if report["alerts"] else "healthy"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _collect_alerts(report: dict) -> list[str]:
    """Turn failed checks into the explicit reasons for retraining."""
    alerts: list[str] = []

    if report["data_quality"]["status"] == "fail":
        failing = ", ".join(report["data_quality"]["failing_features"])
        alerts.append(f"data quality: failing features ({failing})")

    drift = report["drift"]
    if drift["overall_status"] == "significant":
        alerts.append(
            f"drift: {drift['worst_feature']} PSI {drift['worst_psi']} "
            f">= {THRESHOLDS.psi_alert}"
        )

    performance = report["performance"]
    if performance.get("status") == "fail":
        alerts.extend(f"performance: {breach}" for breach in performance["breaches"])

    if report["latency"].get("status") == "fail":
        alerts.append(
            f"latency: p95 {report['latency']['batch_ms_p95']}ms > "
            f"{THRESHOLDS.max_p95_latency_ms}ms"
        )

    if report["model_age"].get("stale"):
        alerts.append(
            f"staleness: model is {report['model_age']['age_days']} days old "
            f"(limit {THRESHOLDS.max_model_age_days})"
        )

    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor churn data, drift and performance")
    parser.add_argument("--source", default="auto", choices=["auto", "postgres", "landing"])
    parser.add_argument("--output", default=str(PATHS.path("monitoring_report")))
    parser.add_argument(
        "--reference",
        default=str(PATHS.path("training_snapshot")),
        help="training feature baseline to measure drift against",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="exit non-zero when any check fails, for use in a scheduled job",
    )
    args = parser.parse_args()

    report = run_monitoring(args.source, Path(args.output), Path(args.reference))
    summary = {
        "status": report["status"],
        "data_quality": report["data_quality"]["status"],
        "drift": report["drift"]["overall_status"],
        "worst_drift": f"{report['drift']['worst_feature']}={report['drift']['worst_psi']}",
        "performance": report["performance"].get("status"),
        "latency_p95_ms": report["latency"].get("batch_ms_p95"),
        "model_age_days": report["model_age"].get("age_days"),
        "retrain_recommended": report["retrain_recommended"],
        "alerts": report["alerts"],
    }
    print(json.dumps(summary, indent=2))
    print(f"monitoring: full report written to {args.output}")

    if args.fail_on_alert and report["retrain_recommended"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
