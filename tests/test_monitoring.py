"""Tests for PSI, the data-quality checks and the retraining decision.

Drift detection is only useful if it is quiet when nothing has changed and
loud when something has, so both directions are tested explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.config import FEATURES, THRESHOLDS
from ml.metrics import classification_metrics, tune_threshold
from ml.monitoring import (
    _collect_alerts,
    data_quality_report,
    drift_report,
    population_stability_index,
    psi_status,
)


@pytest.fixture
def reference_features() -> pd.DataFrame:
    generator = np.random.default_rng(seed=7)
    rows = len(FEATURES) * 0 + 2000
    return pd.DataFrame(
        {
            "customer_id": np.arange(rows),
            **{
                feature: generator.gamma(shape=2.0, scale=30.0, size=rows)
                for feature in FEATURES
            },
        }
    )


def test_psi_is_zero_for_an_identical_distribution(reference_features):
    series = reference_features["monetary"]
    assert population_stability_index(series, series) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_a_shifted_distribution(reference_features):
    series = reference_features["monetary"]
    mild = population_stability_index(series, series * 1.1)
    severe = population_stability_index(series, series * 5.0)
    assert 0 <= mild < severe
    assert severe > THRESHOLDS.psi_alert


def test_psi_handles_a_constant_reference():
    """A feature with no spread cannot drift; it must not raise."""
    constant = pd.Series([3.0] * 100)
    assert population_stability_index(constant, pd.Series([9.0] * 100)) == 0.0


def test_psi_handles_empty_input():
    assert population_stability_index(pd.Series(dtype=float), pd.Series([1.0, 2.0])) == 0.0


def test_psi_status_bands():
    assert psi_status(0.01) == "stable"
    assert psi_status(THRESHOLDS.psi_warn) == "moderate"
    assert psi_status(THRESHOLDS.psi_alert) == "significant"


def test_drift_report_is_stable_against_itself(reference_features):
    report = drift_report(reference_features, reference_features)
    assert report["overall_status"] == "stable"
    assert report["current_rows"] == len(reference_features)
    assert set(report["features"]) == set(FEATURES)


def test_drift_report_flags_a_shifted_feature(reference_features):
    shifted = reference_features.copy()
    shifted["recency_days"] = shifted["recency_days"] * 6.0 + 400
    report = drift_report(reference_features, shifted)

    assert report["overall_status"] == "significant"
    assert report["worst_feature"] == "recency_days"
    assert report["features"]["recency_days"]["status"] == "significant"
    assert report["features"]["monetary"]["status"] == "stable"


def test_drift_report_marks_missing_columns_unavailable(reference_features):
    report = drift_report(reference_features, reference_features.drop(columns=["frequency"]))
    assert report["features"]["frequency"]["status"] == "unavailable"


def test_data_quality_passes_on_a_clean_batch(reference_features):
    clean = reference_features.copy()
    clean["frequency"] = 4
    clean["distinct_active_months"] = 3
    report = data_quality_report(clean)
    assert report["status"] == "pass"
    assert report["failing_features"] == []


def test_data_quality_catches_impossible_values(reference_features):
    broken = reference_features.copy()
    broken.loc[:50, "recency_days"] = -10  # cannot be negative
    broken["frequency"] = 4
    broken["distinct_active_months"] = 3
    report = data_quality_report(broken)

    assert report["status"] == "fail"
    assert "recency_days" in report["failing_features"]
    assert report["checks"]["recency_days"]["domain_violations"] == 51


def test_data_quality_catches_non_finite_values(reference_features):
    broken = reference_features.copy()
    broken.loc[0, "monetary"] = np.inf
    broken["frequency"] = 4
    broken["distinct_active_months"] = 3
    report = data_quality_report(broken)
    assert "monetary" in report["failing_features"]


def test_data_quality_tolerates_structurally_missing_gap(reference_features):
    """avg_interpurchase_days is NaN for single-order customers by design."""
    frame = reference_features.copy()
    frame["avg_interpurchase_days"] = np.nan
    frame["frequency"] = 4
    frame["distinct_active_months"] = 3
    report = data_quality_report(frame)
    assert report["checks"]["avg_interpurchase_days"]["status"] == "pass"
    assert report["status"] == "pass"


def test_data_quality_fails_an_unexpectedly_null_feature(reference_features):
    frame = reference_features.copy()
    frame["monetary"] = np.nan
    frame["frequency"] = 4
    frame["distinct_active_months"] = 3
    report = data_quality_report(frame)
    assert "monetary" in report["failing_features"]


# --------------------------------------------------------------- alert wiring


def _report(**overrides) -> dict:
    """A healthy report, so each test can break exactly one thing."""
    base = {
        "data_quality": {"status": "pass", "failing_features": []},
        "drift": {"overall_status": "stable", "worst_feature": "monetary", "worst_psi": 0.01},
        "performance": {"status": "pass", "breaches": []},
        "latency": {"status": "pass", "batch_ms_p95": 10.0},
        "model_age": {"stale": False, "age_days": 1.0},
    }
    base.update(overrides)
    return base


def test_no_alerts_on_a_healthy_report():
    assert _collect_alerts(_report()) == []


def test_significant_drift_raises_an_alert():
    alerts = _collect_alerts(
        _report(drift={"overall_status": "significant", "worst_feature": "recency_days", "worst_psi": 0.9})
    )
    assert len(alerts) == 1 and "drift" in alerts[0] and "recency_days" in alerts[0]


def test_performance_breach_raises_an_alert():
    alerts = _collect_alerts(
        _report(performance={"status": "fail", "breaches": ["precision 0.300 < 0.55"]})
    )
    assert alerts == ["performance: precision 0.300 < 0.55"]


def test_data_quality_failure_raises_an_alert():
    alerts = _collect_alerts(
        _report(data_quality={"status": "fail", "failing_features": ["monetary"]})
    )
    assert "data quality" in alerts[0] and "monetary" in alerts[0]


def test_stale_model_raises_an_alert():
    alerts = _collect_alerts(_report(model_age={"stale": True, "age_days": 99.0}))
    assert "staleness" in alerts[0]


def test_latency_breach_raises_an_alert():
    alerts = _collect_alerts(_report(latency={"status": "fail", "batch_ms_p95": 900.0}))
    assert "latency" in alerts[0]


def test_alerts_accumulate():
    alerts = _collect_alerts(
        _report(
            data_quality={"status": "fail", "failing_features": ["frequency"]},
            model_age={"stale": True, "age_days": 60.0},
        )
    )
    assert len(alerts) == 2


# -------------------------------------------------------------------- metrics


def test_classification_metrics_on_a_perfect_prediction():
    labels = np.array([0, 0, 1, 1])
    metrics = classification_metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_classification_metrics_handles_a_single_class():
    """A monitoring batch can legitimately contain only one class."""
    metrics = classification_metrics(np.zeros(5), np.array([0.1] * 5))
    assert metrics["roc_auc"] is None
    assert metrics["precision"] == 0.0


def test_tune_threshold_respects_a_precision_floor():
    generator = np.random.default_rng(seed=3)
    labels = generator.integers(0, 2, size=500)
    # Probabilities correlated with the label, so precision is achievable.
    probabilities = np.clip(labels * 0.5 + generator.normal(0.25, 0.15, size=500), 0, 1)

    _, unconstrained = tune_threshold(labels, probabilities)
    _, constrained = tune_threshold(labels, probabilities, min_precision=0.9)

    assert constrained["precision"] >= 0.9
    assert constrained["precision"] >= unconstrained["precision"]
    # Buying precision costs recall - that is the trade the floor makes.
    assert constrained["recall"] <= unconstrained["recall"]


def test_tune_threshold_falls_back_when_floor_is_unreachable():
    labels = np.array([0, 1, 0, 1])
    threshold, metrics = tune_threshold(labels, np.array([0.5, 0.5, 0.5, 0.5]), min_precision=0.99)
    assert threshold == 0.5
    assert metrics["threshold"] == 0.5
