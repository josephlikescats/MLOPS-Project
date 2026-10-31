"""API contract tests.

These run against a small model trained in-process on the synthetic fixture,
not the artifact in artifacts/, so the suite passes on a fresh clone with no
trained model and still exercises the real resolution path in ml.registry.
"""
from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient

from ml.config import FEATURES
from ml.features import build_feature_table, chronological_split
from ml.models import build_candidates

VALID_PAYLOAD = {
    "recency_days": 45,
    "frequency": 6,
    "monetary": 1850.40,
    "average_order_value": 308.40,
    "tenure_days": 520,
    "distinct_products": 73,
    "avg_items_per_order": 18.5,
    "avg_interpurchase_days": 62.0,
    "orders_last_90d": 1,
    "revenue_last_90d": 210.75,
    "revenue_trend_ratio": 0.42,
    "distinct_active_months": 5,
}


@pytest.fixture
def client(tmp_path, transactions, monkeypatch):
    """A TestClient whose model is a small pipeline fit on the fixture data."""
    table = build_feature_table(transactions)
    train, _, _ = chronological_split(table)
    pipeline = build_candidates()["logistic_regression"]
    pipeline.fit(train[FEATURES], train["churned"])

    artifact = tmp_path / "churn_model.joblib"
    joblib.dump(
        {
            "model": pipeline,
            "features": FEATURES,
            "label": "churned",
            "threshold": 0.4,
            "model_family": "logistic_regression",
            "config": {"inactivity_days": 90},
            "metrics": {"test": {"roc_auc": 0.7}},
            "trained_at": "2026-01-01T00:00:00+00:00",
            "tracking": {"registered_model": "retail-churn-classifier", "model_version": "7"},
        },
        artifact,
    )

    import api.main as api_main

    # Point the resolver at the fixture artifact rather than artifacts/, so
    # the real ml.registry resolution path is what gets exercised.
    monkeypatch.setenv("CHURN_MODEL_PATH", str(artifact))
    monkeypatch.delenv("CHURN_MODEL_URI", raising=False)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "retail-churn-classifier/v7"


def test_ready_returns_the_version(client):
    assert client.get("/ready").json()["model_version"] == "retail-churn-classifier/v7"


def test_model_metadata_exposes_the_serving_contract(client):
    body = client.get("/model").json()
    assert body["features"] == FEATURES
    assert body["decision_threshold"] == 0.4
    assert body["model_family"] == "logistic_regression"
    assert body["version"] == "7"


def test_predict_returns_a_calibrated_response(client):
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_band"] in ("low", "medium", "high", "critical")
    assert body["decision_threshold"] == 0.4
    assert body["inference_ms"] >= 0


def test_prediction_agrees_with_the_threshold(client):
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    assert body["churn_prediction"] == int(
        body["churn_probability"] >= body["decision_threshold"]
    )


def test_a_dormant_customer_scores_higher_than_an_active_one(client):
    """Sanity check on direction: recency is the dominant churn signal."""
    dormant = {**VALID_PAYLOAD, "recency_days": 300, "orders_last_90d": 0,
               "revenue_last_90d": 0.0, "revenue_trend_ratio": 0.05}
    active = {**VALID_PAYLOAD, "recency_days": 3, "orders_last_90d": 5,
              "revenue_last_90d": 900.0, "revenue_trend_ratio": 2.5}

    dormant_probability = client.post("/predict", json=dormant).json()["churn_probability"]
    active_probability = client.post("/predict", json=active).json()["churn_probability"]
    assert dormant_probability > active_probability


def test_batch_prediction_counts_match(client):
    body = client.post(
        "/predict/batch", json={"customers": [VALID_PAYLOAD, VALID_PAYLOAD]}
    ).json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["flagged"] == sum(p["churn_prediction"] for p in body["predictions"])


def test_batch_rejects_an_empty_request(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_batch_rejects_an_oversized_request(client, monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "MAX_BATCH_SIZE", 2)
    response = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD] * 3})
    assert response.status_code == 413


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("recency_days", -1),
        ("frequency", 0),
        ("monetary", -5.0),
        ("tenure_days", -10),
        ("revenue_last_90d", -0.01),
    ],
)
def test_impossible_feature_values_are_rejected(client, field, bad_value):
    """The same domain rules the monitoring job enforces, applied at the edge."""
    response = client.post("/predict", json={**VALID_PAYLOAD, field: bad_value})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != "monetary"}
    assert client.post("/predict", json=payload).status_code == 422


def test_optional_interpurchase_gap_may_be_null(client):
    """Single-order customers have no gap; the model imputes it."""
    response = client.post("/predict", json={**VALID_PAYLOAD, "avg_interpurchase_days": None})
    assert response.status_code == 200


def test_unknown_customer_lookup_is_a_404(client):
    response = client.get("/customers/99999999/churn")
    # 404 when a scores file exists but lacks the id, 503 when none has been
    # produced yet - both are correct, and neither is a 500.
    assert response.status_code in (404, 503)


def test_metrics_counts_requests_and_failures(client):
    client.get("/health")
    client.post("/predict", json=VALID_PAYLOAD)
    client.post("/predict", json={**VALID_PAYLOAD, "frequency": 0})  # 422

    body = client.get("/metrics").json()
    assert body["total_requests"] >= 3
    assert body["total_failures"] == 0  # a 422 is the caller's fault, not ours
    assert body["predictions_served"] >= 1
    assert "/predict" in body["endpoints"]
    assert body["endpoints"]["/predict"]["latency_ms"]["p95"] >= 0


def test_prometheus_metrics_expose_the_expected_series(client):
    client.post("/predict", json=VALID_PAYLOAD)
    text = client.get("/metrics/prometheus").text
    for series in (
        "churn_api_requests_total",
        "churn_api_failures_total",
        "churn_api_predictions_total",
        "churn_api_model_loaded 1",
    ):
        assert series in text


def test_response_time_header_is_set(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert float(response.headers["X-Response-Time-Ms"]) >= 0


def test_reload_re_resolves_the_model(client):
    assert client.post("/reload").json()["model_version"] == "retail-churn-classifier/v7"


def test_endpoints_degrade_when_no_model_is_available(tmp_path, monkeypatch):
    """With no artifact and no registry, /health degrades and /predict 503s."""
    import api.main as api_main

    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "absent.joblib"))
    monkeypatch.delenv("CHURN_MODEL_URI", raising=False)
    monkeypatch.setattr(
        api_main, "load_model_bundle", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("none"))
    )

    with TestClient(api_main.app) as degraded:
        health = degraded.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False
        assert degraded.get("/ready").status_code == 503
        assert degraded.post("/predict", json=VALID_PAYLOAD).status_code == 503
        # Observability must survive the model being gone.
        assert degraded.get("/metrics").status_code == 200
