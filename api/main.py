"""
FastAPI inference service for the churn and repeat-purchase models.

Endpoints
    GET  /health                       liveness + whether a model is loaded
    GET  /ready                        readiness: 503 until a model is loaded
    GET  /model                        version, threshold, features, metrics
    POST /predict                      churn risk for one customer's features
    POST /predict/batch                churn risk for many (capped)
    GET  /customers/{id}/churn         latest batch score for a customer
    POST /predict/repeat-purchase      the Part 1 repeat-purchase model
    GET  /metrics                      request counts, failures, latency
    GET  /metrics/prometheus           the same, in Prometheus text format

Two things worth noting about the design:

* The model is resolved once at startup through ml.registry, so the service
  reports a registered model *version* rather than a filename, and can be
  pinned to a specific version with CHURN_MODEL_URI.
* /predict computes nothing from the warehouse - the caller supplies features.
  Serving a customer by id instead reads the latest *batch* score, which is
  the usual split: features are expensive and batch-shaped, lookups are cheap.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ml.config import FEATURES, PATHS
from ml.registry import ModelBundle, load_model_bundle, risk_band

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("churn-api")

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 1000))
REPEAT_PURCHASE_MODEL = PATHS.directory / "repeat_purchase_model.joblib"

# ---------------------------------------------------------------- observability


class ServiceMetrics:
    """In-process request counters and latency percentiles.

    Deliberately in-memory and per-process: it is the right size for a single
    container and needs no extra infrastructure. A horizontally scaled
    deployment would scrape /metrics/prometheus from each replica instead of
    trying to share this state.
    """

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._window = window
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._requests: dict[str, int] = defaultdict(int)
        self._failures: dict[str, int] = defaultdict(int)
        self._status_codes: dict[int, int] = defaultdict(int)
        self.started_at = time.time()
        self.predictions_served = 0

    def observe(self, endpoint: str, duration_ms: float, status_code: int) -> None:
        with self._lock:
            self._latencies[endpoint].append(duration_ms)
            self._requests[endpoint] += 1
            self._status_codes[status_code] += 1
            if status_code >= 500:
                self._failures[endpoint] += 1

    def count_predictions(self, rows: int) -> None:
        with self._lock:
            self.predictions_served += rows

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(round((percentile / 100) * (len(ordered) - 1))), len(ordered) - 1)
        return round(ordered[index], 2)

    def snapshot(self) -> dict:
        with self._lock:
            endpoints = {}
            for endpoint, samples in self._latencies.items():
                values = list(samples)
                endpoints[endpoint] = {
                    "requests": self._requests[endpoint],
                    "failures": self._failures[endpoint],
                    "latency_ms": {
                        "mean": round(sum(values) / len(values), 2) if values else 0.0,
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99),
                        "max": round(max(values), 2) if values else 0.0,
                    },
                }
            total_requests = sum(self._requests.values())
            total_failures = sum(self._failures.values())
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "total_requests": total_requests,
                "total_failures": total_failures,
                "failure_rate": round(total_failures / total_requests, 4) if total_requests else 0.0,
                "predictions_served": self.predictions_served,
                "status_codes": {str(code): count for code, count in sorted(self._status_codes.items())},
                "endpoints": endpoints,
                "latency_window": self._window,
            }


METRICS = ServiceMetrics()

# --------------------------------------------------------------------- schemas


class CustomerFeatures(BaseModel):
    """One customer's features at a point in time.

    Field names and bounds mirror ml/features.py; the bounds are the same
    domain rules the monitoring job's data-quality check enforces, so an
    impossible row is rejected at the edge rather than scored.
    """

    recency_days: int = Field(ge=0, description="days since the customer's last order")
    frequency: int = Field(ge=1, description="distinct orders in history")
    monetary: float = Field(ge=0, description="total revenue in history")
    average_order_value: float = Field(ge=0)
    tenure_days: int = Field(ge=0, description="days since first order")
    distinct_products: int = Field(ge=0)
    avg_items_per_order: float = Field(ge=0)
    avg_interpurchase_days: float | None = Field(
        default=None, ge=0, description="null for single-order customers; median-imputed"
    )
    orders_last_90d: int = Field(ge=0)
    revenue_last_90d: float = Field(ge=0)
    revenue_trend_ratio: float = Field(ge=0, description="(rev last 90d + 1) / (rev prior 90d + 1)")
    distinct_active_months: int = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1)


class ChurnPrediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_band: str
    decision_threshold: float
    model_version: str
    inference_ms: float


class BatchPrediction(BaseModel):
    predictions: list[ChurnPrediction]
    count: int
    flagged: int
    inference_ms: float


class RepeatPurchaseFeatures(BaseModel):
    recency_days: int = Field(ge=0)
    frequency: int = Field(ge=1)
    monetary: float = Field(ge=0)
    average_order_value: float = Field(ge=0)


class RepeatPurchasePrediction(BaseModel):
    repeat_purchase_probability: float
    repeat_purchase_prediction: int
    model_version: str


# ----------------------------------------------------------------- app + state

STATE: dict[str, Any] = {"bundle": None, "load_error": None, "scores": None}


def _load_bundle() -> None:
    """Resolve the churn model at startup; a failure is reported, not fatal.

    The service still starts so /health and /metrics stay scrapeable and the
    cause is visible in one place, instead of a container that crash-loops
    before anything can be asked of it.
    """
    try:
        bundle = load_model_bundle()
        STATE["bundle"] = bundle
        STATE["load_error"] = None
        logger.info(
            "loaded churn model version=%s family=%s origin=%s threshold=%.2f",
            bundle.model_version,
            bundle.model_family,
            bundle.origin,
            bundle.threshold,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced through /health and /ready
        STATE["bundle"] = None
        STATE["load_error"] = f"{exc.__class__.__name__}: {exc}"
        logger.error("churn model unavailable: %s", STATE["load_error"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_bundle()
    yield
    STATE.clear()


app = FastAPI(
    title="Retail Churn & Repeat-Purchase API",
    version="2.0.0",
    description="Serves the registered customer-churn model from the retail MLOps pipeline.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Time every request, count failures, and emit one structured log line."""
    started = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        METRICS.observe(endpoint, duration_ms, 500)
        logger.exception("unhandled error on %s after %.1fms", endpoint, duration_ms)
        raise

    duration_ms = (time.perf_counter() - started) * 1000
    METRICS.observe(endpoint, duration_ms, status_code)
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    if status_code >= 400:
        logger.warning("%s %s -> %s in %.1fms", request.method, endpoint, status_code, duration_ms)
    return response


def get_bundle() -> ModelBundle:
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"churn model unavailable ({STATE.get('load_error')}). Run `python -m ml.train`.",
        )
    return bundle


def _features_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    """Build the model's input frame in the bundle's declared feature order."""
    rows = [customer.model_dump() for customer in customers]
    return pd.DataFrame(rows, columns=FEATURES)


# ------------------------------------------------------------------- endpoints


@app.get("/health", tags=["ops"])
def health() -> dict:
    bundle = STATE.get("bundle")
    return {
        "status": "ok" if bundle else "degraded",
        "model_loaded": bundle is not None,
        "model_version": bundle.model_version if bundle else None,
        "load_error": STATE.get("load_error"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    bundle = get_bundle()
    return {"status": "ready", "model_version": bundle.model_version}


@app.post("/reload", tags=["ops"])
def reload_model() -> dict:
    """Re-resolve the model without restarting, after a retrain or a repoint."""
    _load_bundle()
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail=STATE.get("load_error"))
    return {"status": "reloaded", "model_version": bundle.model_version}


@app.get("/model", tags=["model"])
def model_metadata() -> dict:
    bundle = get_bundle()
    return {
        "model_version": bundle.model_version,
        "registered_model": bundle.registered_model,
        "version": bundle.version,
        "model_family": bundle.model_family,
        "origin": bundle.origin,
        "run_id": bundle.run_id,
        "trained_at": bundle.trained_at,
        "age_days": round(bundle.age_days, 2) if bundle.age_days is not None else None,
        "decision_threshold": bundle.threshold,
        "label": bundle.label,
        "features": bundle.features,
        "churn_config": bundle.config,
        "metrics": bundle.metrics,
    }


@app.post("/predict", response_model=ChurnPrediction, tags=["model"])
def predict(features: CustomerFeatures) -> ChurnPrediction:
    bundle = get_bundle()
    started = time.perf_counter()
    frame = _features_frame([features])
    probability = float(bundle.predict_proba(frame).iloc[0])
    duration_ms = (time.perf_counter() - started) * 1000
    METRICS.count_predictions(1)
    return ChurnPrediction(
        churn_probability=probability,
        churn_prediction=int(probability >= bundle.threshold),
        risk_band=str(risk_band(pd.Series([probability])).iloc[0]),
        decision_threshold=bundle.threshold,
        model_version=bundle.model_version,
        inference_ms=round(duration_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPrediction, tags=["model"])
def predict_batch(request: BatchRequest) -> BatchPrediction:
    bundle = get_bundle()
    if len(request.customers) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(request.customers)} exceeds the limit of {MAX_BATCH_SIZE}",
        )

    started = time.perf_counter()
    frame = _features_frame(request.customers)
    predictions = bundle.predict(frame)
    duration_ms = (time.perf_counter() - started) * 1000
    METRICS.count_predictions(len(frame))

    return BatchPrediction(
        predictions=[
            ChurnPrediction(
                churn_probability=float(row.churn_probability),
                churn_prediction=int(row.churn_prediction),
                risk_band=str(row.risk_band),
                decision_threshold=bundle.threshold,
                model_version=bundle.model_version,
                inference_ms=round(duration_ms / len(frame), 3),
            )
            for row in predictions.itertuples()
        ],
        count=len(frame),
        flagged=int(predictions["churn_prediction"].sum()),
        inference_ms=round(duration_ms, 3),
    )


@app.get("/customers/{customer_id}/churn", tags=["model"])
def customer_churn(customer_id: int) -> dict:
    """Look the customer up in the latest batch-scored file.

    Features are expensive to compute per request, so the scheduled batch job
    (`python -m ml.score_churn`) owns that work and this endpoint serves its
    output. The response states when the batch ran so a stale file is obvious.
    """
    scores = STATE.get("scores")
    scores_path = PATHS.path("scores")
    if scores is None:
        if not scores_path.exists():
            raise HTTPException(
                status_code=503,
                detail="no batch scores available. Run `python -m ml.score_churn` first.",
            )
        scores = pd.read_csv(scores_path)
        STATE["scores"] = scores

    row = scores[scores["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"customer {customer_id} is not in the current scoring population",
        )

    record = row.iloc[0]
    return {
        "customer_id": customer_id,
        "churn_probability": float(record["churn_probability"]),
        "churn_prediction": int(record["churn_prediction"]),
        "risk_band": str(record["risk_band"]),
        "decision_threshold": float(record["decision_threshold"]),
        "model_version": str(record["model_version"]),
        "snapshot_date": str(record.get("snapshot_date")),
        "scored_at": datetime.fromtimestamp(
            scores_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


@app.post("/predict/repeat-purchase", response_model=RepeatPurchasePrediction, tags=["model"])
def predict_repeat_purchase(features: RepeatPurchaseFeatures) -> RepeatPurchasePrediction:
    """The Part 1 repeat-purchase model, kept available alongside churn."""
    if not REPEAT_PURCHASE_MODEL.exists():
        raise HTTPException(
            status_code=503,
            detail="repeat-purchase artifact unavailable; run `python -m ml.train_model` first",
        )
    bundle = joblib.load(REPEAT_PURCHASE_MODEL)
    frame = pd.DataFrame([features.model_dump()], columns=bundle["features"])
    probability = float(bundle["model"].predict_proba(frame)[:, 1][0])
    METRICS.count_predictions(1)
    return RepeatPurchasePrediction(
        repeat_purchase_probability=probability,
        repeat_purchase_prediction=int(probability >= 0.5),
        model_version=REPEAT_PURCHASE_MODEL.name,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> dict:
    bundle = STATE.get("bundle")
    return {
        **METRICS.snapshot(),
        "model_version": bundle.model_version if bundle else None,
        "model_loaded": bundle is not None,
    }


@app.get("/metrics/prometheus", tags=["ops"])
def metrics_prometheus() -> Any:
    """The same counters in Prometheus text format, for a scrape target."""
    from fastapi.responses import PlainTextResponse

    snapshot = METRICS.snapshot()
    lines = [
        "# HELP churn_api_requests_total Total HTTP requests handled.",
        "# TYPE churn_api_requests_total counter",
        f"churn_api_requests_total {snapshot['total_requests']}",
        "# HELP churn_api_failures_total Requests that returned 5xx.",
        "# TYPE churn_api_failures_total counter",
        f"churn_api_failures_total {snapshot['total_failures']}",
        "# HELP churn_api_predictions_total Customer rows scored.",
        "# TYPE churn_api_predictions_total counter",
        f"churn_api_predictions_total {snapshot['predictions_served']}",
        "# HELP churn_api_uptime_seconds Seconds since process start.",
        "# TYPE churn_api_uptime_seconds gauge",
        f"churn_api_uptime_seconds {snapshot['uptime_seconds']}",
        "# HELP churn_api_model_loaded Whether a churn model is loaded.",
        "# TYPE churn_api_model_loaded gauge",
        f"churn_api_model_loaded {1 if STATE.get('bundle') else 0}",
        "# HELP churn_api_latency_ms Request latency percentiles per endpoint.",
        "# TYPE churn_api_latency_ms gauge",
    ]
    for endpoint, stats in snapshot["endpoints"].items():
        for quantile in ("p50", "p95", "p99"):
            lines.append(
                f'churn_api_latency_ms{{endpoint="{endpoint}",quantile="{quantile}"}} '
                f"{stats['latency_ms'][quantile]}"
            )
    return PlainTextResponse("\n".join(lines) + "\n")
