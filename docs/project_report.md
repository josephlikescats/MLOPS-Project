# Project Report Outline

Use this outline for the required 8-10 page report and replace the evidence placeholders with
screenshots from the successful run.

## 1. Executive summary

State the business problem, UCI source, warehouse outcome, dashboard result, and repeat-purchase
prediction objective.

## 2. Dataset and legal access

Source: UCI Online Retail II, https://archive.ics.uci.edu/dataset/502/online+retail+ii.
Explain that it is publicly accessible and downloaded by `scripts/split_landing_files.py`.

## 3. Architecture

Include `docs/architecture.md` as the architecture diagram. Explain landing, raw, staging,
warehouse, marts, dashboard, model training, batch scoring, API, and drift-report layers.

## 4. ETL implementation

Describe Airflow scheduling, incremental file detection, ingestion logging, validation rules,
rejected-record handling, duplicate prevention, and the PostgreSQL star schema.

## 5. Data model and quality

Discuss customer, product, geography, date, and sales fact dimensions. Report accepted and
rejected row counts and attach Airflow logs or a task-graph screenshot.

## 6. Dashboard findings

Include screenshots and interpretations for revenue trends, top products, customer segments,
country sales, and retention/repeat-purchase views.

## 7. Repeat-purchase model

The model uses a 90-day horizon. Features are recency, frequency, monetary value, and average
order value. The label is whether the customer purchases during the future 90-day window.
Explain the chronological split, logistic-regression baseline, threshold, and metrics in
`artifacts/metrics.json`.

## 8. MLOps operationalization

Describe the Joblib artifact, batch scoring command, FastAPI `/predict` endpoint, `/health`
check, and PSI drift report. Explain how a scheduled job can retrain the artifact and retain
metrics for model review.

## 9. Limitations and ethics

Discuss observational data, class imbalance, time-window assumptions, no personally identifiable
enrichment, and threshold review before customer outreach.

## 10. Conclusion and evidence appendix

Include screenshots of the successful DAG run, populated tables, dashboard, training metrics,
API health, and a sample prediction.