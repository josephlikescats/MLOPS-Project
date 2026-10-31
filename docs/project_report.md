# Project Report Outline

Use this outline for the required 8-10 page report and replace the evidence placeholders with
screenshots from the successful run. Numbers quoted below are from the verified run recorded
in the README; re-run the pipeline and update them if the data or configuration changes.

## 1. Executive summary

State the business problem, the UCI source, the warehouse outcome, the dashboard result, and
the churn-prediction objective. Headline figures: 768,873 validated transactions across 25
monthly extracts; a star schema and two marts; a 9-view dashboard; and a churn model scoring
4,269 active customers at ROC-AUC 0.782 on a held-out, strictly later time block.

## 2. Dataset and legal access

Source: UCI Online Retail II, https://archive.ics.uci.edu/dataset/502/online+retail+ii.
Publicly accessible, downloaded by the documented `curl` + `scripts/split_landing_files.py`
step. No final analytical table is edited by hand.

## 3. Architecture

Include the Mermaid diagram from `docs/architecture.md`. Walk the layers: landing, raw,
staging, warehouse star schema, marts, dashboard, then the Part 2 MLOps layer (feature
snapshots, training and selection, MLflow tracking and registry, serving, batch scoring,
monitoring, retraining DAG). Cite the "Why these choices" section for the trade-offs.

## 4. ETL implementation (Part 1)

Airflow scheduling, incremental file detection via `meta.ingestion_log`, validation rules,
rejected-record handling, duplicate prevention on the natural key, and the PostgreSQL star
schema. Note that re-running the DAG reports `skipped=25` and changes no row counts.

## 5. Data model and quality

The customer, product, geography and date dimensions plus the sales fact. Report 768,873
rows passed and 261,822 rejected, broken down by reason (missing_customer_id,
cancelled_order, invalid_quantity, invalid_unit_price). Attach the Airflow task graph and
the dashboard's data-quality panel.

## 6. Dashboard findings

Screenshots and interpretation for the five Part 1 views (revenue/order trend, top products,
customer segmentation, country-wise sales, retention) plus the Part 2 views (churn risk
distribution, retention priority list, model and monitoring status).

## 7. Churn model (Part 2)

The core of the Part 2 marks. Cover, in this order:

- **Label**: churned = no order in the 90 days after a cutoff. State the inactivity window
  explicitly and why 90 days (the median inter-purchase gap is well under it).
- **Population**: only customers with an order in the previous 365 days; a customer silent
  longer has already churned.
- **Snapshots**: month-end cutoffs, 62,607 customer-month rows across 16 cutoffs from 5,245
  customers, overall churn rate 57.7%.
- **Features**: the 12 features in `docs/data_dictionary.md`, all computed from data on or
  before the cutoff. Mention that imputation and scaling sit inside the pipeline.
- **Split**: chronological by cutoff, 9 / 3 / 4 cutoffs for train / validation / test, with
  the test block strictly later in time. State that the leakage test asserts appending a
  year of future transactions changes no feature value.
- **Candidates and selection**: logistic regression, random forest and XGBoost all trained;
  random forest selected on validation ROC-AUC 0.783.
- **Threshold**: 0.34, tuned on validation to maximise F1 — and be explicit that because
  churn is the majority class this favours recall (0.952) over precision (0.677), and that
  `--min-precision` is the knob to shorten the outreach list.
- **Test metrics**: ROC-AUC 0.782, precision 0.677, recall 0.952, F1 0.791 on 17,331 rows.

## 8. MLOps operationalization

MLflow tracking (one parent run, one nested run per candidate; params, metrics, artifacts,
signatures) and the `retail-churn-classifier` registered model. Note why the tracking store
is sqlite/Postgres rather than a file store: the registry is unavailable on the file backend.
Then the resolution order in `ml/registry.py`, the FastAPI endpoints, the Docker images for
the API and dashboard, and the batch scorer that feeds both the dashboard and
`GET /customers/{id}/churn`. Include a `/predict` response and the `/metrics` output.

## 9. Monitoring and retraining design

The four checks (input data quality, feature drift by PSI, model performance on the last
closed snapshot, scoring latency) with their thresholds, and the five retraining criteria.
Explain the drift result honestly: the time-anchored features (tenure 0.53, recency 0.32)
drift over the three-month gap between the last training cutoff and the scoring cutoff,
while the behavioural aggregates (frequency 0.04, monetary 0.03) stay stable — so the alert
means "retrain on recent data", not "the data is broken". Describe the `churn_ml` DAG and
why it short-circuits rather than retraining unconditionally.

## 10. Limitations and ethics

Observational data only; account-level churn; two seasonal cycles; probabilities are ranking
scores rather than calibrated risks; majority-class churn makes accuracy a weak metric. On
ethics: no PII or demographic enrichment, scores for prioritising outreach and not for
per-customer pricing, and the threshold to be reviewed with the campaign owner. The fuller
versions of both lists are in `docs/mlops.md` §10-11 and in the generated model card.

## 11. Conclusion and evidence appendix

Screenshots of: both Airflow DAGs running green, populated mart tables, the dashboard
(Part 1 and Part 2 views), the MLflow experiment comparing the three candidates, the
registered model versions, the training metrics output, the monitoring report, and the API
`/health`, `/predict` and `/metrics` responses. State clearly which pieces were executed and
which were not — the README's "Not yet executed" note lists the Docker services and the
`churn_ml` DAG, which need Docker Desktop running to demonstrate.
