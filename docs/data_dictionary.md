# Data Dictionary

Source: UCI Online Retail II (https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Transaction-level online retail data, Dec 2009 - Dec 2011, UK-based non-store online retailer.
License: public domain / free for research use via the UCI Machine Learning Repository.

## raw.sales_raw
Unmodified landed rows, kept as text so bad data can never be silently coerced away.

| Column | Type | Source column | Notes |
|---|---|---|---|
| invoice_no | text | Invoice | 'C'-prefixed = cancelled order |
| stock_code | text | StockCode | product/item code |
| description | text | Description | product name, free text |
| quantity | text | Quantity | kept as text in raw layer |
| invoice_date | text | InvoiceDate | kept as text in raw layer |
| unit_price | text | Price | kept as text in raw layer, GBP |
| customer_id | text | Customer ID | kept as text in raw layer |
| country | text | Country | customer's country |
| source_file | text | (derived) | landing filename this row came from |
| loaded_at | timestamp | (derived) | load timestamp |

## staging.sales_valid
Typed, validated rows only. Natural key: (invoice_no, stock_code, invoice_date).

| Column | Type | Validation rule |
|---|---|---|
| quantity | integer | must be > 0 |
| unit_price | numeric(12,2) | must be > 0 |
| customer_id | integer | must be present and numeric |
| invoice_date | timestamp | must parse |
| revenue | numeric(14,2) | derived: quantity * unit_price |

## staging.rejected_records
Same raw columns plus `reject_reason` (one of: cancelled_order, invalid_quantity,
invalid_unit_price, missing_customer_id, invalid_invoice_date, missing_country)
and `source_file`.

## dw star schema
- **dw.dim_customer**: customer_id (PK), country, first_purchase_date, last_purchase_date,
  total_orders, customer_segment (One-time / Repeat / Loyal, by distinct order count).
- **dw.dim_product**: stock_code (PK), description (most frequent description seen for that code).
- **dw.dim_date**: date_key (PK), year, month, month_name, quarter, day_of_week.
- **dw.dim_geography**: country (PK), region (currently = country; placeholder for a real
  region mapping if one becomes a requirement).
- **dw.fact_sales**: one row per (invoice_no, stock_code, date_key). quantity, unit_price,
  revenue, FKs to all four dimensions.

## mart (dashboard-facing)
- **mart.customer_rfm**: customer_id, recency_days (since last order, relative to the max
  invoice_date in the data), frequency (distinct orders), monetary (total revenue),
  is_repeat_customer, segment.
- **mart.sales_monthly**: month_key (YYYY-MM), revenue, order_count, avg_order_value.

## meta.ingestion_log
Ingestion audit trail: source_file, extraction_date, row_count, status (success/failed),
error_message. One row per file per status; this is what makes extract.py's incremental
loading skip already-processed files on re-run.

## Part 2: churn model feature table

Built by `ml/features.py` from `staging.sales_valid`; written to
`artifacts/churn_training_features.csv` (training snapshots, also the drift
baseline) and `artifacts/churn_scoring_features.csv` (the current unlabelled
population). One row per customer per month-end cutoff.

| Column | Type | Definition |
|---|---|---|
| customer_id | integer | pseudonymous customer key, from staging |
| snapshot_date | timestamp | the cutoff this row's features were computed at |
| recency_days | integer | days from the customer's last order to the cutoff |
| frequency | integer | distinct invoices on or before the cutoff |
| monetary | numeric | total revenue on or before the cutoff |
| average_order_value | numeric | monetary / frequency |
| tenure_days | integer | days from the customer's first order to the cutoff |
| distinct_products | integer | distinct stock_codes ever purchased |
| avg_items_per_order | numeric | total quantity / frequency |
| avg_interpurchase_days | numeric | mean gap between orders; **null** when frequency = 1, median-imputed inside the model pipeline |
| orders_last_90d | integer | distinct invoices in the 90 days before the cutoff |
| revenue_last_90d | numeric | revenue in the 90 days before the cutoff |
| revenue_trend_ratio | numeric | (revenue_last_90d + 1) / (revenue_prior_90d + 1); > 1 accelerating, < 1 decelerating |
| distinct_active_months | integer | distinct calendar months with at least one order |
| churned | integer | **label**: 1 if the customer places no order in the 90 days after the cutoff, else 0. Absent from the scoring file, whose window has not closed. |

Population rule: a row exists only for customers with at least one order in
the 365 days before the cutoff. A customer silent longer than that has already
churned rather than being at risk of it.

## Part 2: scoring output

`artifacts/churn_scores.csv` — the batch scorer's output, read by the
dashboard and by `GET /customers/{id}/churn`.

| Column | Type | Definition |
|---|---|---|
| (all feature columns above) | | the features the score was computed from |
| churn_probability | numeric | model output, 0-1 |
| churn_prediction | integer | 1 when churn_probability >= decision_threshold |
| risk_band | text | low (<0.25), medium (<0.5), high (<0.75), critical (>=0.75) |
| model_version | text | e.g. `retail-churn-classifier/v1` |
| decision_threshold | numeric | the threshold that produced churn_prediction |

