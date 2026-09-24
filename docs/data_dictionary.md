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
