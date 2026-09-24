-- Retail data warehouse schema: raw -> staging -> dw (star schema) -> mart
-- Applied once at container init, and is idempotent (safe to re-run).

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS dw;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS meta;

-- ---------- meta: ingestion control / audit ----------
CREATE TABLE IF NOT EXISTS meta.ingestion_log (
    id              SERIAL PRIMARY KEY,
    source_file     TEXT NOT NULL,
    extraction_date TIMESTAMP NOT NULL DEFAULT now(),
    row_count       INTEGER,
    status          TEXT NOT NULL,           -- success | failed
    error_message   TEXT,
    UNIQUE (source_file, status)
);

-- ---------- raw: unmodified landed rows + provenance ----------
CREATE TABLE IF NOT EXISTS raw.sales_raw (
    id            BIGSERIAL PRIMARY KEY,
    invoice_no    TEXT,
    stock_code    TEXT,
    description   TEXT,
    quantity      TEXT,      -- kept as text: raw layer must not coerce/lose bad data
    invoice_date  TEXT,
    unit_price    TEXT,
    customer_id   TEXT,
    country       TEXT,
    source_file   TEXT NOT NULL,
    loaded_at     TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sales_raw_source_file ON raw.sales_raw (source_file);

-- ---------- staging: validated + typed rows ----------
CREATE TABLE IF NOT EXISTS staging.sales_valid (
    invoice_no    TEXT NOT NULL,
    stock_code    TEXT NOT NULL,
    description   TEXT,
    quantity      INTEGER NOT NULL,
    invoice_date  TIMESTAMP NOT NULL,
    unit_price    NUMERIC(12,2) NOT NULL,
    customer_id   INTEGER NOT NULL,
    country       TEXT NOT NULL,
    revenue       NUMERIC(14,2) NOT NULL,
    source_file   TEXT NOT NULL,
    loaded_at     TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (invoice_no, stock_code, invoice_date)
);

CREATE TABLE IF NOT EXISTS staging.rejected_records (
    id            BIGSERIAL PRIMARY KEY,
    invoice_no    TEXT,
    stock_code    TEXT,
    description   TEXT,
    quantity      TEXT,
    invoice_date  TEXT,
    unit_price    TEXT,
    customer_id   TEXT,
    country       TEXT,
    reject_reason TEXT NOT NULL,
    source_file   TEXT NOT NULL,
    loaded_at     TIMESTAMP NOT NULL DEFAULT now()
);

-- ---------- dw: star schema ----------
CREATE TABLE IF NOT EXISTS dw.dim_customer (
    customer_id        INTEGER PRIMARY KEY,
    country             TEXT,
    first_purchase_date DATE,
    last_purchase_date  DATE,
    total_orders         INTEGER,
    customer_segment     TEXT
);

CREATE TABLE IF NOT EXISTS dw.dim_product (
    stock_code  TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key   DATE PRIMARY KEY,
    year       INTEGER,
    month      INTEGER,
    month_name TEXT,
    quarter    INTEGER,
    day_of_week TEXT
);

CREATE TABLE IF NOT EXISTS dw.dim_geography (
    country TEXT PRIMARY KEY,
    region  TEXT
);

CREATE TABLE IF NOT EXISTS dw.fact_sales (
    id          BIGSERIAL PRIMARY KEY,
    invoice_no  TEXT NOT NULL,
    customer_id INTEGER REFERENCES dw.dim_customer (customer_id),
    stock_code  TEXT REFERENCES dw.dim_product (stock_code),
    date_key    DATE REFERENCES dw.dim_date (date_key),
    country     TEXT REFERENCES dw.dim_geography (country),
    quantity    INTEGER NOT NULL,
    unit_price  NUMERIC(12,2) NOT NULL,
    revenue     NUMERIC(14,2) NOT NULL,
    UNIQUE (invoice_no, stock_code, date_key)
);

-- ---------- mart: analytical tables consumed by the dashboard ----------
CREATE TABLE IF NOT EXISTS mart.customer_rfm (
    customer_id       INTEGER PRIMARY KEY,
    recency_days      INTEGER,
    frequency         INTEGER,
    monetary          NUMERIC(14,2),
    is_repeat_customer BOOLEAN,
    segment           TEXT
);

CREATE TABLE IF NOT EXISTS mart.sales_monthly (
    month_key      TEXT PRIMARY KEY,
    revenue        NUMERIC(14,2),
    order_count    INTEGER,
    avg_order_value NUMERIC(12,2)
);
