"""
Validate task: reads every row in raw.sales_raw that hasn't been staged yet,
applies data-quality rules, and routes each row to either
staging.sales_valid (typed, clean) or staging.rejected_records (with a reason).

Rules (per the assignment's ETL requirements):
  - drop cancelled orders (invoice_no starting with 'C')
  - quantity must be a positive integer
  - unit_price must be a positive number
  - customer_id must be present and numeric
  - invoice_date must parse
"""
import pandas as pd
from sqlalchemy import text
from db import get_engine


def _to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def validate_and_stage():
    engine = get_engine()

    # pass the engine (not a checked-out connection) to pd.read_sql: it
    # manages the connection itself and stays compatible whether the
    # installed SQLAlchemy is 1.4.x (Airflow's pin) or 2.x (local venv)
    already_staged_files = set(pd.read_sql(
        "SELECT DISTINCT source_file FROM staging.sales_valid", engine
    )["source_file"])

    df = pd.read_sql("SELECT * FROM raw.sales_raw", engine)
    if already_staged_files:
        df = df[~df["source_file"].isin(already_staged_files)]

    if df.empty:
        print("validate: nothing new to validate")
        return {"valid": 0, "rejected": 0}

    df["quantity_num"] = _to_numeric(df["quantity"])
    df["unit_price_num"] = _to_numeric(df["unit_price"])
    df["customer_id_num"] = _to_numeric(df["customer_id"])
    df["invoice_date_parsed"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    reasons = pd.Series([None] * len(df), index=df.index, dtype=object)
    reasons[df["invoice_no"].astype(str).str.startswith("C", na=False)] = "cancelled_order"
    reasons[reasons.isna() & (df["quantity_num"].isna() | (df["quantity_num"] <= 0))] = "invalid_quantity"
    reasons[reasons.isna() & (df["unit_price_num"].isna() | (df["unit_price_num"] <= 0))] = "invalid_unit_price"
    reasons[reasons.isna() & df["customer_id_num"].isna()] = "missing_customer_id"
    reasons[reasons.isna() & df["invoice_date_parsed"].isna()] = "invalid_invoice_date"
    reasons[reasons.isna() & df["country"].isna()] = "missing_country"

    rejected_mask = reasons.notna()
    rejected = df[rejected_mask].copy()
    valid = df[~rejected_mask].copy()

    if not rejected.empty:
        rejected_out = rejected[[
            "invoice_no", "stock_code", "description", "quantity",
            "invoice_date", "unit_price", "customer_id", "country", "source_file",
        ]].copy()
        rejected_out["reject_reason"] = reasons[rejected_mask].values
        rejected_out.to_sql("rejected_records", engine, schema="staging", if_exists="append", index=False)

    if not valid.empty:
        valid_out = pd.DataFrame({
            "invoice_no": valid["invoice_no"],
            "stock_code": valid["stock_code"],
            "description": valid["description"],
            "quantity": valid["quantity_num"].astype(int),
            "invoice_date": valid["invoice_date_parsed"],
            "unit_price": valid["unit_price_num"].astype(float),
            "customer_id": valid["customer_id_num"].astype(int),
            "country": valid["country"],
            "revenue": (valid["quantity_num"] * valid["unit_price_num"]).astype(float),
            "source_file": valid["source_file"],
        })
        # de-dup within this batch on the natural key before upsert
        valid_out = valid_out.drop_duplicates(subset=["invoice_no", "stock_code", "invoice_date"])

        # bulk load via a temp table, then upsert in one statement (fast, still simple)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS staging._tmp_sales_valid"))
        valid_out.to_sql("_tmp_sales_valid", engine, schema="staging", if_exists="replace", index=False)
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO staging.sales_valid
                    (invoice_no, stock_code, description, quantity, invoice_date,
                     unit_price, customer_id, country, revenue, source_file)
                SELECT invoice_no, stock_code, description, quantity, invoice_date,
                       unit_price, customer_id, country, revenue, source_file
                FROM staging._tmp_sales_valid
                ON CONFLICT (invoice_no, stock_code, invoice_date) DO NOTHING
            """))
            conn.execute(text("DROP TABLE staging._tmp_sales_valid"))

    print(f"validate: valid={len(valid)} rejected={len(rejected)}")
    return {"valid": len(valid), "rejected": len(rejected)}


if __name__ == "__main__":
    validate_and_stage()
