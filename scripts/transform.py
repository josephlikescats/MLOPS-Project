"""
Transform task: builds the star schema (dw.*) and analytical marts (mart.*)
from staging.sales_valid.

Incremental loading and duplicate prevention already happen upstream (extract
skips already-loaded files, validate upserts on the natural key into
staging.sales_valid). This layer just recomputes dw/mart from staging on
every run with a full replace/upsert — staging is small enough (~1M rows)
that a full rebuild is simpler and safer than incremental dimension merges.
# ponytail: full rebuild of dw/mart each run, switch to incremental merge if staging grows past a few million rows
"""
import pandas as pd
from sqlalchemy import text
from db import get_engine


def clear_star_schema(engine):
    """Truncate fact + dims together (CASCADE handles FK order) so we can
    reload with to_sql(append) and keep the PK/FK constraints defined in
    schema.sql, instead of letting to_sql(replace) drop/recreate tables
    without them."""
    with engine.begin() as conn:
        conn.execute(text(
            "TRUNCATE TABLE dw.fact_sales, dw.dim_date, dw.dim_geography, "
            "dw.dim_product, dw.dim_customer RESTART IDENTITY CASCADE"
        ))


def build_dimensions(engine, staged: pd.DataFrame):
    # dim_date
    dates = staged["invoice_date"].dt.date.drop_duplicates().to_frame("date_key")
    dates["date_key"] = pd.to_datetime(dates["date_key"])
    dates["year"] = dates["date_key"].dt.year
    dates["month"] = dates["date_key"].dt.month
    dates["month_name"] = dates["date_key"].dt.strftime("%B")
    dates["quarter"] = dates["date_key"].dt.quarter
    dates["day_of_week"] = dates["date_key"].dt.strftime("%A")
    dates.to_sql("dim_date", engine, schema="dw", if_exists="append", index=False)

    # dim_geography (region left equal to country: no reliable public region
    # mapping in-scope; add one if country-grouping becomes a real requirement)
    geo = staged[["country"]].drop_duplicates()
    geo["region"] = geo["country"]
    geo.to_sql("dim_geography", engine, schema="dw", if_exists="append", index=False)

    # dim_product: most common description per stock_code
    prod = (
        staged.groupby("stock_code")["description"]
        .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else None)
        .reset_index()
    )
    prod.to_sql("dim_product", engine, schema="dw", if_exists="append", index=False)

    # dim_customer
    cust = staged.groupby("customer_id").agg(
        country=("country", "last"),
        first_purchase_date=("invoice_date", "min"),
        last_purchase_date=("invoice_date", "max"),
        total_orders=("invoice_no", "nunique"),
    ).reset_index()
    cust["first_purchase_date"] = cust["first_purchase_date"].dt.date
    cust["last_purchase_date"] = cust["last_purchase_date"].dt.date
    cust["customer_segment"] = pd.cut(
        cust["total_orders"], bins=[0, 1, 4, float("inf")],
        labels=["One-time", "Repeat", "Loyal"],
    ).astype(str)
    cust.to_sql("dim_customer", engine, schema="dw", if_exists="append", index=False)


def build_fact(engine, staged: pd.DataFrame):
    fact = staged.copy()
    fact["date_key"] = fact["invoice_date"].dt.date
    fact = fact[["invoice_no", "customer_id", "stock_code", "date_key",
                 "country", "quantity", "unit_price", "revenue"]]
    fact = fact.drop_duplicates(subset=["invoice_no", "stock_code", "date_key"])
    fact.to_sql("fact_sales", engine, schema="dw", if_exists="append", index=False)


def build_marts(engine, staged: pd.DataFrame):
    max_date = staged["invoice_date"].max()

    rfm = staged.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()
    rfm["recency_days"] = (max_date - rfm["last_purchase"]).dt.days
    rfm["is_repeat_customer"] = rfm["frequency"] > 1
    rfm["segment"] = pd.cut(
        rfm["frequency"], bins=[0, 1, 4, float("inf")],
        labels=["One-time", "Repeat", "Loyal"],
    ).astype(str)
    rfm = rfm[["customer_id", "recency_days", "frequency", "monetary",
               "is_repeat_customer", "segment"]]
    rfm.to_sql("customer_rfm", engine, schema="mart", if_exists="replace", index=False)

    monthly = staged.copy()
    monthly["month_key"] = monthly["invoice_date"].dt.strftime("%Y-%m")
    monthly_agg = monthly.groupby("month_key").agg(
        revenue=("revenue", "sum"),
        order_count=("invoice_no", "nunique"),
    ).reset_index()
    monthly_agg["avg_order_value"] = monthly_agg["revenue"] / monthly_agg["order_count"]
    monthly_agg.to_sql("sales_monthly", engine, schema="mart", if_exists="replace", index=False)


def run_transform():
    engine = get_engine()
    staged = pd.read_sql("SELECT * FROM staging.sales_valid", engine, parse_dates=["invoice_date"])
    if staged.empty:
        print("transform: staging.sales_valid is empty, nothing to build")
        return

    clear_star_schema(engine)
    build_dimensions(engine, staged)
    build_fact(engine, staged)
    build_marts(engine, staged)
    print(f"transform: rebuilt dw/mart from {len(staged)} staged rows")


if __name__ == "__main__":
    run_transform()
