"""
Retail sales & retention dashboard (Streamlit).
Reads directly from the mart/dw layers in retail_dw — no business logic here,
all aggregation already happened in scripts/transform.py.

Run: streamlit run dashboard/app.py
(needs .env.local's PG* vars, or Postgres reachable on localhost:5432 —
run `docker compose up -d postgres` first and `python scripts/split_landing_files.py`
+ trigger the Airflow DAG at least once to populate the warehouse.)
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

st.set_page_config(page_title="Retail Sales & Retention", layout="wide")


@st.cache_resource
def get_engine():
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    db = os.environ.get("PGDATABASE", "retail_dw")
    user = os.environ.get("PGUSER", "airflow")
    pw = os.environ.get("PGPASSWORD", "airflow")
    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")


@st.cache_data(ttl=300)
def load_data():
    engine = get_engine()
    sales_monthly = pd.read_sql("SELECT * FROM mart.sales_monthly ORDER BY month_key", engine)
    rfm = pd.read_sql("SELECT * FROM mart.customer_rfm", engine)
    fact = pd.read_sql("""
        SELECT f.invoice_no, f.quantity, f.unit_price, f.revenue, f.country,
               f.date_key, p.description
        FROM dw.fact_sales f
        LEFT JOIN dw.dim_product p ON p.stock_code = f.stock_code
    """, engine, parse_dates=["date_key"])
    return sales_monthly, rfm, fact


engine = get_engine()
try:
    sales_monthly, rfm, fact = load_data()
except Exception as exc:
    st.error(f"Could not read from retail_dw: {exc}\n\nHas the Airflow DAG run at least once?")
    st.stop()

st.title("E-Commerce Sales & Retention Dashboard")
st.caption("Source: UCI Online Retail II, via the retail_etl Airflow pipeline -> PostgreSQL star schema")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total revenue", f"£{fact['revenue'].sum():,.0f}")
col2.metric("Orders", f"{fact['invoice_no'].nunique():,}")
col3.metric("Customers", f"{rfm.shape[0]:,}")
col4.metric("Repeat customer rate", f"{rfm['is_repeat_customer'].mean() * 100:.1f}%")

st.divider()

# 1. Revenue and order trend
st.subheader("1. Revenue and order trend")
c1, c2 = st.columns(2)
c1.plotly_chart(px.line(sales_monthly, x="month_key", y="revenue", markers=True,
                         title="Monthly revenue"), use_container_width=True)
c2.plotly_chart(px.line(sales_monthly, x="month_key", y="order_count", markers=True,
                         title="Monthly order count"), use_container_width=True)

# 2. Top products
st.subheader("2. Top products")
top_products = (fact.groupby("description")["revenue"].sum()
                 .sort_values(ascending=False).head(15).reset_index())
st.plotly_chart(px.bar(top_products, x="revenue", y="description", orientation="h",
                        title="Top 15 products by revenue").update_yaxes(categoryorder="total ascending"),
                 use_container_width=True)

# 3. Customer segmentation (RFM)
st.subheader("3. Customer segmentation (RFM)")
c3, c4 = st.columns(2)
seg_counts = rfm["segment"].value_counts().reset_index()
seg_counts.columns = ["segment", "customers"]
c3.plotly_chart(px.pie(seg_counts, names="segment", values="customers",
                        title="Customers by segment"), use_container_width=True)
c4.plotly_chart(px.scatter(rfm, x="recency_days", y="monetary", color="segment",
                            size="frequency", title="Recency vs. monetary, sized by frequency",
                            hover_data=["customer_id"]), use_container_width=True)

# 4. Country-wise sales
st.subheader("4. Country-wise sales")
country_rev = fact.groupby("country")["revenue"].sum().sort_values(ascending=False).reset_index()
st.plotly_chart(px.bar(country_rev.head(15), x="country", y="revenue",
                        title="Top 15 countries by revenue"), use_container_width=True)

# 5. Retention / repeat-purchase analysis
st.subheader("5. Retention and repeat-purchase analysis")
c5, c6 = st.columns(2)
freq_dist = rfm["frequency"].clip(upper=10).value_counts().sort_index().reset_index()
freq_dist.columns = ["orders_per_customer (10+ capped)", "customers"]
c5.plotly_chart(px.bar(freq_dist, x="orders_per_customer (10+ capped)", y="customers",
                        title="Purchase frequency distribution"), use_container_width=True)
c6.plotly_chart(px.histogram(rfm, x="recency_days", nbins=30,
                              title="Recency distribution (days since last order)"),
                 use_container_width=True)

st.subheader("6. Repeat-purchase prediction")
model_path = Path(__file__).resolve().parent.parent / "artifacts" / "repeat_purchase_model.joblib"
if model_path.exists():
    model_bundle = joblib.load(model_path)
    selected_customer = st.selectbox("Customer", rfm["customer_id"].sort_values().tolist())
    customer = rfm.loc[rfm["customer_id"] == selected_customer].iloc[0]
    model_features = pd.DataFrame([{
        "recency_days": customer["recency_days"],
        "frequency": customer["frequency"],
        "monetary": customer["monetary"],
        "average_order_value": customer["monetary"] / max(customer["frequency"], 1),
    }], columns=model_bundle["features"])
    probability = float(model_bundle["model"].predict_proba(model_features)[:, 1][0])
    st.metric("Predicted repeat-purchase probability", f"{probability:.1%}")
else:
    st.info("Train the repeat-purchase model with `python -m ml.train_model` to enable predictions.")

with st.expander("Data quality: rejected records"):
    rejected = pd.read_sql(
        "SELECT reject_reason, count(*) AS rows FROM staging.rejected_records GROUP BY 1 ORDER BY 2 DESC",
        engine,
    )
    st.dataframe(rejected, use_container_width=True)
