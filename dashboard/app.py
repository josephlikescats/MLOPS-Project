"""
Retail sales & retention dashboard (Streamlit).
Reads directly from the mart/dw layers in retail_dw — no business logic here,
all aggregation already happened in scripts/transform.py.

Run: streamlit run dashboard/app.py
(needs .env.local's PG* vars, or Postgres reachable on localhost:5432 —
run `docker compose up -d postgres` first and `python scripts/split_landing_files.py`
+ trigger the Airflow DAG at least once to populate the warehouse.)
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

st.set_page_config(
    page_title="Retail Sales & Retention",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# --------------------------------------------------------------- design tokens
# Categorical slots are assigned in fixed order and never cycled. Only the first
# three are used for all-pairs forms (scatter/pie), which is the validated cap.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PRIMARY = SERIES[0]
# Risk bands are a *status*, not a series, so they take the reserved status ramp.
RISK_COLORS = {
    "low": "#0ca30c",
    "medium": "#fab219",
    "high": "#ec835a",
    "critical": "#d03b3b",
}
RISK_ORDER = ["low", "medium", "high", "critical"]
# Single-hue sequential ramp, light -> dark, for churn probability magnitude.
RISK_SCALE = ["#fde8e8", "#f8c0c0", "#ef9090", "#e36565", "#d03b3b", "#a82e2e"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

pio.templates["retail"] = pio.templates["plotly_white"]
pio.templates["retail"].layout.update(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=INK),
    colorway=SERIES,
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    margin=dict(l=8, r=8, t=48, b=8),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", pad=dict(b=12)),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID,
               tickfont=dict(color=INK_MUTED, size=12), title_font=dict(color=INK_MUTED, size=12)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID,
               tickfont=dict(color=INK_MUTED, size=12), title_font=dict(color=INK_MUTED, size=12)),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(size=12, color=INK_MUTED), title_text=""),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(size=12, color=INK)),
    transition=dict(duration=350, easing="cubic-in-out"),
)
pio.templates.default = "retail"


def show(fig, height=340, **kwargs):
    """Apply the shared chart chrome, then render."""
    fig.update_layout(height=height, **kwargs)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    # Bars and markers grow on hover rather than only changing the tooltip.
    fig.update_traces(marker=dict(opacity=0.92), selector=dict(type="bar"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


st.markdown(
    f"""
    <style>
      @keyframes fadeUp {{
          from {{ opacity: 0; transform: translateY(14px); }}
          to   {{ opacity: 1; transform: none; }}
      }}
      @keyframes pulse {{
          0%, 100% {{ opacity: 1; transform: scale(1); }}
          50%      {{ opacity: 0.35; transform: scale(0.75); }}
      }}
      @keyframes sweep {{
          0%   {{ background-position: 0% 50%; }}
          100% {{ background-position: 200% 50%; }}
      }}

      .block-container {{ padding-top: 1.6rem; max-width: 1500px; }}
      #MainMenu, footer {{ visibility: hidden; }}

      /* ---- hero ---------------------------------------------------- */
      .hero {{
          border: 1px solid rgba(11,11,11,0.08);
          border-radius: 14px;
          padding: 22px 26px 20px;
          margin-bottom: 20px;
          position: relative;
          overflow: hidden;
          background:
            radial-gradient(1200px 180px at 0% 0%, rgba(42,120,214,0.10), transparent 60%),
            radial-gradient(900px 160px at 100% 100%, rgba(27,175,122,0.09), transparent 60%),
            {SURFACE};
          animation: fadeUp 0.55s cubic-bezier(.2,.7,.3,1) both;
      }}
      .hero::before {{
          content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
          background: linear-gradient(90deg, {SERIES[0]}, {SERIES[2]}, {SERIES[1]}, {SERIES[0]});
          background-size: 200% 100%;
          animation: sweep 7s linear infinite;
      }}
      .hero-title {{
          font-size: 1.95rem; font-weight: 680; letter-spacing: -0.02em;
          color: {INK}; line-height: 1.15;
      }}
      .hero-sub {{
          margin-top: 7px; font-size: 0.88rem; color: {INK_MUTED};
      }}
      .live-dot {{
          display: inline-block; width: 7px; height: 7px; border-radius: 50%;
          background: {RISK_COLORS["low"]}; margin-right: 7px; vertical-align: middle;
          animation: pulse 2.1s ease-in-out infinite;
      }}

      /* ---- metric cards -------------------------------------------- */
      div[data-testid="stMetric"] {{
          background: {SURFACE};
          border: 1px solid rgba(11,11,11,0.10);
          border-radius: 12px;
          padding: 15px 17px;
          position: relative;
          overflow: hidden;
          animation: fadeUp 0.5s cubic-bezier(.2,.7,.3,1) both;
          transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
      }}
      div[data-testid="stMetric"]::after {{
          content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
          background: {PRIMARY}; opacity: 0;
          transition: opacity .18s ease;
      }}
      div[data-testid="stMetric"]:hover {{
          transform: translateY(-3px);
          box-shadow: 0 8px 22px rgba(11,11,11,0.09);
          border-color: rgba(42,120,214,0.45);
      }}
      div[data-testid="stMetric"]:hover::after {{ opacity: 1; }}
      div[data-testid="stMetricLabel"] p {{
          color: {INK_MUTED}; font-size: 0.78rem; font-weight: 500;
          text-transform: uppercase; letter-spacing: 0.04em;
      }}
      div[data-testid="stMetricValue"] {{
          font-size: 1.62rem; color: {INK}; letter-spacing: -0.01em;
      }}

      /* stagger the KPI row so the cards arrive left to right */
      div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] {{ animation-delay: .04s; }}
      div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] {{ animation-delay: .10s; }}
      div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] {{ animation-delay: .16s; }}
      div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] {{ animation-delay: .22s; }}

      /* ---- charts --------------------------------------------------- */
      div[data-testid="stPlotlyChart"] {{
          border: 1px solid rgba(11,11,11,0.07);
          border-radius: 12px;
          padding: 6px 8px 2px;
          background: {SURFACE};
          animation: fadeUp 0.55s cubic-bezier(.2,.7,.3,1) both;
          transition: box-shadow .2s ease, border-color .2s ease;
      }}
      div[data-testid="stPlotlyChart"]:hover {{
          box-shadow: 0 6px 20px rgba(11,11,11,0.07);
          border-color: rgba(11,11,11,0.14);
      }}

      /* ---- tabs ----------------------------------------------------- */
      button[data-baseweb="tab"] {{
          font-size: 0.95rem; font-weight: 500;
          transition: color .15s ease, background .15s ease;
          border-radius: 8px 8px 0 0;
      }}
      button[data-baseweb="tab"]:hover {{ background: rgba(42,120,214,0.07); }}

      /* ---- headings & tables ---------------------------------------- */
      h1 {{ font-size: 1.9rem !important; font-weight: 650 !important; }}
      h2 {{ font-size: 1.3rem !important; padding-top: 0.4rem !important; }}
      h3 {{ font-size: 1.05rem !important; color: {INK}; }}
      div[data-testid="stDataFrame"] {{
          border-radius: 10px; overflow: hidden;
          animation: fadeUp 0.5s cubic-bezier(.2,.7,.3,1) both;
      }}

      /* Respect the OS setting - every animation above is decorative. */
      @media (prefers-reduced-motion: reduce) {{
          *, *::before, *::after {{
              animation: none !important;
              transition: none !important;
          }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


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

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


@st.cache_data(ttl=300)
def load_churn_outputs():
    """Batch scores, model card and monitoring report, if the jobs have run."""

    def read_json(name):
        path = ARTIFACTS / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    scores_path = ARTIFACTS / "churn_scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else None
    return scores, read_json("churn_model_card.json"), read_json("monitoring_report.json")


churn_scores, model_card, monitoring = load_churn_outputs()

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">E-Commerce Sales &amp; Retention</div>
      <div class="hero-sub">
        <span class="live-dot"></span>Live from <b>retail_dw</b> &nbsp;·&nbsp;
        UCI Online Retail II → <code>retail_etl</code> → PostgreSQL star schema → marts
        &nbsp;·&nbsp; churn scores from <code>churn_ml</code>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total revenue", f"£{fact['revenue'].sum():,.0f}")
col2.metric("Orders", f"{fact['invoice_no'].nunique():,}")
col3.metric("Customers", f"{rfm.shape[0]:,}")
col4.metric("Repeat customer rate", f"{rfm['is_repeat_customer'].mean() * 100:.1f}%")

tab_sales, tab_customers, tab_churn, tab_model = st.tabs(
    ["Sales", "Customers & retention", "Churn risk", "Model & data health"]
)

# ───────────────────────────────────────────────────────────────── Sales
with tab_sales:
    st.subheader("1. Revenue and order trend")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sales_monthly, x="month_key", y="revenue", markers=True,
                      title="Monthly revenue", labels={"month_key": "", "revenue": "Revenue (£)"})
        fig.update_traces(line=dict(width=2.5, color=PRIMARY, shape="spline"),
                          marker=dict(size=8),
                          hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>")
        fig.update_layout(hovermode="x unified")
        show(fig)
    with c2:
        fig = px.line(sales_monthly, x="month_key", y="order_count", markers=True,
                      title="Monthly order count", labels={"month_key": "", "order_count": "Orders"})
        fig.update_traces(line=dict(width=2.5, color=SERIES[1], shape="spline"),
                          marker=dict(size=8),
                          hovertemplate="%{x}<br>%{y:,} orders<extra></extra>")
        fig.update_layout(hovermode="x unified")
        show(fig)

    st.subheader("2. Top products")
    top_products = (fact.groupby("description")["revenue"].sum()
                    .sort_values(ascending=False).head(15).reset_index())
    fig = px.bar(top_products, x="revenue", y="description", orientation="h",
                 title="Top 15 products by revenue",
                 labels={"revenue": "Revenue (£)", "description": ""})
    fig.update_traces(marker_color=PRIMARY, hovertemplate="%{y}<br>£%{x:,.0f}<extra></extra>")
    fig.update_yaxes(categoryorder="total ascending")
    show(fig, height=460)

    st.subheader("3. Country-wise sales")
    country_rev = fact.groupby("country")["revenue"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(country_rev.head(15), x="country", y="revenue",
                 title="Top 15 countries by revenue",
                 labels={"country": "", "revenue": "Revenue (£)"})
    fig.update_traces(marker_color=PRIMARY, hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>")
    show(fig, height=400)

# ──────────────────────────────────────────────── Customers & retention
with tab_customers:
    st.subheader("4. Customer segmentation (RFM)")
    c3, c4 = st.columns(2)
    seg_counts = rfm["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "customers"]
    with c3:
        fig = px.pie(seg_counts, names="segment", values="customers", hole=0.55,
                     title="Customers by segment", color_discrete_sequence=SERIES[:3])
        fig.update_traces(textinfo="label+percent", textposition="outside",
                          marker=dict(line=dict(color=SURFACE, width=2)),
                          hovertemplate="%{label}<br>%{value:,} customers<extra></extra>")
        show(fig, showlegend=False)
    with c4:
        fig = px.scatter(rfm, x="recency_days", y="monetary", color="segment", size="frequency",
                         title="Recency vs. monetary, sized by frequency",
                         color_discrete_sequence=SERIES[:3], hover_data=["customer_id"],
                         labels={"recency_days": "Days since last order", "monetary": "Revenue (£)",
                                 "segment": ""})
        fig.update_traces(marker=dict(line=dict(width=1, color=SURFACE), opacity=0.75))
        fig.update_yaxes(type="log", title="Revenue (£, log)")
        show(fig)

    st.subheader("5. Retention and repeat-purchase analysis")
    c5, c6 = st.columns(2)
    freq_dist = rfm["frequency"].clip(upper=10).value_counts().sort_index().reset_index()
    freq_dist.columns = ["orders", "customers"]
    with c5:
        fig = px.bar(freq_dist, x="orders", y="customers",
                     title="Purchase frequency distribution (10+ capped)",
                     labels={"orders": "Orders per customer", "customers": "Customers"})
        fig.update_traces(marker_color=PRIMARY,
                          hovertemplate="%{x} orders<br>%{y:,} customers<extra></extra>")
        show(fig)
    with c6:
        fig = px.histogram(rfm, x="recency_days", nbins=30,
                           title="Recency distribution (days since last order)",
                           labels={"recency_days": "Days since last order"})
        fig.update_traces(marker_color=SERIES[1],
                          hovertemplate="%{x} days<br>%{y:,} customers<extra></extra>")
        fig.update_yaxes(title="Customers")
        show(fig)

    st.subheader("6. Repeat-purchase prediction")
    model_path = ARTIFACTS / "repeat_purchase_model.joblib"
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
        p1, p2, p3 = st.columns(3)
        p1.metric("Repeat-purchase probability", f"{probability:.1%}")
        p2.metric("Orders to date", f"{int(customer['frequency']):,}")
        p3.metric("Revenue to date", f"£{customer['monetary']:,.0f}")
    else:
        st.info("Train the repeat-purchase model with `python -m ml.train_model` to enable predictions.")

# ──────────────────────────────────────────────────────────── Churn risk
with tab_churn:
    if churn_scores is None:
        st.info(
            "No churn scores yet. Run `python -m ml.train` then `python -m ml.score_churn` "
            "to populate this section."
        )
    else:
        st.caption(
            f"Scored population: {len(churn_scores):,} active customers as at "
            f"{pd.to_datetime(churn_scores['snapshot_date']).max():%Y-%m-%d} — "
            f"model {churn_scores['model_version'].iloc[0]}, threshold "
            f"{churn_scores['decision_threshold'].iloc[0]:.2f}. "
            + (model_card or {}).get("label_definition", "")
        )

        flagged = int(churn_scores["churn_prediction"].sum())
        at_risk = churn_scores[churn_scores["risk_band"].isin(["high", "critical"])]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers scored", f"{len(churn_scores):,}")
        k2.metric("Flagged at risk", f"{flagged:,}",
                  f"{flagged / len(churn_scores) * 100:.1f}% of base")
        k3.metric("High/critical risk", f"{len(at_risk):,}")
        k4.metric("Revenue at risk", f"£{at_risk['monetary'].sum():,.0f}",
                  help="Historic revenue of customers in the high and critical bands")

        st.subheader("7. Churn risk distribution")
        c7, c8 = st.columns(2)
        bands = (churn_scores["risk_band"].value_counts()
                 .reindex(RISK_ORDER).fillna(0).reset_index())
        bands.columns = ["risk_band", "customers"]
        with c7:
            fig = px.bar(bands, x="risk_band", y="customers", color="risk_band",
                         category_orders={"risk_band": RISK_ORDER},
                         color_discrete_map=RISK_COLORS, title="Customers by churn-risk band",
                         labels={"risk_band": "", "customers": "Customers"})
            fig.update_traces(hovertemplate="%{x}<br>%{y:,} customers<extra></extra>")
            show(fig, showlegend=False)
        with c8:
            fig = px.histogram(churn_scores, x="churn_probability", nbins=40,
                               title="Predicted churn probability",
                               labels={"churn_probability": "Churn probability"})
            fig.update_traces(marker_color=RISK_COLORS["critical"],
                              hovertemplate="p=%{x:.2f}<br>%{y:,} customers<extra></extra>")
            fig.add_vline(x=float(churn_scores["decision_threshold"].iloc[0]),
                          line_dash="dash", line_color=INK_MUTED, line_width=1,
                          annotation_text="decision threshold",
                          annotation_font=dict(size=11, color=INK_MUTED))
            fig.update_yaxes(title="Customers")
            show(fig)

        st.subheader("8. Retention priority list")
        min_probability = st.slider("Minimum churn probability", 0.0, 1.0, 0.75, 0.05)
        priority = churn_scores[churn_scores["churn_probability"] >= min_probability].sort_values(
            "monetary", ascending=False
        )
        st.caption(
            f"{len(priority):,} customers at or above {min_probability:.0%} churn probability, "
            "ordered by historic revenue — the value that outreach would be protecting."
        )
        st.dataframe(
            priority[[
                "customer_id", "churn_probability", "risk_band", "recency_days",
                "frequency", "monetary", "orders_last_90d", "revenue_trend_ratio",
            ]].head(50),
            width="stretch",
            hide_index=True,
            column_config={
                "customer_id": st.column_config.NumberColumn("Customer", format="%d"),
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"),
                "risk_band": st.column_config.TextColumn("Band"),
                "recency_days": st.column_config.NumberColumn("Recency (d)", format="%d"),
                "frequency": st.column_config.NumberColumn("Orders", format="%d"),
                "monetary": st.column_config.NumberColumn("Revenue", format="£%.0f"),
                "orders_last_90d": st.column_config.NumberColumn("Orders 90d", format="%d"),
                "revenue_trend_ratio": st.column_config.NumberColumn("Trend", format="%.2f"),
            },
        )

        fig = px.scatter(churn_scores, x="recency_days", y="monetary", color="churn_probability",
                         size="frequency", color_continuous_scale=RISK_SCALE,
                         hover_data=["customer_id", "risk_band"],
                         title="Churn probability across the recency / value map",
                         labels={"recency_days": "Days since last order", "monetary": "Revenue (£)",
                                 "churn_probability": "Churn p"})
        fig.update_traces(marker=dict(line=dict(width=1, color=SURFACE), opacity=0.8))
        fig.update_yaxes(type="log", title="Revenue (£, log)")
        show(fig, height=420)

        with st.expander("Look up a single customer"):
            lookup_id = st.selectbox(
                "Customer", churn_scores["customer_id"].sort_values().tolist(), key="churn_lookup"
            )
            record = churn_scores.loc[churn_scores["customer_id"] == lookup_id].iloc[0]
            l1, l2, l3 = st.columns(3)
            l1.metric("Churn probability", f"{record['churn_probability']:.1%}")
            l2.metric("Risk band", str(record["risk_band"]).title())
            l3.metric("Days since last order", f"{int(record['recency_days'])}")
            st.write({
                feature: record[feature]
                for feature in (model_card or {}).get("features", [])
                if feature in record
            })

# ─────────────────────────────────────────────────── Model & data health
with tab_model:
    st.subheader("9. Model and monitoring status")
    if model_card is None:
        st.info("Run `python -m ml.train` to produce the model card.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        test_metrics = model_card["metrics"]["test"]
        m1.metric("Selected model", model_card["model_family"].replace("_", " ").title())
        m2.metric("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
        m3.metric("Test precision", f"{test_metrics['precision']:.3f}")
        m4.metric("Test recall", f"{test_metrics['recall']:.3f}")
        st.caption(
            f"Trained {model_card['trained_at'][:19]}Z using a {model_card['split_strategy']}. "
            f"Decision threshold {model_card['decision_threshold']:.2f}, tuned on the validation block."
        )

        importances = model_card.get("feature_importances")
        if importances:
            importance_frame = pd.DataFrame(
                {"feature": list(importances), "importance": list(importances.values())}
            ).sort_values("importance")
            fig = px.bar(importance_frame, x="importance", y="feature", orientation="h",
                         title="Feature importance of the selected model",
                         labels={"importance": "Importance", "feature": ""})
            fig.update_traces(marker_color=PRIMARY,
                              hovertemplate="%{y}<br>%{x:.3f}<extra></extra>")
            show(fig, height=420)

    if monitoring is not None:
        headline = (
            f"Monitoring: **{monitoring['status']}** — data quality "
            f"{monitoring['data_quality']['status']}, drift "
            f"{monitoring['drift']['overall_status']} (worst: "
            f"{monitoring['drift']['worst_feature']} PSI {monitoring['drift']['worst_psi']}), "
            f"performance {monitoring['performance'].get('status')}, scoring p95 "
            f"{monitoring['latency'].get('batch_ms_p95')}ms."
        )
        (st.warning if monitoring["retrain_recommended"] else st.success)(headline)
        if monitoring["alerts"]:
            st.write("Retraining triggers currently firing:")
            for alert in monitoring["alerts"]:
                st.write(f"- {alert}")

        drift_frame = pd.DataFrame(
            [{"feature": name, **values} for name, values in monitoring["drift"]["features"].items()]
        ).sort_values("psi", ascending=False)
        fig = px.bar(drift_frame.sort_values("psi"), x="psi", y="feature", orientation="h",
                     title="Feature drift (PSI) since the last training cutoff",
                     labels={"psi": "PSI", "feature": ""})
        fig.update_traces(marker_color=SERIES[1], hovertemplate="%{y}<br>PSI %{x:.3f}<extra></extra>")
        fig.add_vline(x=0.25, line_dash="dash", line_color=RISK_COLORS["critical"], line_width=1,
                      annotation_text="retrain threshold",
                      annotation_font=dict(size=11, color=RISK_COLORS["critical"]))
        show(fig, height=420)
        with st.expander("Per-feature drift (PSI) table"):
            st.dataframe(drift_frame, width="stretch", hide_index=True)

    st.subheader("Data quality: rejected records")
    rejected = pd.read_sql(
        "SELECT reject_reason, count(*) AS rows FROM staging.rejected_records GROUP BY 1 ORDER BY 2 DESC",
        engine,
    )
    total_raw = pd.read_sql("SELECT count(*) AS n FROM raw.sales_raw", engine)["n"].iloc[0]
    total_valid = pd.read_sql("SELECT count(*) AS n FROM staging.sales_valid", engine)["n"].iloc[0]
    q1, q2, q3 = st.columns(3)
    q1.metric("Raw rows ingested", f"{total_raw:,}")
    q2.metric("Passed validation", f"{total_valid:,}",
              f"{total_valid / total_raw * 100:.1f}% of raw")
    q3.metric("Rejected", f"{int(rejected['rows'].sum()):,}")

    c9, c10 = st.columns([2, 3])
    with c9:
        st.dataframe(rejected, width="stretch", hide_index=True,
                     column_config={
                         "reject_reason": st.column_config.TextColumn("Reason"),
                         "rows": st.column_config.NumberColumn("Rows", format="%d"),
                     })
    with c10:
        fig = px.bar(rejected.sort_values("rows"), x="rows", y="reject_reason", orientation="h",
                     title="Rejected records by reason",
                     labels={"rows": "Rows", "reject_reason": ""})
        fig.update_traces(marker_color=RISK_COLORS["high"],
                          hovertemplate="%{y}<br>%{x:,} rows<extra></extra>")
        show(fig, height=260)
