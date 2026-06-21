"""Interactive dashboard for the Financial Operations Analytics project.

Run:  streamlit run app.py

Reads the processed parquet tables (build them first with
`python scripts/run_pipeline.py`). Falls back to a friendly message if the
data has not been generated yet.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config

st.set_page_config(page_title="Olist Financial Ops Analytics", layout="wide", page_icon="📊")


@st.cache_data
def load():
    order_level = pd.read_parquet(config.ORDER_LEVEL)
    master = pd.read_parquet(config.ORDERS_MASTER)
    monthly = pd.read_parquet(config.MONTHLY_REVENUE)
    forecast = (
        pd.read_parquet(config.PROCESSED_DIR / "revenue_forecast.parquet")
        if (config.PROCESSED_DIR / "revenue_forecast.parquet").exists()
        else None
    )
    rfm = (
        pd.read_parquet(config.PROCESSED_DIR / "rfm.parquet")
        if (config.PROCESSED_DIR / "rfm.parquet").exists()
        else None
    )
    return order_level, master, monthly, forecast, rfm


if not config.MONTHLY_REVENUE.exists():
    st.error(
        "Processed data not found. Run `python scripts/run_pipeline.py` first "
        "(after unzipping the Olist CSVs into `data/raw/`)."
    )
    st.stop()

order_level, master, monthly, forecast, rfm = load()
valid = order_level[order_level["is_valid_revenue"]]

# Trim to the dense analysis window for time-series views.
m = monthly[(monthly["order_month"] >= config.ANALYSIS_START)
            & (monthly["order_month"] <= config.ANALYSIS_END)]

st.title("📊 Olist — Financial Operations Analytics")
st.caption("Revenue forecasting · churn & retention · profitability  |  Brazilian E-Commerce (2017–2018)")

# --- KPI row ---------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue (GMV)", f"R$ {valid['items_value'].sum()/1e6:.1f}M")
k2.metric("Orders", f"{len(valid):,}")
k3.metric("Customers", f"{valid['customer_unique_id'].nunique():,}")
repeat = valid["customer_unique_id"].value_counts()
k4.metric("Repeat-buyer share", f"{(repeat > 1).mean()*100:.1f}%")
k5.metric("Avg order value", f"R$ {valid['items_value'].mean():.0f}")

tab_rev, tab_churn, tab_profit = st.tabs(
    ["📈 Revenue & Forecast", "🔁 Churn & RFM", "💰 Profitability"]
)

# --- Revenue & forecast ----------------------------------------------------
with tab_rev:
    fig = px.line(m, x="order_month", y="revenue", markers=True,
                  title="Monthly Revenue (realized GMV)",
                  labels={"order_month": "Month", "revenue": "Revenue (R$)"})
    if forecast is not None:
        fc = forecast.reset_index(names="order_month")
        fig.add_scatter(x=fc["order_month"], y=fc["forecast"], mode="lines+markers",
                        name="Forecast", line=dict(dash="dash", color="#b5475d"))
        fig.add_scatter(x=fc["order_month"], y=fc["upper"], mode="lines",
                        line=dict(width=0), showlegend=False)
        fig.add_scatter(x=fc["order_month"], y=fc["lower"], mode="lines", fill="tonexty",
                        line=dict(width=0), name="80% interval",
                        fillcolor="rgba(181,71,93,0.15)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(m, x="order_month", y="n_orders", title="Orders per Month",
                   labels={"order_month": "Month", "n_orders": "Orders"}),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.line(m, x="order_month", y="avg_order_value", markers=True,
                    title="Average Order Value",
                    labels={"order_month": "Month", "avg_order_value": "AOV (R$)"}),
            use_container_width=True,
        )

# --- Churn & RFM -----------------------------------------------------------
with tab_churn:
    if rfm is not None:
        seg = (rfm.groupby("segment")
               .agg(customers=("recency", "size"), revenue=("monetary", "sum"))
               .reset_index().sort_values("revenue", ascending=False))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(seg, x="customers", y="segment", orientation="h",
                       title="Customers per RFM Segment", color="customers",
                       color_continuous_scale="Teal"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.bar(seg, x="revenue", y="segment", orientation="h",
                       title="Revenue per RFM Segment", color="revenue",
                       color_continuous_scale="Greens"),
                use_container_width=True,
            )
        st.dataframe(seg, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.churn_analysis` to generate RFM segments.")

# --- Profitability ---------------------------------------------------------
with tab_profit:
    rev = master[master["is_valid_revenue"]].copy()
    rev["contribution"] = (config.COMMISSION_RATE * rev["price"]
                           - config.PAYMENT_PROCESSING_RATE * (rev["price"] + rev["freight_value"]))
    cat = (rev.groupby("category")
           .agg(contribution=("contribution", "sum"),
                freight_ratio=("freight_value", lambda s: (s / rev.loc[s.index, "price"].clip(lower=0.01)).median()))
           .reset_index().sort_values("contribution", ascending=False).head(15))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(cat.sort_values("contribution"), x="contribution", y="category",
                   orientation="h", title="Top Categories by Contribution",
                   color="contribution", color_continuous_scale="Greens"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.bar(cat.sort_values("freight_ratio"), x="freight_ratio", y="category",
                   orientation="h", title="Freight Drag (freight / price, median)",
                   color="freight_ratio", color_continuous_scale="Reds"),
            use_container_width=True,
        )
    st.caption(
        f"Contribution model: {config.COMMISSION_RATE:.0%} commission − "
        f"{config.PAYMENT_PROCESSING_RATE:.1%} payment fee (assumptions, see src/config.py)."
    )
