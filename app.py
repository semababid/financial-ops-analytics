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


def _optional(name: str):
    """Load a processed table if the stage that writes it has been run."""
    path = config.PROCESSED_DIR / name
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load():
    order_level = pd.read_parquet(config.ORDER_LEVEL)
    master = pd.read_parquet(config.ORDERS_MASTER)
    monthly = pd.read_parquet(config.MONTHLY_REVENUE)
    extras = {
        name: _optional(f"{name}.parquet")
        for name in (
            "revenue_forecast", "rfm", "distance_bands", "distance_delivery",
            "state_distance", "cohort_retention", "segment_value", "customer_value",
        )
    }
    return order_level, master, monthly, extras


if not config.MONTHLY_REVENUE.exists():
    st.error(
        "Processed data not found. Run `python scripts/run_pipeline.py` first "
        "(after unzipping the Olist CSVs into `data/raw/`)."
    )
    st.stop()

order_level, master, monthly, extras = load()
forecast, rfm = extras["revenue_forecast"], extras["rfm"]
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

tab_rev, tab_churn, tab_profit, tab_dist, tab_clv = st.tabs(
    ["📈 Revenue & Forecast", "🔁 Churn & RFM", "💰 Profitability",
     "🚚 Shipping Distance", "💎 Lifetime Value"]
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
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(m, x="order_month", y="n_orders", title="Orders per Month",
                   labels={"order_month": "Month", "n_orders": "Orders"}),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            px.line(m, x="order_month", y="avg_order_value", markers=True,
                    title="Average Order Value",
                    labels={"order_month": "Month", "avg_order_value": "AOV (R$)"}),
            width="stretch",
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
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                px.bar(seg, x="revenue", y="segment", orientation="h",
                       title="Revenue per RFM Segment", color="revenue",
                       color_continuous_scale="Greens"),
                width="stretch",
            )
        st.dataframe(seg, width="stretch", hide_index=True)
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
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            px.bar(cat.sort_values("freight_ratio"), x="freight_ratio", y="category",
                   orientation="h", title="Freight Drag (freight / price, median)",
                   color="freight_ratio", color_continuous_scale="Reds"),
            width="stretch",
        )
    st.caption(
        f"Contribution model: {config.COMMISSION_RATE:.0%} commission − "
        f"{config.PAYMENT_PROCESSING_RATE:.1%} payment fee (assumptions, see src/config.py)."
    )

# --- Shipping distance -----------------------------------------------------
with tab_dist:
    bands, deliv, states = (
        extras["distance_bands"], extras["distance_delivery"], extras["state_distance"]
    )
    if bands is None:
        st.info("Run `python -m src.geo_analysis` to generate the distance analysis.")
    else:
        st.markdown(
            "Freight and delivery both scale with seller→customer distance, which "
            "is what explains the freight burden differences between states."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(bands, x="dist_band", y="avg_freight",
                       title="Avg Freight by Distance Band",
                       labels={"dist_band": "Distance (km)", "avg_freight": "Avg freight (R$)"},
                       color="avg_freight", color_continuous_scale="Reds"),
                width="stretch",
            )
        with c2:
            if deliv is not None:
                st.plotly_chart(
                    px.bar(deliv, x="dist_band", y="avg_delivery_days",
                           title="Avg Delivery Time by Distance Band",
                           labels={"dist_band": "Distance (km)",
                                   "avg_delivery_days": "Avg delivery (days)"},
                           color="avg_delivery_days", color_continuous_scale="Blues"),
                    width="stretch",
                )
        if states is not None:
            s = states.reset_index()
            st.plotly_chart(
                px.scatter(s, x="avg_distance_km", y="freight_burden_pct",
                           size="gmv", text="customer_state", size_max=55,
                           title="Freight Burden vs Shipping Distance by State (bubble = GMV)",
                           labels={"avg_distance_km": "Avg seller→customer distance (km)",
                                   "freight_burden_pct": "Freight as % of GMV"}),
                width="stretch",
            )
        st.caption(
            "Distances are haversine between zip-prefix centroids, so they compare "
            "states well but are too coarse for routing decisions."
        )

# --- Lifetime value --------------------------------------------------------
with tab_clv:
    curve, segval, custval = (
        extras["cohort_retention"], extras["segment_value"], extras["customer_value"]
    )
    if custval is None:
        st.info("Run `python -m src.clv` to generate the lifetime-value analysis.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg GMV per customer", f"R$ {custval['gmv'].mean():.2f}")
        c2.metric("Avg contribution per customer", f"R$ {custval['contribution'].mean():.2f}")
        top10 = custval["contribution"].nlargest(len(custval) // 10).sum()
        c3.metric("Contribution held by top 10%",
                  f"{top10 / custval['contribution'].sum() * 100:.0f}%")
        st.caption(
            "That contribution figure is roughly the ceiling on what can be spent "
            "to acquire or win back a customer."
        )
        c1, c2 = st.columns(2)
        with c1:
            if curve is not None:
                st.plotly_chart(
                    px.line(curve, x="offset", y="retention_pct", markers=True,
                            title="Cohort Retention by Month Since First Order",
                            labels={"offset": "Months after first order",
                                    "retention_pct": "% ordering again"}),
                    width="stretch",
                )
        with c2:
            if segval is not None:
                st.plotly_chart(
                    px.bar(segval.sort_values("contribution"), x="contribution", y="segment",
                           orientation="h", title="Avg Contribution per Customer by Segment",
                           color="contribution", color_continuous_scale="Greens",
                           labels={"contribution": "Contribution (R$)", "segment": ""}),
                    width="stretch",
                )
