"""Interactive dashboard for the Financial Operations Analytics project.

    streamlit run app.py

Reads the small pre-aggregated bundle in data/dashboard/, which is committed to
the repo. That keeps the deployed app independent of the raw Olist CSVs and the
~34 MB of intermediate tables. Rebuild the bundle with:

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config

st.set_page_config(page_title="Olist Financial Ops Analytics", layout="wide", page_icon="📊")


@st.cache_data
def load() -> dict[str, pd.DataFrame | None]:
    names = [
        "kpis", "monthly_revenue", "revenue_forecast", "segment_summary",
        "category_profitability", "distance_bands", "distance_delivery",
        "state_distance", "cohort_retention", "segment_value", "clv_kpis",
    ]
    out = {}
    for n in names:
        path = config.DASHBOARD_DIR / f"{n}.parquet"
        out[n] = pd.read_parquet(path) if path.exists() else None
    return out


d = load()
if d["kpis"] is None:
    st.error(
        "Dashboard data not found. Run `python scripts/run_pipeline.py` first "
        "(after unzipping the Olist CSVs into `data/raw/`)."
    )
    st.stop()

st.title("📊 Olist — Financial Operations Analytics")
st.caption(
    "Revenue forecasting · churn & retention · profitability · logistics · lifetime value"
    "  |  Brazilian E-Commerce (2017–2018)"
)

k = d["kpis"].iloc[0]
c = st.columns(5)
c[0].metric("Revenue (GMV)", f"R$ {k['revenue'] / 1e6:.1f}M")
c[1].metric("Orders", f"{int(k['orders']):,}")
c[2].metric("Customers", f"{int(k['customers']):,}")
c[3].metric("Repeat-buyer share", f"{k['repeat_share_pct']:.1f}%")
c[4].metric("Avg order value", f"R$ {k['avg_order_value']:.0f}")

tab_rev, tab_churn, tab_profit, tab_dist, tab_clv = st.tabs(
    ["📈 Revenue & Forecast", "🔁 Churn & RFM", "💰 Profitability",
     "🚚 Shipping Distance", "💎 Lifetime Value"]
)

# --- Revenue & forecast ----------------------------------------------------
with tab_rev:
    m = d["monthly_revenue"]
    m = m[(m["order_month"] >= config.ANALYSIS_START) & (m["order_month"] <= config.ANALYSIS_END)]
    fig = px.line(m, x="order_month", y="revenue", markers=True,
                  title="Monthly Revenue (realized GMV)",
                  labels={"order_month": "Month", "revenue": "Revenue (R$)"})
    fc = d["revenue_forecast"]
    if fc is not None:
        fc = fc.reset_index(names="order_month")
        fig.add_scatter(x=fc["order_month"], y=fc["forecast"], mode="lines+markers",
                        name="Forecast", line=dict(dash="dash", color="#b5475d"))
        fig.add_scatter(x=fc["order_month"], y=fc["upper"], mode="lines",
                        line=dict(width=0), showlegend=False)
        fig.add_scatter(x=fc["order_month"], y=fc["lower"], mode="lines", fill="tonexty",
                        line=dict(width=0), name="80% interval",
                        fillcolor="rgba(181,71,93,0.15)")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Forecast is SARIMA(0,1,1), picked by lowest MAPE on a 4-month holdout "
        "(10.7%). It projects a flat level, which is the honest read on a series "
        "that plateaued through 2018."
    )

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
    seg = d["segment_summary"]
    if seg is None:
        st.info("Run `python -m src.churn_analysis` to generate RFM segments.")
    else:
        st.markdown(
            "Only ~3% of customers ever order again, so frequency carries almost "
            "no information. Loyalty is therefore a binary repeat flag with its "
            "own segment, and the one-time majority is split on recency and spend."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(seg, x="customers", y="segment", orientation="h",
                       title="Customers per Segment", color="customers",
                       color_continuous_scale="Teal", labels={"segment": ""}),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                px.bar(seg, x="revenue", y="segment", orientation="h",
                       title="Revenue per Segment", color="revenue",
                       color_continuous_scale="Greens", labels={"segment": ""}),
                width="stretch",
            )
        st.dataframe(seg, width="stretch", hide_index=True)

# --- Profitability ---------------------------------------------------------
with tab_profit:
    cat = d["category_profitability"]
    if cat is None:
        st.info("Run `python -m src.profitability` to generate the margin analysis.")
    else:
        cat = cat.reset_index()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(cat.sort_values("contribution"), x="contribution", y="category",
                       orientation="h", title="Top Categories by Contribution",
                       color="contribution", color_continuous_scale="Greens",
                       labels={"category": "", "contribution": "Contribution (R$)"}),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                px.bar(cat.sort_values("freight_ratio"), x="freight_ratio", y="category",
                       orientation="h", title="Freight Drag (freight / price, median)",
                       color="freight_ratio", color_continuous_scale="Reds",
                       labels={"category": "", "freight_ratio": "Freight / price"}),
                width="stretch",
            )
        st.caption(
            f"Contribution model: {config.COMMISSION_RATE:.0%} commission minus "
            f"{config.PAYMENT_PROCESSING_RATE:.1%} payment fee on the amount paid "
            "(assumptions, see src/config.py). Freight is 16.6% of GMV against a "
            "12.1% contribution margin, so logistics outweighs category mix."
        )

# --- Shipping distance -----------------------------------------------------
with tab_dist:
    bands, deliv, states = d["distance_bands"], d["distance_delivery"], d["state_distance"]
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
            "Freight rises about R$1.06 per 100 km; distance and weight together "
            "explain 53% of it. Distances are haversine between zip-prefix "
            "centroids, so they compare states well but are too coarse for routing."
        )

# --- Lifetime value --------------------------------------------------------
with tab_clv:
    clvk, curve, segval = d["clv_kpis"], d["cohort_retention"], d["segment_value"]
    if clvk is None:
        st.info("Run `python -m src.clv` to generate the lifetime-value analysis.")
    else:
        v = clvk.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg GMV per customer", f"R$ {v['avg_gmv']:.2f}")
        c2.metric("Avg contribution per customer", f"R$ {v['avg_contribution']:.2f}")
        c3.metric("Contribution held by top 10%", f"{v['top10_contribution_pct']:.0f}%")
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
