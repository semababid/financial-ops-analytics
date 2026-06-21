"""Exploratory data analysis: generate core charts and print key findings.

Run:  python -m src.eda
Reads processed parquet tables; writes PNGs to reports/figures/.
"""
from __future__ import annotations

import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config, plotting
from .plotting import plt


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not config.MONTHLY_REVENUE.exists():
        raise FileNotFoundError("Processed tables missing. Run: python -m src.data_cleaning")
    master = pd.read_parquet(config.ORDERS_MASTER)
    order_level = pd.read_parquet(config.ORDER_LEVEL)
    monthly = pd.read_parquet(config.MONTHLY_REVENUE)
    return master, order_level, monthly


def _trim_sparse_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """Olist's first and last months are partial; keep the dense window."""
    return monthly[
        (monthly["order_month"] >= config.ANALYSIS_START)
        & (monthly["order_month"] <= config.ANALYSIS_END)
    ].copy()


def plot_monthly_revenue(monthly: pd.DataFrame) -> None:
    m = _trim_sparse_months(monthly)
    fig, ax = plt.subplots()
    ax.plot(m["order_month"], m["revenue"], marker="o", color="#1f6f54", lw=2.2)
    ax.fill_between(m["order_month"], m["revenue"], alpha=0.12, color="#1f6f54")
    ax.set_title("Monthly Revenue (realized GMV)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(plotting.brl))
    plotting.save(fig, "01_monthly_revenue")


def plot_orders_and_aov(monthly: pd.DataFrame) -> None:
    m = _trim_sparse_months(monthly)
    fig, ax1 = plt.subplots()
    ax1.bar(m["order_month"], m["n_orders"], width=20, color="#7fc8a9", alpha=0.7, label="Orders")
    ax1.set_ylabel("Orders / month")
    ax1.set_xlabel("Month")
    ax2 = ax1.twinx()
    ax2.plot(m["order_month"], m["avg_order_value"], color="#b5475d", marker="o", lw=2, label="Avg order value")
    ax2.set_ylabel("Avg order value")
    ax2.yaxis.set_major_formatter(FuncFormatter(plotting.brl))
    ax2.grid(False)
    ax1.set_title("Order Volume vs. Average Order Value")
    plotting.save(fig, "02_orders_vs_aov")


def plot_top_categories(master: pd.DataFrame) -> pd.DataFrame:
    rev = master[master["is_valid_revenue"]]
    top = (
        rev.groupby("category")["item_revenue"].sum().sort_values(ascending=False).head(15)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    top.sort_values().plot.barh(ax=ax, color="#2a8c6f")
    ax.set_title("Top 15 Categories by Revenue")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(plotting.brl))
    plotting.save(fig, "03_top_categories")
    return top


def plot_top_states(order_level: pd.DataFrame) -> pd.DataFrame:
    rev = order_level[order_level["is_valid_revenue"]]
    top = rev.groupby("customer_state")["items_value"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11, 7))
    top.sort_values().plot.barh(ax=ax, color="#3f7d9e")
    ax.set_title("Top 15 States by Revenue (customer location)")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(plotting.brl))
    plotting.save(fig, "04_top_states")
    return top


def plot_payments_and_reviews(order_level: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    pay = order_level["payment_type"].value_counts()
    pay.plot.bar(ax=ax1, color="#2a8c6f")
    ax1.set_title("Payment Type (orders)")
    ax1.set_ylabel("Orders")
    ax1.tick_params(axis="x", rotation=30)

    rev = order_level["review_score"].dropna().round().astype(int)
    rev.value_counts().sort_index().plot.bar(ax=ax2, color="#b5475d")
    ax2.set_title("Review Score Distribution")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Orders")
    plotting.save(fig, "05_payments_reviews")


def plot_delivery(order_level: pd.DataFrame) -> None:
    d = order_level["delivery_days"].dropna()
    d = d[(d >= 0) & (d <= 60)]
    fig, ax = plt.subplots()
    ax.hist(d, bins=40, color="#3f7d9e", alpha=0.85)
    ax.axvline(d.median(), color="#b5475d", ls="--", lw=2, label=f"median {d.median():.0f}d")
    ax.set_title("Delivery Time (purchase → customer)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Orders")
    ax.legend()
    plotting.save(fig, "06_delivery_days")


def print_findings(master: pd.DataFrame, order_level: pd.DataFrame, monthly: pd.DataFrame) -> None:
    m = _trim_sparse_months(monthly)
    valid = order_level[order_level["is_valid_revenue"]]
    print("\n" + "=" * 60)
    print("KEY EDA FINDINGS")
    print("=" * 60)
    print(f"Analysis window      : {m['order_month'].min():%Y-%m} .. {m['order_month'].max():%Y-%m} ({len(m)} months)")
    print(f"Realized revenue     : R$ {valid['items_value'].sum():,.0f}")
    print(f"Realized orders      : {len(valid):,}")
    print(f"Unique customers     : {valid['customer_unique_id'].nunique():,}")
    repeat = valid['customer_unique_id'].value_counts()
    print(f"Repeat-buyer share   : {(repeat > 1).mean() * 100:.1f}%  (most customers order once)")
    print(f"Avg order value      : R$ {valid['items_value'].mean():,.2f}")
    print(f"Median delivery time : {order_level['delivery_days'].median():.0f} days")
    print(f"Late-delivery share  : {order_level['is_late'].mean() * 100:.1f}%")
    growth = m['revenue'].iloc[-1] / m['revenue'].iloc[0] - 1
    print(f"Revenue {m['order_month'].iloc[0]:%b%y}->{m['order_month'].iloc[-1]:%b%y} : {growth * 100:+.0f}%")
    top_cat = master[master['is_valid_revenue']].groupby('category')['item_revenue'].sum().sort_values(ascending=False)
    print(f"Top category         : {top_cat.index[0]} (R$ {top_cat.iloc[0]:,.0f}, {top_cat.iloc[0]/top_cat.sum()*100:.1f}% of rev)")
    print("=" * 60)


def run() -> None:
    master, order_level, monthly = _load_processed()
    plot_monthly_revenue(monthly)
    plot_orders_and_aov(monthly)
    plot_top_categories(master)
    plot_top_states(order_level)
    plot_payments_and_reviews(order_level)
    plot_delivery(order_level)
    print_findings(master, order_level, monthly)
    print(f"\nFigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    run()
