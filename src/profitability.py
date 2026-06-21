"""Profitability analysis for the Olist marketplace.

Olist is a platform, not a retailer: it does not keep item revenue, it earns a
*commission* on GMV and incurs payment-processing cost. We model a stylised
contribution margin per order item:

    commission_revenue = COMMISSION_RATE      * item_price
    payment_cost       = PAYMENT_PROCESSING_RATE * (item_price + freight)
    contribution       = commission_revenue - payment_cost

Freight is shown separately because it is largely pass-through to the carrier
but still a lever (free-shipping promos, heavy/low-value items erode margin).

These rates are assumptions (see config.py) — the analysis is about *relative*
profitability across categories, sellers and regions, which is robust to the
exact take-rate chosen.

Run:  python -m src.profitability
"""
from __future__ import annotations

import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config, plotting
from .plotting import plt


def load_items() -> pd.DataFrame:
    df = pd.read_parquet(config.ORDERS_MASTER)
    return df[df["is_valid_revenue"]].copy()


def add_economics(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    items["commission_revenue"] = config.COMMISSION_RATE * items["price"]
    items["payment_cost"] = config.PAYMENT_PROCESSING_RATE * (items["price"] + items["freight_value"])
    items["contribution"] = items["commission_revenue"] - items["payment_cost"]
    # Freight as a share of item price flags shipping-heavy, margin-thin items.
    items["freight_ratio"] = items["freight_value"] / items["price"].clip(lower=0.01)
    return items


def by_category(items: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    g = (
        items.groupby("category")
        .agg(
            gmv=("price", "sum"),
            commission_revenue=("commission_revenue", "sum"),
            contribution=("contribution", "sum"),
            items=("price", "size"),
            freight_ratio=("freight_ratio", "median"),
        )
    )
    g["margin_pct"] = g["contribution"] / g["gmv"] * 100
    return g.sort_values("contribution", ascending=False).head(top)


def by_state(items: pd.DataFrame) -> pd.DataFrame:
    g = (
        items.groupby("customer_state")
        .agg(
            gmv=("price", "sum"),
            contribution=("contribution", "sum"),
            freight=("freight_value", "sum"),
            items=("price", "size"),
        )
    )
    g["margin_pct"] = g["contribution"] / g["gmv"] * 100
    g["freight_burden_pct"] = g["freight"] / g["gmv"] * 100
    return g.sort_values("contribution", ascending=False)


def by_seller(items: pd.DataFrame, min_items: int = 50) -> pd.DataFrame:
    g = (
        items.groupby("seller_id")
        .agg(gmv=("price", "sum"), contribution=("contribution", "sum"), items=("price", "size"))
    )
    g = g[g["items"] >= min_items]
    g["margin_pct"] = g["contribution"] / g["gmv"] * 100
    return g.sort_values("contribution", ascending=False)


def freight_drag(items: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Categories where freight is heaviest relative to item value."""
    g = (
        items.groupby("category")
        .agg(median_freight_ratio=("freight_ratio", "median"), items=("price", "size"),
             avg_price=("price", "mean"))
    )
    return g[g["items"] >= 200].sort_values("median_freight_ratio", ascending=False).head(top)


def plot_category_margin(cat: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    c = cat.sort_values("contribution")
    c["contribution"].plot.barh(ax=ax1, color="#2a8c6f")
    ax1.set_title("Top Categories by Contribution")
    ax1.set_xlabel("Contribution (R$)")
    ax1.set_ylabel("")
    ax1.xaxis.set_major_formatter(FuncFormatter(plotting.brl))
    c["margin_pct"].plot.barh(ax=ax2, color="#3f7d9e")
    ax2.set_title("Contribution Margin %")
    ax2.set_xlabel("Margin (% of GMV)")
    ax2.set_ylabel("")
    plotting.save(fig, "12_category_profitability")


def plot_freight_drag(fr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    (fr["median_freight_ratio"] * 100).sort_values().plot.barh(ax=ax, color="#b5475d")
    ax.set_title("Freight Drag — median freight as % of item price")
    ax.set_xlabel("Freight / price (%)")
    ax.set_ylabel("")
    plotting.save(fig, "13_freight_drag")


def run() -> None:
    items = add_economics(load_items())

    total_gmv = items["price"].sum()
    total_contrib = items["contribution"].sum()
    total_freight = items["freight_value"].sum()

    print("\n" + "=" * 66)
    print("PROFITABILITY (marketplace contribution model)")
    print("=" * 66)
    print(f"Assumptions: commission {config.COMMISSION_RATE:.0%} of price, "
          f"payment fee {config.PAYMENT_PROCESSING_RATE:.1%} of paid")
    print(f"GMV (item value)     : R$ {total_gmv:,.0f}")
    print(f"Commission revenue   : R$ {items['commission_revenue'].sum():,.0f}")
    print(f"Payment cost         : R$ {items['payment_cost'].sum():,.0f}")
    print(f"Contribution         : R$ {total_contrib:,.0f}  ({total_contrib/total_gmv*100:.1f}% of GMV)")
    print(f"Freight handled      : R$ {total_freight:,.0f}  ({total_freight/total_gmv*100:.1f}% of GMV)")

    cat = by_category(items)
    print("\nTop categories by contribution:")
    print(f"{'category':24s} {'contribution':>13s} {'margin%':>8s} {'freight_ratio':>14s}")
    for name, row in cat.iterrows():
        print(f"{name:24s} {row['contribution']:>13,.0f} {row['margin_pct']:>7.1f}% {row['freight_ratio']*100:>13.0f}%")

    states = by_state(items)
    print("\nMargin & freight burden by top-5 states (by contribution):")
    print(f"{'state':6s} {'contribution':>13s} {'margin%':>8s} {'freight_burden%':>16s}")
    for st, row in states.head(5).iterrows():
        print(f"{st:6s} {row['contribution']:>13,.0f} {row['margin_pct']:>7.1f}% {row['freight_burden_pct']:>15.1f}%")

    sellers = by_seller(items)
    print(f"\nSellers (>=50 items): {len(sellers):,}  |  "
          f"top 10% drive {sellers['contribution'].head(len(sellers)//10).sum()/sellers['contribution'].sum()*100:.0f}% of seller contribution")

    fr = freight_drag(items)
    print("\nHighest freight-drag categories (median freight / price):")
    for name, row in fr.head(5).iterrows():
        print(f"  {name:24s} {row['median_freight_ratio']*100:>5.0f}%  (avg price R$ {row['avg_price']:,.0f})")

    plot_category_margin(cat)
    plot_freight_drag(fr)
    cat.to_parquet(config.PROCESSED_DIR / "category_profitability.parquet")
    states.to_parquet(config.PROCESSED_DIR / "state_profitability.parquet")
    print(f"\nFigures written to {config.FIGURES_DIR}")
    print("=" * 66)


if __name__ == "__main__":
    run()
