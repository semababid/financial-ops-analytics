"""Build clean, analysis-ready tables from the raw Olist star schema.

Outputs (written to data/processed/ as parquet):
  - orders_master.parquet : one row per ORDER ITEM, fully enriched (the grain
    at which price/freight live). Use for category/seller/profitability work.
  - order_level.parquet    : one row per ORDER, with item totals, payment info,
    customer key, review score and delivery timing. Use for churn/forecasting.
  - monthly_revenue.parquet: monthly revenue time series for forecasting.

Design notes
------------
* Revenue is recognized from ``price`` (item value), not ``payment_value``
  (which includes freight and is affected by installments/vouchers).
* ``customer_unique_id`` is the true person; ``customer_id`` is per-order. All
  customer-level work keys on ``customer_unique_id``.
* Only ``VALID_REVENUE_STATUSES`` count as realized revenue. Canceled and
  unavailable orders are flagged but excluded from revenue aggregates.
"""
from __future__ import annotations

import pandas as pd

from . import config, data_loader


def _payments_per_order(payments: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple payment rows into one row per order."""
    grp = payments.groupby("order_id")
    out = pd.DataFrame(
        {
            "payment_value": grp["payment_value"].sum(),
            "payment_installments": grp["payment_installments"].max(),
            "n_payments": grp.size(),
        }
    )
    # Dominant payment type = the one with the largest value share.
    dominant = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        .set_index("order_id")["payment_type"]
    )
    out["payment_type"] = dominant
    return out.reset_index()


def _reviews_per_order(reviews: pd.DataFrame) -> pd.DataFrame:
    """One review score per order (mean if duplicated)."""
    return (
        reviews.groupby("order_id")["review_score"].mean().round(2).reset_index()
    )


def build_orders_master(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Item-grain table: every order item enriched with all dimensions."""
    t = tables or data_loader.load_all()

    products = t["products"].merge(
        t["category_translation"], on="product_category_name", how="left"
    )
    # Prefer the English category; fall back to the Portuguese name.
    products["category"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    ).fillna("unknown")

    df = (
        t["order_items"]
        .merge(t["orders"], on="order_id", how="left")
        .merge(t["customers"], on="customer_id", how="left")
        .merge(
            products[["product_id", "category", "product_weight_g"]],
            on="product_id",
            how="left",
        )
        .merge(
            t["sellers"][["seller_id", "seller_city", "seller_state"]],
            on="seller_id",
            how="left",
        )
    )

    # Time features off the purchase timestamp.
    ts = df["order_purchase_timestamp"]
    df["order_month"] = ts.dt.to_period("M").dt.to_timestamp()
    df["order_year"] = ts.dt.year
    df["order_dow"] = ts.dt.dayofweek

    df["is_valid_revenue"] = df["order_status"].isin(config.VALID_REVENUE_STATUSES)
    df["item_revenue"] = df["price"]
    df["item_total_paid"] = df["price"] + df["freight_value"]
    return df


def build_order_level(
    master: pd.DataFrame,
    tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Order-grain table for churn and forecasting."""
    t = tables or data_loader.load_all()

    agg = (
        master.groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            n_sellers=("seller_id", "nunique"),
            items_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            order_purchase_timestamp=("order_purchase_timestamp", "first"),
            order_delivered_customer_date=("order_delivered_customer_date", "first"),
            order_estimated_delivery_date=("order_estimated_delivery_date", "first"),
            order_status=("order_status", "first"),
            customer_unique_id=("customer_unique_id", "first"),
            customer_state=("customer_state", "first"),
            order_month=("order_month", "first"),
            is_valid_revenue=("is_valid_revenue", "first"),
        )
    )

    agg = (
        agg.merge(_payments_per_order(t["payments"]), on="order_id", how="left")
        .merge(_reviews_per_order(t["reviews"]), on="order_id", how="left")
    )

    # Delivery timing (days). Negative = delivered earlier than estimated.
    delivered = agg["order_delivered_customer_date"]
    purchase = agg["order_purchase_timestamp"]
    estimate = agg["order_estimated_delivery_date"]
    agg["delivery_days"] = (delivered - purchase).dt.days
    agg["delivery_delay_days"] = (delivered - estimate).dt.days
    agg["is_late"] = agg["delivery_delay_days"] > 0
    return agg


def build_monthly_revenue(order_level: pd.DataFrame) -> pd.DataFrame:
    """Monthly realized revenue + order/customer counts for forecasting."""
    rev = order_level[order_level["is_valid_revenue"]].copy()
    monthly = (
        rev.groupby("order_month")
        .agg(
            revenue=("items_value", "sum"),
            freight=("freight_value", "sum"),
            n_orders=("order_id", "count"),
            n_customers=("customer_unique_id", "nunique"),
        )
        .reset_index()
        .sort_values("order_month")
    )
    monthly["avg_order_value"] = monthly["revenue"] / monthly["n_orders"]
    return monthly


def run(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Build all processed tables and persist them as parquet."""
    tables = data_loader.load_all()

    master = build_orders_master(tables)
    order_level = build_order_level(master, tables)
    monthly = build_monthly_revenue(order_level)

    master.to_parquet(config.ORDERS_MASTER, index=False)
    order_level.to_parquet(config.ORDER_LEVEL, index=False)
    monthly.to_parquet(config.MONTHLY_REVENUE, index=False)

    if verbose:
        print(f"orders_master   : {master.shape[0]:,} item rows -> {config.ORDERS_MASTER.name}")
        print(f"order_level     : {order_level.shape[0]:,} orders  -> {config.ORDER_LEVEL.name}")
        print(f"monthly_revenue : {monthly.shape[0]:,} months  -> {config.MONTHLY_REVENUE.name}")
        print(f"  revenue span  : {monthly['order_month'].min():%Y-%m} .. {monthly['order_month'].max():%Y-%m}")
        print(f"  total revenue : R$ {monthly['revenue'].sum():,.0f}")
    return {"master": master, "order_level": order_level, "monthly": monthly}


if __name__ == "__main__":
    run()
