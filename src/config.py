"""Paths and a few business constants, kept in one place so the modules agree."""
from __future__ import annotations

from pathlib import Path

# directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
# Small pre-aggregated tables for the dashboard. Unlike processed/, these are
# committed, so the deployed Streamlit app has data without the raw CSVs.
DASHBOARD_DIR = DATA_DIR / "dashboard"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, DASHBOARD_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# raw CSVs
RAW_FILES = {
    "customers": RAW_DIR / "olist_customers_dataset.csv",
    "geolocation": RAW_DIR / "olist_geolocation_dataset.csv",
    "order_items": RAW_DIR / "olist_order_items_dataset.csv",
    "order_payments": RAW_DIR / "olist_order_payments_dataset.csv",
    "order_reviews": RAW_DIR / "olist_order_reviews_dataset.csv",
    "orders": RAW_DIR / "olist_orders_dataset.csv",
    "products": RAW_DIR / "olist_products_dataset.csv",
    "sellers": RAW_DIR / "olist_sellers_dataset.csv",
    "category_translation": RAW_DIR / "product_category_name_translation.csv",
}

# processed outputs
ORDERS_MASTER = PROCESSED_DIR / "orders_master.parquet"   # one row per order item, enriched
ORDER_LEVEL = PROCESSED_DIR / "order_level.parquet"        # one row per order
MONTHLY_REVENUE = PROCESSED_DIR / "monthly_revenue.parquet"

# Olist takes a commission on GMV rather than keeping the item price, so these
# rates drive the profitability model. They're assumptions - the relative
# rankings hold regardless of the exact numbers.
COMMISSION_RATE = 0.15
PAYMENT_PROCESSING_RATE = 0.025

# data is sparse before 2017 and after Aug 2018, so the time series sticks to this window
ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-08-31"

# what counts as realized revenue (drop canceled/unavailable)
VALID_REVENUE_STATUSES = {"delivered", "shipped", "invoiced"}

RANDOM_STATE = 42
