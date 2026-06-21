"""Central paths and constants for the Financial Operations Analytics project."""
from __future__ import annotations

from pathlib import Path

# --- Directories -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Raw CSV files ---------------------------------------------------------
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

# --- Processed outputs -----------------------------------------------------
ORDERS_MASTER = PROCESSED_DIR / "orders_master.parquet"   # one row per order item, enriched
ORDER_LEVEL = PROCESSED_DIR / "order_level.parquet"        # one row per order
MONTHLY_REVENUE = PROCESSED_DIR / "monthly_revenue.parquet"

# --- Business constants ----------------------------------------------------
# Olist is a marketplace; the platform earns a commission on GMV rather than the
# full item price. We model a representative take-rate for profitability work.
COMMISSION_RATE = 0.15          # platform commission on item price (assumption)
PAYMENT_PROCESSING_RATE = 0.025  # payment processor fee on total paid (assumption)

# Reliable date window for time-series work (Olist data is sparse before/after).
ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-08-31"

# Only these order statuses represent realized revenue.
VALID_REVENUE_STATUSES = {"delivered", "shipped", "invoiced"}

RANDOM_STATE = 42
