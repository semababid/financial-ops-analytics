"""Load raw Olist CSVs into pandas with correct dtypes and parsed dates.

Each loader is small and explicit so the schema is documented in code.
"""
from __future__ import annotations

import pandas as pd

from . import config

# Date columns per table (parsed on load).
_DATE_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}


def _load(name: str) -> pd.DataFrame:
    path = config.RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw file for '{name}': {path}\n"
            "Extract archive.zip into data/raw/ (see README)."
        )
    parse_dates = _DATE_COLS.get(name)
    return pd.read_csv(path, parse_dates=parse_dates)


def load_orders() -> pd.DataFrame:
    return _load("orders")


def load_order_items() -> pd.DataFrame:
    return _load("order_items")


def load_payments() -> pd.DataFrame:
    return _load("order_payments")


def load_reviews() -> pd.DataFrame:
    return _load("order_reviews")


def load_customers() -> pd.DataFrame:
    return _load("customers")


def load_products() -> pd.DataFrame:
    return _load("products")


def load_sellers() -> pd.DataFrame:
    return _load("sellers")


def load_category_translation() -> pd.DataFrame:
    df = _load("category_translation")
    # Strip any BOM that slipped into the header.
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Return every raw table keyed by short name."""
    return {
        "orders": load_orders(),
        "order_items": load_order_items(),
        "payments": load_payments(),
        "reviews": load_reviews(),
        "customers": load_customers(),
        "products": load_products(),
        "sellers": load_sellers(),
        "category_translation": load_category_translation(),
    }


if __name__ == "__main__":
    for key, frame in load_all().items():
        print(f"{key:22s} {frame.shape[0]:>8,} rows  {frame.shape[1]:>2} cols")
