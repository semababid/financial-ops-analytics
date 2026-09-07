"""Pre-aggregate everything the dashboard needs into a small committed bundle.

The processed tables are ~34 MB, mostly because orders_master and order_level
are row-per-item and row-per-order. Those are too big to commit and the raw
CSVs aren't in the repo either, so a deployed Streamlit app would have nothing
to read. This step rolls them up into the handful of tiny tables the dashboard
actually displays (a few tens of KB in total) and writes them to
data/dashboard/, which is committed.

Run after the analysis stages:

    python -m src.dashboard_data
"""
from __future__ import annotations

import shutil

import pandas as pd

from . import config

# Tables already small enough to copy across untouched.
_COPY_AS_IS = [
    "monthly_revenue.parquet",
    "revenue_forecast.parquet",
    "category_profitability.parquet",
    "distance_bands.parquet",
    "distance_delivery.parquet",
    "state_distance.parquet",
    "cohort_retention.parquet",
    "segment_value.parquet",
]


def _headline_kpis(order_level: pd.DataFrame) -> pd.DataFrame:
    valid = order_level[order_level["is_valid_revenue"]]
    per_customer = valid["customer_unique_id"].value_counts()
    return pd.DataFrame([{
        "revenue": valid["items_value"].sum(),
        "orders": len(valid),
        "customers": valid["customer_unique_id"].nunique(),
        "repeat_share_pct": (per_customer > 1).mean() * 100,
        "avg_order_value": valid["items_value"].mean(),
    }])


def _segment_summary() -> pd.DataFrame | None:
    path = config.PROCESSED_DIR / "rfm.parquet"
    if not path.exists():
        return None
    rfm = pd.read_parquet(path)
    return (
        rfm.groupby("segment")
        .agg(customers=("recency", "size"), revenue=("monetary", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )


def _clv_kpis() -> pd.DataFrame | None:
    path = config.PROCESSED_DIR / "customer_value.parquet"
    if not path.exists():
        return None
    cv = pd.read_parquet(path)
    top10 = cv["contribution"].nlargest(len(cv) // 10).sum()
    return pd.DataFrame([{
        "avg_gmv": cv["gmv"].mean(),
        "avg_contribution": cv["contribution"].mean(),
        "top10_contribution_pct": top10 / cv["contribution"].sum() * 100,
    }])


def run(verbose: bool = True) -> None:
    if not config.ORDER_LEVEL.exists():
        raise FileNotFoundError(
            "Processed tables missing. Run: python scripts/run_pipeline.py"
        )

    order_level = pd.read_parquet(
        config.ORDER_LEVEL, columns=["is_valid_revenue", "items_value", "customer_unique_id"]
    )
    _headline_kpis(order_level).to_parquet(config.DASHBOARD_DIR / "kpis.parquet", index=False)

    written = ["kpis.parquet"]
    for builder, name in ((_segment_summary, "segment_summary.parquet"),
                          (_clv_kpis, "clv_kpis.parquet")):
        df = builder()
        if df is not None:
            df.to_parquet(config.DASHBOARD_DIR / name, index=False)
            written.append(name)

    for name in _COPY_AS_IS:
        src = config.PROCESSED_DIR / name
        if src.exists():
            shutil.copyfile(src, config.DASHBOARD_DIR / name)
            written.append(name)

    if verbose:
        total = sum((config.DASHBOARD_DIR / n).stat().st_size for n in written)
        print(f"dashboard bundle: {len(written)} tables, {total / 1024:.0f} KB "
              f"-> {config.DASHBOARD_DIR.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
