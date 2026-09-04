"""Shipping distance analysis - does distance explain the freight problem?

The profitability work found freight (16.6% of GMV) costs more than the
platform's contribution margin (12.1%), and that SP has the lowest freight
burden. This module uses the geolocation table to test the obvious explanation:
seller-to-customer distance.

Steps: collapse the 1M-row geolocation file to one coordinate per zip prefix,
attach seller and customer coordinates to every order item, compute the
haversine distance, then look at how freight, delivery time and lateness move
with it. A small OLS puts a R$-per-100km number on it while controlling for
product weight.

    python -m src.geo_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config, data_loader, plotting
from .plotting import plt

# Brazil bounding box - the raw geolocation file has a handful of points in
# Europe/Asia that would otherwise drag a zip's centroid across the planet.
BRAZIL_LAT = (-34.0, 5.3)
BRAZIL_LNG = (-74.0, -34.7)

EARTH_RADIUS_KM = 6371.0

# Distance bands used for the summary tables/plots.
DIST_BINS = [0, 50, 150, 400, 800, 1500, 10000]
DIST_LABELS = ["<50", "50-150", "150-400", "400-800", "800-1500", "1500+"]


def build_zip_lookup() -> pd.DataFrame:
    """One (lat, lng) per zip prefix, median-averaged and outlier-filtered."""
    g = data_loader._load("geolocation")
    g = g[
        g["geolocation_lat"].between(*BRAZIL_LAT)
        & g["geolocation_lng"].between(*BRAZIL_LNG)
    ]
    # Median rather than mean: robust to the remaining scatter within a prefix.
    return (
        g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance in km between two arrays of coordinates."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_geo_items() -> pd.DataFrame:
    """Order items enriched with seller/customer coordinates and distance."""
    items = pd.read_parquet(config.ORDERS_MASTER)
    items = items[items["is_valid_revenue"]].copy()

    # orders_master keeps the customer zip but not the seller's, so pull it in.
    sellers = data_loader.load_sellers()[["seller_id", "seller_zip_code_prefix"]]
    items = items.merge(sellers, on="seller_id", how="left")

    zipgeo = build_zip_lookup()
    cust = zipgeo.rename(columns={"lat": "cust_lat", "lng": "cust_lng"})
    sell = zipgeo.rename(columns={"lat": "sell_lat", "lng": "sell_lng"})

    items = items.merge(cust, left_on="customer_zip_code_prefix", right_index=True, how="left")
    items = items.merge(sell, left_on="seller_zip_code_prefix", right_index=True, how="left")

    items["distance_km"] = haversine_km(
        items["sell_lat"], items["sell_lng"], items["cust_lat"], items["cust_lng"]
    )
    items["freight_ratio"] = items["freight_value"] / items["price"].clip(lower=0.01)
    items["dist_band"] = pd.cut(items["distance_km"], DIST_BINS, labels=DIST_LABELS)
    return items


def by_distance_band(geo: pd.DataFrame) -> pd.DataFrame:
    d = geo.dropna(subset=["distance_km"])
    out = d.groupby("dist_band", observed=True).agg(
        items=("freight_value", "size"),
        avg_freight=("freight_value", "mean"),
        avg_price=("price", "mean"),
        freight_pct_of_price=("freight_ratio", "median"),
    )
    out["freight_pct_of_price"] = (out["freight_pct_of_price"] * 100).round(1)
    return out


def delivery_by_distance(geo: pd.DataFrame) -> pd.DataFrame:
    """Delivery time and lateness by distance band (needs order-level dates)."""
    order_level = pd.read_parquet(config.ORDER_LEVEL)[
        ["order_id", "delivery_days", "is_late"]
    ]
    # One row per order: take the max distance across its items (worst leg).
    per_order = (
        geo.dropna(subset=["distance_km"])
        .groupby("order_id")
        .agg(distance_km=("distance_km", "max"))
        .reset_index()
        .merge(order_level, on="order_id", how="left")
    )
    per_order["dist_band"] = pd.cut(per_order["distance_km"], DIST_BINS, labels=DIST_LABELS)
    return (
        per_order.groupby("dist_band", observed=True)
        .agg(
            orders=("order_id", "size"),
            avg_delivery_days=("delivery_days", "mean"),
            late_rate=("is_late", "mean"),
        )
        .assign(late_rate=lambda d: (d["late_rate"] * 100).round(1))
        .round(1)
    )


def freight_regression(geo: pd.DataFrame):
    """OLS: freight ~ distance + weight. Gives an interpretable R$ per 100 km."""
    import statsmodels.api as sm

    d = geo.dropna(subset=["distance_km", "product_weight_g", "freight_value"]).copy()
    d = d[d["freight_value"].between(0, 400)]  # trim extreme freight outliers
    X = pd.DataFrame(
        {
            "distance_100km": d["distance_km"] / 100.0,
            "weight_kg": d["product_weight_g"] / 1000.0,
        }
    )
    X = sm.add_constant(X)
    model = sm.OLS(d["freight_value"], X).fit()
    return model


def state_distance(geo: pd.DataFrame) -> pd.DataFrame:
    d = geo.dropna(subset=["distance_km"])
    out = (
        d.groupby("customer_state")
        .agg(
            items=("freight_value", "size"),
            avg_distance_km=("distance_km", "mean"),
            gmv=("price", "sum"),
            freight=("freight_value", "sum"),
        )
    )
    out["freight_burden_pct"] = out["freight"] / out["gmv"] * 100
    return out[out["items"] >= 200].sort_values("gmv", ascending=False)


def plot_distance_effects(band: pd.DataFrame, deliv: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(band.index.astype(str), band["avg_freight"], color="#b5475d", alpha=0.85)
    ax1.set_title("Freight Cost by Shipping Distance")
    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("Avg freight (R$)")

    ax2.bar(deliv.index.astype(str), deliv["avg_delivery_days"], color="#3f7d9e", alpha=0.85)
    ax2.set_title("Delivery Time by Shipping Distance")
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Avg delivery (days)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=25)
    plotting.save(fig, "14_distance_freight_delivery")


def plot_state_scatter(states: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(states["avg_distance_km"], states["freight_burden_pct"],
               s=states["gmv"] / 8000, color="#2a8c6f", alpha=0.65, edgecolor="k", lw=0.5)
    for st, row in states.iterrows():
        ax.annotate(st, (row["avg_distance_km"], row["freight_burden_pct"]),
                    fontsize=9, ha="center", va="center")
    ax.set_title("Freight Burden vs. Shipping Distance by State\n(bubble = GMV)")
    ax.set_xlabel("Avg seller→customer distance (km)")
    ax.set_ylabel("Freight as % of GMV")
    plotting.save(fig, "15_state_distance_vs_freight")


def run() -> None:
    geo = build_geo_items()
    matched = geo["distance_km"].notna().mean()

    print("\n" + "=" * 66)
    print("SHIPPING DISTANCE ANALYSIS")
    print("=" * 66)
    print(f"Order items with a usable distance : {matched*100:.1f}%")
    print(f"Median seller→customer distance    : {geo['distance_km'].median():,.0f} km")
    print(f"Mean                               : {geo['distance_km'].mean():,.0f} km")

    band = by_distance_band(geo)
    print("\nFreight by distance band:")
    print(f"{'band(km)':>10s} {'items':>9s} {'avg freight':>12s} {'avg price':>10s} {'freight/price':>14s}")
    for b, row in band.iterrows():
        print(f"{str(b):>10s} {int(row['items']):>9,} {row['avg_freight']:>12.2f} "
              f"{row['avg_price']:>10.2f} {row['freight_pct_of_price']:>13.1f}%")

    deliv = delivery_by_distance(geo)
    print("\nDelivery by distance band:")
    print(f"{'band(km)':>10s} {'orders':>9s} {'avg days':>10s} {'late rate':>11s}")
    for b, row in deliv.iterrows():
        print(f"{str(b):>10s} {int(row['orders']):>9,} {row['avg_delivery_days']:>10.1f} {row['late_rate']:>10.1f}%")

    model = freight_regression(geo)
    per100 = model.params["distance_100km"]
    perkg = model.params["weight_kg"]
    print("\nOLS  freight ~ distance + weight")
    print(f"  +R$ {per100:.2f} per 100 km   (t={model.tvalues['distance_100km']:.0f})")
    print(f"  +R$ {perkg:.2f} per kg        (t={model.tvalues['weight_kg']:.0f})")
    print(f"  R² = {model.rsquared:.3f}")

    states = state_distance(geo)
    print("\nStates - distance vs freight burden (top 8 by GMV):")
    print(f"{'state':>6s} {'avg km':>9s} {'freight % of GMV':>18s}")
    for st, row in states.head(8).iterrows():
        print(f"{st:>6s} {row['avg_distance_km']:>9,.0f} {row['freight_burden_pct']:>17.1f}%")
    corr = states["avg_distance_km"].corr(states["freight_burden_pct"])
    print(f"\nCorrelation (state avg distance vs freight burden): {corr:.2f}")

    plot_distance_effects(band, deliv)
    plot_state_scatter(states)
    states.to_parquet(config.PROCESSED_DIR / "state_distance.parquet")
    print(f"\nFigures written to {config.FIGURES_DIR}")
    print("=" * 66)


if __name__ == "__main__":
    run()
