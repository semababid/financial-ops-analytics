"""Churn / retention analysis.

Two parts. First the usual RFM segmentation (recency/frequency/monetary per
customer_unique_id, bucketed into Champions / At Risk / etc.).

Second, a repeat-purchase model. Almost everyone here buys exactly once, so
"churn" doesn't really fit the classic definition - I framed it the other way
round: given only what we know from a customer's *first* order, can we predict
whether they ever come back? That's the thing Olist could actually act on.

    python -m src.churn_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, plotting
from .plotting import plt

# As-of date for recency: day after the last order in the dataset.
SNAPSHOT_OFFSET_DAYS = 1


def _load_orders() -> pd.DataFrame:
    df = pd.read_parquet(config.ORDER_LEVEL)
    return df[df["is_valid_revenue"]].copy()


# --- RFM segmentation ---
def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    snapshot = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=SNAPSHOT_OFFSET_DAYS)
    rfm = (
        orders.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda s: (snapshot - s.max()).days),
            frequency=("order_id", "count"),
            monetary=("items_value", "sum"),
        )
    )

    # Recency and Monetary have real spread, so quintile scores are meaningful
    # (recency reverse-scored: recent = 5).
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Frequency is degenerate here - ~97% of customers order exactly once - so a
    # 5-way F score would just be noise. Treat loyalty as a binary repeat flag
    # and segment the one-time majority on recency x monetary instead.
    rfm["is_repeat"] = rfm["frequency"] > 1
    rfm["segment"] = rfm.apply(_segment, axis=1)
    return rfm


def _segment(row) -> str:
    # Repeat buyers are the genuinely loyal minority; everyone else ordered once
    # and is placed by how recently they bought and how much they spent.
    if row["is_repeat"]:
        return "Loyal / Repeat"
    r, m = row["r_score"], row["m_score"]
    if r >= 4:
        return "Champions" if m >= 4 else "Promising"
    if r <= 2:
        return "At Risk" if m >= 4 else "Hibernating"
    return "Needs Attention"  # middle recency band


def plot_segments(rfm: pd.DataFrame) -> None:
    seg = rfm.groupby("segment").agg(customers=("recency", "size"), revenue=("monetary", "sum"))
    seg = seg.sort_values("revenue", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    seg["customers"].plot.barh(ax=ax1, color="#3f7d9e")
    ax1.set_title("Customers per Segment")
    ax1.set_xlabel("Customers")
    ax1.set_ylabel("")
    seg["revenue"].plot.barh(ax=ax2, color="#2a8c6f")
    ax2.set_title("Revenue per Segment")
    ax2.set_xlabel("Revenue (R$)")
    ax2.set_ylabel("")
    plotting.save(fig, "09_rfm_segments")


# --- Repeat-purchase model ---
def build_repeat_dataset(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per customer; features from their FIRST order, label = repeated?"""
    orders = orders.sort_values("order_purchase_timestamp")
    counts = orders.groupby("customer_unique_id")["order_id"].count()
    # Take the actual first *row* per customer. groupby().first() would instead
    # take the first non-null value column-by-column, which can splice a later
    # order's review/delivery into the "first order" features.
    first = orders.drop_duplicates("customer_unique_id", keep="first").set_index("customer_unique_id")

    df = first.copy()
    df["repeat"] = (counts > 1).astype(int)
    df["first_order_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["first_order_month"] = df["order_purchase_timestamp"].dt.month
    feature_cols = [
        "items_value", "freight_value", "n_items", "n_sellers",
        "payment_installments", "review_score", "delivery_days",
        "delivery_delay_days", "first_order_dow", "first_order_month",
        "payment_type", "customer_state",
    ]
    return df[feature_cols + ["repeat"]].dropna(subset=["review_score", "delivery_days"])


def train_repeat_model(data: pd.DataFrame) -> dict:
    y = data["repeat"]
    X = data.drop(columns="repeat")
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes("object").columns.tolist()

    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat),
        ]
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE, stratify=y
    )

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gbm": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }
    fitted, scores = {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        scores[name] = roc_auc_score(y_te, proba)
        fitted[name] = pipe

    best = max(scores, key=scores.get)
    best_pipe = fitted[best]
    proba = best_pipe.predict_proba(X_te)[:, 1]
    return {
        "scores": scores, "best": best, "pipe": best_pipe,
        "y_te": y_te.reset_index(drop=True), "proba": proba,
        "num": num, "cat": cat,
        "lift": decile_lift(y_te.values, proba),
        "base_rate": y.mean(),
    }


def decile_lift(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Rank customers by predicted propensity, bucket into deciles, measure lift.

    Lift = repeat rate in the decile / overall repeat rate. A lift of 3 in the
    top decile means targeting that 10% finds repeaters 3x more efficiently
    than random outreach.
    """
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=range(10, 0, -1)).astype(int)
    base = df["y"].mean()
    out = (
        df.groupby("decile")
        .agg(customers=("y", "size"), repeaters=("y", "sum"), repeat_rate=("y", "mean"))
        .sort_index()
    )
    out["lift"] = out["repeat_rate"] / base
    out["cum_repeaters_pct"] = out["repeaters"].cumsum() / out["repeaters"].sum() * 100
    return out


def feature_importance(res: dict) -> pd.Series:
    """Permutation-free importance: |coef| for logreg, impurity for GBM."""
    pipe = res["pipe"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
    else:
        vals = clf.feature_importances_
    return pd.Series(vals, index=names).sort_values(ascending=False)


def plot_importance(imp: pd.Series) -> None:
    top = imp.head(12)
    fig, ax = plt.subplots(figsize=(11, 7))
    top.sort_values().plot.barh(ax=ax, color="#b5475d")
    ax.set_title("Top Drivers of Repeat Purchase")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    plotting.save(fig, "10_repeat_drivers")


def plot_lift(lift: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.bar(lift.index, lift["lift"], color="#2a8c6f", alpha=0.85)
    ax.axhline(1.0, color="#b5475d", ls="--", lw=2, label="random (lift = 1)")
    ax.set_title("Repeat-Purchase Model — Lift by Decile")
    ax.set_xlabel("Decile (1 = highest predicted propensity)")
    ax.set_ylabel("Lift vs. base rate")
    ax.set_xticks(range(1, 11))
    ax.legend()
    plotting.save(fig, "11_repeat_lift")


def run() -> None:
    orders = _load_orders()

    rfm = build_rfm(orders)
    plot_segments(rfm)
    seg_summary = rfm.groupby("segment").agg(
        customers=("recency", "size"), revenue=("monetary", "sum"), avg_freq=("frequency", "mean")
    ).sort_values("revenue", ascending=False)

    print("\n" + "=" * 64)
    print("RFM SEGMENTS")
    print("=" * 64)
    print(f"{'segment':18s} {'customers':>10s} {'revenue':>14s} {'rev share':>10s}")
    tot = seg_summary["revenue"].sum()
    for seg, row in seg_summary.iterrows():
        print(f"{seg:18s} {int(row['customers']):>10,} {row['revenue']:>14,.0f} {row['revenue']/tot*100:>9.1f}%")

    data = build_repeat_dataset(orders)
    res = train_repeat_model(data)
    print("\n" + "=" * 64)
    print("REPEAT-PURCHASE MODEL")
    print("=" * 64)
    print(f"Base repeat rate : {res['base_rate']*100:.1f}%  ({len(data):,} customers)")
    for name, auc in res["scores"].items():
        print(f"  {name:8s} ROC-AUC = {auc:.3f}")
    print(f"Best model       : {res['best']}")

    lift = res["lift"]
    print("\nDecile lift (rank by predicted repeat propensity):")
    print(f"{'decile':>6s} {'customers':>10s} {'repeat_rate':>12s} {'lift':>6s} {'cum%repeaters':>14s}")
    for dec, row in lift.iterrows():
        print(f"{dec:>6d} {int(row['customers']):>10,} {row['repeat_rate']*100:>11.1f}% "
              f"{row['lift']:>6.2f} {row['cum_repeaters_pct']:>13.1f}%")
    top_lift = lift.loc[1, "lift"]
    print(f"\n-> Top decile finds repeaters {top_lift:.1f}x better than random outreach.")
    plot_lift(lift)

    imp = feature_importance(res)
    plot_importance(imp)
    print("\nTop repeat-purchase drivers:")
    for name, val in imp.head(8).items():
        print(f"  {name:40s} {val:.3f}")

    rfm.to_parquet(config.PROCESSED_DIR / "rfm.parquet")
    print(f"\nFigures written to {config.FIGURES_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    run()
