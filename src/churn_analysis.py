"""Customer churn & retention analysis for Olist.

Two complementary views:

1. RFM segmentation
   Recency / Frequency / Monetary scores per `customer_unique_id`, bucketed
   into actionable segments (Champions, Loyal, At Risk, Hibernating, ...).

2. Repeat-purchase ("anti-churn") model
   Olist customers overwhelmingly buy once, so "churn" here is framed as the
   inverse of *repeat purchase within the observation window*. We engineer
   features from a customer's FIRST order (the only signal available at
   acquisition) and train a classifier to predict whether they buy again.
   This is the lever Olist can actually pull: which first-order experiences
   predict a second order.

Run:  python -m src.churn_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
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


# --------------------------------------------------------------------------- #
# 1. RFM segmentation
# --------------------------------------------------------------------------- #
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

    # Score 1-5. Recency is reverse-scored (recent = high). Frequency is mostly
    # 1, so rank with ties spread out before cutting into quintiles.
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["segment"] = rfm.apply(_segment, axis=1)
    return rfm


def _segment(row) -> str:
    r, f = row["r_score"], row["f_score"]
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    return "Needs Attention"


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


# --------------------------------------------------------------------------- #
# 2. Repeat-purchase model
# --------------------------------------------------------------------------- #
def build_repeat_dataset(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per customer; features from their FIRST order, label = repeated?"""
    orders = orders.sort_values("order_purchase_timestamp")
    first = orders.groupby("customer_unique_id").first()
    counts = orders.groupby("customer_unique_id")["order_id"].count()

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

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num),
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
        "y_te": y_te, "proba": proba, "num": num, "cat": cat,
        "report": classification_report(y_te, (proba >= 0.5).astype(int), digits=3),
        "base_rate": y.mean(),
    }


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
    print("\nClassification report (threshold 0.5):")
    print(res["report"])

    imp = feature_importance(res)
    plot_importance(imp)
    print("Top repeat-purchase drivers:")
    for name, val in imp.head(8).items():
        print(f"  {name:40s} {val:.3f}")

    rfm.to_parquet(config.PROCESSED_DIR / "rfm.parquet")
    print(f"\nFigures written to {config.FIGURES_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    run()
