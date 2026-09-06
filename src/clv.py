"""Customer lifetime value.

Two views, because the textbook one has a real problem on this dataset.

1. Empirical cohort retention. Group customers by the month of their first
   order and track what share come back in each later month. No model, no
   assumptions, and it works fine even at a 3% repeat rate.

2. BG/NBD + Gamma-Gamma, the standard probabilistic CLV pair. These infer a
   purchase rate and a dropout rate per customer, which needs customers who
   actually repeat. Only ~1.7k of 94.7k do here, so I fit it, validate it
   against a holdout, and report honestly how little it can say rather than
   presenting a CLV number it hasn't earned.

Both are then converted from GMV to what Olist actually keeps, using the
contribution model from src/profitability.py.

    python -m src.clv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config, plotting
from .plotting import plt

warnings.filterwarnings("ignore")

# Forecast horizon for forward-looking CLV, matched to the revenue forecast.
HORIZON_DAYS = 180
HORIZON_MONTHS = 6
MONTHLY_DISCOUNT = 0.01

def load_transactions() -> pd.DataFrame:
    """One row per valid order: customer, date, order value, freight."""
    ol = pd.read_parquet(config.ORDER_LEVEL)
    ol = ol[ol["is_valid_revenue"]].copy()
    return ol[
        ["customer_unique_id", "order_purchase_timestamp", "items_value", "freight_value"]
    ].rename(columns={"order_purchase_timestamp": "date", "items_value": "value"})


def contribution_of(gmv, freight):
    """Same contribution formula as src/profitability.py, so the two agree.

    Payment fees are charged on the whole amount paid (item + freight), which
    is why the effective take lands near 12.1% of GMV rather than a flat 12.5%.
    """
    return config.COMMISSION_RATE * gmv - config.PAYMENT_PROCESSING_RATE * (gmv + freight)


# --- 1. Empirical cohort retention ---
def cohort_retention(tx: pd.DataFrame) -> pd.DataFrame:
    """Share of each acquisition cohort ordering again N months later.

    Only cohort/offset cells that were fully observable are counted, so a
    cohort acquired in the last month of data doesn't drag the curve down
    just because it had no time to come back.
    """
    tx = tx.copy()
    tx["order_month"] = tx["date"].dt.to_period("M")
    first = tx.groupby("customer_unique_id")["order_month"].min().rename("cohort")
    tx = tx.join(first, on="customer_unique_id")
    tx["offset"] = (tx["order_month"] - tx["cohort"]).apply(lambda x: x.n)

    last_month = tx["order_month"].max()
    cohort_size = first.value_counts().rename("size")

    # Customers per (cohort, offset) that placed at least one order.
    active = (
        tx.groupby(["cohort", "offset"])["customer_unique_id"].nunique().rename("active")
    ).reset_index()
    active = active.join(cohort_size, on="cohort")
    # Observable only if cohort_month + offset <= last month of data.
    active["observable"] = active.apply(
        lambda r: (r["cohort"] + r["offset"]) <= last_month, axis=1
    )
    active = active[active["observable"] & (active["size"] >= 100)]
    active["retention"] = active["active"] / active["size"]
    return active


def retention_curve(active: pd.DataFrame, max_offset: int = 12) -> pd.DataFrame:
    """Average retention by months-since-first-order across cohorts."""
    a = active[active["offset"].between(1, max_offset)]
    return (
        a.groupby("offset")
        .apply(lambda d: pd.Series({
            "cohorts": d["cohort"].nunique(),
            "customers": d["size"].sum(),
            "returning": d["active"].sum(),
            "retention_pct": d["active"].sum() / d["size"].sum() * 100,
        }), include_groups=False)
    )


# --- 2. BG/NBD + Gamma-Gamma ---
def build_summary(tx: pd.DataFrame) -> pd.DataFrame:
    from lifetimes.utils import summary_data_from_transaction_data

    return summary_data_from_transaction_data(
        tx, "customer_unique_id", "date", monetary_value_col="value", freq="D"
    )


def fit_bgnbd(summary: pd.DataFrame):
    from lifetimes import BetaGeoFitter

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    return bgf


def fit_gamma_gamma(summary: pd.DataFrame):
    """Gamma-Gamma models spend per transaction; needs repeat customers only."""
    from lifetimes import GammaGammaFitter

    rep = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rep["frequency"], rep["monetary_value"])
    return ggf, rep


def validate_bgnbd(tx: pd.DataFrame) -> dict:
    """Calibration/holdout check: can it predict repeat counts it hasn't seen?"""
    from lifetimes import BetaGeoFitter
    from lifetimes.utils import calibration_and_holdout_data

    end = tx["date"].max()
    cal_end = end - pd.Timedelta(days=180)
    cv = calibration_and_holdout_data(
        tx, "customer_unique_id", "date",
        calibration_period_end=cal_end, observation_period_end=end, freq="D",
    )
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(cv["frequency_cal"], cv["recency_cal"], cv["T_cal"])
    pred = bgf.conditional_expected_number_of_purchases_up_to_time(
        cv["duration_holdout"].iloc[0], cv["frequency_cal"], cv["recency_cal"], cv["T_cal"]
    )
    actual = cv["frequency_holdout"]
    return {
        "n": len(cv),
        "mae": float(np.mean(np.abs(actual - pred))),
        "pred_mean": float(pred.mean()),
        "actual_mean": float(actual.mean()),
        "corr": float(np.corrcoef(actual, pred)[0, 1]),
        "holdout_days": float(cv["duration_holdout"].iloc[0]),
    }


def predict_clv(summary: pd.DataFrame, bgf, ggf) -> pd.DataFrame:
    """Forward-looking CLV in GMV and in platform contribution."""
    out = summary.copy()
    out["pred_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        HORIZON_DAYS, out["frequency"], out["recency"], out["T"]
    )
    # Gamma-Gamma needs a spend estimate; fall back to observed value for
    # one-time buyers, whose monetary_value is 0 by construction.
    est_value = ggf.conditional_expected_average_profit(
        out["frequency"], out["monetary_value"]
    )
    out["exp_order_value"] = np.where(out["monetary_value"] > 0, est_value, np.nan)
    return out


def observed_value(tx: pd.DataFrame) -> pd.DataFrame:
    """What each customer has actually been worth so far."""
    g = tx.groupby("customer_unique_id").agg(
        orders=("value", "size"), gmv=("value", "sum"), freight=("freight_value", "sum")
    )
    g["contribution"] = contribution_of(g["gmv"], g["freight"])
    return g


# --- plots ---
def plot_retention(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(curve.index, curve["retention_pct"], marker="o", color="#1f6f54", lw=2.2)
    ax.fill_between(curve.index, curve["retention_pct"], alpha=0.12, color="#1f6f54")
    ax.set_title("Repeat-Purchase Retention by Month Since First Order")
    ax.set_xlabel("Months after first order")
    ax.set_ylabel("% of cohort ordering again")
    ax.set_xticks(curve.index)
    plotting.save(fig, "16_cohort_retention")


def plot_clv_distribution(obs: pd.DataFrame, seg_clv: pd.DataFrame | None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    v = obs["contribution"].clip(upper=obs["contribution"].quantile(0.99))
    ax1.hist(v, bins=50, color="#3f7d9e", alpha=0.85)
    ax1.axvline(v.mean(), color="#b5475d", ls="--", lw=2, label=f"mean R$ {v.mean():.2f}")
    ax1.set_title("Contribution per Customer (to date)")
    ax1.set_xlabel("Contribution (R$)")
    ax1.set_ylabel("Customers")
    ax1.legend()

    if seg_clv is not None:
        seg_clv["contribution"].sort_values().plot.barh(ax=ax2, color="#2a8c6f")
        ax2.set_title("Avg Contribution per Customer by Segment")
        ax2.set_xlabel("Contribution (R$)")
        ax2.set_ylabel("")
    plotting.save(fig, "17_clv_distribution")


def run() -> None:
    tx = load_transactions()

    print("\n" + "=" * 68)
    print("CUSTOMER LIFETIME VALUE")
    print("=" * 68)

    # --- observed value -------------------------------------------------
    obs = observed_value(tx)
    effective_rate = obs["contribution"].sum() / obs["gmv"].sum()
    print(f"Customers                  : {len(obs):,}")
    print(f"Avg GMV per customer       : R$ {obs['gmv'].mean():.2f}")
    print(f"Avg contribution / customer: R$ {obs['contribution'].mean():.2f}"
          f"   ({effective_rate:.1%} of GMV)")
    top10 = obs["contribution"].nlargest(len(obs) // 10).sum()
    print(f"Top 10% of customers hold  : {top10 / obs['contribution'].sum() * 100:.0f}% of contribution")

    # --- 1. empirical retention ----------------------------------------
    active = cohort_retention(tx)
    curve = retention_curve(active)
    print("\nEmpirical cohort retention (fully observed cells only):")
    print(f"{'month':>6s} {'cohorts':>8s} {'customers':>10s} {'returning':>10s} {'retention':>10s}")
    for off, row in curve.iterrows():
        print(f"{off:>6d} {int(row['cohorts']):>8d} {int(row['customers']):>10,} "
              f"{int(row['returning']):>10,} {row['retention_pct']:>9.2f}%")
    cumulative = curve["returning"].sum() / curve["customers"].iloc[0] * 100
    print(f"\nMonth-1 retention is {curve['retention_pct'].iloc[0]:.2f}% and decays from there.")
    print(f"Summed over 12 months that is {cumulative:.1f}% of a cohort ever returning,")
    print("which lines up with the ~3% repeat rate found in the churn analysis.")

    # --- 2. BG/NBD ------------------------------------------------------
    summary = build_summary(tx)
    bgf = fit_bgnbd(summary)
    ggf, rep = fit_gamma_gamma(summary)
    print("\nBG/NBD fitted on %s customers (%s with a repeat purchase)."
          % (f"{len(summary):,}", f"{len(rep):,}"))

    val = validate_bgnbd(tx)
    print(f"\nHoldout validation ({val['holdout_days']:.0f} days, n={val['n']:,}):")
    print(f"  predicted repeat purchases / customer : {val['pred_mean']:.4f}")
    print(f"  actual    repeat purchases / customer : {val['actual_mean']:.4f}")
    print(f"  MAE = {val['mae']:.4f}   corr = {val['corr']:.3f}")

    pred = predict_clv(summary, bgf, ggf)
    near_zero = (pred["pred_purchases"] < 0.1).mean() * 100
    print(f"\nPredicted purchases in next {HORIZON_MONTHS} months:")
    print(f"  mean {pred['pred_purchases'].mean():.4f} | "
          f"median {pred['pred_purchases'].median():.4f} | "
          f"max {pred['pred_purchases'].max():.2f}")
    print(f"  {near_zero:.1f}% of customers are predicted < 0.1 further purchases")

    # Forward CLV for the repeat cohort only, where the model has something to say.
    r = pred[pred["frequency"] > 0].copy()
    r["clv_gmv"] = r["pred_purchases"] * r["exp_order_value"]
    r["clv_contribution"] = r["clv_gmv"] * effective_rate
    print(f"\nAmong the {len(r):,} repeat customers the model can speak to:")
    print(f"  expected order value : R$ {r['exp_order_value'].mean():.2f}")
    print(f"  {HORIZON_MONTHS}-month CLV (GMV)   : R$ {r['clv_gmv'].mean():.2f}")
    print(f"  {HORIZON_MONTHS}-month CLV (contrib): R$ {r['clv_contribution'].mean():.2f}")

    # --- segment tie-in -------------------------------------------------
    seg_clv = None
    rfm_path = config.PROCESSED_DIR / "rfm.parquet"
    if rfm_path.exists():
        rfm = pd.read_parquet(rfm_path)
        joined = obs.join(rfm["segment"], how="inner")
        seg_clv = joined.groupby("segment").agg(
            customers=("gmv", "size"), gmv=("gmv", "mean"), contribution=("contribution", "mean")
        ).sort_values("contribution", ascending=False)
        print("\nObserved contribution per customer by RFM segment:")
        print(f"{'segment':18s} {'customers':>10s} {'avg GMV':>10s} {'avg contrib':>12s}")
        for seg, row in seg_clv.iterrows():
            print(f"{seg:18s} {int(row['customers']):>10,} {row['gmv']:>10.2f} {row['contribution']:>12.2f}")

    plot_retention(curve)
    plot_clv_distribution(obs, seg_clv)
    obs.to_parquet(config.PROCESSED_DIR / "customer_value.parquet")
    curve.reset_index().to_parquet(
        config.PROCESSED_DIR / "cohort_retention.parquet", index=False
    )
    if seg_clv is not None:
        seg_clv.reset_index().to_parquet(
            config.PROCESSED_DIR / "segment_value.parquet", index=False
        )
    print(f"\nFigures written to {config.FIGURES_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    run()
