"""Monthly revenue forecasting for Olist.

Approach
--------
The series is short (20 dense months), so we keep models simple and honest:

  * Baseline   : seasonal-naive-ish — last value carried forward with the
    average month-over-month growth of the training window.
  * SARIMA     : statsmodels SARIMAX on log revenue. Order kept small given the
    short series; we validate with a holdout backtest rather than trusting AIC.

We backtest on the last `TEST_MONTHS` months, report MAPE/RMSE, then refit on
the full series and project `FORECAST_MONTHS` ahead with a confidence band.

Run:  python -m src.revenue_forecast
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config, plotting
from .plotting import plt

warnings.filterwarnings("ignore")  # statsmodels emits many convergence warnings

TEST_MONTHS = 4
FORECAST_MONTHS = 6
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (0, 1, 0, 12)  # yearly seasonality, minimal params


def load_series() -> pd.Series:
    monthly = pd.read_parquet(config.MONTHLY_REVENUE)
    monthly = monthly[
        (monthly["order_month"] >= config.ANALYSIS_START)
        & (monthly["order_month"] <= config.ANALYSIS_END)
    ]
    s = monthly.set_index("order_month")["revenue"].astype(float)
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    return s


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual, pred = np.asarray(actual), np.asarray(pred)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return {"rmse": rmse, "mape": mape}


def baseline_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Carry last level forward, compounding by mean MoM growth of train."""
    growth = train.pct_change().dropna().mean()
    last = train.iloc[-1]
    return np.array([last * (1 + growth) ** (h + 1) for h in range(horizon)])


def fit_sarima(train: pd.Series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        np.log(train),
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def backtest(series: pd.Series) -> dict:
    """Hold out the last TEST_MONTHS and compare baseline vs SARIMA."""
    train, test = series.iloc[:-TEST_MONTHS], series.iloc[-TEST_MONTHS:]

    base = baseline_forecast(train, len(test))
    res = fit_sarima(train)
    sar = np.exp(res.forecast(steps=len(test)).values)

    return {
        "train": train,
        "test": test,
        "baseline": base,
        "sarima": sar,
        "baseline_metrics": _metrics(test.values, base),
        "sarima_metrics": _metrics(test.values, sar),
    }


def project(series: pd.Series, horizon: int = FORECAST_MONTHS) -> pd.DataFrame:
    """Refit SARIMA on the full series and forecast forward with a band."""
    res = fit_sarima(series)
    fc = res.get_forecast(steps=horizon)
    mean = np.exp(fc.predicted_mean)
    ci = np.exp(fc.conf_int(alpha=0.20))  # 80% interval
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    return pd.DataFrame(
        {
            "forecast": mean.values,
            "lower": ci.iloc[:, 0].values,
            "upper": ci.iloc[:, 1].values,
        },
        index=idx,
    )


def plot_backtest(bt: dict) -> None:
    fig, ax = plt.subplots()
    full = pd.concat([bt["train"], bt["test"]])
    ax.plot(full.index, full.values, color="#444", marker="o", label="Actual", lw=2)
    ax.plot(bt["test"].index, bt["baseline"], color="#b5475d", ls="--", marker="s", label="Baseline")
    ax.plot(bt["test"].index, bt["sarima"], color="#1f6f54", ls="--", marker="^", label="SARIMA")
    ax.axvline(bt["train"].index[-1], color="#999", ls=":", lw=1)
    ax.set_title(f"Backtest — last {TEST_MONTHS} months held out")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(plotting.brl))
    ax.legend()
    plotting.save(fig, "07_forecast_backtest")


def plot_projection(series: pd.Series, proj: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, color="#1f6f54", marker="o", lw=2, label="Actual")
    ax.plot(proj.index, proj["forecast"], color="#b5475d", marker="o", ls="--", lw=2, label="Forecast")
    ax.fill_between(proj.index, proj["lower"], proj["upper"], color="#b5475d", alpha=0.15, label="80% interval")
    ax.set_title(f"Revenue Forecast — next {FORECAST_MONTHS} months")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(plotting.brl))
    ax.legend()
    plotting.save(fig, "08_forecast_projection")


def run() -> None:
    series = load_series()
    bt = backtest(series)

    print("\n" + "=" * 60)
    print("REVENUE FORECAST — BACKTEST")
    print("=" * 60)
    bm, sm = bt["baseline_metrics"], bt["sarima_metrics"]
    print(f"{'model':10s} {'MAPE':>8s} {'RMSE':>12s}")
    print(f"{'baseline':10s} {bm['mape']:>7.1f}% {bm['rmse']:>12,.0f}")
    print(f"{'SARIMA':10s} {sm['mape']:>7.1f}% {sm['rmse']:>12,.0f}")
    winner = "SARIMA" if sm["mape"] <= bm["mape"] else "baseline"
    print(f"-> better holdout MAPE: {winner}")

    proj = project(series)
    print("\nProjection (next months):")
    for ts, row in proj.iterrows():
        print(f"  {ts:%Y-%m}: R$ {row['forecast']:>10,.0f}  [{row['lower']:,.0f} .. {row['upper']:,.0f}]")
    print("=" * 60)

    plot_backtest(bt)
    plot_projection(series, proj)
    proj.to_parquet(config.PROCESSED_DIR / "revenue_forecast.parquet")
    print(f"\nFigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    run()
