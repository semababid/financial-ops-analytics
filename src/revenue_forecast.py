"""Monthly revenue forecasting for Olist.

Approach
--------
The series is short (20 dense months) and trends steeply through 2017 before
flattening in 2018, so we run a small, honest bake-off and let a holdout
backtest pick the winner rather than trusting AIC on so few points:

  * Baseline      : last value carried forward by the train window's mean
    month-over-month growth.
  * SARIMA grid   : a few small SARIMAX orders on log revenue.
  * Holt-Winters  : exponential smoothing with a *damped* additive trend, which
    is well suited to a series that decelerates into a plateau.

We backtest on the last `TEST_MONTHS` months (MAPE/RMSE), select the model with
the best holdout MAPE, refit it on the full series, and project
`FORECAST_MONTHS` ahead with a confidence band.

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

# Candidate SARIMA orders (order, seasonal_order). Kept small for a short series.
SARIMA_CANDIDATES = [
    ((1, 1, 1), (0, 0, 0, 0)),   # non-seasonal ARIMA
    ((1, 1, 0), (0, 0, 0, 0)),
    ((0, 1, 1), (0, 0, 0, 0)),
    ((1, 1, 1), (1, 0, 0, 12)),  # mild seasonality, no seasonal differencing
]


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


def _fit_sarima(train: pd.Series, order, seasonal):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(
        np.log(train), order=order, seasonal_order=seasonal,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)


def _fit_holt(train: pd.Series):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    return ExponentialSmoothing(
        train, trend="add", damped_trend=True, seasonal=None,
    ).fit()


def _candidate_forecast(name: str, train: pd.Series, horizon: int) -> np.ndarray:
    """Point forecast for a named candidate model over `horizon` steps."""
    if name == "baseline":
        return baseline_forecast(train, horizon)
    if name == "holt-winters":
        return np.asarray(_fit_holt(train).forecast(horizon))
    # SARIMA candidate, name like "sarima(1,1,1)(1,0,0,12)"
    order, seasonal = _SARIMA_BY_NAME[name]
    return np.exp(_fit_sarima(train, order, seasonal).forecast(steps=horizon).values)


def _sarima_name(order, seasonal) -> str:
    return f"sarima{tuple(order)}{tuple(seasonal)}"


_SARIMA_BY_NAME = {_sarima_name(o, s): (o, s) for o, s in SARIMA_CANDIDATES}
CANDIDATES = ["baseline", "holt-winters", *_SARIMA_BY_NAME.keys()]


def backtest(series: pd.Series) -> dict:
    """Hold out the last TEST_MONTHS; score every candidate by MAPE/RMSE."""
    train, test = series.iloc[:-TEST_MONTHS], series.iloc[-TEST_MONTHS:]
    results = {}
    for name in CANDIDATES:
        try:
            pred = _candidate_forecast(name, train, len(test))
            results[name] = {"pred": pred, **_metrics(test.values, pred)}
        except Exception as exc:  # a candidate that fails to fit is just skipped
            results[name] = {"pred": None, "rmse": np.inf, "mape": np.inf, "error": str(exc)}
    best = min(results, key=lambda k: results[k]["mape"])
    return {"train": train, "test": test, "results": results, "best": best}


def project(series: pd.Series, model_name: str, horizon: int = FORECAST_MONTHS) -> pd.DataFrame:
    """Refit the chosen model on the full series and forecast with an 80% band."""
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")

    if model_name.startswith("sarima"):
        order, seasonal = _SARIMA_BY_NAME[model_name]
        fc = _fit_sarima(series, order, seasonal).get_forecast(steps=horizon)
        mean = np.exp(fc.predicted_mean.values)
        ci = np.exp(fc.conf_int(alpha=0.20).values)
        lower, upper = ci[:, 0], ci[:, 1]
    elif model_name == "holt-winters":
        res = _fit_holt(series)
        mean = np.asarray(res.forecast(horizon))
        resid_sd = np.std(res.resid)
        lower, upper = mean - 1.28 * resid_sd, mean + 1.28 * resid_sd
    else:  # baseline
        mean = baseline_forecast(series, horizon)
        resid_sd = np.std(series.pct_change().dropna()) * series.iloc[-1]
        lower, upper = mean - 1.28 * resid_sd, mean + 1.28 * resid_sd

    return pd.DataFrame({"forecast": mean, "lower": lower, "upper": upper}, index=idx)


def plot_backtest(bt: dict) -> None:
    fig, ax = plt.subplots()
    full = pd.concat([bt["train"], bt["test"]])
    ax.plot(full.index, full.values, color="#444", marker="o", label="Actual", lw=2)
    # Show the baseline and the winning model for a clean comparison.
    base = bt["results"]["baseline"]["pred"]
    ax.plot(bt["test"].index, base, color="#b5475d", ls="--", marker="s", label="Baseline")
    best_pred = bt["results"][bt["best"]]["pred"]
    ax.plot(bt["test"].index, best_pred, color="#1f6f54", ls="--", marker="^", label=f"Best: {bt['best']}")
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
    print("REVENUE FORECAST — BACKTEST (last %d months held out)" % TEST_MONTHS)
    print("=" * 60)
    print(f"{'model':28s} {'MAPE':>8s} {'RMSE':>12s}")
    for name in sorted(bt["results"], key=lambda k: bt["results"][k]["mape"]):
        r = bt["results"][name]
        mark = "  <- best" if name == bt["best"] else ""
        print(f"{name:28s} {r['mape']:>7.1f}% {r['rmse']:>12,.0f}{mark}")

    proj = project(series, bt["best"])
    print(f"\nProjection using '{bt['best']}' (next months):")
    for ts, row in proj.iterrows():
        print(f"  {ts:%Y-%m}: R$ {row['forecast']:>10,.0f}  [{row['lower']:,.0f} .. {row['upper']:,.0f}]")
    print("=" * 60)

    plot_backtest(bt)
    plot_projection(series, proj)
    proj.to_parquet(config.PROCESSED_DIR / "revenue_forecast.parquet")
    print(f"\nFigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    run()
