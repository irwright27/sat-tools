from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL


DEFAULT_PERIOD = 365
DEFAULT_NS = 7
DEFAULT_INNER_ITER = 2
DEFAULT_OUTER_ITER = 5


def _validate_integer(value: int, name: str, minimum: int) -> None:
    """Reject fractional and boolean parameters before passing them to STL."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _next_odd(value: float) -> int:
    """Return the smallest odd integer greater than or equal to value."""
    n = math.ceil(value)
    return n if n % 2 == 1 else n + 1


def calculate_trend_window(ns: int, period: int = DEFAULT_PERIOD) -> int:
    """
    Calculate the STL trend-window length from Cleveland et al. (1990).

    nt >= 1.5 * period / (1 - 1.5 / ns)

    The returned value is the smallest odd integer satisfying the expression.
    """
    _validate_integer(period, "period", 2)
    _validate_integer(ns, "ns", 3)
    if ns % 2 == 0:
        raise ValueError("ns must be an odd integer >= 3.")

    nt_min = 1.5 * period / (1.0 - 1.5 / ns)
    return _next_odd(nt_min)


def calculate_low_pass_window(period: int = DEFAULT_PERIOD) -> int:
    """
    Return an odd low-pass window strictly greater than period.

    statsmodels.STL requires low_pass > period and low_pass to be odd.
    """
    _validate_integer(period, "period", 2)

    nl = period + 1
    return nl if nl % 2 == 1 else nl + 1


def stl_parameters(
    ns: int = DEFAULT_NS,
    period: int = DEFAULT_PERIOD,
    inner_iter: int = DEFAULT_INNER_ITER,
    outer_iter: int = DEFAULT_OUTER_ITER,
) -> dict[str, int]:
    """Return the explicit STL parameters used by :func:`decompose_stl`."""
    _validate_integer(inner_iter, "inner_iter", 1)
    _validate_integer(outer_iter, "outer_iter", 0)

    return {
        "period": period,
        "seasonal": ns,
        "trend": calculate_trend_window(ns, period),
        "low_pass": calculate_low_pass_window(period),
        "inner_iter": inner_iter,
        "outer_iter": outer_iter,
    }


def prepare_daily_series(
    series: pd.Series,
    interpolate: bool = True,
) -> pd.Series:
    """
    Normalize a VI time series to one value per day and a complete daily index.

    Duplicate observations on the same date are averaged. Missing days are
    optionally filled by time interpolation.

    Parameters
    ----------
    series : pandas.Series
        VI values indexed by datetime.
    interpolate : bool, default True
        If True, fill missing daily values using time interpolation in both
        directions. Leading/trailing missing values use the nearest valid
        value; interior gaps are linearly interpolated in time.

    Returns
    -------
    pandas.Series
        Daily series with a DatetimeIndex.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series index must be a pandas DatetimeIndex.")
    if series.empty:
        raise ValueError("series is empty.")

    if series.index.hasnans:
        raise ValueError("series index contains NaT.")
    if np.isinf(series.to_numpy(dtype=float)).any():
        raise ValueError("series contains infinite values.")
    if not series.notna().any():
        raise ValueError("series has no non-missing observations.")

    out = series.astype(float).sort_index().copy()
    out.index = out.index.normalize()
    out = out.groupby(level=0).mean()

    daily_index = pd.date_range(
        start=out.index.min(),
        end=out.index.max(),
        freq="1D",
        tz=out.index.tz,
    )
    out = out.reindex(daily_index)

    if interpolate:
        out = out.interpolate(method="time", limit_direction="both")

    return out


def smooth_daily_series(
    series: pd.Series,
    window_days: int = 31,
    polyorder: int = 2,
) -> pd.Series:
    """Smooth a complete daily series with a Savitzky–Golay filter.

    Defaults to a 31-day quadratic fit. Prepare and gap-fill irregular
    observations with ``prepare_daily_series`` first. The window must be odd,
    longer than the polynomial order, and no longer than the series.

    Returns a new Series preserving its name, dates, and existing metadata;
    ``attrs['smoothing']`` records the filter settings. Values are not clipped.
    Edge values use polynomial fits (``mode='interp'``). Inspect endpoints,
    long interpolated gaps, and peak attenuation when choosing a window.
    Keep the original observations separately: downstream partition residuals
    are relative to this smoothed input.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series index must be a pandas DatetimeIndex.")
    if series.empty:
        raise ValueError("series is empty.")
    if series.index.hasnans or not series.index.is_unique:
        raise ValueError("series index must contain unique dates without NaT.")
    expected = pd.date_range(series.index[0], periods=len(series), freq="1D")
    if (not series.index.equals(expected)
            or not series.index.equals(series.index.normalize())):
        raise ValueError("series must have an increasing, complete daily index at midnight.")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Prepare/gap-fill the series before smoothing; values must be finite.")
    _validate_integer(window_days, "window_days", 1)
    _validate_integer(polyorder, "polyorder", 0)
    if window_days % 2 == 0:
        raise ValueError("window_days must be odd.")
    if polyorder >= window_days:
        raise ValueError("polyorder must be less than window_days.")
    if window_days > len(series):
        raise ValueError("window_days must not exceed the series length.")

    out = series.astype(float).copy(deep=True)
    out.iloc[:] = savgol_filter(values, window_days, polyorder, mode="interp")
    out.attrs["smoothing"] = {
        "method": "savgol",
        "window_days": int(window_days),
        "polyorder": int(polyorder),
        "mode": "interp",
    }
    return out


def decompose_stl(
    series: pd.Series,
    ns: int = DEFAULT_NS,
    period: int = DEFAULT_PERIOD,
    inner_iter: int = DEFAULT_INNER_ITER,
    outer_iter: int = DEFAULT_OUTER_ITER,
) -> pd.DataFrame:
    """
    Decompose a complete daily VI series into STL components.

    Gap filling is intentionally kept separate. Use :func:`prepare_daily_series`
    first when working from irregular Sentinel-2 observations. Optionally call
    :func:`smooth_daily_series` next; no smoothing is applied implicitly.

    Returns
    -------
    pandas.DataFrame
        Columns: observed, trend, seasonal, remainder.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series index must be a pandas DatetimeIndex.")
    if series.empty:
        raise ValueError("series is empty.")

    if series.index.hasnans or not series.index.is_unique:
        raise ValueError("series index must contain unique dates without NaT.")
    expected = pd.date_range(series.index[0], periods=len(series), freq="1D")
    if (
        not series.index.equals(expected)
        or not series.index.equals(series.index.normalize())
    ):
        raise ValueError("series must have an increasing, complete daily index at midnight.")

    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "series contains NaN or infinite values. "
            "Prepare/gap-fill the series before STL decomposition."
        )

    params = stl_parameters(
        ns=ns,
        period=period,
        inner_iter=inner_iter,
        outer_iter=outer_iter,
    )

    model = STL(
        series,
        period=params["period"],
        seasonal=params["seasonal"],
        trend=params["trend"],
        low_pass=params["low_pass"],
        robust=outer_iter > 0,
    )

    fit = model.fit(
        inner_iter=inner_iter,
        outer_iter=outer_iter,
    )

    out = pd.DataFrame(
        {
            "observed": series,
            "trend": fit.trend,
            "seasonal": fit.seasonal,
            "remainder": fit.resid,
        },
        index=series.index,
    )

    out.attrs.update(series.attrs)
    out.attrs["stl_parameters"] = params
    return out
