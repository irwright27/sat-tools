from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


# Match the decay time of Eq. 14's p=0.99 on 10-day data at daily cadence.
_DAILY_DECAY = (0.99 / (2.0 - 0.99)) ** 0.1
DEFAULT_ENVELOPE_P = 2.0 * _DAILY_DECAY / (1.0 + _DAILY_DECAY)


def _recursive_envelopes(values: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 14, forward/backward/forward, seeded with each pass's endpoint.

    Keep the published coefficients literally: they do not sum to one.
    Each envelope feeds its own preceding pass into the next pass.
    """
    low, high = values.copy(), values.copy()
    a, b = p / (2.0 - p), (1.0 - p) / (2.0 - p)
    for reverse in (False, True, False):
        if reverse:
            low, high = low[::-1].copy(), high[::-1].copy()
        for i in range(1, len(values)):
            low[i] = min(low[i], a * low[i - 1] + b * low[i])
            high[i] = max(high[i], a * high[i - 1] + b * high[i])
        if reverse:
            low, high = low[::-1].copy(), high[::-1].copy()
    return low, high


def isolate_olive_ndvi(
    stl_result: pd.DataFrame,
    *,
    soil_ndvi: float,
    lambda_woody: float,
    envelope_p: float = DEFAULT_ENVELOPE_P,
) -> pd.DataFrame:
    """Partition daily STL results using Lu et al. (2003), Eqs. 11, 13-15.

    ``soil_ndvi`` (d) and ``lambda_woody`` (lambda) are required, finite scalar
    calibrations for this series, e.g. from UAV measurements. Lambda is >= 0
    and measures woody seasonal variation relative to the woody baseline;
    soil NDVI must be in [-1, 1]. No calibration defaults are assumed.

    Positive remainder is retained in seasonal_adjusted (Eq. 13). Recursive
    trough/peak envelopes use Eq. 14's literal coefficients in three passes:
    forward, backward, forward. Each pass is seeded at its first input value.
    ``envelope_p`` must be in (0, 1), and is per sample. The daily default
    preserves the decay time of the paper's p=0.99 at 10-day cadence. Datetime
    indexes must be complete daily sequences; other indexes are treated as
    ordered daily samples. The backward pass makes this an offline filter.

    With A=peak-trough, S=(seasonal_adjusted-trough)/A and s=-trough/A,
    B=trend-s*A=trend+trough. Using the local trough identity from Eq. 15
    avoids a whole-record mean S as the envelopes evolve. For A=0, S=s=0;
    B remains trend+trough (also defined for a constant nonzero component).

    ndvi_olive = (1 + lambda*S)*(B-d)              [woody only]
    ndvi_olive_background = ndvi_olive + d        [woody plus soil]
    ndvi_cover = S*(A-lambda*(B-d))               [herbaceous]
    ndvi_modeled = trend + seasonal_adjusted
    partition_residual = observed - ndvi_modeled

    Olive + soil + cover closes to the modeled signal, not necessarily the
    input observations. Outputs are not clipped, including negative cover
    estimates when calibrations are incompatible with the inferred amplitude.
    These are model contributions to mixed-pixel NDVI, not canopy-only NDVI
    measurements. NDVI is not physically additive between vegetation classes.

    Returns a copy with the components above, ndvi_soil, ndvi_base, and the
    diagnostics seasonal_adjusted, trough, peak, amplitude, S and s. Input
    metadata is preserved and calibration/filter settings are recorded.
    """
    if not isinstance(stl_result, pd.DataFrame):
        raise TypeError("stl_result must be a pandas DataFrame.")
    if stl_result.empty:
        raise ValueError("stl_result is empty.")
    required = {"observed", "trend", "seasonal", "remainder"}
    missing = required.difference(stl_result.columns)
    if missing:
        raise ValueError(f"stl_result is missing required columns: {sorted(missing)}")
    for name, value in (("soil_ndvi", soil_ndvi), ("lambda_woody", lambda_woody),
                        ("envelope_p", envelope_p)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar.")
    if not -1 <= soil_ndvi <= 1:
        raise ValueError("soil_ndvi must be in [-1, 1].")
    if lambda_woody < 0:
        raise ValueError("lambda_woody must be >= 0.")
    if not 0 < envelope_p < 1:
        raise ValueError("envelope_p must be in (0, 1).")
    if not np.isfinite(stl_result[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("stl_result contains NaN or infinite values.")
    index = stl_result.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("stl_result index must be unique and increasing.")
    if isinstance(index, pd.DatetimeIndex):
        if index.hasnans or not index.equals(pd.date_range(index[0], periods=len(index), freq="D")):
            raise ValueError("stl_result must have a complete daily index.")

    out = stl_result.copy()
    adjusted = (out["seasonal"] + out["remainder"].clip(lower=0)).to_numpy(dtype=float)
    trough, peak = _recursive_envelopes(adjusted, float(envelope_p))
    amplitude = peak - trough
    S = np.divide(adjusted - trough, amplitude, out=np.zeros_like(amplitude), where=amplitude > 0)
    s = np.divide(-trough, amplitude, out=np.zeros_like(amplitude), where=amplitude > 0)
    base = out["trend"] + trough
    olive = (1 + lambda_woody * S) * (base - soil_ndvi)
    modeled = out["trend"] + adjusted
    for name, value in {
        "seasonal_adjusted": adjusted, "trough": trough, "peak": peak,
        "amplitude": amplitude, "S": S, "s": s, "ndvi_base": base,
        "ndvi_olive": olive, "ndvi_soil": soil_ndvi,
        "ndvi_olive_background": olive + soil_ndvi,
        "ndvi_cover": S * (amplitude - lambda_woody * (base - soil_ndvi)),
        "ndvi_modeled": modeled, "partition_residual": out["observed"] - modeled,
    }.items():
        out[name] = value
    out.attrs["partition"] = {
        "method": "Lu et al. (2003), equations 11, 13-15",
        "soil_ndvi": float(soil_ndvi), "lambda_woody": float(lambda_woody),
        "envelope_p": float(envelope_p), "sample_days": 1,
        "envelope_passes": ["forward", "backward", "forward"],
        "envelope_initialization": "first input of each pass",
        "baseline": "trend + trough",
    }
    return out
