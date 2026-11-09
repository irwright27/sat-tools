# s2-tools

Python tools for processing and analyzing Sentinel-2 imagery and time series.

## Overview

`s2-tools` provides reusable workflows for Sentinel-2 data processing,
including raster operations, vegetation indices, time-series analysis,
and spectral decomposition.

The package is designed primarily for environmental and agricultural
remote-sensing applications.

## Installation

Clone the repository:

```bash
git clone git@github.com:YOUR-USERNAME/s2-tools.git
cd s2-tools
uv sync
```

## Saving gridded results

`s2_tools.io` saves compressed NetCDF4 files and reopens them with processing
settings restored. Install the updated dependencies before using it.

```python
from s2_tools import io

# Inputs are xarray objects already produced by your processing workflow.
dataset = io.build_results_dataset(
    observations=ndvi_observed,  # DataArray: time (actual acquisitions), y, x
    daily_results=pixel_results,  # Dataset: time (daily), y, x
    orchard_mask=orchard_mask,  # Boolean DataArray: y, x
    metadata={"qa_exclusions": [0, 1, 3, 8, 9, 10, 11]},
)
io.save_netcdf(dataset, "orchard_results.nc")

with io.open_netcdf("orchard_results.nc") as saved:
    olive = saved["ndvi_olive"].load()
```

`daily_results` must contain `ndvi_daily`, `ndvi_smoothed`, `ndvi_olive`, and
`ndvi_cover`. Include `smoothing_residual`, `trend`, `seasonal`, `remainder`,
and any other diagnostics you want to retain. An optional integer
`processing_status(y, x)` can distinguish outside-orchard pixels, successful
results, and failed processing; document its codes in `flag_values` and
`flag_meanings` attributes. Missing scientific values remain NaN.

The dataset must carry the original scalar `spatial_ref` coordinate with CRS
WKT and any `GeoTransform` metadata. Preserve these from the loaded imagery.
All inputs must use exactly the same x/y coordinates. Inputs carrying their own
`spatial_ref` must match; otherwise the builder assumes the reference grid.
Acquisition timestamps are stored on `acquisition_time`, separate from daily
`time`. Dates must be timezone-naive UTC, unique, and increasing. The builder
does not interpolate, reproject, or run per-pixel STL; the existing STL functions
return pandas objects that your processing workflow must assemble into a grid.

Save smoothing, STL, partition, calibration, and QA settings in dataset attributes
or pass them through `metadata`. Existing settings cannot be overwritten by
`metadata`. Nested settings are stored as JSON and restored by `open_netcdf`;
tuples and NumPy arrays within JSON reopen as lists. Other NetCDF readers can
read the variables and standard CRS attributes directly; boolean masks appear
as integer 0/1 and JSON settings remain strings in those readers.

By default, floating data variables are stored as float32 with lossless gzip
compression level 4. The float32 conversion introduces small rounding changes;
use `dtype="float64"` to retain double precision. Coordinate precision is retained.
Optional `chunks={"time": 128, "y": 32, "x": 32}` controls storage chunks;
`io.open_netcdf(path, chunks={})` uses Dask-backed arrays. Saving computes any lazy
inputs and leaves them unchanged. Existing files require `overwrite=True`, and
a failed write leaves the previous file intact. The output directory must exist.

Run the storage and processing tests with:

```bash
python -m unittest discover -s tests
```


## Woody, soil, and cover partition

`isolate_olive_ndvi` now implements Lu et al. (2003), equations 11 and 13-15.
The previous `gamma` anomaly-allocation rule has been removed. Supply measured
orchard-specific scalar calibrations from uav-tools; neither is defaulted:

```python
from s2_tools.partition import isolate_olive_ndvi

partitioned = isolate_olive_ndvi(
    stl_result,
    soil_ndvi=measured_soil_ndvi,
    lambda_woody=measured_lambda_woody,
)
```

`soil_ndvi` is the soil background d, and `lambda_woody` is the fractional
woody seasonal increase relative to its baseline (nonnegative). These must
represent the calibration quantities in the paper, rather than arbitrary
canopy NDVI measurements.

For B = trend + trough, A = peak - trough, and normalized phenology S:

- `ndvi_olive = (1 + lambda_woody*S)*(B - soil_ndvi)` is woody only.
- `ndvi_olive_background = ndvi_olive + soil_ndvi` includes background soil.
- `ndvi_cover = S*(A - lambda_woody*(B - soil_ndvi))` is herbaceous.
- `ndvi_modeled = trend + seasonal_adjusted` is the modeled signal.
- `partition_residual = observed - ndvi_modeled` retains the input difference.

Thus olive + soil + cover + partition residual reconstructs the input to STL.
Do not use the old olive + cover closure check. Save the new components and
partition metadata alongside the existing daily results. Negative components
are retained for diagnosis, not clipped. These remain mixed-pixel model
contributions, not spatially isolated canopy NDVI.

The seasonal adjustment retains positive STL remainder. Peaks and troughs now
follow the paper's literal Eq. 14 recursion, with forward/backward/forward
passes, each initialized from its first input value. The daily default for
`envelope_p` preserves the decay time of p=0.99 for the paper's 10-day samples:
a_daily = (0.99/1.01)**(1/10), p_daily = 2*a_daily/(1+a_daily).
The other coefficient remains (1-p)/(2-p), as printed in the paper; the two
coefficients are not normalized to sum to one. This cadence adjustment is an
adaptation of the paper, not a claim of identical daily and 10-day results.

`trough`, `peak`, `amplitude`, `S`, and `s` are time-varying columns. The baseline
uses Eq. 15's local identity s=-trough/A, hence B=trend+trough, rather than the
old whole-record mean S. At zero amplitude, S and s are set to zero and the
baseline remains defined. The backward pass uses future observations, so
endpoints and prior estimates can change when the record is extended.

The example notebook requires calibration values before it will partition;
old partition outputs have been cleared. Envelope filtering takes linear time
in the number of dates per pixel; the implementation processes one series per
call, as before.
