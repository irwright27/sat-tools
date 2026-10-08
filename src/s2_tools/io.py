"""Build and store gridded Sentinel-2 results as compressed NetCDF4.

Saving computes lazy arrays but does not modify the input dataset. Opening is
lazy; use the returned Dataset as a context manager, or call its close method.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

_JSON_ATTRS = '_s2_tools_json_attrs'
_BOOL = '_s2_tools_bool'


def _check_time(obj, name, *, daily=False):
    if name not in obj.coords or obj[name].dims != (name,):
        raise ValueError(f'Missing one-dimensional {name} coordinate.')
    values = obj[name].values
    if not len(values) or not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError(f'{name} must contain timezone-naive datetime64 values in UTC.')
    if np.isnat(values).any() or (np.diff(values) <= np.timedelta64(0, 'ns')).any():
        raise ValueError(f'{name} must be finite, unique, and increasing.')
    if daily and (np.diff(values) != np.timedelta64(1, 'D')).any():
        raise ValueError('Daily time coordinate must have one-day spacing.')


def _check_grid(reference, other):
    for dim in ('x', 'y'):
        if dim not in other.coords or other[dim].dims != (dim,):
            raise ValueError(f'Missing one-dimensional grid coordinate {dim}.')
        if not np.array_equal(reference[dim].values, other[dim].values):
            raise ValueError(f'Grid coordinate {dim} differs; reproject before building results.')
    if 'spatial_ref' in other.coords:
        try:
            xr.testing.assert_identical(reference.spatial_ref, other.spatial_ref)
        except AssertionError as error:
            raise ValueError('Conflicting spatial_ref CRS or transform.') from error


def build_results_dataset(*, observations: xr.DataArray, daily_results: xr.Dataset,
                          orchard_mask: xr.DataArray,
                          metadata: Mapping | None = None) -> xr.Dataset:
    """Combine results without resampling or changing either time axis.

    observations must have (time, y, x) or (acquisition_time, y, x) dimensions.
    daily_results must contain ndvi_daily, ndvi_smoothed, ndvi_olive, ndvi_cover
    on (time, y, x), and a scalar spatial_ref coordinate containing CRS WKT.
    Other daily diagnostics and a (y, x) processing_status can be included.
    orchard_mask must be boolean on (y, x). All spatial coordinates must match
    exactly. Inputs without their own spatial_ref inherit the reference grid;
    callers must ensure those inputs were produced on that grid.

    Metadata is stored as global attributes. Nested JSON-compatible settings
    (including NumPy scalars/arrays) are supported by save_netcdf. Datetime
    coordinates must already be timezone-naive UTC. No per-pixel STL is run.
    """
    if not isinstance(daily_results, xr.Dataset):
        raise TypeError('daily_results must be an xarray Dataset.')
    if not isinstance(observations, xr.DataArray) or not isinstance(orchard_mask, xr.DataArray):
        raise TypeError('observations and orchard_mask must be xarray DataArrays.')
    required = {'ndvi_daily', 'ndvi_smoothed', 'ndvi_olive', 'ndvi_cover'}
    if required.difference(daily_results):
        raise ValueError(f'Missing daily variables: {sorted(required.difference(daily_results))}')
    for name in required:
        if set(daily_results[name].dims) != {'time', 'y', 'x'}:
            raise ValueError(f'{name} must have time, y, x dimensions.')
    _check_time(daily_results, 'time', daily=True)
    for dim in ('x', 'y'):
        if dim not in daily_results.coords or daily_results[dim].dims != (dim,):
            raise ValueError(f'Missing grid coordinate {dim}.')
        values = daily_results[dim].values
        if not len(values) or not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise ValueError(f'Grid coordinate {dim} must be nonempty, finite, and unique.')
    if ('spatial_ref' not in daily_results.coords or daily_results.spatial_ref.ndim != 0
            or not any(k in daily_results.spatial_ref.attrs for k in ('spatial_ref', 'crs_wkt'))):
        raise ValueError('daily_results needs a scalar spatial_ref coordinate with CRS WKT.')
    if 'acquisition_time' in daily_results.dims or {'ndvi_observed', 'orchard_mask'} & set(daily_results):
        raise ValueError('daily_results contains reserved observation or mask names.')
    acquisition = 'acquisition_time' if 'acquisition_time' in observations.dims else 'time'
    if set(observations.dims) != {acquisition, 'y', 'x'}:
        raise ValueError('observations must have acquisition time, y, x dimensions.')
    _check_time(observations, acquisition)
    if set(orchard_mask.dims) != {'y', 'x'} or orchard_mask.dtype != np.dtype(bool):
        raise ValueError('orchard_mask must be boolean with y, x dimensions.')
    _check_grid(daily_results, observations)
    _check_grid(daily_results, orchard_mask)
    obs = observations.rename({acquisition: 'acquisition_time'}) if acquisition == 'time' else observations
    out = xr.merge([daily_results, obs.rename('ndvi_observed'), orchard_mask.rename('orchard_mask')],
                   join='exact', compat='identical', combine_attrs='override')
    out.attrs = dict(daily_results.attrs)
    if metadata:
        if set(metadata) & set(out.attrs):
            raise ValueError('Metadata would overwrite existing processing settings.')
        out.attrs.update(metadata)
    for variable in out.data_vars.values():
        variable.attrs = dict(variable.attrs)
        variable.encoding = dict(variable.encoding)
        if {'x', 'y'}.issubset(variable.dims):
            variable.attrs.pop('grid_mapping', None)
            variable.encoding['grid_mapping'] = 'spatial_ref'
    return out


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}')


def _encode_attrs(attrs):
    if {_JSON_ATTRS, _BOOL} & set(attrs):
        raise ValueError('Attributes use a reserved s2-tools serialization name.')
    result, encoded = {}, []
    for key, value in attrs.items():
        if not isinstance(key, str):
            raise TypeError('Attribute names must be strings.')
        if value is None or isinstance(value, (Mapping, list, tuple, bool, np.bool_)):
            result[key] = json.dumps(value, default=_json_default)
            encoded.append(key)
        else:
            result[key] = value
    if encoded:
        result[_JSON_ATTRS] = json.dumps(encoded)
    return result


def _decode_attrs(attrs):
    result = dict(attrs)
    for key in json.loads(result.pop(_JSON_ATTRS, '[]')):
        result[key] = json.loads(result[key])
    return result


def save_netcdf(dataset: xr.Dataset, path: str | os.PathLike, *,
                dtype: str = 'float32', compression: int = 4,
                chunks: Mapping[str, int] | None = None,
                overwrite: bool = False) -> Path:
    """Write a complete dataset atomically, returning its path.

    Floating data variables use float32 by default (small rounding error), or
    float64. Coordinates retain their precision; integer flags and boolean masks
    retain their meaning. Compression is lossless gzip, level 0-9. Chunk lengths
    default to 128 for time axes and 64 for spatial axes, capped to actual sizes.
    Custom chunks maps dimension names to positive lengths. Lazy inputs compute
    during the write; this is a single-writer operation, not an append interface.

    Dictionary/list/tuple/bool/None attributes are JSON encoded and restored by
    open_netcdf (tuples and NumPy arrays inside JSON become lists). Standard CRS
    attributes remain readable by other NetCDF readers. Stale packing encodings
    are discarded so already-decoded data are not scaled again. The destination
    directory must exist; a failed write never replaces an existing file.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError('dataset must be an xarray Dataset.')
    if dtype not in ('float32', 'float64'):
        raise ValueError('dtype must be float32 or float64.')
    if isinstance(compression, bool) or not isinstance(compression, int) or not 0 <= compression <= 9:
        raise ValueError('compression must be an integer from 0 to 9.')
    chunks = dict(chunks or {})
    for dim, size in chunks.items():
        if dim not in dataset.dims or isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('chunks must map existing dimensions to positive integers.')
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    out = dataset.copy(deep=False)
    out.attrs = _encode_attrs(dataset.attrs)
    encoding = {}
    for name in out.variables:
        source = dataset[name]
        out[name].attrs = _encode_attrs(source.attrs)
        # Retain CF coordinate relationships, but not source file packing/chunks.
        out[name].encoding = {k: v for k, v in source.encoding.items()
                              if k in ('grid_mapping', 'coordinates', 'bounds')}
        spec = dict(out[name].encoding)
        if source.dtype.kind == 'b':
            out[name] = out[name].astype('int8')
            out[name].attrs[_BOOL] = 1
        if name in out.data_vars and source.dtype.kind == 'f':
            spec['dtype'] = dtype
        if source.dtype.kind in 'biu':
            spec['_FillValue'] = None
        if source.ndim and source.size and source.dtype.kind in 'biuf':
            spec.update(zlib=compression > 0, complevel=compression, shuffle=True)
            spec['chunksizes'] = tuple(min(source.sizes[d], chunks.get(
                d, 128 if d in ('time', 'acquisition_time') else 64)) for d in source.dims)
        encoding[name] = spec
    handle, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp.nc', dir=path.parent)
    os.close(handle)
    try:
        out.to_netcdf(temporary, engine='h5netcdf', format='NETCDF4', encoding=encoding)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Unlike replace, link fails if another writer created the target.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def open_netcdf(path: str | os.PathLike, *, chunks=None) -> xr.Dataset:
    """Open lazily, restoring JSON settings, boolean masks, and CF grid mappings.

    Pass chunks={} for Dask-backed arrays using the on-disk chunks. Close the
    returned dataset when finished, preferably with a with statement.
    """
    source = xr.open_dataset(path, engine='h5netcdf', decode_coords='all', chunks=chunks)
    try:
        out = source.copy(deep=False)
        out.attrs = _decode_attrs(source.attrs)
        for name in list(out.variables):
            out[name].attrs = _decode_attrs(source[name].attrs)
            if out[name].attrs.pop(_BOOL, False):
                encoding = dict(out[name].encoding)
                out[name] = out[name].astype(bool)
                out[name].encoding = encoding
        out.set_close(source.close)
        return out
    except Exception:
        source.close()
        raise
