"""Storage contract: actual disk round trips, georeferencing, and scientific closure."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from rasterio.crs import CRS
import xarray as xr

from s2_tools import io
from s2_tools.partition import isolate_olive_ndvi
from s2_tools.stl import decompose_stl, smooth_daily_series


class NetCDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range('2022-01-01', periods=365 * 3)
        t = np.arange(len(dates))
        daily = pd.Series(0.5 + 0.2 * np.cos(2 * np.pi * t / 365)
                          + 0.01 * np.sin(t), index=dates)
        smooth = smooth_daily_series(daily)
        fitted = isolate_olive_ndvi(decompose_stl(smooth), soil_ndvi=0.15, lambda_woody=0.2)
        coords = {'time': dates, 'y': [4415015., 4415005.],
                  'x': [571005.123456789, 571015.123456789, 571025.123456789]}
        cls.results = xr.Dataset(coords=coords, attrs=fitted.attrs)
        columns = dict(fitted.items())
        columns.update(ndvi_daily=daily, ndvi_smoothed=smooth,
                       smoothing_residual=daily - smooth)
        for name, series in columns.items():
            values = np.broadcast_to(np.asarray(series)[:, None, None], (len(t), 2, 3)).copy()
            values[:, 0, 0] = np.nan  # Outside the orchard.
            values[:, 1, 2] = np.nan  # Inside, but processing failed.
            cls.results[name] = (('time', 'y', 'x'), values)
            cls.results[name].attrs['units'] = '1'
        cls.results['processing_status'] = (('y', 'x'), np.array([[0, 1, 1], [1, 1, 2]], dtype='uint8'))
        cls.results.processing_status.attrs.update(
            flag_values=np.array([0, 1, 2], dtype='uint8'),
            flag_meanings='outside_orchard success failed')
        cls.results = cls.results.assign_coords(spatial_ref=0)
        cls.results.spatial_ref.attrs = {
            'spatial_ref': CRS.from_epsg(32610).to_wkt(),
            'GeoTransform': '571000.123456789 10 0 4415020 0 -10',
        }
        for var in cls.results.data_vars.values():
            var.encoding['grid_mapping'] = 'spatial_ref'
        cls.observations = cls.results.ndvi_daily.isel(time=[0, 7, 21, 40]).copy(deep=True)
        cls.observations = cls.observations.assign_coords(
            time=cls.observations.time + np.timedelta64(18, 'h'))
        cls.observations.values[1, 0, 1] = np.nan  # Cloud at one acquisition.
        cls.mask = xr.DataArray([[False, True, True], [True, True, True]],
                               dims=('y', 'x'), coords={k: cls.results[k] for k in ('y', 'x')})

    def build(self):
        return io.build_results_dataset(
            observations=self.observations, daily_results=self.results,
            orchard_mask=self.mask,
            metadata={'calibration': {'reflectance': True, 'offsets': [-0.1, 0.]},
                      'qa_exclusions': [0, 1, 3, 8, 9, 10, 11]})

    def test_round_trip_values_times_crs_settings_and_closure(self):
        original = self.build()
        before = original.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            path = io.save_netcdf(original, Path(directory) / 'orchard.nc')
            with io.open_netcdf(path) as restored:
                restored.load()
                self.assertEqual(restored.ndvi_olive.dtype, np.dtype('float32'))
                for name in original.data_vars:
                    np.testing.assert_allclose(restored[name], original[name], rtol=1e-6, atol=1e-7, equal_nan=True)
                for name in ('time', 'acquisition_time', 'x', 'y', 'spatial_ref'):
                    xr.testing.assert_equal(restored[name], original[name])
                self.assertEqual(restored.attrs, original.attrs)
                self.assertEqual(restored.spatial_ref.attrs, original.spatial_ref.attrs)
                self.assertEqual(restored.ndvi_olive.encoding['grid_mapping'], 'spatial_ref')
                self.assertEqual(restored.orchard_mask.dtype, np.dtype('bool'))
                self.assertEqual(restored.orchard_mask.encoding['grid_mapping'], 'spatial_ref')
                self.assertEqual(restored.processing_status.dtype, np.dtype('uint8'))
                np.testing.assert_array_equal(restored.processing_status.attrs['flag_values'], [0, 1, 2])
                np.testing.assert_allclose(restored.ndvi_olive_background + restored.ndvi_cover + restored.partition_residual,
                                           restored.ndvi_smoothed, rtol=1e-6, atol=1e-7, equal_nan=True)
                np.testing.assert_allclose(restored.ndvi_daily - restored.ndvi_smoothed,
                                           restored.smoothing_residual, rtol=1e-5, atol=1e-7, equal_nan=True)
            xr.testing.assert_identical(original, before)

    def test_float64_and_dask_and_compression(self):
        import h5netcdf
        original = self.build().chunk({'time': 100})
        with tempfile.TemporaryDirectory() as directory:
            path = io.save_netcdf(original, Path(directory) / 'test.nc',
                                  dtype='float64', chunks={'time': 90, 'y': 1, 'x': 2})
            with io.open_netcdf(path, chunks={}) as restored:
                self.assertIsNotNone(restored.ndvi_olive.chunks)
                np.testing.assert_array_equal(restored.ndvi_olive, original.ndvi_olive)
            with h5netcdf.File(path) as raw:
                self.assertEqual(raw.variables['ndvi_olive'].compression, 'gzip')
                self.assertEqual(raw.variables['ndvi_olive'].compression_opts, 4)
                self.assertEqual(raw.variables['ndvi_olive'].chunks, (90, 1, 2))

    def test_rejects_grid_and_time_mismatch(self):
        for bad in (self.observations.assign_coords(x=self.observations.x + 1),
                    self.observations.isel(x=slice(1, None))):
            with self.assertRaisesRegex(ValueError, 'grid|coordinate'):
                io.build_results_dataset(observations=bad, daily_results=self.results,
                                         orchard_mask=self.mask)
        for times in ([0, 1, 1], [2, 1, 0]):
            with self.assertRaises(ValueError):
                io.build_results_dataset(observations=self.observations.isel(time=times),
                                         daily_results=self.results, orchard_mask=self.mask)
        bad_crs = self.observations.copy(deep=True)
        bad_crs.spatial_ref.attrs['GeoTransform'] = '0 20 0 0 0 -20'
        with self.assertRaisesRegex(ValueError, 'spatial_ref|CRS|transform'):
            io.build_results_dataset(observations=bad_crs, daily_results=self.results,
                                     orchard_mask=self.mask)

    def test_overwrite_and_failure_leave_original_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.nc'
            io.save_netcdf(self.build(), path)
            saved = path.read_bytes()
            with self.assertRaises(FileExistsError):
                io.save_netcdf(self.build(), path)
            with patch.object(xr.Dataset, 'to_netcdf', side_effect=RuntimeError('write failed')):
                with self.assertRaisesRegex(RuntimeError, 'write failed'):
                    io.save_netcdf(self.build(), path, overwrite=True)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            io.save_netcdf(self.build(), path, overwrite=True, dtype='float64')
            with io.open_netcdf(path) as restored:
                self.assertEqual(restored.ndvi_olive.dtype, np.dtype('float64'))

    def test_decoded_input_is_not_packed_again_and_can_be_resaved(self):
        original = self.build()
        original.ndvi_olive.encoding.update(dtype='int16', scale_factor=0.0001,
                                             add_offset=-0.1, _FillValue=-9999)
        original.attrs['note'] = '{"ordinary": "string"}'
        original.ndvi_olive.attrs['settings'] = {'enabled': True, 'optional': None}
        with tempfile.TemporaryDirectory() as directory:
            first = io.save_netcdf(original, Path(directory) / 'first.nc', dtype='float64')
            with io.open_netcdf(first) as reopened:
                second = io.save_netcdf(reopened, Path(directory) / 'second.nc', dtype='float64')
            with io.open_netcdf(second) as restored:
                np.testing.assert_array_equal(restored.ndvi_olive, original.ndvi_olive)
                self.assertEqual(restored.attrs['note'], original.attrs['note'])
                self.assertEqual(restored.ndvi_olive.attrs['settings'], original.ndvi_olive.attrs['settings'])
            self.assertEqual(original.ndvi_olive.encoding['scale_factor'], 0.0001)

    def test_builder_leaves_inputs_unchanged(self):
        before = self.results.copy(deep=True)
        observations = self.observations.copy(deep=True)
        mask = self.mask.copy(deep=True)
        self.build()
        xr.testing.assert_identical(self.results, before)
        xr.testing.assert_identical(self.observations, observations)
        xr.testing.assert_identical(self.mask, mask)


if __name__ == '__main__':
    unittest.main()
