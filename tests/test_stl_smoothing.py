"""Scientific and input-contract checks for daily smoothing."""
import unittest

import numpy as np
import pandas as pd

from s2_tools.stl import smooth_daily_series, decompose_stl
from s2_tools.partition import isolate_olive_ndvi


class SmoothingTests(unittest.TestCase):
    def series(self, values):
        return pd.Series(values, index=pd.date_range('2022-01-01', periods=len(values), tz='UTC'), name='NDVI')

    def test_preserves_polynomials_and_input(self):
        t = np.linspace(-1, 1, 101)
        for values in [np.full(101, 0.4), 0.4 + 0.1*t + 0.05*t*t]:
            raw = self.series(values)
            raw.attrs['units'] = '1'
            before = raw.copy(deep=True)
            smooth = smooth_daily_series(raw)
            np.testing.assert_allclose(smooth, raw, atol=1e-12)
            pd.testing.assert_series_equal(raw, before)
            self.assertNotIn('smoothing', raw.attrs)
            self.assertEqual(smooth.attrs['units'], '1')
            self.assertEqual(smooth.name, raw.name)
            self.assertTrue(smooth.index.equals(raw.index))

    def test_noise_reduction_and_partition_provenance(self):
        t = np.arange(365*3)
        truth = 0.5 + 0.15*np.cos(2*np.pi*t/365)
        raw = self.series(truth + np.random.default_rng(42).normal(0, 0.02, len(t)))
        smooth = smooth_daily_series(raw)
        self.assertLess(np.mean((smooth-truth)**2), np.mean((raw-truth)**2))
        result = isolate_olive_ndvi(decompose_stl(smooth), soil_ndvi=0.15, lambda_woody=0.2)
        np.testing.assert_allclose(result.ndvi_olive_background + result.ndvi_cover + result.partition_residual, smooth, atol=1e-12)
        self.assertEqual(result.attrs['smoothing'], smooth.attrs['smoothing'])

    def test_invalid_windows(self):
        raw = self.series(np.full(61, 0.5))
        for kwargs in [dict(window_days=30), dict(window_days=63), dict(window_days=True),
                       dict(window_days=31.0), dict(polyorder=-1), dict(polyorder=31),
                       dict(polyorder=True), dict(polyorder=2.5)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                smooth_daily_series(raw, **kwargs)

    def test_invalid_series(self):
        raw = self.series(np.full(61, 0.5))
        bad = [raw.iloc[:0], raw.drop(raw.index[20]), raw.iloc[::-1],
               pd.concat([raw, raw.iloc[-1:]])]
        shifted = raw.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=1)
        bad.append(shifted)
        for value in [np.nan, np.inf]:
            nonfinite = raw.copy()
            nonfinite.iloc[20] = value
            bad.append(nonfinite)
        for series in bad:
            with self.assertRaises(ValueError):
                smooth_daily_series(series)
        with self.assertRaises(TypeError):
            smooth_daily_series([0.5]*61)
        with self.assertRaises(TypeError):
            smooth_daily_series(pd.Series([0.5]*61))


if __name__ == '__main__':
    unittest.main()
