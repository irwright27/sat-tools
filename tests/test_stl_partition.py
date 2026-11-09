"""Regression checks for daily preparation, STL constraints, and proxy bookkeeping."""
import unittest

import numpy as np
import pandas as pd

from s2_tools.partition import isolate_olive_ndvi
from s2_tools.stl import decompose_stl, prepare_daily_series, stl_parameters


class STLPartitionTests(unittest.TestCase):
    def test_same_day_mean_and_linear_interpolation(self):
        raw = pd.Series([0.6, 0.2, 0.8], index=pd.to_datetime([
            "2024-01-01 18:00Z", "2024-01-01 10:00Z", "2024-01-03 12:00Z"
        ]))
        daily = prepare_daily_series(raw)
        np.testing.assert_allclose(daily, [0.4, 0.6, 0.8])
        self.assertEqual(str(daily.index.tz), "UTC")
        self.assertTrue(np.isnan(prepare_daily_series(raw, interpolate=False).iloc[1]))

    def test_invalid_input_and_parameters(self):
        for kwargs in [{"ns": 8}, {"ns": 7.0}, {"period": True}, {"inner_iter": 1.5}, {"outer_iter": -1}]:
            with self.assertRaises(ValueError):
                stl_parameters(**kwargs)
        irregular = pd.Series([0.3, 0.4], index=pd.to_datetime(["2024-01-01", "2024-01-03"]))
        with self.assertRaises(ValueError):
            decompose_stl(irregular)
        for values in [[np.nan, np.nan], [0.3, np.inf]]:
            with self.assertRaises(ValueError):
                prepare_daily_series(pd.Series(values, index=pd.date_range("2024-01-01", periods=2)))
        with self.assertRaises(ValueError):
            isolate_olive_ndvi(pd.DataFrame(columns=["observed", "trend", "seasonal", "remainder"]), soil_ndvi=0.15, lambda_woody=0.2)

    def test_candidates_and_closure(self):
        dates = pd.date_range("2021-01-01", "2025-12-31")
        values = 0.4 + 0.1 * np.cos(2 * np.pi * np.arange(len(dates)) / 365)
        values += np.where(dates < "2022-01-01", 0.2, 0.0)
        series = pd.Series(values, index=dates)
        for ns in [7, 15, 31, 61, 91]:
            result = decompose_stl(series, ns=ns)
            params = result.attrs["stl_parameters"]
            self.assertEqual(params["low_pass"], 367)
            self.assertEqual(params["trend"] % 2, 1)
            minimum = 1.5 * 365 / (1 - 1.5 / ns)
            self.assertGreaterEqual(params["trend"], minimum)
            self.assertLess(params["trend"] - 2, minimum)
            proxy = isolate_olive_ndvi(result, soil_ndvi=0.15, lambda_woody=0.2)
            np.testing.assert_allclose(result[["trend", "seasonal", "remainder"]].sum(axis=1), series, atol=1e-12)
            np.testing.assert_allclose(proxy["ndvi_olive_background"] + proxy["ndvi_cover"] + proxy["partition_residual"], series, atol=1e-12)
            self.assertTrue(proxy["S"].between(0, 1).all())

    def test_equation_11_and_residual(self):
        frame = pd.DataFrame({"observed": [0.3, 1.3, 0.2], "trend": [0.4]*3,
                              "seasonal": [-0.1, 0.2, 0.0], "remainder": [0.0, 0.7, -0.2]})
        proxy = isolate_olive_ndvi(frame, soil_ndvi=0.15, lambda_woody=0.2)
        np.testing.assert_allclose(proxy.seasonal_adjusted, [-0.1, 0.9, 0])
        np.testing.assert_allclose(proxy.ndvi_olive, (1 + 0.2*proxy.S)*(proxy.ndvi_base-0.15))
        np.testing.assert_allclose(proxy.ndvi_cover, proxy.S*(proxy.amplitude-0.2*(proxy.ndvi_base-0.15)))
        np.testing.assert_allclose(proxy.ndvi_olive_background + proxy.ndvi_cover, proxy.ndvi_modeled)
        np.testing.assert_allclose(proxy.partition_residual, [0, 0, -0.2], atol=1e-12)
        np.testing.assert_allclose(proxy.ndvi_base, frame.trend + proxy.trough)
        self.assertNotIn("w_olive", proxy)

    def test_zero_amplitude_and_no_clipping(self):
        frame = pd.DataFrame({"observed": [1.2]*3, "trend": [1.2]*3,
                              "seasonal": [0.0]*3, "remainder": [0.0]*3})
        proxy = isolate_olive_ndvi(frame, soil_ndvi=0.15, lambda_woody=0.2)
        np.testing.assert_allclose(proxy.ndvi_olive, 1.05)
        np.testing.assert_allclose(proxy.ndvi_olive_background, 1.2)
        np.testing.assert_allclose(proxy[["S", "s", "amplitude", "ndvi_cover"]], 0)

    def test_recursive_hand_calculation(self):
        from s2_tools.partition import _recursive_envelopes
        # p=0.5 gives coefficients a=b=1/3. Three literal Eq. 14 passes.
        low, high = _recursive_envelopes(np.array([-1., 1., -1.]), 0.5)
        np.testing.assert_allclose(low, [-1, -4/9, -1])
        np.testing.assert_allclose(high, [0, 1, 1/3])

    def test_envelopes_adapt_and_bound_signal(self):
        from s2_tools.partition import _recursive_envelopes, DEFAULT_ENVELOPE_P
        t = np.arange(365*12)
        signal = np.where(t < 365*4, 0.3, 0.1)*np.cos(2*np.pi*t/365)
        low, high = _recursive_envelopes(signal, DEFAULT_ENVELOPE_P)
        self.assertTrue(np.all(low <= signal))
        self.assertTrue(np.all(high >= signal))
        self.assertGreater(np.mean((high-low)[365:365*3]), np.mean((high-low)[365*9:365*11]))
        a = DEFAULT_ENVELOPE_P/(2-DEFAULT_ENVELOPE_P)
        self.assertAlmostEqual(a**10, 0.99/1.01)

    def test_calibration_validation(self):
        frame = pd.DataFrame({"observed": [0.4], "trend": [0.4], "seasonal": [0.], "remainder": [0.]})
        for name, values in {"soil_ndvi": [np.nan, 2, True], "lambda_woody": [-1, np.inf, True],
                             "envelope_p": [0, 1, np.nan]}.items():
            for value in values:
                args = dict(soil_ndvi=0.15, lambda_woody=0.2)
                args[name] = value
                with self.assertRaises(ValueError):
                    isolate_olive_ndvi(frame, **args)
        with self.assertRaises(TypeError):
            isolate_olive_ndvi(frame)


if __name__ == "__main__":
    unittest.main()
