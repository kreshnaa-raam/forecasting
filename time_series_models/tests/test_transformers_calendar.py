import unittest
import numpy as np

from time_series_models.transformers_calendar import (
    delta_value,
    harmonic,
    harmonic_transform_pipeline,
    is_business_day,
    is_business_day_pipeline,
    one_hot_encode_day_of_week_pipeline,
)


class TestCalendarTransformers(unittest.TestCase):
    def test_delta_value__daily_timeseries(self):
        x = np.arange(
            np.datetime64("2021-07-01"),
            np.datetime64("2021-07-08"),
            step=np.timedelta64(1, "D"),
        ).reshape(-1, 1)
        result = delta_value(x)
        expected = np.arange(0, 7, dtype=np.int64).reshape(-1, 1)
        np.testing.assert_array_equal(result, expected)

        result = delta_value(x, zero_value=np.datetime64("2021-06-01"))
        expected = np.arange(0, 7, dtype=np.int64).reshape(-1, 1) + 30
        np.testing.assert_array_equal(result, expected)

    def test_delta_value__ms_timeseries(self):
        x = np.arange(
            np.datetime64("2021-07-01T12:34:56.000"),
            np.datetime64("2021-07-01T12:34:58.100"),
            step=np.timedelta64(300, "ms"),
        ).reshape(-1, 1)
        result = delta_value(x)
        expected = np.arange(0, 2100, 300, dtype=np.int64).reshape(-1, 1)
        np.testing.assert_array_equal(result, expected)

        result = delta_value(x, zero_value=np.datetime64("2021-07-01"))
        expected = np.arange(0, 2100, 300, dtype=np.int64).reshape(-1, 1) + 45296000
        np.testing.assert_array_equal(result, expected)

    def test_delta_value_float64(self):
        x = np.arange(12.0, 24.0, step=3.0, dtype=np.float64).reshape(-1, 1)
        result = delta_value(x)
        expected = np.arange(0.0, 12.0, 3.0, dtype=np.float64).reshape(-1, 1)
        np.testing.assert_array_equal(result, expected)

        result = delta_value(x, zero_value=np.pi + 12.0)
        expected = np.arange(0.0, 12.0, 3.0, dtype=np.float64).reshape(-1, 1) - np.pi
        np.testing.assert_array_equal(result, expected)

    @unittest.skip("not implemented yet!")
    def test_harmonic(self):
        pass

    @unittest.skip("not implemented yet!")
    def test_harmonic_transform_pipeline(self):
        pass

    def test_is_business_day(self):
        x = np.arange(
            np.datetime64("2021-07-01"),
            np.datetime64("2021-07-08"),
            step=np.timedelta64(1, "D"),
        ).reshape(-1, 1)

        result = is_business_day(x)

        expected = np.array(
            [
                1,
                1,
                0,
                0,
                0,
                1,
                1,
            ]
        ).reshape(-1, 1)
        np.testing.assert_equal(result, expected)

    @unittest.skip("not implemented yet!")
    def test_is_business_day_pipeline(self):
        pass

    @unittest.skip("not implemented yet!")
    def test_one_hot_encode_day_of_week_pipeline(self):
        pass


if __name__ == "__main__":
    unittest.main()
