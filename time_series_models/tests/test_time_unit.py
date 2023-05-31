import unittest

import numpy as np

from time_series_models.time_unit import TimeUnitEnum


class TimeUnitEnumTests(unittest.TestCase):
    def test_convert_to_year(self):
        test_array = np.arange(
            np.datetime64("2020-01"),
            np.datetime64("2022-02"),
            step=np.timedelta64(6, "M"),
        )
        result = TimeUnitEnum.YEAR.as_unit(test_array)
        np.testing.assert_array_equal(result, np.array((2020, 2020, 2021, 2021, 2022)))

    def test_convert_to_month(self):
        test_array = np.arange(
            np.datetime64("2019-12-01"),
            np.datetime64("2020-04-01"),
            step=np.timedelta64(15, "D"),
        )
        result = TimeUnitEnum.MONTH.as_unit(test_array)
        np.testing.assert_array_equal(result, np.array([12, 12, 12, 1, 1, 2, 2, 3, 3]))

    def test_convert_to_day(self):
        test_array = np.arange(
            np.datetime64("2019-12-28"),
            np.datetime64("2020-01-05"),
            step=np.timedelta64(12, "h"),
        )
        result = TimeUnitEnum.DAY.as_unit(test_array)
        np.testing.assert_array_equal(
            result, np.array([28, 28, 29, 29, 30, 30, 31, 31, 1, 1, 2, 2, 3, 3, 4, 4])
        )

    def test_convert_to_leap_day(self):
        test_array = np.arange(
            np.datetime64("2020-02-27"),
            np.datetime64("2020-03-02"),
            step=np.timedelta64(12, "h"),
        )
        result = TimeUnitEnum.DAY.as_unit(test_array)
        np.testing.assert_array_equal(result, np.array([27, 27, 28, 28, 29, 29, 1, 1]))

    def test_convert_to_hour(self):
        test_array = np.arange(
            np.datetime64("2020-02-29 20:00:00"),
            np.datetime64("2020-03-01 04:00:00"),
            step=np.timedelta64(120, "m"),
        )
        result = TimeUnitEnum.HOUR.as_unit(test_array)
        np.testing.assert_array_equal(result, np.array([20, 22, 0, 2]))

    def test_convert_to_minute(self):
        test_array = np.arange(
            np.datetime64("2019-12-31 23:50:00"),
            np.datetime64("2020-01-01 00:10:00"),
            step=np.timedelta64(120, "s"),
        )
        result = TimeUnitEnum.MINUTE.as_unit(test_array)
        np.testing.assert_array_equal(
            result, np.array([50, 52, 54, 56, 58, 00, 2, 4, 6, 8])
        )

    def test_convert_to_second(self):
        test_array = np.arange(
            np.datetime64("2019-12-31 23:59:58"),
            np.datetime64("2020-01-01 00:00:02"),
            step=np.timedelta64(1_000_000, "us"),
        )
        result = TimeUnitEnum.SECOND.as_unit(test_array)
        np.testing.assert_array_equal(result, np.array([58, 59, 0, 1]))

    def test_convert_to_microsecond(self):
        test_array = np.arange(
            np.datetime64("2019-12-31 23:59:59.5"),
            np.datetime64("2020-01-01 00:00:00.4"),
            step=np.timedelta64(100_000, "us"),
        )
        result = TimeUnitEnum.MICROSECOND.as_unit(test_array)
        np.testing.assert_array_equal(
            result, np.array([v * 100_000 for v in (5, 6, 7, 8, 9, 0, 1, 2, 3)])
        )

    def test_convert_to_int_day_of_week(self):
        test_array = np.arange(
            np.datetime64("2021-01-01"),
            np.datetime64("2021-01-12"),
            step=np.timedelta64(24, "h"),
        )
        result = TimeUnitEnum.INT_DAY_OF_WEEK.as_unit(test_array)
        np.testing.assert_array_equal(
            result, np.array([4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0])
        )

    def test_convert_to_day_of_week(self):
        test_array = np.arange(
            np.datetime64("2021-01-01"),
            np.datetime64("2021-01-12"),
            step=np.timedelta64(24, "h"),
        )
        result = TimeUnitEnum.DAY_OF_WEEK.as_unit(test_array)
        np.testing.assert_array_equal(
            result,
            np.array(
                [
                    "Friday",
                    "Saturday",
                    "Sunday",
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                    "Monday",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
