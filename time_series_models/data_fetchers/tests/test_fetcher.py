import unittest
from unittest.mock import patch, Mock

import numpy as np

from time_series_models.constants import LOCATION, DATE_TIME
from time_series_models.data_fetchers.fetcher import Fetcher
from time_series_models.exceptions import (
    EmptyFetchException,
    FetcherConfigurationError,
)
from time_series_models.transformers import make_domain


class TestFetcher(unittest.TestCase):
    def test_monthly_gcs_names(self):
        p_format = "year/month: {year:04d}/{month:02d}"

        start = np.datetime64("2021-01-03T00")
        end = np.datetime64("2021-01-03T23")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(p_format, start, end), ["year/month: 2021/01"]
        )

        # handle kwargs
        p_format_with_kwarg = (
            p_format + "; with kwargs foo={foo} and kwarg_count={kwarg_count}"
        )
        start = np.datetime64("2021-01-03T00")
        end = np.datetime64("2021-01-03T23")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(
                p_format_with_kwarg, start, end, foo="bar", kwarg_count=2
            ),
            ["year/month: 2021/01; with kwargs foo=bar and kwarg_count=2"],
        )

        # handle date with no time
        start = np.datetime64("2021-01-03")
        end = np.datetime64("2021-01-03")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(p_format, start, end), ["year/month: 2021/01"]
        )

        # even a one-hour interval should return a set of labels
        start = np.datetime64("2021-08-22T00")
        end = np.datetime64("2021-08-22T01")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(p_format, start, end), ["year/month: 2021/08"]
        )

        # Note: this method doesn't care whether "start" actually precedes "end"!
        start = np.datetime64("2021-08-22T01")
        end = np.datetime64("2021-08-22T00")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(p_format, start, end), ["year/month: 2021/08"]
        )

        # at start of month, tstart gets tipped into preceding month
        start = np.datetime64("2021-01-01T00")
        end = np.datetime64("2021-01-01T23")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(p_format, start, end),
            ["year/month: 2020/12", "year/month: 2021/01"],
        )

        # at end of month, tend gets tipped into next month
        start = np.datetime64("2021-01-30T00")
        end = np.datetime64("2021-01-31T23")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(p_format, start, end),
            ["year/month: 2021/01", "year/month: 2021/02"],
        )

        # starting somewhere in middle of a month and ending a year later in same month, so we should have 13 months
        start = np.datetime64("2020-10-15T10")
        end = np.datetime64("2021-10-08T02")
        self.assertListEqual(
            Fetcher.monthly_gcs_names(p_format, start, end),
            [
                "year/month: 2020/10",
                "year/month: 2020/11",
                "year/month: 2020/12",
                "year/month: 2021/01",
                "year/month: 2021/02",
                "year/month: 2021/03",
                "year/month: 2021/04",
                "year/month: 2021/05",
                "year/month: 2021/06",
                "year/month: 2021/07",
                "year/month: 2021/08",
                "year/month: 2021/09",
                "year/month: 2021/10",
            ],
        )

    def test_check_domain(self):
        dtype = np.dtype([(LOCATION, np.unicode_, 36), (DATE_TIME, np.dtype("<M8[h]"))])
        domain = np.empty([6, 1], dtype=dtype)

        domain[DATE_TIME] = np.arange(
            np.datetime64("2021-05-17T00:00:00"),
            np.datetime64("2021-05-17T06:00:00"),
            step=np.timedelta64(1, "h"),
        ).reshape(-1, 1)

        # properly arranged multi-location domain
        domain[LOCATION] = np.array(
            ("Zeno",) * 2 + ("Socrates",) * 2 + ("Arendt",) * 2, dtype="U36"
        ).reshape(-1, 1)
        Fetcher._check_domain(domain)  # should not raise RuntimeError

        # one location in domain
        domain[LOCATION] = np.array(("Many Zeno",) * 6, dtype="U36").reshape(-1, 1)
        Fetcher._check_domain(domain)  # should not raise RuntimeError

        # domain with improper interleaving
        domain[LOCATION] = np.array(
            (
                "Zeno",
                "Socrates",
                "Arendt",
            )
            * 2,
            dtype="U36",
        ).reshape(-1, 1)
        with self.assertRaises(RuntimeError):
            Fetcher._check_domain(domain)

        # edge case: another improper interleaving
        domain[LOCATION] = np.array(
            (
                "Duck",
                "Duck",
                "Duck",
                "Duck",
                "Goose",
                "Duck",
            ),
            dtype="U36",
        ).reshape(-1, 1)
        with self.assertRaises(RuntimeError):
            Fetcher._check_domain(domain)

    @patch.multiple(Fetcher, __abstractmethods__=set())
    def test_check_conflicting_variables(self):
        fetcher = Fetcher(None, None, None)
        fetcher._check_conflicting_variables()  # should not raise (no attributes to conflict)
        fetcher.selector_args = {
            "method": "transform",
            "variables": [1, 2, 3],
            "kwargs": {"func": {1: min, 2: [min, max]}},
        }
        fetcher._check_conflicting_variables()  # should not raise ("transform" ignores "kwargs")

        fetcher.selector_args["method"] = "aggregate"
        with self.assertRaises(FetcherConfigurationError):
            fetcher._check_conflicting_variables()
        del fetcher.selector_args["variables"]
        fetcher._check_conflicting_variables()  # should not raise

        # restore potentially offending variables
        fetcher.selector_args["variables"] = [1, 2, 3]
        with self.assertRaises(FetcherConfigurationError):
            fetcher._check_conflicting_variables()
        fetcher.selector_args["kwargs"]["func"] = np.median
        fetcher._check_conflicting_variables()  # should not raise
        fetcher.selector_args["kwargs"]["func"] = [min, "max"]
        fetcher._check_conflicting_variables()  # should not raise
        fetcher.selector_args["kwargs"]["func"] = dict()
        with self.assertRaises(FetcherConfigurationError):
            fetcher._check_conflicting_variables()

    def test_deconstruct_domain(self):
        domain = make_domain(
            "2005-02-01T10", "2005-02-01T19", np.timedelta64(1, "h"), "Alpaca"
        )
        result = Fetcher._deconstruct_domain(domain)
        self.assertTupleEqual(
            result,
            (np.datetime64("2005-02-01T10"), np.datetime64("2005-02-01T19"), "Alpaca"),
        )

        result = Fetcher._deconstruct_domain(domain, return_all_locs=True)
        self.assertTupleEqual(
            result,
            (np.datetime64("2005-02-01T10"), np.datetime64("2005-02-01T19"), "Alpaca"),
        )

    def test_extract_start_stop_loc__multi_loc(self):
        domain = make_domain(
            "2005-02-01T10",
            "2005-02-01T19",
            np.timedelta64(1, "h"),
            "Alpaca",
            "Bobcat",
            "Capybara",
        )
        result = Fetcher._deconstruct_domain(domain)
        self.assertTupleEqual(
            result,
            (np.datetime64("2005-02-01T10"), np.datetime64("2005-02-01T19"), "Alpaca"),
        )

        result = Fetcher._deconstruct_domain(domain, return_all_locs=True)
        self.assertTupleEqual(
            result,
            (
                np.datetime64("2005-02-01T10"),
                np.datetime64("2005-02-01T19"),
                "Alpaca",
                "Bobcat",
                "Capybara",
            ),
        )

    @patch.multiple(Fetcher, __abstractmethods__=set())
    def test_location_mapper_is_identity(self):
        domain = make_domain(
            "2021-01-01", "2023-01-01", np.timedelta64(1, "h"), "a", "b", "c"
        )

        # with identity mapping:
        fetcher = Fetcher(
            location_mapper=lambda x: x,
            selector="select",
            selector_args={},
        )
        result = fetcher._location_mapper_is_identity(domain)
        self.assertTrue(result)

        # with many-to-one mapping:
        def location_mapper(x):
            if x == "a":
                return "a"
            else:
                return "b"

        fetcher = Fetcher(
            location_mapper=location_mapper,
            selector="select",
            selector_args={},
        )
        result = fetcher._location_mapper_is_identity(domain)
        self.assertFalse(result)

        # note that a many-to-one mapper might still be effectively an identity mapper over a narrow domain
        domain = make_domain(
            "2021-01-01", "2023-01-01", np.timedelta64(1, "h"), "a", "b"
        )
        result = fetcher._location_mapper_is_identity(domain)
        self.assertTrue(result)

    def test_pad_hourly_datetime_array(self):
        dt_array = np.arange("2005-02-01T10", "2005-02-01T19", dtype="datetime64[h]")
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.arange("2005-02-01T09", "2005-02-01T20", dtype="datetime64[h]")
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(dt_array.dtype, result.dtype)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=0)
        np.testing.assert_array_equal(result, dt_array)
        self.assertEqual(dt_array.dtype, result.dtype)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=24)
        expected = np.arange("2005-01-31T10", "2005-02-02T19", dtype="datetime64[h]")
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(dt_array.dtype, result.dtype)

    def test_pad_hourly_datetime_array_generic_datetime(self):
        dt_array = np.arange("2005-02-01T10", "2005-02-01T19", dtype="datetime64")
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.arange("2005-02-01T09", "2005-02-01T20", dtype="datetime64")
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(dt_array.dtype, result.dtype)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=0)
        np.testing.assert_array_equal(result, dt_array)
        self.assertEqual(dt_array.dtype, result.dtype)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=24)
        expected = np.arange("2005-01-31T10", "2005-02-02T19", dtype="datetime64")
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(dt_array.dtype, result.dtype)

    def test_pad_daily_datetime_array(self):
        dt_array = np.arange("2005-02-01", "2005-02-02", dtype="datetime64[D]")
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.arange("2005-01-31", "2005-02-03", dtype="datetime64[D]")
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(dt_array.dtype, result.dtype)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=0)
        np.testing.assert_array_equal(result, dt_array)
        self.assertEqual(dt_array.dtype, result.dtype)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=3)
        expected = np.arange("2005-01-29", "2005-02-05", dtype="datetime64[D]")
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(dt_array.dtype, result.dtype)

    def test_pad_daily_datetime_array_generic_datetime(self):
        dt_array = np.arange("2005-02-01", "2005-02-02", dtype="datetime64")
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.arange("2005-01-31", "2005-02-03", dtype="datetime64")
        np.testing.assert_array_equal(result, expected)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=0)
        np.testing.assert_array_equal(result, dt_array)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=3)
        expected = np.arange("2005-01-29", "2005-02-05", dtype="datetime64")
        np.testing.assert_array_equal(result, expected)

    def test_pad_hourly_datetime_array_non_monotonic(self):
        dt_array = np.array(
            [
                "2005-02-01T10",
                "2005-02-01T11",
                "2005-02-01T12",
                "2005-02-01T13",
                "2005-02-01T12",
            ],
            dtype="datetime64[h]",
        )
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.array(
            [
                "2005-02-01T09",
                "2005-02-01T10",
                "2005-02-01T11",
                "2005-02-01T12",
                "2005-02-01T13",
                "2005-02-01T12",
                "2005-02-01T13",
            ],
            dtype="datetime64[h]",
        )
        np.testing.assert_array_equal(result, expected)

    def test_pad_daily_datetime_array_non_monotonic(self):
        dt_array = np.array(
            [
                "2005-02-04",
                "2005-02-02",
                "2005-02-03",
                "2005-02-04",
                "2005-02-01",
            ],
            dtype="datetime64[D]",
        )
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.array(
            [
                "2005-02-03",
                "2005-02-04",
                "2005-02-02",
                "2005-02-03",
                "2005-02-04",
                "2005-02-01",
                "2005-02-02",
            ],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(result, expected)

    def test_pad_hourly_datetime_ndarray(self):
        dt_array = np.array(
            [
                np.arange("2005-02-01T10", "2005-02-01T19", dtype="datetime64[h]"),
                np.arange("2005-02-01T10", "2005-02-01T19", dtype="datetime64[h]"),
            ]
        )
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.array(
            [
                np.arange("2005-02-01T09", "2005-02-01T20", dtype="datetime64[h]"),
                np.arange("2005-02-01T09", "2005-02-01T20", dtype="datetime64[h]"),
            ]
        )
        np.testing.assert_array_equal(result, expected)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=0)
        np.testing.assert_array_equal(result, dt_array)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=24)
        expected = np.array(
            [
                np.arange("2005-01-31T10", "2005-02-02T19", dtype="datetime64[h]"),
                np.arange("2005-01-31T10", "2005-02-02T19", dtype="datetime64[h]"),
            ]
        )
        np.testing.assert_array_equal(result, expected)

        dt_array = np.array(
            [
                np.arange("2005-02-01T10", "2005-02-01T19", dtype="datetime64[h]"),
                np.arange("2005-02-01T10", "2005-02-01T19", dtype="datetime64[h]"),
                np.arange("2005-02-01T10", "2005-02-01T19", dtype="datetime64[h]"),
            ]
        )
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.array(
            [
                np.arange("2005-02-01T09", "2005-02-01T20", dtype="datetime64[h]"),
                np.arange("2005-02-01T09", "2005-02-01T20", dtype="datetime64[h]"),
                np.arange("2005-02-01T09", "2005-02-01T20", dtype="datetime64[h]"),
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_pad_daily_datetime_ndarray(self):
        dt_array = np.array(
            [
                np.arange("2005-02-01", "2005-02-02", dtype="datetime64[D]"),
                np.arange("2005-02-01", "2005-02-02", dtype="datetime64[D]"),
            ]
        )
        result = Fetcher._pad_datetime_array(dt_array)
        expected = np.array(
            [
                np.arange("2005-01-31", "2005-02-03", dtype="datetime64[D]"),
                np.arange("2005-01-31", "2005-02-03", dtype="datetime64[D]"),
            ]
        )
        np.testing.assert_array_equal(result, expected)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=0)
        np.testing.assert_array_equal(result, dt_array)

        result = Fetcher._pad_datetime_array(dt_array, pad_width=3)
        expected = np.array(
            [
                np.arange("2005-01-29", "2005-02-05", dtype="datetime64[D]"),
                np.arange("2005-01-29", "2005-02-05", dtype="datetime64[D]"),
            ]
        )
        np.testing.assert_array_equal(result, expected)

        dt_array = np.array(
            [
                np.arange("2005-02-01", "2005-02-02", dtype="datetime64[D]"),
                np.arange("2005-02-01", "2005-02-02", dtype="datetime64[D]"),
                np.arange("2005-02-01", "2005-02-02", dtype="datetime64[D]"),
            ]
        )
        result = Fetcher._pad_datetime_array(dt_array, pad_width=3)
        expected = np.array(
            [
                np.arange("2005-01-29", "2005-02-05", dtype="datetime64[D]"),
                np.arange("2005-01-29", "2005-02-05", dtype="datetime64[D]"),
                np.arange("2005-01-29", "2005-02-05", dtype="datetime64[D]"),
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_pad_hourly_datetime_array_from_domain(self):
        locations = ["a", "b", "c"]
        dt_array = make_domain(
            "2020-01-01T12", "2021-02-05T05", np.timedelta64(1, "h"), *locations
        )[DATE_TIME].reshape(len(locations), -1)
        result = Fetcher._pad_datetime_array(dt_array, pad_width=3)
        expected = make_domain(
            "2020-01-01T09", "2021-02-05T08", np.timedelta64(1, "h"), *locations
        )[DATE_TIME].reshape(len(locations), -1)
        np.testing.assert_array_equal(result, expected)

    def test_pad_daily_datetime_array_from_domain(self):
        locations = ["a", "b", "c"]
        dt_array = make_domain(
            "2020-01-01", "2021-02-05", np.timedelta64(1, "D"), *locations
        )[DATE_TIME].reshape(len(locations), -1)
        result = Fetcher._pad_datetime_array(dt_array, pad_width=3)
        expected = make_domain(
            "2019-12-29", "2021-02-08", np.timedelta64(1, "D"), *locations
        )[DATE_TIME].reshape(len(locations), -1)
        np.testing.assert_array_equal(result, expected)

    def test_pad_domain_dt_hourly_no_op(self):
        domain = make_domain(
            "2020-02-05T04", "2020-02-05T05", np.timedelta64(1, "h"), "single location"
        )

        # hour pad
        result = Fetcher._pad_domain_dt(domain, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

    def test_pad_domain_dt_daily_no_op(self):
        domain = make_domain(
            "2020-02-05", "2020-02-08", np.timedelta64(1, "D"), "single location"
        )

        # day pad
        result = Fetcher._pad_domain_dt(domain, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

    def test_pad_domain_dt_hourly_one_hour(self):
        domain = make_domain(
            "2020-02-05T04", "2020-02-05T05", np.timedelta64(1, "h"), "single location"
        )

        # hour pad
        result = Fetcher._pad_domain_dt(domain)
        expected = make_domain(
            "2020-02-05T03", "2020-02-05T06", np.timedelta64(1, "h"), "single location"
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan)
        expected = make_domain(
            "2020-02-05T03", "2020-02-05T06", np.timedelta64(1, "h"), "single location"
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][-1] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_domain_dt_daily_one_hour(self):
        domain = make_domain(
            "2020-02-05", "2020-02-08", np.timedelta64(1, "D"), "single location"
        )

        # hour pad
        result = Fetcher._pad_domain_dt(domain)
        expected = make_domain(
            "2020-02-04", "2020-02-09", np.timedelta64(1, "D"), "single location"
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan)
        expected = make_domain(
            "2020-02-04", "2020-02-09", np.timedelta64(1, "D"), "single location"
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][-1] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_domain_dt_hourly_two_hours(self):
        domain = make_domain(
            "2020-02-05T04", "2020-02-05T05", np.timedelta64(1, "h"), "single location"
        )
        # hour pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2)
        expected = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), "single location"
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2, pad_value=np.nan)
        expected = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), "single location"
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][1] = np.datetime64("NaT")
        expected[DATE_TIME][-1] = np.datetime64("NaT")
        expected[DATE_TIME][-2] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_domain_dt_daily_two_hours(self):
        domain = make_domain(
            "2020-02-05", "2020-02-08", np.timedelta64(1, "D"), "single location"
        )
        # hour pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2)
        expected = make_domain(
            "2020-02-03", "2020-02-10", np.timedelta64(1, "D"), "single location"
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2, pad_value=np.nan)
        expected = make_domain(
            "2020-02-03", "2020-02-10", np.timedelta64(1, "D"), "single location"
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][1] = np.datetime64("NaT")
        expected[DATE_TIME][-1] = np.datetime64("NaT")
        expected[DATE_TIME][-2] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_domain_dt_hourly_no_op_multi_loc(self):
        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05T04", "2020-02-05T05", np.timedelta64(1, "h"), *locations
        )

        # hour pad
        result = Fetcher._pad_domain_dt(domain, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

    def test_pad_domain_dt_daily_no_op_multi_loc(self):
        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05", "2020-02-08", np.timedelta64(1, "D"), *locations
        )

        # hour pad
        result = Fetcher._pad_domain_dt(domain, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=0)
        expected = domain.copy()
        np.testing.assert_array_equal(result, expected)

    def test_pad_domain_dt_one_hour_multi_loc(self):
        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05T04", "2020-02-05T05", np.timedelta64(1, "h"), *locations
        )

        # hour pad
        result = Fetcher._pad_domain_dt(domain)
        expected = make_domain(
            "2020-02-05T03", "2020-02-05T06", np.timedelta64(1, "h"), *locations
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan)
        expected = make_domain(
            "2020-02-05T03", "2020-02-05T06", np.timedelta64(1, "h"), *locations
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][3] = np.datetime64("NaT")
        expected[DATE_TIME][4] = np.datetime64("NaT")
        expected[DATE_TIME][7] = np.datetime64("NaT")
        expected[DATE_TIME][8] = np.datetime64("NaT")
        expected[DATE_TIME][11] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_domain_dt_one_day_multi_loc(self):
        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05", "2020-02-08", np.timedelta64(1, "D"), *locations
        )

        # hour pad
        result = Fetcher._pad_domain_dt(domain)
        expected = make_domain(
            "2020-02-04", "2020-02-09", np.timedelta64(1, "D"), *locations
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_value=np.nan)
        expected = make_domain(
            "2020-02-04", "2020-02-09", np.timedelta64(1, "D"), *locations
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][5] = np.datetime64("NaT")
        expected[DATE_TIME][6] = np.datetime64("NaT")
        expected[DATE_TIME][11] = np.datetime64("NaT")
        expected[DATE_TIME][12] = np.datetime64("NaT")
        expected[DATE_TIME][17] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_domain_dt_two_hours_multi_loc(self):
        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05T04", "2020-02-05T05", np.timedelta64(1, "h"), *locations
        )
        # hour pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2)
        expected = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), *locations
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2, pad_value=np.nan)
        expected = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), *locations
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][1] = np.datetime64("NaT")
        expected[DATE_TIME][4] = np.datetime64("NaT")
        expected[DATE_TIME][5] = np.datetime64("NaT")
        expected[DATE_TIME][6] = np.datetime64("NaT")
        expected[DATE_TIME][7] = np.datetime64("NaT")
        expected[DATE_TIME][10] = np.datetime64("NaT")
        expected[DATE_TIME][11] = np.datetime64("NaT")
        expected[DATE_TIME][12] = np.datetime64("NaT")
        expected[DATE_TIME][13] = np.datetime64("NaT")
        expected[DATE_TIME][16] = np.datetime64("NaT")
        expected[DATE_TIME][17] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_domain_dt_two_days_multi_loc(self):
        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05", "2020-02-08", np.timedelta64(1, "D"), *locations
        )
        # hour pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2)
        expected = make_domain(
            "2020-02-03", "2020-02-10", np.timedelta64(1, "D"), *locations
        )
        np.testing.assert_array_equal(result, expected)

        # nan pad
        result = Fetcher._pad_domain_dt(domain, pad_width=2, pad_value=np.nan)
        expected = make_domain(
            "2020-02-03", "2020-02-10", np.timedelta64(1, "D"), *locations
        )
        expected[DATE_TIME][0] = np.datetime64("NaT")
        expected[DATE_TIME][1] = np.datetime64("NaT")
        expected[DATE_TIME][6] = np.datetime64("NaT")
        expected[DATE_TIME][7] = np.datetime64("NaT")
        expected[DATE_TIME][8] = np.datetime64("NaT")
        expected[DATE_TIME][9] = np.datetime64("NaT")
        expected[DATE_TIME][14] = np.datetime64("NaT")
        expected[DATE_TIME][15] = np.datetime64("NaT")
        expected[DATE_TIME][16] = np.datetime64("NaT")
        expected[DATE_TIME][17] = np.datetime64("NaT")
        expected[DATE_TIME][22] = np.datetime64("NaT")
        expected[DATE_TIME][23] = np.datetime64("NaT")
        self.assertListEqual(result.tolist(), expected.tolist())

    def test_pad_mask_integrated_no_op(self):
        domain = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), "single location"
        )
        padded_domain = Fetcher._pad_domain_dt(domain, pad_width=0)
        pad_mask = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=0)
        np.testing.assert_array_equal(
            domain, padded_domain[~np.isnat(pad_mask[DATE_TIME])].reshape(-1, 1)
        )

        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), *locations
        )
        padded_domain = Fetcher._pad_domain_dt(domain, pad_width=0)
        pad_mask = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=0)
        np.testing.assert_array_equal(
            domain, padded_domain[~np.isnat(pad_mask[DATE_TIME])].reshape(-1, 1)
        )

    def test_pad_mask_integrated(self):
        domain = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), "single location"
        )
        padded_domain = Fetcher._pad_domain_dt(domain, pad_width=4)
        pad_mask = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=4)
        np.testing.assert_array_equal(
            domain, padded_domain[~np.isnat(pad_mask[DATE_TIME])].reshape(-1, 1)
        )

        locations = ["a", "b", "c"]
        domain = make_domain(
            "2020-02-05T02", "2020-02-05T07", np.timedelta64(1, "h"), *locations
        )
        padded_domain = Fetcher._pad_domain_dt(domain, pad_width=23)
        pad_mask = Fetcher._pad_domain_dt(domain, pad_value=np.nan, pad_width=23)
        np.testing.assert_array_equal(
            domain, padded_domain[~np.isnat(pad_mask[DATE_TIME])].reshape(-1, 1)
        )

    def test_get_data_on_raise_empty(self):
        domain = make_domain(
            "2020-02-05T02",
            "2020-02-05T07",
            np.timedelta64(1, "h"),
            "Newton",
            "Leibniz",
        )
        with patch.multiple(
            Fetcher,
            __abstractmethods__=set(),
            _load_data=Mock(side_effect=EmptyFetchException("expected test exception")),
            get_feature_names=lambda _: ["var1", "var2"],
        ):
            instance = Fetcher(
                location_mapper=Mock(), selector="select", selector_args=Mock()
            )
            result = instance.get_data(domain)

        expected = np.empty((12, 2))
        expected[:] = np.nan
        np.testing.assert_array_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
