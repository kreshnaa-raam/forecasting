import unittest
import sklearn
import sklearn.compose
import sklearn.pipeline

import numpy as np
import pandas as pd

from unittest.mock import patch, MagicMock

from time_series_models.sklearn import sklearn_monkey_patch

sklearn_monkey_patch.apply_patches()

from time_series_models import transformers
from time_series_models.constants import DATE_TIME, LOCATION
from time_series_models.data_fetchers.ami_fetcher import AmiFetcher
from time_series_models.data_monitor import ForecastDataMonitor
from time_series_models.transformers import (
    is_iterable,
    decorate_feature_names,
    make_domain_type,
    make_domain,
    multiindex_from_domain,
    revise_pipeline,
    shift_time,
    uniques_from_sublists,
    convolution_transform,
    make_lookup,
    make_domain_mask_from_dict,
    masking_domain_from_dict,
    nan_like,
    squash_mask,
    add_arrays,
    subtract_arrays,
    interpolate_interior,
    interpolate_array_interior,
    interpolate_array_pandas_default,
    interpolate_array_by_group,
    nullify_cols,
    count_domain_locs_into_column,
    drop_first_column,
    first_column_only,
    fill_nan_columns,
    map_stack_domain,
    overfetched_range_pipeline,
    slice_and_hstack,
    monitor_fetcher,
    net_energy_pipeline,
    RowFilteringFunctionTransformer,
    ColumnTypeTransformer,
    split_domain,
    domain_from_multiindex,
)


class TransformerTests(unittest.TestCase):
    def test_is_iterable(self):
        # list
        self.assertTrue(is_iterable([1, 2, "dog"]))
        # empty list
        self.assertTrue(is_iterable([]))
        # tuple
        self.assertTrue(is_iterable(("x", "y")))
        # generator
        self.assertTrue(is_iterable((i for i in list(range(10)))))
        # numpy array
        self.assertTrue(is_iterable(np.array(["x", "y"])))

        # str fails!
        self.assertFalse(is_iterable("Hello, world"))
        # so does np.str_
        np_str = np.array(["hi", "there"])[0]
        self.assertEqual(np_str, "hi")
        self.assertIsInstance(np_str, np.str_)
        self.assertFalse(is_iterable(np_str))

        # NoneType
        self.assertFalse(is_iterable(None))

        # numeric
        self.assertFalse(is_iterable(3))

        # pandas
        self.assertFalse(is_iterable(pd.DataFrame))
        self.assertTrue(is_iterable(pd.DataFrame().iterrows()))

    def test_uniques_from_sublist(self):
        result = uniques_from_sublists(
            [["a", "c", "e"], ["d"], ["a", "b"], [], ["d", "c"]]
        )
        expected = {"a", "b", "c", "d", "e"}
        self.assertSetEqual(result, expected)

    def test_decorate_feature_names(self):
        def old_func(x):
            return x**2

        def gfn():
            return ["foo", "bar"]

        new_func = decorate_feature_names(old_func, feature_names_func=gfn)
        self.assertEqual(new_func(2), 4)
        self.assertEqual(new_func(-3), 9)

        self.assertTrue(hasattr(new_func, "get_feature_names"))
        self.assertListEqual(new_func.get_feature_names(), ["foo", "bar"])

        # note, the new function *is* the original function, so both now have a get_feature_names method
        self.assertIs(new_func, old_func)
        self.assertTrue(hasattr(old_func, "get_feature_names"))
        self.assertListEqual(old_func.get_feature_names(), ["foo", "bar"])

    def test_decorate_feature_names__misc_gfn_types(self):
        def my_func(x):
            return x**2

        # singleton list works
        def gfn():
            return ["foo"]

        decorate_feature_names(my_func, gfn)
        result = my_func.get_feature_names()
        self.assertListEqual(result, ["foo"])

        # tuple works
        def gfn():
            return "foo", "bar"

        decorate_feature_names(my_func, gfn)
        result = my_func.get_feature_names()
        self.assertTupleEqual(result, ("foo", "bar"))

        # numpy array works
        def gfn():
            return np.array([1, 2, 3], dtype="timedelta64[h]")

        decorate_feature_names(my_func, gfn)
        result = my_func.get_feature_names()
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

        # note, a generator will also pass the iterable check
        def gfn():
            yield "foo"

        decorate_feature_names(my_func, gfn)
        result = my_func.get_feature_names()
        self.assertEqual(next(result), "foo")

        # but a simple str return will raise
        def gfn():
            return "foo"

        with self.assertRaises(RuntimeError):
            decorate_feature_names(my_func, gfn)

        # as will any other non-iterable
        def gfn():
            pass

        with self.assertRaises(RuntimeError):
            decorate_feature_names(my_func, gfn)

    def test_multiindex_from_domain(self):
        domain = make_domain(
            "2021-01-01T00", "2021-01-01T09", np.timedelta64(3, "h"), "a", "b", "c"
        )
        result = multiindex_from_domain(domain)
        self.assertIsInstance(result, pd.MultiIndex)
        self.assertEqual(12, len(result))
        self.assertListEqual(result.names, [LOCATION, DATE_TIME])
        loc_idx_expected = pd.Index(["a"] * 4 + ["b"] * 4 + ["c"] * 4, name=LOCATION)
        pd.testing.assert_index_equal(
            result.get_level_values(LOCATION), loc_idx_expected
        )
        date_idx_expected = pd.Index(
            np.tile(
                pd.date_range(
                    start="2021-01-01T00", end="2021-01-01T09", freq="3H"
                ).to_numpy(),
                3,
            ),
            name=DATE_TIME,
        )
        pd.testing.assert_index_equal(
            result.get_level_values(DATE_TIME), date_idx_expected
        )

        # works on a slice of a domain, too
        result = multiindex_from_domain(domain[:5])
        self.assertIsInstance(result, pd.MultiIndex)
        self.assertEqual(5, len(result))
        self.assertListEqual(result.names, [LOCATION, DATE_TIME])

        loc_idx_expected = pd.Index(["a"] * 4 + ["b"], name=LOCATION)
        pd.testing.assert_index_equal(
            result.get_level_values(LOCATION), loc_idx_expected
        )

        date_idx_expected = pd.DatetimeIndex(
            [
                "2021-01-01T00",
                "2021-01-01T03",
                "2021-01-01T06",
                "2021-01-01T09",
                "2021-01-01T00",
            ],
            dtype="datetime64[ns]",
            name=DATE_TIME,
        )
        pd.testing.assert_index_equal(
            result.get_level_values(DATE_TIME), date_idx_expected
        )

    def test_make_domain__single(self):
        result = make_domain(
            "2021-01-01T00", "2021-01-02T00", np.timedelta64(1, "h"), "camus"
        )
        np.testing.assert_equal(
            result[LOCATION], np.array(("camus",) * 25, dtype="U36").reshape(-1, 1)
        )
        np.testing.assert_equal(
            result[DATE_TIME],
            np.arange(
                np.datetime64("2021-01-01T00"),
                np.datetime64("2021-01-02T01"),
                step=np.timedelta64(1, "h"),
            ).reshape(-1, 1),
        )

    def test_make_domain__multi(self):
        result = make_domain(
            "2021-01-01T00",
            "2021-01-02T00",
            np.timedelta64(1, "h"),
            "kant",
            "nietzsche",
        )
        np.testing.assert_equal(
            result[LOCATION],
            np.array(("kant",) * 25 + ("nietzsche",) * 25, dtype="U36").reshape(-1, 1),
        )
        np.testing.assert_equal(
            result[DATE_TIME],
            np.tile(
                np.arange(
                    np.datetime64("2021-01-01T00"),
                    np.datetime64("2021-01-02T01"),
                    step=np.timedelta64(1, "h"),
                ).reshape(-1, 1),
                (2, 1),
            ),
        )

    def test_make_domain__too_long(self):
        self.assertRaisesRegexp(
            ValueError,
            r"Invalid locations are too long: \['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'\]",
            make_domain,
            "2021-01-01T00",
            "2021-01-02T00",
            np.timedelta64(1, "h"),
            "a" * 37,
            "b" * 37,
            "c" * 5,
        )

    def test_domain_from_multiindex(self):
        loc_idx = pd.Index(["a"] * 4 + ["b"] * 4 + ["c"] * 4, name=LOCATION)
        date_idx = pd.Index(
            np.tile(
                pd.date_range(
                    start="2021-01-01T00", end="2021-01-01T09", freq="3H"
                ).to_numpy(),
                3,
            ),
            name=DATE_TIME,
        )
        domain_like_index = pd.MultiIndex.from_arrays((loc_idx, date_idx))
        result = domain_from_multiindex(domain_like_index, np.timedelta64(3, "h"))

        expected = make_domain(
            "2021-01-01T00", "2021-01-01T09", np.timedelta64(3, "h"), "a", "b", "c"
        )
        np.testing.assert_array_equal(result, expected)

    def test_domain_from_mutliindex__irregular_long_locations(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                [
                    "a" * 37,
                    "a" * 37,
                ],
                pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                    ]
                ),
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            r"Irregular multiindex is not a valid domain because the following locations are more than 36 characters: \['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\]",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_domain_from_mutliindex__irregular_not_timestamps(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                [
                    "a",
                    "a",
                ],
                [
                    "2023-01-01",
                    "2023-01-02",
                ],
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            "Irregular multiindex is not a valid domain because the multiindex dtypes are not correct",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_domain_from_mutliindex__irregular_not_rectangular(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                ["alpha", "alpha", "beta"],
                pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04"]),
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            "Irregular multiindex is not a valid domain because the length 3 isn't divisible by the number of unique locations 2",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_domain_from_mutliindex__irregular_location_order(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                ["alpha", "beta", "alpha", "beta"],
                pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
                ),
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            "Irregular multiindex is not a valid domain: locations are not regular",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_domain_from_mutliindex__irregular_locations(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                ["alpha", "alpha", "alpha", "beta"],
                pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
                ),
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            "Irregular multiindex is not a valid domain: locations are not regular",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_domain_from_mutliindex__irregular_duplicate_locations(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                ["alpha", "alpha", "beta", "beta", "beta", "beta"],
                pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                        "2023-01-06",
                    ]
                ),
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            "Irregular multiindex is not a valid domain: locations are not regular",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_domain_from_mutliindex__irregular_datetimes(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                ["alpha", "alpha", "beta", "beta"],
                pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
                ),
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            "Irregular multiindex is not a valid domain: timesteps are not regular",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_domain_from_mutliindex__irregular_time_steps(self):
        undomain_like_index = pd.MultiIndex.from_arrays(
            (
                ["alpha", "alpha", "beta", "beta"],
                pd.to_datetime(
                    ["2023-01-01", "2023-01-02T01", "2023-01-01", "2023-01-02T01"]
                ),
            ),
            names=[LOCATION, DATE_TIME],
        )
        self.assertRaisesRegexp(
            ValueError,
            "Irregular multiindex is not a valid domain: timesteps are not uniform",
            domain_from_multiindex,
            undomain_like_index,
            np.timedelta64(1, "D"),
        )

    def test_rank(self):
        result = transformers.rank(
            np.array([1, 5, 2, 7, 3]), np.array(["a", "b", "c", "d", "e"])
        )
        np.testing.assert_equal(result, np.array(["e", "b", "d", "a", "c"]))

    def test_rank_long(self):
        result = transformers.rank(
            np.array([1, 5, 2, 7, 3]), np.array(["a", "b", "c", "d", "e", "f", "g"])
        )
        np.testing.assert_equal(result, np.array(["e", "b", "d", "a", "c"]))

    def test_shift_time(self):
        location_list = []
        for philosopher in ["Voltaire", "Karl Marx"]:
            location_list.append(philosopher)
            for start, stop in [
                ("2021-05-17", "2021-05-25"),
                ("2021-05-17", "2021-05-17"),
            ]:
                with self.subTest(
                    "shift time params test",
                    start=start,
                    stop=stop,
                    locations=location_list,
                ):
                    domain = make_domain(
                        start, stop, np.timedelta64(1, "D"), *location_list
                    )
                    result = shift_time(domain, np.timedelta64(2, "D"))

                    self.assertIsNot(domain, result)

                    new_start = np.datetime64(start) - np.timedelta64(2, "D")
                    new_stop = np.datetime64(stop) - np.timedelta64(2, "D")
                    np.testing.assert_array_equal(
                        result,
                        make_domain(
                            new_start, new_stop, np.timedelta64(1, "D"), *location_list
                        ),
                    )

    def test_shift_time__extend(self):
        location_list = []
        for philosopher in ["Voltaire", "Karl Marx"]:
            location_list.append(philosopher)
            for start, stop in [
                ("2021-05-17", "2021-05-25"),
                ("2021-05-17", "2021-05-17"),
            ]:
                with self.subTest(
                    "shift time extend params test",
                    start=start,
                    stop=stop,
                    locations=location_list,
                ):
                    domain = make_domain(
                        start, stop, np.timedelta64(1, "D"), *location_list
                    )
                    result = shift_time(
                        domain,
                        np.timedelta64(2, "D"),
                        extend=True,
                        tstep=np.timedelta64(1, "D"),
                    )

                    self.assertIsNot(domain, result)

                    new_start = np.datetime64(start) - np.timedelta64(2, "D")
                    new_stop = np.datetime64(stop)
                    np.testing.assert_array_equal(
                        result,
                        make_domain(
                            new_start, new_stop, np.timedelta64(1, "D"), *location_list
                        ),
                    )

        # make the time step non-uniform
        domain[DATE_TIME][-1, 0] = np.datetime64("2021-06-01")
        with self.assertRaises(AssertionError):
            shift_time(domain, np.timedelta64(2, "D"), extend=True)

    def test_convolution_transform(self):
        domain = make_domain(
            "2021-05-17", "2021-05-30", np.timedelta64(1, "D"), "Rousseau", "Chomsky"
        )

        data = np.zeros([domain.shape[0], 2])
        data[0:14, 0] = 1
        data[0:14, 1] = 2
        data[14:, 0] = -1
        data[14:, 1] = -2
        filt = np.array([-5.0, 5.0, -5.0])
        result = convolution_transform(data, domain, filt)

        expected = np.zeros([24, 2])
        expected[0:12, 0] = -5
        expected[0:12, 1] = -10
        expected[12:, 0] = 5
        expected[12:, 1] = 10

        np.testing.assert_almost_equal(result, expected)

    def test_location_lookup(self):
        reference_location = "Pacific Ocean"
        mappers = []
        for city in ["LA", "Tokyo", "Sydney"]:
            mappers.append(make_lookup(city))
        result = [m(reference_location) for m in mappers]
        expected = ["LA", "Tokyo", "Sydney"]
        self.assertListEqual(result, expected)

        # counter-example: mis-scoped lambdas
        mappers = []
        for city in ["LA", "Tokyo", "Sydney"]:
            mappers.append(lambda x: city)
        result = [m(reference_location) for m in mappers]
        expected = ["Sydney", "Sydney", "Sydney"]
        self.assertListEqual(result, expected)

        # lambdas w/ proper scoping:
        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        mappers = []
        for city in ["LA", "Tokyo", "Sydney"]:
            mappers.append(lambda dom_loc, lookup=city: lookup)
        result = [m(reference_location) for m in mappers]
        expected = ["LA", "Tokyo", "Sydney"]
        self.assertListEqual(result, expected)

    def test_interpolate_interior(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, np.nan],
                "b": [2, np.nan, np.nan, 8],
                "c": [np.nan, np.nan, 10, 20],
                "d": [np.nan, 4, 3, 2],
                "e": [2, 4, np.nan, 8],
                "f": [3, 5, 7, np.nan],
                "g": [np.nan, 2, np.nan, 4],
                "h": [0, np.nan, 2, np.nan],
            }
        )
        copy_original = df.copy()
        result = interpolate_interior(df)
        expected = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, np.nan],
                "b": [2.0, 4.0, 6.0, 8.0],
                "c": [np.nan, np.nan, 10.0, 20.0],
                "d": [np.nan, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, np.nan],
                "g": [np.nan, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, np.nan],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

        # assert original is unmodified
        pd.testing.assert_frame_equal(df, copy_original)

    @unittest.skip("TODO")
    def test_interpolate_array_interior(self):
        pass

    @unittest.skip("TODO")
    def test_interpolate_array_pandas_default(self):
        pass

    @staticmethod
    def make_example_array(n_locs):
        return np.array(
            [
                [n_locs, 1.0, 22.0, 100.0],
                [n_locs, 2.0, 23.0, np.nan],
                [n_locs, np.nan, 24.0, np.nan],
                [n_locs, 4.0, np.nan, 103.0],
                [n_locs, 11.0, np.nan, 200.0],
                [n_locs, np.nan, 31.0, np.nan],
                [n_locs, 13.0, 32.0, 202.0],
                [n_locs, 14.0, 33.0, np.nan],
            ]
        )

    def test_interpolate_array_by_group__default_func__default_cols(self):
        # default is to interpolate all columns, grouped by the value of the first column

        array = self.make_example_array(n_locs=1)
        # first column valued 1.0 --> interpolate array as a single group
        result = interpolate_array_by_group(array)
        expected = np.array(
            [
                [1.0, 1.0, 22.0, 100.0],
                [1.0, 2.0, 23.0, 101.0],
                [1.0, 3.0, 24.0, 102.0],
                [1.0, 4.0, 24 + 1 / 3 * (31 - 24), 103.0],
                [1.0, 11.0, 24 + 2 / 3 * (31 - 24), 200.0],
                [1.0, 12.0, 31.0, 201.0],
                [1.0, 13.0, 32.0, 202.0],
                [1.0, 14.0, 33.0, 202.0],  # extrapolate trailing nan
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        array = self.make_example_array(n_locs=2)
        # first column valued 2.0 --> interpolate array as two groups (split row-wise)
        result = interpolate_array_by_group(array)
        expected = np.array(
            [
                [2.0, 1.0, 22.0, 100.0],
                [2.0, 2.0, 23.0, 101.0],
                [2.0, 3.0, 24.0, 102.0],
                [2.0, 4.0, 24.0, 103.0],  # extrapolate trailing nan
                [2.0, 11.0, np.nan, 200.0],  # leading nan
                [2.0, 12.0, 31.0, 201.0],
                [2.0, 13.0, 32.0, 202.0],
                [2.0, 14.0, 33.0, 202.0],  # extrapolate trailing nan
            ]
        )
        np.testing.assert_almost_equal(result, expected)

    def test_interpolate_array_by_group__default_func__ncol(self):
        arr_single_group = self.make_example_array(n_locs=1)
        arr_two_groups = self.make_example_array(n_locs=2)

        # bad value for n_col should raise regardless of value of arr
        with self.assertRaises(ValueError):
            interpolate_array_by_group(arr_single_group, n_col=-2)
        with self.assertRaises(ValueError):
            interpolate_array_by_group(arr_two_groups, n_col=0)

        # interpolate only the first (non-location-count) column
        result = interpolate_array_by_group(arr_two_groups, n_col=1)
        expected = np.array(
            [
                [2.0, 1.0, 22.0, 100.0],
                [2.0, 2.0, 23.0, np.nan],
                [2.0, 3.0, 24.0, np.nan],
                [2.0, 4.0, np.nan, 103.0],
                [2.0, 11.0, np.nan, 200.0],
                [2.0, 12.0, 31.0, np.nan],
                [2.0, 13.0, 32.0, 202.0],
                [2.0, 14.0, 33.0, np.nan],
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        # interpolate first and second (non-location-count) columns
        expected = np.array(
            [
                [2.0, 1.0, 22.0, 100.0],
                [2.0, 2.0, 23.0, np.nan],
                [2.0, 3.0, 24.0, np.nan],
                [2.0, 4.0, 24.0, 103.0],  # extrapolate trailing nan
                [2.0, 11.0, np.nan, 200.0],  # leading nan
                [2.0, 12.0, 31.0, np.nan],
                [2.0, 13.0, 32.0, 202.0],
                [2.0, 14.0, 33.0, np.nan],
            ]
        )
        result = interpolate_array_by_group(arr_two_groups, n_col=2)
        np.testing.assert_almost_equal(result, expected)

        # with 1 location/group, the second (non-location-count) column will be interpolated
        result = interpolate_array_by_group(arr_single_group, n_col=2)
        expected = np.array(
            [
                [1.0, 1.0, 22.0, 100.0],
                [1.0, 2.0, 23.0, np.nan],  # last column not interpolated
                [1.0, 3.0, 24.0, np.nan],  # last column not interpolated
                [1.0, 4.0, 24 + 1 / 3 * (31 - 24), 103.0],
                [1.0, 11.0, 24 + 2 / 3 * (31 - 24), 200.0],
                [1.0, 12.0, 31.0, np.nan],  # last column not interpolated
                [1.0, 13.0, 32.0, 202.0],
                [1.0, 14.0, 33.0, np.nan],  # last column not interpolated
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        # with n_col >= arr.shape[1] - 1, expect same result as with default setting, interpolating all columns by group
        result = interpolate_array_by_group(arr_two_groups, n_col=3)
        expected = np.array(
            [
                [2.0, 1.0, 22.0, 100.0],
                [2.0, 2.0, 23.0, 101.0],
                [2.0, 3.0, 24.0, 102.0],
                [2.0, 4.0, 24.0, 103.0],  # extrapolate trailing nan
                [2.0, 11.0, np.nan, 200.0],  # leading nan
                [2.0, 12.0, 31.0, 201.0],
                [2.0, 13.0, 32.0, 202.0],
                [2.0, 14.0, 33.0, 202.0],  # extrapolate trailing nan
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        # n_col > arr.shape[1] - 1 will raise
        with self.assertRaises(ValueError):
            interpolate_array_by_group(arr_two_groups, n_col=100)
        np.testing.assert_almost_equal(result, expected)

    def test_interpolate_array_by_group__interp_interior__default(self):
        # default is to interpolate all columns, grouped by the value of the first column

        array = self.make_example_array(n_locs=1)
        # first column valued 1.0 --> interpolate array as a single group
        result = interpolate_array_by_group(
            array, interp_func=interpolate_array_interior
        )
        expected = np.array(
            [
                [1.0, 1.0, 22.0, 100.0],
                [1.0, 2.0, 23.0, 101.0],
                [1.0, 3.0, 24.0, 102.0],
                [1.0, 4.0, 24 + 1 / 3 * (31 - 24), 103.0],
                [1.0, 11.0, 24 + 2 / 3 * (31 - 24), 200.0],
                [1.0, 12.0, 31.0, 201.0],
                [1.0, 13.0, 32.0, 202.0],
                [1.0, 14.0, 33.0, np.nan],  # no extrapolation
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        array = self.make_example_array(n_locs=2)
        # first column valued 2.0 --> interpolate array as two groups (split row-wise)
        result = interpolate_array_by_group(
            array, interp_func=interpolate_array_interior
        )
        expected = np.array(
            [
                [2.0, 1.0, 22.0, 100.0],
                [2.0, 2.0, 23.0, 101.0],
                [2.0, 3.0, 24.0, 102.0],
                [2.0, 4.0, np.nan, 103.0],  # no extrapolation
                [2.0, 11.0, np.nan, 200.0],  # leading nan
                [2.0, 12.0, 31.0, 201.0],
                [2.0, 13.0, 32.0, 202.0],
                [2.0, 14.0, 33.0, np.nan],  # no extrapolation
            ]
        )
        np.testing.assert_almost_equal(result, expected)

    def test_interpolate_array_by_group__interp_interior__ncol(self):
        arr_single_group = self.make_example_array(n_locs=1)
        arr_two_groups = self.make_example_array(n_locs=2)

        # bad value for n_col should raise regardless of value of arr
        with self.assertRaises(ValueError):
            interpolate_array_by_group(
                arr_single_group, interp_func=interpolate_array_interior, n_col=-2
            )
        with self.assertRaises(ValueError):
            interpolate_array_by_group(
                arr_two_groups, interp_func=interpolate_array_interior, n_col=0
            )

        # interpolate only the first (non-location-count) column
        result = interpolate_array_by_group(
            arr_two_groups, interp_func=interpolate_array_interior, n_col=1
        )
        expected = np.array(
            [
                [2.0, 1.0, 22.0, 100.0],
                [2.0, 2.0, 23.0, np.nan],  # last column not interpolated
                [2.0, 3.0, 24.0, np.nan],  # last column not interpolated
                [2.0, 4.0, np.nan, 103.0],  # no extrapolation
                [2.0, 11.0, np.nan, 200.0],  # leading nan
                [2.0, 12.0, 31.0, np.nan],  # last column not interpolated
                [2.0, 13.0, 32.0, 202.0],
                [2.0, 14.0, 33.0, np.nan],
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        # interpolate first and second (non-location-count) columns...
        # but with 2 locations, the second column should not be modified since nulls are at location boundaries
        # so expect same result as for transformation with n_col=1
        result = interpolate_array_by_group(
            arr_two_groups, interp_func=interpolate_array_interior, n_col=2
        )
        np.testing.assert_almost_equal(result, expected)

        # with 1 location/group, the second (non-location-count) column will be interpolated
        result = interpolate_array_by_group(
            arr_single_group, interp_func=interpolate_array_interior, n_col=2
        )
        expected = np.array(
            [
                [1.0, 1.0, 22.0, 100.0],
                [1.0, 2.0, 23.0, np.nan],  # last column not interpolated
                [1.0, 3.0, 24.0, np.nan],  # last column not interpolated
                [1.0, 4.0, 24 + 1 / 3 * (31 - 24), 103.0],
                [1.0, 11.0, 24 + 2 / 3 * (31 - 24), 200.0],
                [1.0, 12.0, 31.0, np.nan],  # last column not interpolated
                [1.0, 13.0, 32.0, 202.0],
                [1.0, 14.0, 33.0, np.nan],
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        # with n_col == arr.shape[1] - 1, expect same result as with default setting, interpolating all columns by group
        result = interpolate_array_by_group(
            arr_two_groups, interp_func=interpolate_array_interior, n_col=3
        )
        expected = np.array(
            [
                [2.0, 1.0, 22.0, 100.0],
                [2.0, 2.0, 23.0, 101.0],
                [2.0, 3.0, 24.0, 102.0],
                [2.0, 4.0, np.nan, 103.0],  # no extrapolation
                [2.0, 11.0, np.nan, 200.0],  # leading nan
                [2.0, 12.0, 31.0, 201.0],
                [2.0, 13.0, 32.0, 202.0],
                [2.0, 14.0, 33.0, np.nan],  # no extrapolation
            ]
        )
        np.testing.assert_almost_equal(result, expected)

        # n_col > arr.shape[1] - 1 will raise
        with self.assertRaises(ValueError):
            interpolate_array_by_group(
                arr_two_groups, interp_func=interpolate_array_interior, n_col=100
            )

    def test_masking_domain_from_dict__no_overlaps(self):
        mask_dict = dict(
            resource_1=[
                ("2005-05-01T00", "2005-05-01T02"),
                ("2020-01-01T18", "2020-01-02"),
                ("2020-01-03", "2020-01-04T04"),
            ],
            resource_2=[("2020-01-01T02", "2020-01-01T05")],
        )
        tstep = np.timedelta64(1, "h")
        result = masking_domain_from_dict(tstep, mask_dict)

        expected_shape = (3 + 7 + 29 + 4, 1)
        expected = np.empty(expected_shape, dtype=make_domain_type(tstep))
        expected[LOCATION] = np.array(
            ["resource_1"] * (3 + 7 + 29) + ["resource_2"] * 4
        ).reshape(-1, 1)
        expected[DATE_TIME] = np.concatenate(
            (
                np.arange(
                    np.datetime64("2005-05-01T00"),
                    np.datetime64("2005-05-01T03"),
                    tstep,
                ),
                np.arange(
                    np.datetime64("2020-01-01T18"),
                    np.datetime64("2020-01-02T01"),
                    tstep,
                ),
                np.arange(
                    np.datetime64("2020-01-03"),
                    np.datetime64("2020-01-04T05"),
                    tstep,
                ),
                np.arange(
                    np.datetime64("2020-01-01T02"),
                    np.datetime64("2020-01-01T06"),
                    tstep,
                ),
            )
        ).reshape(-1, 1)
        np.testing.assert_array_equal(result, expected)

    def test_masking_domain_from_dict__overlapping_time__across_location(self):
        mask_dict = dict(
            resource_1=[
                ("2005-05-01T00", "2005-05-01T02"),
                ("2020-01-01T18", "2020-01-02"),
                ("2020-01-03", "2020-01-04T04"),
            ],
            resource_2=[
                ("2020-01-03", "2020-01-03T03")
            ],  # overlaps last time range of resource_1
        )
        tstep = np.timedelta64(1, "h")
        result = masking_domain_from_dict(tstep, mask_dict)

        # both domain locations will be masked at once where they overlap
        expected_shape = (3 + 7 + 29 + 4, 1)
        expected = np.empty(expected_shape, dtype=make_domain_type(tstep))
        expected[LOCATION] = np.array(
            ["resource_1"] * (3 + 7 + 29) + ["resource_2"] * 4
        ).reshape(-1, 1)
        expected[DATE_TIME] = np.concatenate(
            (
                np.arange(
                    np.datetime64("2005-05-01T00"),
                    np.datetime64("2005-05-01T03"),
                    tstep,
                ),
                np.arange(
                    np.datetime64("2020-01-01T18"),
                    np.datetime64("2020-01-02T01"),
                    tstep,
                ),
                np.arange(
                    np.datetime64("2020-01-03"),
                    np.datetime64("2020-01-04T05"),
                    tstep,
                ),
                np.arange(
                    np.datetime64("2020-01-03"),
                    np.datetime64("2020-01-03T04"),
                    tstep,
                ),
            )
        ).reshape(-1, 1)
        np.testing.assert_array_equal(result, expected)

    def test_masking_domain_from_dict__overlapping_time__within_location(self):
        mask_dict = dict(
            resource_1=[
                ("2020-01-02T00", "2020-01-02T03"),
                ("2020-01-02T02", "2020-01-02T05"),
            ],
        )
        tstep = np.timedelta64(1, "h")
        result = masking_domain_from_dict(tstep, mask_dict)

        # overlaps will be repeated
        expected_shape = (4 + 4, 1)
        expected = np.empty(expected_shape, dtype=make_domain_type(tstep))
        expected[LOCATION] = np.array(["resource_1"] * 8).reshape(-1, 1)
        expected[DATE_TIME] = np.array(
            [
                [np.datetime64("2020-01-02T00")],
                [np.datetime64("2020-01-02T01")],
                [np.datetime64("2020-01-02T02")],
                [np.datetime64("2020-01-02T03")],
                [np.datetime64("2020-01-02T02")],  # note overlap!
                [np.datetime64("2020-01-02T03")],  # note overlap!
                [np.datetime64("2020-01-02T04")],
                [np.datetime64("2020-01-02T05")],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_masking_domain_from_dict__empty_dict(self):
        mask_dict = {}
        tstep = np.timedelta64(1, "h")
        result = masking_domain_from_dict(tstep, mask_dict)
        self.assertTupleEqual(result.shape, (0, 1))
        expected = np.empty((0, 1), dtype=make_domain_type(tstep))
        np.testing.assert_array_equal(result, expected)

    def test_make_domain_mask_from_dict__no_overlap(self):
        domain = make_domain(
            "2020-01-01",
            "2020-01-02",
            np.timedelta64(1, "h"),
            "resource_1",
            "resource_2",
            "resource_3",
        )  # 25 hrs * 3 locations
        mask_dict = dict(
            resource_1=[
                ("2005-05-01T00", "2005-05-01T02"),  # time range outside of domain!
                ("2020-01-01T02", "2020-01-01T04"),
                ("2020-01-01T16", "2020-01-01T18"),
            ],
            resource_2=[("2020-01-01T00", "2020-01-01T01")],
        )
        result = make_domain_mask_from_dict(domain, mask_dict)
        self.assertTupleEqual(result.shape, domain.shape)

        expected = np.empty(domain.shape, dtype=bool)
        expected[:, 0] = False
        expected[2:5, 0] = True
        expected[16:19, 0] = True
        expected[25:27, 0] = True
        np.testing.assert_array_equal(result, expected)

    def test_make_domain_mask_from_dict__overlapping_time_within_location(self):
        domain = make_domain(
            "2020-01-01", "2020-01-02", np.timedelta64(1, "h"), "resource_1"
        )
        mask_dict = dict(
            resource_1=[
                ("2020-01-01T02", "2020-01-01T06"),
                ("2020-01-01T04", "2020-01-01T08"),
            ],
        )
        result = make_domain_mask_from_dict(domain, mask_dict)
        self.assertTupleEqual(result.shape, domain.shape)

        # "double-masked" times are still masked
        expected = np.empty(domain.shape, dtype=bool)
        expected[:, 0] = False
        expected[2:9, 0] = True
        np.testing.assert_array_equal(result, expected)

    def test_squash_mask(self):
        arr = np.array(
            [
                [1, 0],
                [2, 0],
                [3, 1],
                [4, 1],
            ]
        )
        result = squash_mask(arr)
        expected = np.array(
            [
                [0.0],
                [0.0],
                [3.0],
                [4.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        result = squash_mask(arr, masked_value=100)
        expected = np.array(
            [
                [100.0],
                [100.0],
                [3.0],
                [4.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_add_arrays(self):
        arr = np.array(
            [
                [10, 1],
                [10, 2],
                [10, 3],
            ]
        )

        result = add_arrays(arr)
        expected = np.array(
            [
                [11.0],
                [12.0],
                [13.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        names = add_arrays.get_feature_names()
        self.assertListEqual(names, ["components_sum"])

    def test_subtract_arrays(self):
        arr = np.array(
            [
                [10, 1],
                [10, 2],
                [10, 3],
            ]
        )
        result = subtract_arrays(arr)
        expected = np.array(
            [
                [9.0],
                [8.0],
                [7.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        names = subtract_arrays.get_feature_names()
        self.assertListEqual(names, ["components_difference"])

    def test_subtract_arrays__n_arrays(self):
        arr = np.array(
            [
                [10, 1, 4],
                [10, 2, 5],
                [10, 3, 6],
            ]
        )
        result = subtract_arrays(arr)
        expected = np.array(
            [
                [5.0],
                [3.0],
                [1.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_map_stack_domain(self):
        start = np.datetime64("2022-08-08")
        end = np.datetime64("2022-08-09")
        tstep = np.timedelta64(6, "h")
        domain = make_domain(start, end, tstep, "a", "b", "c")

        # missing key --> KeyError
        mapping = {"a": ["one", "two"], "b": ["three"]}
        with self.assertRaises(KeyError):
            map_stack_domain(domain, mapping)

        # every key must have mapping, at least an empty list
        mapping = {"a": ["one", "two"], "b": ["three"], "c": [], "d": ["four"]}
        result = map_stack_domain(domain, mapping)
        self.assertIsInstance(result, tuple)
        self.assertEqual(2, len(result))
        self.assertEqual(len(result[0]), len(result[1]))

        expected_domain = make_domain(start, end, tstep, "one", "two", "three")
        np.testing.assert_array_equal(result[0], expected_domain)

        expected_key = np.array(["a"] * 10 + ["b"] * 5)
        np.testing.assert_array_equal(result[1], expected_key)

    def test_slice_and_hstack(self):
        lags = np.array([1, 2], dtype="datetime64[h]")

        # example values representing four time steps fetched for each of three domain locations;
        # fetched range has already been back-extended in time by 2 steps, since max(lags) = 2
        arr = np.array(
            [
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 10],
                [3, 11],
                [3, 12],
                [3, 13],
                [3, 20],
                [3, 21],
                [3, 22],
                [3, 23],
            ]
        )
        slice_and_stack = slice_and_hstack(lags)
        result = slice_and_stack(arr)
        expected = np.array(
            [
                [1, 0],
                [2, 1],
                [11, 10],
                [12, 11],
                [21, 20],
                [22, 21],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        # array to be sliced must already account for requested lags, since slicing logic uses max(lags):
        slice_and_stack = slice_and_hstack(np.array([1], dtype="datetime64[h]"))
        result = slice_and_stack(arr)
        expected = np.array(
            [
                [0],
                [1],
                [2],
                [10],
                [11],
                [12],
                [20],
                [21],
                [22],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        slice_and_stack = slice_and_hstack(np.array([1, 3], dtype="datetime64[h]"))
        result = slice_and_stack(arr)
        expected = np.array(
            [
                [2, 0],
                [12, 10],
                [22, 20],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        # lag 0 should result in slice being aligned with end of each location's date range
        slice_and_stack = slice_and_hstack(np.array([0, 3], dtype="datetime64[h]"))
        result = slice_and_stack(arr)
        expected = np.array(
            [
                [3, 0],
                [13, 10],
                [23, 20],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        # note that the value of the first column determines how domain locations are identified and handled:
        arr = np.array(
            [
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
                [2, 10],
                [2, 11],
                [2, 12],
                [2, 13],
                [2, 20],
                [2, 21],
                [2, 22],
                [2, 23],
            ]
        )
        slice_and_stack = slice_and_hstack(np.array([1, 2], dtype="datetime64[h]"))
        # two locations --> the first half of the observations is assigned to the first location
        result = slice_and_stack(arr)
        expected = np.array(
            [
                [1, 0],
                [2, 1],
                [3, 2],
                [10, 3],
                [13, 12],
                [20, 13],
                [21, 20],
                [22, 21],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_column_type_transform_simple(self):
        def myfunc(X):
            return X.astype(np.int64)

        myfunc.get_feature_names = lambda: [
            "one",
        ]

        ctf = ColumnTypeTransformer(
            [
                ("pass", "passthrough", [0]),
                ("funky", sklearn.preprocessing.FunctionTransformer(func=myfunc), [1]),
            ]
        )
        X = np.random.standard_normal((12, 2)) - 2
        result = ctf.fit_transform(X)
        expected_dtype = np.dtype([("pass", np.float64), ("one", np.int64)])
        self.assertEqual(result.dtype, expected_dtype)
        np.testing.assert_array_equal(result["pass"], X[:, [0]])
        np.testing.assert_array_equal(result["one"], X[:, [1]].astype(np.int64))

    def test_column_type_transform(self):
        start = np.datetime64("2022-08-08")
        end = np.datetime64("2022-08-09")
        tstep = np.timedelta64(6, "h")
        domain = make_domain(start, end, tstep, "a", "b", "c")

        def function_with_feature_names(x):
            return pd.DataFrame(
                np.arange(x.shape[0] * 3, dtype=np.int32).reshape(x.shape[0], 3),
                index=x,
                columns=["one", "two", "three"],
            )

        function_with_feature_names.get_feature_names = lambda: ["one", "two", "three"]

        # Can only attach one function which returns one value - can't use the kwargs it is called with
        split_domain.get_feature_names = lambda: ["domain_split"]

        array_type = np.dtype([("array", np.int32, (3, 3))])

        def array_function_with_single_name(x):
            data = np.zeros(x.shape, array_type)
            data["array"][:, 0] = np.arange(x.shape[0] * 3 * 3).reshape(
                x.shape[0], 3, 3
            )
            return data

        array_function_with_single_name.get_feature_names = lambda: ["magic"]

        ctf = ColumnTypeTransformer(
            [
                ("pass", "passthrough", [0]),
                (
                    "split_date",
                    sklearn.preprocessing.FunctionTransformer(
                        func=split_domain, kw_args={"key": DATE_TIME}
                    ),
                    [0],
                ),
                (
                    "named_features",
                    sklearn.preprocessing.FunctionTransformer(
                        func=function_with_feature_names
                    ),
                    [0],
                ),
                (
                    "array",
                    sklearn.preprocessing.FunctionTransformer(
                        func=array_function_with_single_name
                    ),
                    [0],
                ),
            ],
            n_jobs=3,
        )

        result = ctf.fit_transform(domain)
        result_dtype = result.dtype
        expected_dtype = np.dtype(
            [
                ("pass", domain.dtype),
                ("domain_split", "datetime64[h]"),
                ("one", np.int32),
                ("two", np.int32),
                ("three", np.int32),
                ("magic", array_type),
            ]
        )

        self.assertEqual(result_dtype, expected_dtype)

        expected_array = np.ndarray((15, 1), dtype=expected_dtype)
        expected_array["pass"] = domain
        expected_array["domain_split"] = domain[DATE_TIME]
        expected_array["one"][:, 0] = np.arange(15, dtype=np.int32) * 3
        expected_array["two"][:, 0] = np.arange(15, dtype=np.int32) * 3 + 1
        expected_array["three"][:, 0] = np.arange(15, dtype=np.int32) * 3 + 2

        expected_array["magic"]["array"][:, 0] = np.arange(15 * 3 * 3).reshape(15, 3, 3)

        np.testing.assert_array_equal(result, expected_array)

        function_with_feature_names.get_feature_names = lambda: ["one", "two"]

        with self.assertRaises(NotImplementedError):
            ctf.fit_transform(domain)

    def test_nullify_cols__1d_raises(self):
        arr = np.arange(0, 9)
        with self.assertRaises(ValueError):
            nullify_cols(arr)

    def test_nullify_cols__array(self):
        arr = np.arange(0, 9).reshape(-1, 1)
        res = nullify_cols(arr)
        np.testing.assert_array_equal(arr, res)
        self.assertIsNot(arr, res)
        self.assertIs(res.dtype, np.dtype(int))

        res = nullify_cols(arr, [])
        np.testing.assert_array_equal(arr, res)
        self.assertIsNot(arr, res)
        self.assertIs(res.dtype, np.dtype(int))

        res = nullify_cols(arr, cols=[0])
        expected = np.repeat(np.nan, 9).reshape(-1, 1)
        np.testing.assert_array_equal(res, expected)
        self.assertIs(res.dtype, np.dtype(float))

    def test_nullify_cols__bad_dtype_raises(self):
        arr = np.array([["one", "two"], ["three", "four"]])
        np.testing.assert_array_equal(
            arr, nullify_cols(arr)
        )  # works if no modification required
        with self.assertRaises(ValueError):  # else raises
            nullify_cols(arr, cols=[0])

    def test_nullify_cols__ndarray(self):
        arr = np.arange(0, 9).reshape(3, 3)
        res = nullify_cols(arr)
        np.testing.assert_array_equal(arr, res)
        self.assertIsNot(arr, res)
        self.assertIs(res.dtype, np.dtype(int))

        res = nullify_cols(arr, [])
        np.testing.assert_array_equal(arr, res)
        self.assertIsNot(arr, res)
        self.assertIs(res.dtype, np.dtype(int))

        res = nullify_cols(arr, cols=[0, 2])
        expected = np.array(
            [[np.nan, 1.0, np.nan], [np.nan, 4, np.nan], [np.nan, 7.0, np.nan]]
        )
        np.testing.assert_array_equal(res, expected)
        self.assertIs(res.dtype, np.dtype(float))

    def nan_like_tester(self, arr):
        res = nan_like(arr)
        self.assertTrue(np.all(np.isnan(res)))
        self.assertTupleEqual(arr.shape, res.shape)
        self.assertIs(res.dtype, np.dtype(float))

    def test_nan_like__feature_names(self):
        result = nan_like.get_feature_names()
        self.assertListEqual(result, ["nan"])

    def test_nan_like__array(self):
        # an array should be replaced with nan values while retaining the original shape
        arr = np.zeros(5)
        self.nan_like_tester(arr)

    def test_nan_like__ndarray(self):
        # an ndarray should be replaced with nan values while retaining the original shape
        arr = np.ones((2, 3, 4))
        self.nan_like_tester(arr)

    def test_nan_like__int_array(self):
        arr = np.array([1, 2, 3])
        self.nan_like_tester(arr)

    def test_nan_like__datetime_array(self):
        arr = np.arange(
            "2021-01-01", "2021-12-31", np.timedelta64(1, "M"), dtype="datetime64[M]"
        )
        self.nan_like_tester(arr)

    def test_nan_like__object_array(self):
        arr = np.ones(3, dtype=object)
        self.nan_like_tester(arr)

    def test_drop_first_column(self):
        arr = np.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            drop_first_column(arr)

        arr = np.array([[1, 2, 3, 4]])
        result = drop_first_column(arr)
        np.testing.assert_array_equal(result, np.array([[2, 3, 4]]))

        arr = np.array([[1, 2], [3, 4]])
        result = drop_first_column(arr)
        np.testing.assert_array_equal(result, np.array([[2], [4]]))
        np.testing.assert_array_equal(result, arr[:, 1:])

    def test_first_column_only(self):
        arr = np.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            first_column_only(arr)

        arr = np.array([[1, 2, 3, 4]])
        result = first_column_only(arr)
        np.testing.assert_array_equal(result, np.array([[1]]))

        arr = np.array([[1, 2], [3, 4]])
        result = first_column_only(arr)
        np.testing.assert_array_equal(result, np.array([[1], [3]]))
        # expect result is a copy, not a view
        np.testing.assert_array_equal(result, arr[:, [0]])
        self.assertIsNot(result, arr[:, [0]])

    def test_fill_nan_columns__default(self):
        arr = np.array(
            [
                [np.nan, 1, 2, np.nan],
                [4, np.nan, np.nan, 7],
            ]
        )
        result = fill_nan_columns(arr)
        expected = np.array(
            [
                [0.0, 1.0, 2.0, 0.0],
                [4.0, 0.0, 0.0, 7.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        # note, we are modifying the original array
        self.assertIs(result, arr)

    def test_fill_nan_columns__col_to(self):
        arr = np.array(
            [
                [np.nan, 1, 2, np.nan],
                [4, np.nan, np.nan, 7],
            ]
        )
        result = fill_nan_columns(arr, col_to=2)
        expected = np.array(
            [
                [0.0, 1.0, 2.0, np.nan],
                [4.0, 0.0, np.nan, 7.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_fill_nan_columns__col_from(self):
        arr = np.array(
            [
                [np.nan, 1, 2, np.nan],
                [4, np.nan, np.nan, 7],
            ]
        )
        result = fill_nan_columns(arr, col_from=2)
        expected = np.array(
            [
                [np.nan, 1.0, 2.0, 0.0],
                [4.0, np.nan, 0.0, 7.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_fill_nan_columns__col_from_to(self):
        arr = np.array(
            [
                [np.nan, 1, 2, np.nan],
                [4, np.nan, np.nan, 7],
            ]
        )
        result = fill_nan_columns(arr, col_from=1, col_to=3)
        expected = np.array(
            [
                [np.nan, 1.0, 2.0, np.nan],
                [4.0, 0.0, 0.0, 7.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_count_domain_locs_into(self):
        def _test_count_domain_locs(domain, n_expected):
            res = count_domain_locs_into_column(domain)
            self.assertIsInstance(res, np.ndarray)
            self.assertTrue((res == n_expected).all())
            self.assertTupleEqual(res.shape, (len(domain), 1))

        d = make_domain("2021-01-01", "2022-01-01", np.timedelta64(1, "W"), "a")
        _test_count_domain_locs(domain=d, n_expected=1)

        d = make_domain(
            "2021-01-01T04", "2022-01-01T04", np.timedelta64(1, "D"), "a", "b"
        )
        _test_count_domain_locs(domain=d, n_expected=2)

        d = make_domain(
            "2021-01-01", "2022-01-01", np.timedelta64(1, "h"), "a", "b", "c"
        )
        _test_count_domain_locs(domain=d, n_expected=3)

    @patch("time_series_models.transformers.autoregressive_features_pipeline")
    @patch("time_series_models.transformers.first_column_only")
    def test_overfetched_range_pipeline(self, mock_first_column, mock_ar_pipeline):
        mock_ar_pipeline_fit_transform_output = MagicMock()
        mock_ar_pipeline.return_value.fit_transform.return_value = (
            mock_ar_pipeline_fit_transform_output
        )

        fetcher = AmiFetcher(location_mapper=lambda x: x)
        p = overfetched_range_pipeline(
            fetcher=fetcher,
            time_step=np.timedelta64(1, "h"),
            lags=np.array([24, 25, 26, 168, 169], dtype=np.timedelta64(1, "h")),
            n_jobs=2,
        )
        self.assertIsInstance(p, sklearn.pipeline.Pipeline)

        # expect that the autoregressive_features_pipeline is called only with lag 0 and max(lags),
        # along with fetcher and tstep_dtype
        last_call = mock_ar_pipeline.call_args_list[-1]
        self.assertEqual(len(last_call), 2)
        self.assertEqual(len(last_call[0]), 2)
        self.assertIs(last_call[0][0], fetcher)
        np.testing.assert_array_equal(
            # expect lags 0 and 169 here
            last_call[0][1],
            np.array([0, 169], dtype=np.timedelta64(1, "h")),
        )
        self.assertDictEqual(
            last_call[1], {"time_step": np.timedelta64(1, "h"), "n_jobs": 2}
        )

        p.fit_transform("camus")
        # second pipeline step should be called with output of first step
        mock_first_column.assert_called_with(mock_ar_pipeline_fit_transform_output)

        self.assertListEqual(
            list(p.named_steps.keys()), ["overfetch_and_cache", "just_the_range"]
        )
        self.assertIsInstance(
            p.named_steps["just_the_range"], sklearn.preprocessing.FunctionTransformer
        )

    @patch("time_series_models.transformers.autoregressive_features_pipeline")
    def test_overfetched_range_pipeline__one_lag(self, mock_ar_pipeline):
        mock_ar_pipeline_fit_transform_output = MagicMock()
        mock_ar_pipeline.return_value.fit_transform.return_value = (
            mock_ar_pipeline_fit_transform_output
        )

        fetcher = AmiFetcher(location_mapper=lambda x: x)
        p = overfetched_range_pipeline(
            fetcher=fetcher,
            time_step=np.timedelta64(1, "D"),
            lags=np.array([24], dtype=np.timedelta64(1, "D")),
        )
        self.assertIsInstance(p, sklearn.pipeline.Pipeline)

        # expect that the autoregressive_features_pipeline is called only with lag 0 and max(lags),
        # along with fetcher and tstep_dtype
        last_call = mock_ar_pipeline.call_args_list[-1]
        self.assertEqual(len(last_call), 2)
        self.assertEqual(len(last_call[0]), 2)
        self.assertIs(last_call[0][0], fetcher)
        np.testing.assert_array_equal(
            # expect lags 0 and 24 here
            last_call[0][1],
            np.array([0, 24], dtype=np.timedelta64(1, "D")),
        )
        self.assertDictEqual(
            last_call[1], {"n_jobs": None, "time_step": np.timedelta64(1, "D")}
        )

    @patch("time_series_models.transformers.autoregressive_features_pipeline")
    def test_overfetched_range_pipeline__none_lag(self, mock_ar_pipeline):
        mock_ar_pipeline_fit_transform_output = MagicMock()
        mock_ar_pipeline.return_value.fit_transform.return_value = (
            mock_ar_pipeline_fit_transform_output
        )

        fetcher = AmiFetcher(location_mapper=lambda x: x)
        p = overfetched_range_pipeline(
            fetcher=fetcher,
            time_step=np.timedelta64(1, "D"),
            lags=None,
        )
        self.assertIsInstance(p, sklearn.pipeline.Pipeline)

        # expect that the autoregressive_features_pipeline is called only with lag 0 and max(lags),
        # along with fetcher and tstep_dtype
        last_call = mock_ar_pipeline.call_args_list[-1]
        self.assertEqual(len(last_call), 2)
        self.assertEqual(len(last_call[0]), 2)
        self.assertIs(last_call[0][0], fetcher)
        np.testing.assert_array_equal(
            # expect lags 0 and 0 here
            last_call[0][1],
            np.array([0, 0], dtype=np.timedelta64(1, "D")),
        )
        self.assertDictEqual(
            last_call[1], {"n_jobs": None, "time_step": np.timedelta64(1, "D")}
        )

    def test_overfetched_range_pipeline__misspecified_lags(self):
        fetcher = AmiFetcher(location_mapper=lambda x: x)

        # empty array raises
        with self.assertRaises(ValueError):
            overfetched_range_pipeline(
                fetcher,
                lags=np.array([], dtype="datetime64[D]"),
                time_step=np.timedelta64(1, "h"),
            )

        # empty list raises
        with self.assertRaises(ValueError):
            overfetched_range_pipeline(
                fetcher,
                lags=[],
                time_step=np.timedelta64(1, "h"),
            )

    def test_revise_pipeline(self):
        # revising a pipeline with a "passthrough" or None transformer should result in the step being replaced.
        first_transformer = sklearn.preprocessing.FunctionTransformer(func=np.mean)
        new_step = ("step_one", first_transformer)

        # check both None and "passthrough" values of the transformer step
        for transform in [None, "passthrough"]:
            with self.subTest(transform=transform):
                pipe = sklearn.pipeline.Pipeline([("any_name_here", transform)])
                result = revise_pipeline(pipe, new_step)
                # we should get a new pipeline with a single (new) step
                self.assertIsInstance(result, sklearn.pipeline.Pipeline)
                self.assertIsNot(result, pipe)
                self.assertEqual(len(result.steps), 1)
                self.assertEqual(result.steps[0][0], "step_one")
                self.assertIs(result.steps[0][1], first_transformer)

        # try appending a step to a pipeline with a non-"passthrough" transformer
        pipe = sklearn.pipeline.Pipeline([("step_one", first_transformer)])
        second_transformer = sklearn.preprocessing.FunctionTransformer(func=np.sqrt)
        new_step = ("step_two", second_transformer)
        next_result = revise_pipeline(pipe, new_step)

        # since we are revising a pipeline with a non-"passthrough" transformer, new result should be a pipeline with
        # two steps: the first is the same as from the old pipeline, and the second should be the latest step
        self.assertIsInstance(next_result, sklearn.pipeline.Pipeline)
        self.assertIsNot(next_result, result)
        self.assertEqual(len(next_result.steps), 2)

        # the first step
        self.assertEqual(next_result.steps[0][0], "step_one")
        self.assertIs(next_result.steps[0][1], first_transformer)

        # the second step:
        self.assertEqual(next_result.steps[1][0], "step_two")
        self.assertIs(next_result.steps[1][1], second_transformer)

    def assert_equal_metrics_dict(self, d1, d2):
        # copied from test_data_monitor.py
        self.assertSetEqual(set(d1.keys()), set(d2.keys()))
        for k in d1.keys():
            if type(d1[k]) in {np.array, np.ndarray}:
                np.testing.assert_almost_equal(d1[k], d2[k])
            else:
                self.assertEqual(d1[k], d2[k])

    @patch.object(
        AmiFetcher,
        "get_data",
        return_value=np.array([[10.0, 0.0], [8.0, 0.0], [3.0, 1.0]]),
    )
    def test_monitor_fetcher__default_monitor(self, mock_get_data):
        fetcher = AmiFetcher(location_mapper=lambda x: x, units="energy")
        monitor = ForecastDataMonitor()
        pipe = monitor_fetcher(fetcher, monitor)
        self.assertIsInstance(pipe, sklearn.pipeline.Pipeline)
        self.assertListEqual(list(pipe.named_steps.keys()), ["fetch", "monitor"])

        domain = make_domain(
            "2023-02-10", "2023-02-12", np.timedelta64(1, "D"), "Ocelot"
        )
        result = pipe.fit_transform(domain)
        np.testing.assert_array_equal(
            result,
            np.array([[10.0, 0.0], [8.0, 0.0], [3.0, 1.0]]),
        )

        fit_stats = pipe.named_steps["monitor"].fit_stats
        self.assertIsInstance(fit_stats, dict)
        self.assert_equal_metrics_dict(
            fit_stats,
            {
                "feature_n_obs": 3,
                "feature_n_missing": np.array([0.0, 0.0]),
                "feature_missing": np.array([0.0, 0.0]),
                "feature_trailing_n_missing": np.array([0, 0]),
                "feature_mean": np.array([7.0, 1 / 3]),
                "feature_min": np.array([3.0, 0.0]),
                "feature_max": np.array([10.0, 1.0]),
                "feature_q0.05": np.array([3.5, 0.0]),
                "feature_q0.25": np.array([5.5, 0.0]),
                "feature_q0.75": np.array([9.0, 0.5]),
                "feature_q0.95": np.array([9.8, 0.9]),
            },
        )
        transform_stats = pipe.named_steps["monitor"].transform_stats
        self.assertIsInstance(transform_stats, dict)
        self.assert_equal_metrics_dict(
            transform_stats,
            {
                "feature_n_obs": 3,
                "feature_n_missing": np.array([0.0, 0.0]),
                "feature_missing": np.array([0.0, 0.0]),
                "feature_trailing_n_missing": np.array([0, 0]),
                "feature_min": np.array([3.0, 0.0]),
                "feature_max": np.array([10.0, 1.0]),
            },
        )

        self.assertDictEqual(pipe.named_steps["monitor"].fit_stats_by_loc, {})
        self.assertDictEqual(pipe.named_steps["monitor"].transform_stats_by_loc, {})

    @patch.object(
        AmiFetcher,
        "get_data",
        return_value=np.array([[10.0, 0.0], [8.0, 0.0], [3.0, 1.0]]),
    )
    def test_monitor_fetcher__use_locs(self, mock_get_data):
        fetcher = AmiFetcher(location_mapper=lambda x: x, units="watts")
        monitor = ForecastDataMonitor(use_locs=True)
        pipe = monitor_fetcher(fetcher, monitor)
        self.assertIsInstance(pipe, sklearn.pipeline.Pipeline)
        self.assertListEqual(
            list(pipe.named_steps.keys()), ["count_and_fetch", "monitor", "drop_column"]
        )
        self.assertIsInstance(
            pipe.named_steps["count_and_fetch"], sklearn.compose.ColumnTransformer
        )
        # no named_transformers_ until after calling fit, so check those later

        domain = make_domain(
            "2023-02-10", "2023-02-12", np.timedelta64(1, "D"), "Ocelot"
        )
        result = pipe.fit_transform(domain)

        self.assertListEqual(
            list(pipe.named_steps["count_and_fetch"].named_transformers_.keys()),
            ["count", "fetch"],
        )

        # expect same result overall
        np.testing.assert_array_equal(
            result,
            np.array([[10.0, 0.0], [8.0, 0.0], [3.0, 1.0]]),
        )

        fit_stats = pipe.named_steps["monitor"].fit_stats
        self.assertIsInstance(fit_stats, dict)
        self.assert_equal_metrics_dict(
            fit_stats,
            {
                "feature_n_obs": 3,
                "feature_n_missing": np.array([0.0, 0.0, 0.0]),
                "feature_missing": np.array([0.0, 0.0, 0.0]),
                "feature_trailing_n_missing": np.array([0, 0, 0]),
                "feature_mean": np.array([1.0, 7.0, 1 / 3]),
                "feature_min": np.array([1.0, 3.0, 0.0]),
                "feature_max": np.array([1.0, 10.0, 1.0]),
                "feature_q0.05": np.array([1.0, 3.5, 0.0]),
                "feature_q0.25": np.array([1.0, 5.5, 0.0]),
                "feature_q0.75": np.array([1.0, 9.0, 0.5]),
                "feature_q0.95": np.array([1.0, 9.8, 0.9]),
            },
        )
        transform_stats = pipe.named_steps["monitor"].transform_stats
        self.assertIsInstance(transform_stats, dict)
        self.assert_equal_metrics_dict(
            transform_stats,
            {
                "feature_n_obs": 3,
                "feature_n_missing": np.array([0.0, 0.0, 0.0]),
                "feature_missing": np.array([0.0, 0.0, 0.0]),
                "feature_trailing_n_missing": np.array([0, 0, 0]),
                "feature_min": np.array([1.0, 3.0, 0.0]),
                "feature_max": np.array([1.0, 10.0, 1.0]),
            },
        )

        fit_stats_by_loc = pipe.named_steps["monitor"].fit_stats_by_loc
        self.assertIsInstance(fit_stats_by_loc, dict)
        self.assertSetEqual(
            set(fit_stats_by_loc.keys()),
            {
                "feature_n_obs",
                "feature_missing",
                "feature_n_missing",
                "feature_trailing_n_missing",
                "feature_mean",
                "feature_min",
                "feature_max",
            },
        )
        # expect one location -> one row in array
        self.assertEqual(len(fit_stats_by_loc["feature_missing"]), 1)
        np.testing.assert_array_equal(
            fit_stats_by_loc["feature_max"], np.array([[1.0, 10.0, 1.0]])
        )

        transform_stats_by_loc = pipe.named_steps["monitor"].transform_stats_by_loc
        self.assertIsInstance(transform_stats_by_loc, dict)
        self.assertSetEqual(
            set(transform_stats_by_loc.keys()),
            {
                "feature_n_obs",
                "feature_missing",
                "feature_n_missing",
                "feature_trailing_n_missing",
                "feature_min",
                "feature_max",
            },
        )

    @patch.object(
        AmiFetcher,
        "get_data",
        return_value=np.array([[10.0, 0.0], [8.0, 0.0], [3.0, 1.0]]),
    )
    def test_net_energy_pipeline(self, mock_get_data):
        fetcher = AmiFetcher(location_mapper=lambda x: x, units="energy")
        p = net_energy_pipeline(fetcher)
        self.assertIsInstance(p, sklearn.pipeline.Pipeline)
        self.assertEqual(list(p.named_steps.keys()), ["fetch", "combine"])
        self.assertIs(p.named_steps["fetch"], fetcher)
        self.assertIsInstance(
            p.named_steps["combine"], sklearn.preprocessing.FunctionTransformer
        )

        some_domain = make_domain(
            "2023-02-08T01", "2023-02-08T03", np.timedelta64(1, "h"), "Capy"
        )
        fetch_result = p.named_steps["fetch"].fit_transform(some_domain)
        np.testing.assert_array_equal(
            fetch_result, np.array([[10.0, 0.0], [8.0, 0.0], [3.0, 1.0]])
        )

        transform_result = p.named_steps["combine"].fit_transform(fetch_result)
        expected = np.array([[10.0 - 0.0], [8.0 - 0.0], [3.0 - 1.0]])
        np.testing.assert_array_equal(transform_result, expected)

        pipeline_result = p.fit_transform(some_domain)
        np.testing.assert_array_equal(pipeline_result, expected)


class RowFilteringFunctionTransformerTests(unittest.TestCase):
    def test_init(self):
        ft = RowFilteringFunctionTransformer()
        self.assertIsInstance(ft, sklearn.preprocessing.FunctionTransformer)
        self.assertIsNone(ft.func)
        self.assertIsNone(ft.inverse_func)

    def test_fit_transform(self):
        x = np.array([[1, 10], [2, 20], [3, 30]])
        y = np.array(["a", "b"])
        range_nan = np.array([False, True, False])

        ft = RowFilteringFunctionTransformer()
        result = ft.fit_transform(x, y=y, range_nan=range_nan)
        expected = np.array(
            [
                [1, 10],
                [3, 30],
            ]
        )
        # second row should be dropped from x, then FunctionTransformer fit_transform on the filtered result is a no-op
        np.testing.assert_array_equal(result, expected)

    def test_fit_transform__assertionerror(self):
        x = np.array([[1, 10], [2, 20], [3, 30]])
        y = np.array(["a", "b", "c"])
        range_nan = np.array([False, True, False])
        ft = RowFilteringFunctionTransformer()
        with self.assertRaises(AssertionError):
            # len(y) != len(x[~range_nan])
            ft.fit_transform(x, y=y, range_nan=range_nan)

        x = np.array([[1, 10], [2, 20], [3, 30]])
        y = np.array(["a", "b"])
        range_nan = np.array([True, False])
        ft = RowFilteringFunctionTransformer()
        with self.assertRaises(AssertionError):
            # len(x) != len(range_nan)
            ft.fit_transform(x, y=y, range_nan=range_nan)


if __name__ == "__main__":
    import logging
    import time

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    unittest.main()
