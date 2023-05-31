import datetime
import unittest

import numpy as np
import pandas as pd

from time_series_models.gap_filler import (
    backfill_singleton,
    get_tstep,
    return_last_observation,
    GapFiller,
)


def assert_df_changes(df_old, df_new, df_exp_changes):
    """
    Helper method to show where a dataframe should or should not have been modified
    :param df_old: the original dataframe
    :param df_new: the resulting dataframe
    :param df_exp_changes: a dataframe containing just the columns with expected changes
    :return:
    """
    pd.testing.assert_index_equal(df_old.columns, df_new.columns)
    pd.testing.assert_index_equal(df_old.index, df_new.index)
    for col in df_new.columns:
        if col in df_exp_changes:
            # columns where we expect changes -> new matches expected changes
            pd.testing.assert_series_equal(df_new[col], df_exp_changes[col])
        else:
            # columns where we don't expect changes -> new matches old
            pd.testing.assert_series_equal(df_new[col], df_old[col])


class TestGapRunnerFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
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

        self.complete_df = pd.DataFrame(
            {
                "x": [5, 4, 3, 2],
                "y": [2, 4, 6, 8],
            }
        )

    def test_backfill_singleton(self):
        # singleton gaps should be filled unless they are the last observation in the series
        result1 = backfill_singleton(self.df)
        expected = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, np.nan],
                "b": [2.0, np.nan, np.nan, 8.0],
                "c": [np.nan, np.nan, 10.0, 20.0],
                "d": [4.0, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, np.nan],
                "g": [2.0, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, np.nan],
            }
        )
        pd.testing.assert_frame_equal(result1, expected)

        # the columns with expected changes:
        df_expected_changes = pd.DataFrame(
            {
                "d": [4.0, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, np.nan],
                "g": [2.0, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, np.nan],
            }
        )
        assert_df_changes(self.df, result1, df_expected_changes)

        # repeating the call should result in no-op since no singletons remain
        result2 = backfill_singleton(result1)
        pd.testing.assert_frame_equal(result2, expected)
        assert_df_changes(result1, result2, pd.DataFrame())

        # backfilling a complete df is a no-op
        result3 = backfill_singleton(self.complete_df)
        pd.testing.assert_frame_equal(result3, self.complete_df)

    def test_get_tstep(self):
        def compare_tstep(array, expected_tstep):
            """helper function to make_tstep from array
            and compare to expected_tstep"""
            tstep = get_tstep(array)
            self.assertEqual(tstep, expected_tstep)

        arr = pd.date_range(start="2022-05-04", end="2022-05-06", freq="D")
        compare_tstep(arr, pd.Timedelta(1, "D"))

        arr = pd.date_range(start="2022-05-04", end="2022-05-06", freq="H")
        compare_tstep(arr, pd.Timedelta(1, "H"))

        arr = np.array(
            [
                datetime.date(2022, 5, 4),
                datetime.date(2022, 5, 6),
                datetime.date(2022, 6, 4),
            ]
        )
        with self.assertWarns(RuntimeWarning):
            compare_tstep(arr, datetime.timedelta(days=2))

        # get_tstep currently does not play nice with pd.Series with RangeIndex
        series = pd.Series([1, 2])
        with self.assertRaises(KeyError):
            get_tstep(series)

    def test_return_last_observation(self):
        arr = np.array([20.0, 10.0, 30.0])
        result = return_last_observation(arr)
        self.assertEqual(result, 30.0)

        arr[-1] = np.nan
        result = return_last_observation(arr)
        self.assertEqual(result, 10.0)

        arr[:] = np.nan
        result = return_last_observation(arr)
        self.assertIs(np.nan, result)

        arr = np.array(
            [
                [1, 2],
                [3, 4],
            ],
        )
        with self.assertRaises(RuntimeError):
            return_last_observation(arr)


class TestGapFiller(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
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

        self.complete_df = pd.DataFrame(
            {
                "x": [5, 4, 3, 2],
                "y": [2, 4, 6, 8],
            }
        )

        self.gap_filler = GapFiller(self.df)

    def test_init(self):
        gap_filler = GapFiller(self.df)
        pd.testing.assert_frame_equal(gap_filler._original, self.df)
        self.assertIs(gap_filler._original, self.df)
        self.assertDictEqual(gap_filler.models, {})

    def test_original(self):
        # test the self.original property
        gap_filler = GapFiller(self.df)
        pd.testing.assert_frame_equal(gap_filler.original, self.df)
        self.assertIsNot(gap_filler.original, self.df)

    def test_fill_gaps_rolling_1d(self):
        df = self.df.copy()
        df.index = pd.date_range(start=1, end=4, periods=4)

        # each np.nan should be replaced by the last observed (or gap-filled) preceding value in the series
        expected = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 2.0, 2.0],
                "b": [2.0, 2.0, 2.0, 8.0],
                "c": [np.nan, np.nan, 10.0, 20.0],
                "d": [np.nan, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 4.0, 8.0],
                "f": [3.0, 5.0, 7.0, 7.0],
                "g": [np.nan, 2.0, 2.0, 4.0],
                "h": [0.0, 0.0, 2.0, 2.0],
            },
            index=df.index,
        )

        for col in df.columns:
            series = df[col]
            result = GapFiller.fill_gaps_rolling_1d(
                series, fill_method=return_last_observation, fit_len=2
            )
            pd.testing.assert_series_equal(result, expected[col])

    def test_fill_gaps_rolling_1d__complete_df(self):
        df = self.complete_df.copy()
        df.index = pd.date_range(start=1, end=4, periods=4)
        for col in df.columns:
            series = df[col]
            result = GapFiller.fill_gaps_rolling_1d(
                series, fill_method=return_last_observation, fit_len=2
            )
            # expect no changes to an already-complete series
            pd.testing.assert_series_equal(result, series)

    def test_pre_interpolated(self):
        gap_filler = GapFiller.pre_interpolated(self.df)
        self.assertIsInstance(gap_filler, GapFiller)
        # expect that GapFiller is instantiated with the return value of the
        # original data having been pre-processed with 'backfill_singleton'
        expected = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, np.nan],
                "b": [2.0, np.nan, np.nan, 8.0],
                "c": [np.nan, np.nan, 10.0, 20.0],
                "d": [4.0, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, np.nan],
                "g": [2.0, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, np.nan],
            }
        )
        pd.testing.assert_frame_equal(expected, gap_filler.original)

        # the columns with expected changes:
        df_expected_changes = pd.DataFrame(
            {
                "d": [4.0, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, np.nan],
                "g": [2.0, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, np.nan],
            }
        )
        assert_df_changes(self.df, gap_filler.original, df_expected_changes)

    def test_fill_with_interpolation__default(self):
        result = self.gap_filler.fill_with_interpolation()
        # the default setting is pd.DataFrame.interpolate with limit_direction="both"
        expected = pd.DataFrame(
            {
                "a": [1.0, 2.0, 2.0, 2.0],
                "b": [2.0, 4.0, 6.0, 8.0],
                "c": [10.0, 10.0, 10.0, 20.0],
                "d": [4.0, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, 7.0],
                "g": [2.0, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, 2.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

        # the columns with expected changes: all columns have gaps to fill
        df_expected_changes = pd.DataFrame(
            {
                "a": [1.0, 2.0, 2.0, 2.0],
                "b": [2.0, 4.0, 6.0, 8.0],
                "c": [10.0, 10.0, 10.0, 20.0],
                "d": [4.0, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, 7.0],
                "g": [2.0, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, 2.0],
            }
        )
        assert_df_changes(self.df, result, df_expected_changes)

    def test_fill_with_interpolation__forward(self):
        result = self.gap_filler.fill_with_interpolation(limit_direction="forward")
        # expect that kwargs can be passed to pd.DataFrame.interpolate correctly, e.g. to do forward-fill only
        expected = pd.DataFrame(
            {
                "a": [1.0, 2.0, 2.0, 2.0],
                "b": [2.0, 4.0, 6.0, 8.0],
                "c": [np.nan, np.nan, 10.0, 20.0],
                "d": [np.nan, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, 7.0],
                "g": [np.nan, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, 2.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

        # the columns with expected changes:
        df_expected_changes = pd.DataFrame(
            {
                "a": [1.0, 2.0, 2.0, 2.0],
                "b": [2.0, 4.0, 6.0, 8.0],
                "e": [2.0, 4.0, 6.0, 8.0],
                "f": [3.0, 5.0, 7.0, 7.0],
                "g": [np.nan, 2.0, 3.0, 4.0],
                "h": [0.0, 1.0, 2.0, 2.0],
            }
        )
        assert_df_changes(self.df, result, df_expected_changes)

    def test_fill_with_rolling_series_method(self):
        df = self.df.copy()
        df.index = pd.date_range(start=1, end=4, periods=4)
        # get_tstep (a required helper method) does not like RangeIndex,
        # so instantiate a gap_filler with a modified df
        gap_filler = GapFiller(df)
        result = gap_filler.fill_with_rolling_series_method(
            return_last_observation, fit_len=2
        )
        # expect that return_last_observation has been rolled through each series in the dataframe,
        # so all gaps following an observation persist the last observed value
        expected = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 2.0, 2.0],
                "b": [2.0, 2.0, 2.0, 8.0],
                "c": [np.nan, np.nan, 10.0, 20.0],
                "d": [np.nan, 4.0, 3.0, 2.0],
                "e": [2.0, 4.0, 4.0, 8.0],
                "f": [3.0, 5.0, 7.0, 7.0],
                "g": [np.nan, 2.0, 2.0, 4.0],
                "h": [0.0, 0.0, 2.0, 2.0],
            },
            index=df.index,
        )
        pd.testing.assert_frame_equal(result, expected)

        # the columns with expected changes:
        df_expected_changes = pd.DataFrame(
            {
                "a": [1.0, 2.0, 2.0, 2.0],
                "b": [2.0, 2.0, 2.0, 8.0],
                "e": [2.0, 4.0, 4.0, 8.0],
                "f": [3.0, 5.0, 7.0, 7.0],
                "g": [np.nan, 2.0, 2.0, 4.0],
                "h": [0.0, 0.0, 2.0, 2.0],
            },
            index=pd.date_range(start=1, end=4, periods=4),
        )
        assert_df_changes(df, result, df_expected_changes)


if __name__ == "__main__":
    unittest.main()
