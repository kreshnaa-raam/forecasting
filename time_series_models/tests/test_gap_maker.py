import unittest

import numpy as np
import pandas as pd

from time_series_models.gap_maker import (
    GapMaker1D,
    GapMaker,
)


class TestGapMaker1D(unittest.TestCase):
    def test_init(self):
        series = pd.Series([1.0, np.nan], name="test data")
        gm = GapMaker1D(series)

        pd.testing.assert_series_equal(series, gm._series)
        self.assertIs(series, gm._series)
        self.assertSetEqual(gm.series_nan_idx_set, {1})
        pd.testing.assert_index_equal(gm.series_notna_idx, pd.Index([0]))
        self.assertIsNone(gm.synth_gap_idx)

    def test_series(self):
        # test that the series property returns a copy of the Series used in the constructor
        series = pd.Series([1.0, np.nan], name="test data")
        gm = GapMaker1D(series)
        pd.testing.assert_series_equal(series, gm.series)
        self.assertIsNot(series, gm.series)

    def test_make_random_gaps__frac(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        gm1 = GapMaker1D(series)
        series_with_gaps = gm1._make_random_gaps(frac=0.4, random_state=1)

        # expect synth_gap_idx shows where gapmaker made random gaps
        expected_idx = series_with_gaps.loc[series_with_gaps.isna()].index
        pd.testing.assert_index_equal(gm1.synth_gap_idx.sort_values(), expected_idx)

        # the original Series had no gaps so there should now be 4 gaps (frac=0.4 with 10 observations)
        total_nan = series_with_gaps.isna().sum()
        self.assertEqual(4, total_nan)
        self.assertEqual(4, len(gm1.synth_gap_idx))

        gm2 = GapMaker1D(series)
        # the same sampling method with a different random state can return a different arrangement of 4 gaps
        series_with_other_gaps = gm2._make_random_gaps(frac=0.4, random_state=2)
        self.assertEqual(4, len(gm2.synth_gap_idx))
        with self.assertRaises(AssertionError):
            pd.testing.assert_series_equal(series_with_gaps, series_with_other_gaps)
        with self.assertRaises(AssertionError):
            pd.testing.assert_index_equal(gm1.synth_gap_idx, gm2.synth_gap_idx)

    def test_make_random_gaps__n(self):
        # same as test_make_random_gaps__frac, but this time specify number of gaps with 'n' parameter
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        gm = GapMaker1D(series)
        series_with_gaps = gm._make_random_gaps(n=5, random_state=1)

        expected_idx = series_with_gaps.loc[series_with_gaps.isna()].index
        pd.testing.assert_index_equal(gm.synth_gap_idx.sort_values(), expected_idx)

        total_nan = series_with_gaps.isna().sum()
        self.assertEqual(5, total_nan)
        self.assertEqual(5, len(gm.synth_gap_idx))

    def test_make_random_gaps__with_prior_gaps__frac(self):
        series = pd.Series([1, 2, 3, 4, np.nan, 6, 7, 8, 9, np.nan, 11, 12])
        gm = GapMaker1D(series)
        series_with_gaps = gm._make_random_gaps(frac=0.2, random_state=1)

        # since the series already had missing values, synth_gap_idx should show only the locations of new gaps
        expected_idx = series_with_gaps.loc[
            series_with_gaps.isna() & series.notna()
        ].index
        pd.testing.assert_index_equal(gm.synth_gap_idx, expected_idx)

        # 2 original missing values + 2 synthetic gaps = 4 total missing values
        total_nan = series_with_gaps.isna().sum()
        self.assertEqual(4, total_nan)
        self.assertEqual(2, len(gm.synth_gap_idx))

    def test_make_random_gaps__with_prior_gaps__n(self):
        # same as test_make_random_gaps__with_prior_gaps__frac, but this time specify number of gaps with 'n' parameter
        series = pd.Series([1, 2, 3, 4, np.nan, 6, 7, 8, 9, np.nan, 11, 12])
        gm = GapMaker1D(series)
        series_with_gaps = gm._make_random_gaps(n=1, random_state=1)

        expected_idx = series_with_gaps.loc[
            series_with_gaps.isna() & series.notna()
        ].index
        pd.testing.assert_index_equal(gm.synth_gap_idx, expected_idx)

        # 2 original missing values + 1 synthetic gap = 3 total missing values
        total_nan = series_with_gaps.isna().sum()
        self.assertEqual(3, total_nan)
        self.assertEqual(1, len(gm.synth_gap_idx))

    def test_make_block_gaps(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        gm = GapMaker1D(series)

        # default params result in no-op
        series_with_gaps = gm._make_block_gaps()
        pd.testing.assert_series_equal(series, series_with_gaps)

        # can also be a no-op due to rounding, except for type conversion
        series_with_gaps = gm._make_block_gaps(block_size=4, frac=0.1, random_state=0)
        pd.testing.assert_series_equal(series.astype(float), series_with_gaps)

        # manually control n_gaps:
        series_with_gaps = gm._make_block_gaps(block_size=4, n_blocks=2, random_state=0)
        expected = pd.Series(
            [
                1.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                6.0,
                7.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )
        pd.testing.assert_series_equal(series_with_gaps, expected)

        # finding a solution is not guaranteed:
        # in current configuration, random_state=3 does not succeed after 3 attempts
        with self.assertRaises(RuntimeError):
            gm._make_block_gaps(block_size=4, n_blocks=2, random_state=3)

        # but more attempts can help (we don't necessarily use all 15 attempts here, see DEBUG logs)
        gm._make_block_gaps(block_size=4, n_blocks=2, random_state=3, n_attempts=15)
        self.assertTrue(True)

    def test_make_block_gaps__with_prior_gaps(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, np.nan, 8, np.nan, 10, 11])
        gm = GapMaker1D(series)
        series_with_gaps = gm._make_block_gaps(block_size=4, n_blocks=1, random_state=1)

        expected_idx = pd.Index(
            [1, 2, 3, 4]
        )  # the only place where a buffered gap of size 4 can fit
        pd.testing.assert_index_equal(gm.synth_gap_idx, expected_idx)

        total_nan = series_with_gaps.isna().sum()
        self.assertEqual(6, total_nan)

        # where multiple arrangements of gaps are possible, different random states can have different outcomes
        series_1 = gm._make_block_gaps(block_size=1, n_blocks=2, random_state=1)
        series_2 = gm._make_block_gaps(block_size=1, n_blocks=2, random_state=2)
        with self.assertRaises(AssertionError):
            pd.testing.assert_series_equal(series_1, series_2)

        # if a solution is not found in n_attempts, expect RuntimeError
        with self.assertRaises(RuntimeError):
            gm._make_block_gaps(n_blocks=3, block_size=2, random_state=1)


class TestGapMaker(unittest.TestCase):
    def test_init(self):
        df = pd.DataFrame({"a": [0, 1], "b": [2, 3]})
        gm = GapMaker(df)
        pd.testing.assert_frame_equal(gm._original, df)
        self.assertIs(gm._original, df)
        self.assertIsNone(gm.synth_gaps)

    def test_original(self):
        df = pd.DataFrame({"a": [0, 1], "b": [2, 3]})
        gm = GapMaker(df)
        # expect the original property is a copy of the dataframe used in the constructor
        pd.testing.assert_frame_equal(gm.original, df)
        self.assertIsNot(gm.original, df)

    def test_make_synth_gaps(self):
        df = pd.DataFrame({"a": np.zeros(1000), "b": np.ones(1000)})
        gm = GapMaker(df)

        # make three dataframes with synthetic gaps, using the same gapmaker
        df_gaps_1 = gm.make_synth_gaps(random_state=1, frac=0.1, method="random")
        df_gaps_2 = gm.make_synth_gaps(random_state=2, frac=0.1, method="random")
        df_gaps_3 = gm.make_synth_gaps(random_state=1, frac=0.1, method="random")

        # in all three cases, we should get the same number of gaps in each constituent series (since frac=0.1)
        expected_nan_counts = pd.Series([100, 100], index=["a", "b"])
        actual_nan_counts_1 = df_gaps_1.isna().sum()
        actual_nan_counts_2 = df_gaps_2.isna().sum()
        actual_nan_counts_3 = df_gaps_3.isna().sum()
        pd.testing.assert_series_equal(expected_nan_counts, actual_nan_counts_1)
        pd.testing.assert_series_equal(expected_nan_counts, actual_nan_counts_2)
        pd.testing.assert_series_equal(expected_nan_counts, actual_nan_counts_3)

        # synth_gaps reflects last set of synthetic gaps, since we re-used the same gapmaker
        pd.testing.assert_frame_equal(gm.synth_gaps, df_gaps_3.isna())

        # unique sampling per column --> gaps don't always coincide across series within a DataFrame
        nan_counts_rowwise = df_gaps_1.isna().sum(axis=1)
        self.assertEqual(nan_counts_rowwise.max(), 2)
        self.assertEqual(nan_counts_rowwise.min(), 0)

        # across first two dataframes with synthetic gaps: different random state, different sample
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(df_gaps_1, df_gaps_2)

        # full reproducibility via random state: third dataframe with synth gaps is identical to first one
        pd.testing.assert_frame_equal(df_gaps_1, df_gaps_3)

    def test_make_synth_gaps__prior_gaps(self):
        df = pd.DataFrame({"a": np.zeros(100), "b": np.ones(100)})
        df.iloc[10, 0] = np.nan  # one missing value in "a"
        df.iloc[20:30, 1] = np.nan  # ten missing values in "b"

        gm = GapMaker(df)
        df_gaps = gm.make_synth_gaps(random_state=1, n=10, method="random")

        # expect that if we add ten synth gaps to each series in df, total missing will increment accordingly
        all_gaps_count = df_gaps.isna().sum()
        expected_gaps_count = pd.Series([11, 20], index=["a", "b"])
        pd.testing.assert_series_equal(all_gaps_count, expected_gaps_count)

        # expect that we correctly added ten synthetic gaps to each series
        synth_gaps_count = gm.synth_gaps.sum()
        expected_synth_gaps_count = pd.Series([10, 10], index=["a", "b"])
        pd.testing.assert_series_equal(expected_synth_gaps_count, synth_gaps_count)


if __name__ == "__main__":
    unittest.main()
