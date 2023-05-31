import unittest

import numpy as np
import pandas as pd

from time_series_models.constants import DATE_TIME, LOCATION
from time_series_models.data_fetchers import numba_groupby_functions
from time_series_models.transformers import (
    make_domain,
    multiindex_from_domain,
)


def numba_groupby_agg(df, func, variable="value", pd_grouper_freq="1H"):
    return (
        df.reset_index(level=LOCATION)
        .groupby([LOCATION, pd.Grouper(freq=pd_grouper_freq)])[variable]
        .aggregate(func, engine="numba")
    )


class NumbaGroupbyFunctionsTests(unittest.TestCase):
    def setUp(self) -> None:
        atlantic = [1, 2, 3, np.nan, 4, np.nan]  # <- mode = 1
        pacific = [6, np.nan, 8, 8, 10, 0]  # <- mode = 8
        aral = [np.nan] * 6  # <- mode = np.nan
        citnalta = [np.nan, 4, np.nan, 3, 2, 1]  # <- mode = 4 (the reverse of atlantic)

        self.test_df = pd.DataFrame(
            data={"value": atlantic + pacific + aral + citnalta},
            index=multiindex_from_domain(
                make_domain(
                    "2021-01-01T00",
                    "2021-01-01T00:50",
                    np.timedelta64(10, "m"),
                    "Atlantic",
                    "Pacific",
                    "Aral",
                    "citnalta",
                )
            ),
        )

    def test_nanmode(self):
        result = numba_groupby_agg(self.test_df, numba_groupby_functions.nanmode())
        # note, in a simple df.groupby, the index is sorted by location (A-Z, a-z order!)
        expected = pd.Series(
            data=np.array([np.nan, 1.0, 8.0, 4.0]),
            index=pd.MultiIndex.from_tuples(
                [
                    ("Aral", np.datetime64("2021-01-01T00")),
                    ("Atlantic", np.datetime64("2021-01-01T00")),
                    ("Pacific", np.datetime64("2021-01-01T00")),
                    ("citnalta", np.datetime64("2021-01-01T00")),
                ],
                names=[LOCATION, DATE_TIME],
            ),
            name="value",
        )
        pd.testing.assert_series_equal(result, expected)

    def test_impute_sum__nan(self):
        result = numba_groupby_agg(self.test_df, numba_groupby_functions.impute_sum())
        # note, in a simple df.groupby, the index is sorted by location (A-Z, a-z order!)
        expected = pd.Series(
            data=np.array([np.nan, 15.0, 38.4, 15.0]),
            index=pd.MultiIndex.from_tuples(
                [
                    ("Aral", np.datetime64("2021-01-01T00")),
                    ("Atlantic", np.datetime64("2021-01-01T00")),
                    ("Pacific", np.datetime64("2021-01-01T00")),
                    ("citnalta", np.datetime64("2021-01-01T00")),
                ],
                names=[LOCATION, DATE_TIME],
            ),
            name="value",
        )
        pd.testing.assert_series_equal(result, expected)

    def test_impute_sum__nan_fill(self):
        result = numba_groupby_agg(
            self.test_df, numba_groupby_functions.impute_sum(all_nan_fillval=99.9)
        )
        # note, in a simple df.groupby, the index is sorted by location (A-Z, a-z order!)
        expected = pd.Series(
            data=np.array([99.9, 15.0, 38.4, 15.0]),
            index=pd.MultiIndex.from_tuples(
                [
                    ("Aral", np.datetime64("2021-01-01T00")),
                    ("Atlantic", np.datetime64("2021-01-01T00")),
                    ("Pacific", np.datetime64("2021-01-01T00")),
                    ("citnalta", np.datetime64("2021-01-01T00")),
                ],
                names=[LOCATION, DATE_TIME],
            ),
            name="value",
        )
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
