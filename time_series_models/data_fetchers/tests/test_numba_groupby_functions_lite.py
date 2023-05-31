import unittest
from functools import partial

import numpy as np
import pandas as pd
from time_series_models.data_fetchers import numba_groupby_functions


def denumbafy(numba_func, **kwargs):
    """
    Extract the inner groubpy function to test its behavior without dealing with the numba overhead.
    :param numba_func: a callable that returns a groupby function for use with numba
    :param kwargs: optional keyword args for the numba function constructor
    :return: the internal groupby function
    """
    return partial(numba_func(**kwargs), index=None)


class SansNumbaGroupbyFunctionsTests(unittest.TestCase):
    def test_impute_sum__default(self):
        impute_sum = denumbafy(numba_groupby_functions.impute_sum)
        values = [1, 2, 3]
        result = impute_sum(values)
        self.assertEqual(result, 6.0)

        values = [1, 2, np.nan]
        result = impute_sum(values)
        self.assertEqual(result, 4.5)

    def test_impute_sum__n_expected(self):
        values = [1, 2, 3]
        impute_sum = denumbafy(numba_groupby_functions.impute_sum, n_expected=4)
        result = impute_sum(values)
        self.assertEqual(result, 8.0)

        # note if more values are seen than expected, the imputed sum could be biased low
        impute_sum = denumbafy(numba_groupby_functions.impute_sum, n_expected=2)
        result = impute_sum(values)
        self.assertEqual(result, 4.0)

    def test_impute_sum__fillna(self):
        # Test that impute sum imputes nan values
        impute_sum = denumbafy(numba_groupby_functions.impute_sum)
        result = impute_sum([np.nan, 2.0])
        self.assertEqual(result, 4.0)

        # By default, it returns nan when all values are nan
        impute_sum = denumbafy(numba_groupby_functions.impute_sum)
        result = impute_sum([np.nan, np.nan])
        np.testing.assert_equal(result, np.nan)

        # using all_nan_fillval you can set the return value
        impute_sum = denumbafy(numba_groupby_functions.impute_sum, all_nan_fillval=1.0)
        result = impute_sum([np.nan, np.nan])
        self.assertEqual(result, 1.0)

    def test_nanmode(self):
        nanmode = denumbafy(numba_groupby_functions.nanmode)

        # does not work with list (requires ability to slice!)
        with self.assertRaises(TypeError):
            nanmode([1, 2, 2, np.nan, np.nan, np.nan])

        # works with array
        result = nanmode(np.array([1, 2, 2, np.nan, np.nan, np.nan]))
        self.assertEqual(result, 2.0)

        # works with pandas
        result = nanmode(pd.Series([1, 2, 2, np.nan, np.nan, np.nan]))
        self.assertEqual(result, 2.0)

    def test_nanmode__all_nan(self):
        nanmode = denumbafy(numba_groupby_functions.nanmode)
        result = nanmode(np.array([np.nan, np.nan, np.nan]))
        self.assertTrue(np.isnan(result))

    def test_nanmode__multimodal(self):
        # if result is multimodal, the first mode (in order of appearance) will be returned
        nanmode = denumbafy(numba_groupby_functions.nanmode)
        result = nanmode(np.array([2, np.nan, 2, 1, np.nan, 1, np.nan]))
        self.assertEqual(result, 2.0)

        result = nanmode(np.array([1, 2, np.nan, 2, np.nan, 1, np.nan]))
        self.assertEqual(result, 1.0)

        result = nanmode(np.array([1, 0, np.nan, 0, np.nan, 1, np.nan]))
        self.assertEqual(result, 1.0)

        result = nanmode(np.array([0, np.nan, 0, 1, np.nan, 1, np.nan]))
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
