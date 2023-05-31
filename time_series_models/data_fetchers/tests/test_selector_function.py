import unittest

import numpy as np
import pandas as pd

from time_series_models.data_fetchers.selector_functions import (
    GroupHours,
    diff,
)


class GroupHoursTest(unittest.TestCase):
    def test_multi_hour_mean(self):
        data = pd.Series(
            range(49), index=pd.date_range("2021-05-17", "2021-05-19", freq="H")
        )

        group_hours = GroupHours([6, 7, 8], pd.Series.max)

        result = group_hours(data)
        expected = 32
        self.assertEqual(expected, result)

    def test_daily_diff(self):
        data = pd.Series(
            range(49), index=pd.date_range("2021-05-17", "2021-05-19", freq="H")
        )

        group_hours = GroupHours([6, 11], diff)

        result = data.groupby(pd.Grouper(freq="1D")).aggregate(func=group_hours)
        expected = pd.Series(
            [5.0, 5.0, np.nan],
            index=pd.date_range("2021-05-17", "2021-05-19", freq="1D"),
        )
        pd.testing.assert_series_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
