import unittest

import numpy as np
import sklearn.preprocessing
import sklearn.utils.estimator_checks
from time_series_models.data_monitor import ForecastDataMonitor


class ForecastDataMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [np.nan, np.nan, 0.0],
                [1.0, np.nan, np.nan],
                [3.0, 4.0, 0.0],
                [5.0, np.nan, 10.0],
                [1.0, 8.0, 20.0],
            ]
        )
        self.y = np.array(
            [
                [np.nan],
                [2.0],
                [np.nan],
                [6.0],
                [5.0],
            ]
        )
        self.x_with_loc = np.array(
            [
                [2.0, 1.0, np.nan, np.nan],
                [2.0, 3.0, 4.0, 0.0],
                [2.0, 5.0, np.nan, 10.0],
                [2.0, np.nan, 8.0, 20.0],
            ]
        )
        self.y_with_loc = np.array(
            [
                [2.0],
                [np.nan],
                [6.0],
                [5.0],
            ]
        )

    def assert_equal_metrics_dict(self, d1, d2):
        self.assertSetEqual(set(d1.keys()), set(d2.keys()))
        for k in d1.keys():
            if type(d1[k]) in {np.array, np.ndarray}:
                np.testing.assert_almost_equal(d1[k], d2[k])
            else:
                self.assertEqual(d1[k], d2[k])

    def test_init(self):
        dm = ForecastDataMonitor()
        self.assertIsInstance(dm, sklearn.preprocessing.FunctionTransformer)
        self.assertIsNone(dm.quantiles)

        dm = ForecastDataMonitor(quantiles=[0.95, 0.85, 0.75])
        self.assertListEqual(dm.quantiles, [0.95, 0.85, 0.75])

    def test_sklearn(self):
        # need to skip saving transform_stats to evaluate the
        # rest of ForecastDataMonitor against the sklearn API
        dm = ForecastDataMonitor(no_transform_stats=True)
        # the next call triggers a comprehensive test suite for adherence to the sklearn API;
        # for more details, see https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
        sklearn.utils.estimator_checks.check_estimator(dm)

        # still expect a no-op fit and transform
        result = dm.fit_transform(self.x, self.y)
        np.testing.assert_almost_equal(result, self.x)

        # but, of course, we don't get transform stats, only fit stats!
        self.assertDictEqual(dm.transform_stats_, {})
        self.assertGreater(len(dm.fit_stats_), 1)

    def test_fit(self):
        dm = ForecastDataMonitor(quantiles=[0.5])
        with self.assertRaises(AttributeError):
            dm.fit_stats_
        with self.assertRaises(AttributeError):
            dm.fit_stats_by_loc_
        dm.fit_stats_ = {"old": "values", "should be": "overwritten at next call"}
        dm.fit(self.x, y=self.y)
        expected = {
            "feature_n_obs": 5,
            "feature_n_missing": np.array([1.0, 3.0, 1.0]),
            "feature_missing": np.array([0.2, 0.6, 0.2]),
            "feature_trailing_n_missing": np.array([0, 0, 0]),
            "feature_min": np.array([1.0, 4.0, 0.0]),
            "feature_max": np.array([5.0, 8.0, 20.0]),
            "feature_mean": np.array([2.5, 6.0, 7.5]),
            "feature_q0.25": np.array([1.0, 5.0, 0.0]),
            "feature_q0.5": np.array([2.0, 6.0, 5.0]),
            "feature_q0.75": np.array([3.5, 7.0, 12.5]),
        }
        self.assert_equal_metrics_dict(dm.fit_stats_, expected)
        self.assert_equal_metrics_dict(dm.fit_stats, expected)

    def test_fit__use_locs(self):
        dm = ForecastDataMonitor(quantiles=[0.5], use_locs=True)
        with self.assertRaises(AttributeError):
            dm.fit_stats
        with self.assertRaises(AttributeError):
            dm.fit_stats_by_loc
        dm.fit_stats_ = {"old": "values", "should be": "overwritten at next call"}
        dm.fit_stats_by_loc_ = {"these": "too"}
        dm.fit(self.x_with_loc, y=self.y_with_loc)
        expected = {
            "feature_n_obs": 4,
            "feature_n_missing": np.array([0.0, 1.0, 2.0, 1.0]),
            "feature_missing": np.array([0.0, 0.25, 0.5, 0.25]),
            "feature_trailing_n_missing": np.array([0, 1, 0, 0]),
            "feature_min": np.array([2.0, 1.0, 4.0, 0.0]),
            "feature_max": np.array([2.0, 5.0, 8.0, 20.0]),
            "feature_mean": np.array([2.0, 3.0, 6.0, 10.0]),
            "feature_q0.25": np.array([2.0, 2.0, 5.0, 5.0]),
            "feature_q0.5": np.array([2.0, 3.0, 6.0, 10.0]),
            "feature_q0.75": np.array([2.0, 4.0, 7.0, 15.0]),
        }
        self.assert_equal_metrics_dict(dm.fit_stats, expected)
        expected_by_loc = {
            "feature_n_obs": 2,
            "feature_n_missing": np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
            "feature_missing": np.array([[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.5, 0.0]]),
            "feature_trailing_n_missing": np.array([[0, 0, 0, 0], [0, 1, 0, 0]]),
            "feature_min": np.array([[2.0, 1.0, 4.0, 0.0], [2.0, 5.0, 8.0, 10.0]]),
            "feature_max": np.array([[2.0, 3.0, 4.0, 0.0], [2.0, 5.0, 8.0, 20.0]]),
            "feature_mean": np.array([[2.0, 2.0, 4.0, 0.0], [2.0, 5.0, 8.0, 15.0]]),
        }
        self.assert_equal_metrics_dict(dm.fit_stats_by_loc, expected_by_loc)

    def test_transform(self):
        dm = ForecastDataMonitor(quantiles=[0.5])
        with self.assertRaises(AttributeError):
            dm.transform_stats
        with self.assertRaises(AttributeError):
            dm.transform_stats_by_loc
        dm.transform_stats_ = {
            "this": "should",
            "go": "away",
        }
        result = dm.transform(self.x)
        self.assertIs(result, self.x)
        np.testing.assert_almost_equal(result, self.x)
        expected = {
            "feature_n_obs": 5,
            "feature_n_missing": np.array([1.0, 3.0, 1.0]),
            "feature_missing": np.array([0.2, 0.6, 0.2]),
            "feature_trailing_n_missing": np.array([0, 0, 0]),
            "feature_min": np.array([1.0, 4.0, 0.0]),
            "feature_max": np.array([5.0, 8.0, 20.0]),
        }
        self.assert_equal_metrics_dict(dm.transform_stats, expected)

    def test_transform__use_locs(self):
        dm = ForecastDataMonitor(quantiles=[0.5], use_locs=True)
        with self.assertRaises(AttributeError):
            dm.transform_stats
        with self.assertRaises(AttributeError):
            dm.transform_stats_by_loc
        dm.transform_stats_by_loc_ = {
            "this": "should",
            "go": "away",
        }
        result = dm.transform(self.x_with_loc)
        self.assertIs(result, self.x_with_loc)
        np.testing.assert_almost_equal(result, self.x_with_loc)
        expected = {
            "feature_n_obs": 4,
            "feature_n_missing": np.array([0.0, 1.0, 2.0, 1.0]),
            "feature_missing": np.array([0.0, 0.25, 0.5, 0.25]),
            "feature_trailing_n_missing": np.array([0, 1, 0, 0]),
            "feature_min": np.array([2.0, 1.0, 4.0, 0.0]),
            "feature_max": np.array([2.0, 5.0, 8.0, 20.0]),
        }
        self.assert_equal_metrics_dict(dm.transform_stats, expected)

        expected_by_loc = {
            "feature_n_obs": 2,
            "feature_missing": np.array([[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.5, 0.0]]),
            "feature_n_missing": np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
            "feature_trailing_n_missing": np.array([[0, 0, 0, 0], [0, 1, 0, 0]]),
            "feature_min": np.array([[2.0, 1.0, 4.0, 0.0], [2.0, 5.0, 8.0, 10.0]]),
            "feature_max": np.array([[2.0, 3.0, 4.0, 0.0], [2.0, 5.0, 8.0, 20.0]]),
        }
        self.assert_equal_metrics_dict(dm.transform_stats_by_loc, expected_by_loc)

    def test_fit_transform(self):
        dm = ForecastDataMonitor(quantiles=[0.5])
        result = dm.fit_transform(self.x, self.y)
        self.assertIs(result, self.x)
        np.testing.assert_almost_equal(result, self.x)

        expected_fit_stats = {
            "feature_n_obs": 5,
            "feature_n_missing": np.array([1.0, 3.0, 1.0]),
            "feature_missing": np.array([0.2, 0.6, 0.2]),
            "feature_trailing_n_missing": np.array([0, 0, 0]),
            "feature_mean": np.array([2.5, 6.0, 7.5]),
            "feature_min": np.array([1.0, 4.0, 0.0]),
            "feature_max": np.array([5.0, 8.0, 20.0]),
            "feature_q0.25": np.array([1.0, 5.0, 0.0]),
            "feature_q0.5": np.array([2.0, 6.0, 5.0]),
            "feature_q0.75": np.array([3.5, 7.0, 12.5]),
        }
        self.assert_equal_metrics_dict(expected_fit_stats, dm.fit_stats)

        expected_transform_stats = {
            "feature_n_obs": 5,
            "feature_n_missing": np.array([1.0, 3.0, 1.0]),
            "feature_missing": np.array([0.2, 0.6, 0.2]),
            "feature_trailing_n_missing": np.array([0, 0, 0]),
            "feature_min": np.array([1.0, 4.0, 0.0]),
            "feature_max": np.array([5.0, 8.0, 20.0]),
        }
        self.assert_equal_metrics_dict(expected_transform_stats, dm.transform_stats)

    def test_count_trailing_nan(self):
        arr = np.array(
            [
                [np.nan, 3, 1, 2, 8, np.nan, 1],
                [np.nan, 1, 1, 1, 1, 1, np.nan],
                [np.nan, 5, np.nan, 4, 9, 1, np.nan],
                [np.nan, 7, 6, np.nan, 10, 1, np.nan],
                [np.nan, 11, np.nan, np.nan, np.nan, 1, np.nan],
            ]
        )
        result = ForecastDataMonitor.count_trailing_nan(arr)
        expected = np.array([5, 0, 1, 2, 1, 0, 4])
        np.testing.assert_array_equal(result, expected)

    def test_count_trailing_nan__no_trailing_nan(self):
        arr = np.array(
            [
                [3, np.nan],
                [1, np.nan],
                [5, 3],
                [7, np.nan],
                [11, 1],
            ]
        )
        result = ForecastDataMonitor.count_trailing_nan(arr)
        expected = np.array([0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_get_quantiles__default(self):
        dm = ForecastDataMonitor()
        quantiles = dm.get_quantiles()
        self.assertListEqual(quantiles, [0.05, 0.25, 0.75, 0.95])

    def test_get_quantiles__custom_quantiles(self):
        dm = ForecastDataMonitor(quantiles=[0.75, 0.85, 0.95])
        quantiles = dm.get_quantiles()
        self.assertListEqual(quantiles, [0.25, 0.75, 0.85, 0.95])

    def test_get_quantiles__empty_quantiles(self):
        dm = ForecastDataMonitor(quantiles=[])
        quantiles = dm.get_quantiles()
        self.assertListEqual(quantiles, [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
