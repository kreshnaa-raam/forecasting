import io
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock

import xgboost as xgb
import numpy as np
import numpy.testing
import pandas as pd

import sklearn
import sklearn.datasets
import sklearn.compose
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error

from time_series_models.constants import LOCATION, DATE_TIME
from time_series_models.estimators import (
    LinearRegressor,
    XgbRegressor,
)
from time_series_models.processes import ExampleProcess
from time_series_models.time_series_models import (
    RegularTimeSeriesModel,
    MonthSelector,
    ForecastDataMonitor,
)
from time_series_models.transformers import (
    make_domain,
    multiindex_from_domain,
)

logging.basicConfig()
logging.getLogger().setLevel(
    logging.INFO
)  # set level to INFO for code commit, DEBUG for test development


@patch.multiple(RegularTimeSeriesModel, __abstractmethods__=set())
class RegularTimeSeriesModelTest(unittest.TestCase):
    def dtype(self, ttype):
        return np.dtype([(LOCATION, np.unicode_, 36), (DATE_TIME, ttype)])

    def test_domain_type(self):
        tsm = RegularTimeSeriesModel(np.timedelta64(5, "h"))
        self.assertEqual(tsm.domain_type, self.dtype("datetime64[h]"))

        tsm = RegularTimeSeriesModel(np.timedelta64(2, "D"))
        self.assertEqual(tsm.domain_type, self.dtype("datetime64[D]"))

        tsm = RegularTimeSeriesModel(np.timedelta64(10, "s"))
        self.assertEqual(tsm.domain_type, self.dtype("datetime64[s]"))

    def test_domain(self):
        tsm = RegularTimeSeriesModel(np.timedelta64(6, "h"))

        actual = tsm.domain("2021-05-17", "2021-05-28", "capybara")
        self.assertEqual(actual.dtype, self.dtype("datetime64[h]"))
        numpy.testing.assert_equal(
            actual[LOCATION], np.array(("capybara",) * 45, dtype="U36").reshape(-1, 1)
        )
        numpy.testing.assert_equal(
            actual[DATE_TIME],
            np.arange(
                np.datetime64("2021-05-17"),
                np.datetime64("2021-05-28T01"),
                step=tsm._time_step,
            ).reshape(-1, 1),
        )

        actual = tsm.domain("2021-05-17", "2021-05-28", *["capybara", "sisyphus"])
        self.assertEqual(actual.dtype, self.dtype("datetime64[h]"))
        numpy.testing.assert_equal(
            actual[LOCATION],
            np.array(("capybara",) * 45 + ("sisyphus",) * 45, dtype="U36").reshape(
                -1, 1
            ),
        )
        numpy.testing.assert_equal(
            actual[DATE_TIME],
            np.tile(
                np.arange(
                    np.datetime64("2021-05-17"),
                    np.datetime64("2021-05-28T01"),
                    step=tsm._time_step,
                ),
                2,
            ).reshape(-1, 1),
        )

    def test_domain__dupe_locs_raises(self):
        tsm = RegularTimeSeriesModel(np.timedelta64(6, "h"))

        # regular domain will not raise
        tsm.domain("2021-05-17", "2021-05-28", "a", "b", "c", "d")

        # duplicate domain locations will raise ValueError
        with self.assertRaises(ValueError):
            tsm.domain("2021-05-17", "2021-05-28", "a", "b", "c", "a")

    def test_fit(self):
        tsm = RegularTimeSeriesModel(np.timedelta64(4, "h"))
        tsm._model = Mock()

        tsm.fit("2021-05-17", "2021-05-28", "capybara")
        # Assert that the argument to the mock objects fit method is array equal to the domain returned
        # by the fit_domain helper using the (start, end and locations) values persisted by the domain_wrangler
        np.testing.assert_array_equal(tsm.model.fit.call_args.args[0], tsm.fit_domain)
        self.assertEqual(tsm.model.fit.call_args.args[1], tsm.fit_range)

        numpy.testing.assert_equal(
            tsm.fit_domain[LOCATION],
            np.array(("capybara",) * 67, dtype="U36").reshape(-1, 1),
        )

    def test_fit__multi_location(self):
        tstep = np.timedelta64(4, "h")
        start = "2021-05-17"
        end = "2021-05-28"
        tsm = RegularTimeSeriesModel(tstep)
        tsm._model = Mock()

        tsm.fit("2021-05-17", "2021-05-28", "capybara", "nutria", "muskrat")
        # Assert that the argument to the mock objects fit method is array equal to the domain returned
        # by the fit_domain helper using the (start, end and locations) values persisted by the domain_wrangler
        np.testing.assert_array_equal(tsm.model.fit.call_args.args[0], tsm.fit_domain)
        self.assertEqual(tsm.model.fit.call_args.args[1], tsm.fit_range)

        numpy.testing.assert_equal(
            tsm.fit_domain[LOCATION],
            np.concatenate(
                [
                    np.array(("capybara",) * 67, dtype="U36"),
                    np.array(("nutria",) * 67, dtype="U36"),
                    np.array(("muskrat",) * 67, dtype="U36"),
                ]
            ).reshape(-1, 1),
        )
        numpy.testing.assert_equal(
            tsm.fit_domain[DATE_TIME],
            np.concatenate(
                [
                    np.arange(
                        start=np.datetime64(start),
                        stop=np.datetime64(end) + tstep,
                        step=tstep,
                    ),
                ]
                * 3
            ).reshape(-1, 1),
        )

    def test_predict_dataframe(self):
        tsm = RegularTimeSeriesModel(np.timedelta64(1, "h"))
        tsm._model = MagicMock()
        locations = [
            "capybara",
            "nutria",
            "muskrat",
        ]
        expected_domain = make_domain(
            "2023-01-01", "2023-01-02", np.timedelta64(1, "h"), *locations
        )
        test_predictions = np.full(expected_domain.shape, 1.0)
        tsm.model.predict.return_value = test_predictions

        ## Test the no kwargs case
        result = tsm.predict_dataframe("2023-01-01", "2023-01-02", *locations)
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame(
                test_predictions.reshape(-1),
                columns=["predicted"],
                index=multiindex_from_domain(expected_domain),
            ),
        )
        tsm._model.predict.assert_called_once()

        tsm._model.reset_mock()

        ## Test the range kwargs
        tsm._model.predict.return_value = test_predictions
        tsm.get_range = Mock()
        test_range = np.full(expected_domain.shape, -1.0)
        tsm.get_range.return_value = test_range

        result = tsm.predict_dataframe(
            "2023-01-01", "2023-01-02", *locations, range=True
        )
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame(
                np.concatenate((test_predictions, test_range), axis=1),
                columns=["predicted", "true"],
                index=multiindex_from_domain(expected_domain),
            ),
        )
        tsm._model.predict.assert_called_once()
        tsm.get_range.assert_called_once()

        tsm._model.reset_mock()
        tsm.get_range.reset_mock()

        ## Test the feature_score kwarg
        tsm._model.predict.return_value = test_predictions
        tsm.get_range.return_value = test_range

        test_feature_data = np.full((len(expected_domain), 4), 2.0)
        # First column is dropped
        test_feature_data[24, 1] = np.nan
        test_feature_data[49, 2] = np.nan
        test_feature_data[74, 2] = np.nan
        test_feature_data[74, 3] = np.nan
        tsm._model["feature_builder"].transform.return_value = test_feature_data

        tsm.get_feature_names = Mock()
        tsm.get_feature_names.return_value = ["a", "b", "c"]

        ### With valid shap model
        with patch("time_series_models.shap_viz.ShapViz") as mock_shapviz:
            mock_shapviz.return_value.values_dataframe.return_value = pd.DataFrame(
                np.tile(np.array([12, 2, 5], dtype=float).reshape(1, 3), (75, 1)),
                columns=["a", "b", "c"],
                index=multiindex_from_domain(expected_domain),
            )
            result = tsm.predict_dataframe(
                "2023-01-01", "2023-01-02", *locations, range=True, feature_score=True
            )

        expected_scores = np.full(expected_domain.shape, 1.0)
        expected_scores[24, 0] = 0.368421
        expected_scores[49, 0] = 0.894737
        expected_scores[74, 0] = 0.631579

        expected_result = pd.DataFrame(
            np.concatenate((test_predictions, test_range, expected_scores), axis=1),
            columns=["predicted", "true", "feature_score"],
            index=multiindex_from_domain(expected_domain),
        )

        pd.testing.assert_frame_equal(
            result, expected_result, check_exact=False, atol=1e-6
        )

        ### If shap throws a TypeError
        with patch("time_series_models.shap_viz.ShapViz") as mock_shapviz:
            mock_shapviz.side_effect = TypeError
            result = tsm.predict_dataframe(
                "2023-01-01", "2023-01-02", *locations, range=True, feature_score=True
            )

        expected_scores = np.full(expected_domain.shape, 1.0)
        expected_scores[24, 0] = 0.0
        expected_scores[49, 0] = 0.0
        expected_scores[74, 0] = 0.0

        expected_result = pd.DataFrame(
            np.concatenate((test_predictions, test_range, expected_scores), axis=1),
            columns=["predicted", "true", "feature_score"],
            index=multiindex_from_domain(expected_domain),
        )

        pd.testing.assert_frame_equal(
            result, expected_result, check_exact=False, atol=1e-6
        )

    def test_dump_load(self):
        """
        Prove the RegularTimeSeriesModel with a trained xbg model is serializable using dump/load with pickle
        """
        tsm = RegularTimeSeriesModel(np.timedelta64(4, "h"))
        iris = sklearn.datasets.load_iris()
        Y = iris["target"]
        X = iris["data"]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, Y, test_size=0.33, random_state=42
        )
        tsm._model = xgb.XGBRegressor(n_jobs=1).fit(X_train, y_train)
        predictions = tsm._model.predict(X_test)
        original_mse = mean_squared_error(y_test, predictions)

        with io.BytesIO() as bio:
            tsm.dump(bio)
            bio.seek(0)
            zombie = RegularTimeSeriesModel.load(bio)

        predictions = zombie._model.predict(X_test)
        zombie_mse = mean_squared_error(y_test, predictions)
        self.assertEqual(original_mse, zombie_mse)
        self.assertEqual(tsm._time_step, zombie._time_step)

    def test_filter(self):
        # ensure the feature_filter is a no-op ColumnTransfomer
        tsm = RegularTimeSeriesModel(np.timedelta64(4, "h"))
        feature_filter = tsm.filters
        self.assertIsInstance(feature_filter, sklearn.pipeline.Pipeline)
        self.assertEqual(feature_filter.named_steps["passthrough"], "passthrough")

        iris = sklearn.datasets.load_iris()
        X = iris["data"]
        result = feature_filter.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_fit_range_stats(self):
        with patch.multiple(
            RegularTimeSeriesModel,
            fit_domain=make_domain(
                "2022-01-01",
                "2022-01-03",
                np.timedelta64(1, "D"),
                *["capybara", "caiman", "ocelot"],
            ),
            fit_range=np.array(
                # capybara is missing the last observation, and ocelot is missing all three observations
                [2.0, 1.0, np.nan, 3.0, 4.0, 6.0, np.nan, np.nan, np.nan]
            ).reshape(-1, 1),
        ):
            tsm = RegularTimeSeriesModel(np.timedelta64(1, "D"))
            with self.assertWarns(RuntimeWarning):
                result = tsm.fit_range_stats
        self.assertIsInstance(result, pd.DataFrame)
        expected = pd.DataFrame(
            data={
                # datetime indices of most recent non-missing values, by location, are 2022-01-02, 2022-01-03, and NaT
                # and 2022-01-03 overall
                "most_recent_non_null": [
                    np.datetime64("2022-01-03"),
                    np.datetime64("2022-01-02"),
                    np.datetime64("2022-01-03"),
                    np.datetime64("NaT"),
                ],
                "freshness": np.array([0.0, 86400.0, 0.0, np.nan], dtype="float64"),
                "fraction_missing": np.array([4 / 9, 1 / 3, 0.0, 1.0]),
                "mean": np.array([3.2, 1.5, 13 / 3, np.nan]),
                "min": np.array([1.0, 1.0, 3.0, np.nan]),
                "q0.25": np.array([2.0, 1.25, 3.5, np.nan]),
                "q0.75": np.array([4.0, 1.75, 5.0, np.nan]),
                "max": np.array([6.0, 2.0, 6.0, np.nan]),
                "n_stats": np.array([9.0, 3.0, 3.0, 3.0]),
                "n_missing": np.array([4.0, 1.0, 0.0, 3.0]),
                "n_present": np.array([5.0, 2.0, 3.0, 0.0]),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("all", "all"),
                    ("by_location", "capybara"),
                    ("by_location", "caiman"),
                    ("by_location", "ocelot"),
                ],
                names=["grouping", LOCATION],
            ),
        )
        pd.testing.assert_frame_equal(result, expected)

    @unittest.skip("TODO")
    def test_prettify_stats_by_loc(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_unfiltered_fit_stats(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_filtered_fit_stats(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_unfiltered_fit_stats_by_loc(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_filtered_fit_stats_by_loc(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_unfiltered_predict_stats(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_filtered_predict_stats(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_unfiltered_predict_stats_by_loc(self):
        pass

    @unittest.skip("TODO")
    def test_monitor_filtered_predict_stats_by_loc(self):
        pass


class XgbExampleTest(unittest.TestCase):
    # TODO: Fix xgboost regressor on CircleCI https://app.asana.com/0/0/1202477655607180/f
    class Model(ExampleProcess, LinearRegressor, RegularTimeSeriesModel):
        pass

    def test_range(self):
        tsm = self.Model(np.timedelta64(3, "h"))

        rng0 = tsm.range("2021-05-17", "2021-05-31", *["capybara", "sisyphus", "camus"])
        rng1 = tsm.range("2021-05-17", "2021-05-31", *["capybara", "sisyphus", "camus"])

        # Show that the range is reproducible (random seed is set correctly)
        np.testing.assert_almost_equal(rng0, rng1)

    def test_model(self):
        tsm = self.Model(np.timedelta64(2, "h"))

        tsm.fit("2021-05-17", "2021-06-30", *["capybara", "sisyphus", "camus"])

        expected_feature_names = [
            "location__x0_camus",
            "location__x0_capybara",
            "location__x0_sisyphus",
            "24hour__sin",
            "24hour__cos",
            "168hour__sin",
            "168hour__cos",
        ]

        self.assertEqual(expected_feature_names, tsm.get_feature_names())

        res1 = tsm.score("2021-07-01", "2021-07-14", *["capybara", "sisyphus", "camus"])
        np.testing.assert_almost_equal(res1, 0.50, decimal=2)

        with io.BytesIO() as bio:
            tsm.dump(bio)
            bio.seek(0)
            zombie = self.Model.load(bio)

        self.assertEqual(expected_feature_names, zombie.get_feature_names())

        res2 = zombie.score(
            "2021-07-01", "2021-07-14", *["capybara", "sisyphus", "camus"]
        )
        np.testing.assert_almost_equal(res1, res2)


class MonthSelectorTest(unittest.TestCase):
    class Model(ExampleProcess, XgbRegressor, MonthSelector, RegularTimeSeriesModel):
        pass

    def test_domain(self):
        model = self.Model(np.timedelta64(1, "D"), include_only_months=[7, 8])
        domain = model.domain("2020-01-01", "2022-01-01", "Camus")
        range = model.range("2020-01-01", "2022-01-01", "Camus")
        self.assertEqual(domain.shape, range.shape)
        expected = np.concatenate(
            [
                np.arange(
                    np.datetime64("2020-07-01"), np.datetime64("2020-09-01")
                ),  # open interval!!!
                np.arange(np.datetime64("2021-07-01"), np.datetime64("2021-09-01")),
            ]
        ).reshape(-1, 1)
        np.testing.assert_equal(domain[DATE_TIME], expected)


if __name__ == "__main__":
    unittest.main()
