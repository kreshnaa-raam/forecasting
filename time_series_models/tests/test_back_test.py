import functools
import sklearn
import unittest

import numpy as np
import pandas as pd

from unittest.mock import patch

from time_series_models.constants import DATE_TIME, LOCATION
from time_series_models.back_test import BackTest


class BackTestTestTests(unittest.TestCase):
    def sum_metric(self, x):
        # add 'em all up; note, excluding null values is equivalent to counting as zero in summation
        assert ("true" in x.columns) & ("predicted" in x.columns)
        return pd.Series({"sum": x.sum().sum()})

    def setUp(self) -> None:
        self.predictions_df = pd.DataFrame(
            data={
                "true": [0, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan, 11],
                "predicted": np.arange(0, 24, 2),
            },
            index=pd.MultiIndex.from_arrays(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                    [
                        "up",
                        "up",
                        "down",
                        "down",
                        "up",
                        "up",
                        "down",
                        "down",
                        "up",
                        "up",
                        "down",
                        "down",
                    ],
                    np.array(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                            "2021-01-06",
                            "2021-01-05",
                            "2021-01-06",
                        ],
                        dtype="datetime64[D]",
                    ),
                ],
                names=["fold", LOCATION, DATE_TIME],
            ),
        )

        class Model:
            def __init__(self, **config_dict):
                self.config_dict = config_dict
                self._tstep = np.timedelta64(1, "h")

            @property
            def tstep(self):
                return self._tstep

        self.model_class = Model

    def test_metrics_set(self):
        example_df = pd.DataFrame(
            {"true": [4, 5, np.nan, 0], "predicted": [4, 6, 3, 1]}
        )
        with self.assertLogs(level="WARNING") as logger_warnings:
            result = BackTest._metrics_set(example_df)
            records = logger_warnings.records
            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].getMessage(), "Excluding 1 observations where y_true is NaN!"
            )

        self.assertIsInstance(result, pd.Series)

        expected_index = pd.Index(
            [
                "mean_absolute_percentage_error",
                "mean_absolute_error",
                "median_absolute_error",
                "mean_squared_error",
                "root_mean_squared_error",
                "r2_score",
                "explained_variance_score",
            ]
        )
        pd.testing.assert_index_equal(result.index, expected_index)

        expected_metrics_set = [
            lambda x, y: np.nan,  # due to 0 in "true" we will expect np.nan
            sklearn.metrics.mean_absolute_error,
            sklearn.metrics.median_absolute_error,
            sklearn.metrics.mean_squared_error,
            functools.partial(sklearn.metrics.mean_squared_error, squared=False),
            sklearn.metrics.r2_score,
            sklearn.metrics.explained_variance_score,
        ]
        # expect that rows containing np.nan in "true" have been filtered already
        example_notna = example_df.loc[example_df["true"].notna()]
        expected = pd.Series(
            [
                m(example_notna["true"], example_notna["predicted"])
                for m in expected_metrics_set
            ],
            index=expected_index,
        )
        pd.testing.assert_series_equal(result, expected)

    def test_metrics_set__missing_y_true(self):
        example_df = pd.DataFrame(
            data={"true": [np.nan, np.nan, np.nan, np.nan], "predicted": [4, 6, 3, 1]},
            index=pd.MultiIndex.from_product(
                [
                    [1],
                    ["up", "down"],
                    [np.datetime64("2020-01-01"), np.datetime64("2022-12-31")],
                ],
                names=["fold", "location", "date_time"],
            ),
        )
        with self.assertLogs(level="WARNING") as logger_warnings:
            result = BackTest._metrics_set(example_df)
            records = logger_warnings.records
            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].getMessage(),
                f"No y_true values for fold [1], location ['up', 'down'], "
                f"date_time [2020-01-01 00:00:00 - 2022-12-31 00:00:00]",
            )

        expected = pd.Series(
            {
                "mean_absolute_percentage_error": np.nan,
                "mean_absolute_error": np.nan,
                "median_absolute_error": np.nan,
                "mean_squared_error": np.nan,
                "root_mean_squared_error": np.nan,
                "r2_score": np.nan,
                "explained_variance_score": np.nan,
            }
        )
        pd.testing.assert_series_equal(result, expected)

    def test_get_metrics__no_args(self):
        bt = BackTest.from_default_splitter(self.model_class, {})
        with patch.multiple(
            BackTest,
            predictions_df=self.predictions_df,
            _metrics_set=self.sum_metric,
        ):
            result = bt.get_metrics()

        expected = pd.DataFrame(
            data={
                "sum": [
                    (0.0 + 2.0 + 4.0 + 6.0) + (0.0 + 1.0 + 2.0 + 3.0),
                    (8.0 + 10.0 + 12.0 + 14.0) + (6.0 + 7.0),
                    (16.0 + 18.0 + 20.0 + 22.0) + (8.0 + 9.0 + 11.0),
                ]
            },
            index=pd.Index([0, 1, 2], name="fold"),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_get_metrics__fold(self):
        bt = BackTest.from_default_splitter(self.model_class, {})
        with patch.multiple(
            BackTest,
            predictions_df=self.predictions_df,
            _metrics_set=self.sum_metric,
        ):
            result = bt.get_metrics(how="fold")

        expected = pd.DataFrame(
            data={
                "sum": [
                    (0.0 + 2.0 + 4.0 + 6.0) + (0.0 + 1.0 + 2.0 + 3.0),
                    (8.0 + 10.0 + 12.0 + 14.0) + (6.0 + 7.0),
                    (16.0 + 18.0 + 20.0 + 22.0) + (8.0 + 9.0 + 11.0),
                ]
            },
            index=pd.Index([0, 1, 2], name="fold"),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_get_metrics__location(self):
        bt = BackTest.from_default_splitter(self.model_class, {})
        with patch.multiple(
            BackTest,
            predictions_df=self.predictions_df,
            _metrics_set=self.sum_metric,
        ):
            result = bt.get_metrics(how="location")

        expected = pd.DataFrame(
            data={
                "sum": [
                    (2 + 3 + 6 + 7 + 0 + 11) + (4 + 6 + 12 + 14 + 20 + 22),
                    (0 + 1 + 8 + 9) + (0 + 2 + 8 + 10 + 16 + 18),
                ],
            },
            index=pd.Index(["down", "up"], name=LOCATION),
        ).astype(float)
        pd.testing.assert_frame_equal(result, expected)

    def test_get_metrics__location_fold(self):
        bt = BackTest.from_default_splitter(self.model_class, {})
        with patch.multiple(
            BackTest,
            predictions_df=self.predictions_df,
            _metrics_set=self.sum_metric,
        ):
            result = bt.get_metrics(how="location-fold")

        expected = pd.DataFrame(
            data={
                "sum": [
                    2 + 3 + 4 + 6,
                    6 + 7 + 12 + 14,
                    0 + 11 + 20 + 22,
                    0 + 1 + 0 + 2,
                    8 + 10,
                    8 + 9 + 16 + 18,
                ]
            },
            index=pd.MultiIndex.from_arrays(
                [
                    ["down", "down", "down", "up", "up", "up"],
                    [0, 1, 2, 0, 1, 2],
                ],
                names=[LOCATION, "fold"],
            ),
        ).astype(float)
        pd.testing.assert_frame_equal(result, expected)

    def test_construct_model(self):
        bt = BackTest.from_default_splitter(self.model_class, {"this": "that"})
        result = bt.construct_model()
        self.assertIsInstance(result, self.model_class)
        self.assertDictEqual(result.config_dict, {"this": "that"})

    @unittest.skip("TODO - not implemented yet!")
    def test_run(self):
        # TODO (Michael H): maybe construct with a default splitter and assert calls on mocked fit and predict methods,
        #  along with structure of resulting metadata attribute
        pass


if __name__ == "__main__":
    unittest.main()
