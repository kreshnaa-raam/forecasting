import unittest
from unittest.mock import (
    create_autospec,
    Mock,
    patch,
)

import sklearn
import sklearn.compose
import numpy as np

from time_series_models.estimators import (
    LinearRegressor,
    IdentityEstimator,
    IdentityRegressor,
    Estimator,
    SequentialFeatureSelection,
    XgbRegressor,
    XgbClassifier,
    TransformedTargetRegressor,
    RandomizedSearch,
)
from time_series_models.time_series_models import (
    RegularTimeSeriesModel,
)
from time_series_models.processes import Process


class IdentityEstimatorTests(unittest.TestCase):
    def test_check_input_shape__1d_raises(self):
        with self.assertRaises(IndexError):
            with self.assertLogs(level="ERROR") as logger_warnings:
                IdentityEstimator._check_input_shape(np.array([1, 2, 3]))

        records = logger_warnings.records
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].getMessage()[:34],
            "Feature data must be 2-dimensional",
        )

    def test_check_input_shape__multi_columns_raises(self):
        with self.assertRaises(AssertionError):
            with self.assertLogs(level="ERROR") as logger_warnings:
                IdentityEstimator._check_input_shape(np.array([[1, 2, 3]]))
        records = logger_warnings.records
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].getMessage()[:45],
            "Feature data must consist of a single column!",
        )

    def test_check_input_shape__okay(self):
        arr = np.array([1, 2, 3]).reshape(-1, 1)
        result = IdentityEstimator._check_input_shape(arr)
        self.assertIsNone(result)

    @patch.object(IdentityEstimator, "_check_input_shape")
    def test_fit(self, mock_check_shape):
        x = np.array([1, 2, 3]).reshape(-1, 1)
        y = np.array([1, 2, 3]).reshape(-1, 1) * 3

        e = IdentityEstimator()
        f = e.fit(x, y)

        self.assertIs(f, e)

        mock_check_shape.assert_called_once()
        mock_check_shape.assert_called_with(x)

    @patch.object(IdentityEstimator, "_check_input_shape")
    def test_predict(self, mock_check_shape):
        e = IdentityEstimator()
        arr = np.array([1, 2, 3]).reshape(-1, 1)
        result = e.predict(arr)
        np.testing.assert_array_equal(arr, result)
        mock_check_shape.assert_called_once()
        mock_check_shape.assert_called_with(arr)

    def test_transform(self):
        e = IdentityEstimator()
        arr = np.array([1, 2, 3]).reshape(-1, 1)
        result = e.transform(arr)
        np.testing.assert_array_equal(arr, result)


class EstimatorTests:
    TEST_CLASS = None
    # TODO add optional kwargs to test with

    def test_is_an_estimator_mixin(self):
        self.assertTrue(
            issubclass(self.TEST_CLASS, Estimator),
            f"Test class {self.TEST_CLASS} is not a {Estimator} mixin",
        )

    def test_implements_make_estimator(self):
        result = self.TEST_CLASS().make_estimator()
        self.assertIsInstance(result, sklearn.base.BaseEstimator)


class TestIdentityRegressor(EstimatorTests, unittest.TestCase):
    TEST_CLASS = IdentityRegressor


class TestLinearRegressor(EstimatorTests, unittest.TestCase):
    TEST_CLASS = LinearRegressor


class TestXgbClassifier(EstimatorTests, unittest.TestCase):
    TEST_CLASS = XgbClassifier


class TestXgbRegressor(EstimatorTests, unittest.TestCase):
    TEST_CLASS = XgbRegressor


class EstimatorWrapperTest(EstimatorTests):
    # Define test constants
    MOCK_ESTIMATOR = create_autospec(sklearn.base.BaseEstimator, fit=Mock())
    EXPECTED_WRAPPER = None

    # Create a test harness class that implements Estimator & Process used defined Mocks from the outer scope
    class TestHarnessProcessEstimator(Estimator, Process):
        def make_estimator(self, **kwargs):
            return EstimatorWrapperTest.MOCK_ESTIMATOR

        def get_range(self, domain):
            return Mock("Range")

        def make_preprocessor(self, **kwagrs):
            return sklearn.compose.ColumnTransformer([])

    def test_implements_make_estimator(self):
        result = self.TEST_CLASS(Mock("TimeStep")).make_estimator()
        self.assertIsInstance(result, self.EXPECTED_WRAPPER)

    def test_estimator_property(self):
        test_instance = self.TEST_CLASS(Mock("TimeStep"))
        self.assertIs(test_instance.estimator, self.MOCK_ESTIMATOR)


class TestSequentialFeatureSelection(EstimatorWrapperTest, unittest.TestCase):
    class ModelTest(
        SequentialFeatureSelection,
        EstimatorWrapperTest.TestHarnessProcessEstimator,
        RegularTimeSeriesModel,
    ):
        pass

    TEST_CLASS = ModelTest
    EXPECTED_WRAPPER = sklearn.pipeline.Pipeline


class TestTransformedTargetRegressor(EstimatorWrapperTest, unittest.TestCase):
    class ModelTest(
        TransformedTargetRegressor,
        EstimatorWrapperTest.TestHarnessProcessEstimator,
        RegularTimeSeriesModel,
    ):
        pass

    TEST_CLASS = ModelTest
    EXPECTED_WRAPPER = sklearn.compose.TransformedTargetRegressor

    def test_estimator_property(self):
        test_instance = self.TEST_CLASS(Mock("TimeStep"))

        # Assert that the estimator property is None before fitting
        self.assertIs(test_instance.estimator, None)

        # Use the unbound fget method of the property from the class to get the sklearn transformed target regressor instance
        transform_target_regressor = RegularTimeSeriesModel.estimator.fget(
            test_instance
        )

        # Set internal state of the transformed target regressor as though the estimator is fitted
        transform_target_regressor.regressor_ = self.MOCK_ESTIMATOR

        # Assert that the estimator property is getting the correct object
        self.assertIs(test_instance.estimator, self.MOCK_ESTIMATOR)


class TestRandomizedSearch(EstimatorWrapperTest, unittest.TestCase):
    class ModelTest(
        RandomizedSearch,
        EstimatorWrapperTest.TestHarnessProcessEstimator,
        RegularTimeSeriesModel,
    ):
        pass

    TEST_CLASS = ModelTest
    EXPECTED_WRAPPER = sklearn.model_selection.RandomizedSearchCV

    def test_estimator_property(self):
        test_instance = self.TEST_CLASS(Mock("TimeStep"))

        # Assert that the estimator property is None before fitting
        self.assertIs(test_instance.estimator, None)

        # Use the unbound fget method of the property from the class to get the sklearn transformed target regressor instance
        random_search_cv = RegularTimeSeriesModel.estimator.fget(test_instance)

        # Set internal state as though the estimator is fit
        random_search_cv.best_estimator_ = self.MOCK_ESTIMATOR

        # Assert that the estimator property is getting the correct object
        self.assertIs(test_instance.estimator, self.MOCK_ESTIMATOR)


if __name__ == "__main__":
    unittest.main()
