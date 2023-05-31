import unittest
from abc import ABC, abstractmethod
from unittest.mock import Mock

import sklearn
import sklearn.datasets
import sklearn.pipeline

import numpy as np
import pandas as pd

from time_series_models.filters import (
    DropFeatures,
    NullifyFeatures,
)


class DropFeaturesTest(unittest.TestCase):
    iris = sklearn.datasets.load_iris()
    X = iris["data"]
    y = iris["target"]
    # check dimensions of test dataset
    assert X.shape == (150, 4)

    class Tester(ABC):
        def __init__(self, time_step, **kwargs):
            self.time_step = time_step
            self.model = sklearn.pipeline.Pipeline(  # we must have a pipeline to access "named_steps"
                steps=[
                    ("filter", self.filter_features(**kwargs)),
                ]
            )

        @abstractmethod
        def filter_features(self, **kwargs):
            pass

        @property
        def filters(self):
            return self.model.named_steps["filter"]

    def test_tester(self):  # TODO(Michael): extract Tester class (and associated test)
        # confirm the test harness class raises TypeError when abstractmethod isn't overwritten
        with self.assertRaises(TypeError):
            self.Tester(np.timedelta64(1, "h"))

    class IntegrationTester(ABC):
        def __init__(self, time_step, **kwargs):
            self.time_step = time_step
            self.model = sklearn.pipeline.Pipeline(
                steps=[
                    ("filter", self.filter_features(**kwargs)),
                    ("estimator", self.make_estimator()),
                ]
            )

        @abstractmethod
        def filter_features(self, **kwargs):
            pass

        def make_estimator(self, **kwargs):
            return Mock()

        @property
        def filters(self):
            return self.model.named_steps["filter"]

    def test_integration_tester(
        self,
    ):  # TODO(Michael): extract IntegrationTester class (and associated test)
        # confirm the test harness class raises TypeError when abstractmethod isn't overwritten
        with self.assertRaises(TypeError):
            self.Tester(np.timedelta64(1, "h"))

    class FilterTester(DropFeatures, Tester):
        pass

    class IntegratedFilterTester(DropFeatures, IntegrationTester):
        pass

    def test_drop_features__drop_one(self):
        ft = self.FilterTester(np.timedelta64(1, "h"), drop_features=[1])
        self.assertListEqual(ft.drop_features, [1])
        res = ft.model.fit_transform(self.X)
        self.assertTupleEqual(res.shape, (150, 3))
        np.testing.assert_array_equal(res, self.X[:, np.r_[0, 2, 3]])

    def test_drop_features__empty_noop(self):
        # create concrete class with no drop_features -> result should be no-op
        ft = self.FilterTester(np.timedelta64(1, "h"))
        self.assertEqual(ft.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ft.drop_features, [])
        self.assertIsInstance(
            ft.model.named_steps["filter"], sklearn.compose.ColumnTransformer
        )
        self.assertEqual(ft.model.named_steps["filter"].remainder, "passthrough")
        res = ft.model.fit_transform(self.X)
        np.testing.assert_array_equal(res, self.X)

    def test_drop_features__illdefined_raises(self):
        # create concrete class with improperly defined drop_features
        ft = self.FilterTester(np.timedelta64(1, "h"), drop_features=["bingo"])
        self.assertEqual(ft.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ft.drop_features, ["bingo"])
        self.assertIsInstance(
            ft.model.named_steps["filter"], sklearn.compose.ColumnTransformer
        )
        # string specification of columns is only supported for pandas DataFrame
        with self.assertRaises(ValueError):
            with self.assertRaises(AttributeError):
                ft.model.fit_transform(self.X)

    def test_drop_features__drop_multiple(self):
        # drop three columns
        ft = self.FilterTester(np.timedelta64(1, "h"), drop_features=[0, 1, 3])
        self.assertListEqual(ft.drop_features, [0, 1, 3])
        res = ft.model.fit_transform(self.X)
        np.testing.assert_array_equal(res, self.X[:, [2]])

    def test_drop_features__pandas_input(self):
        # use pandas DataFrame input: works with ColumnTransformer
        df = pd.DataFrame(data=self.X, columns=["one", "two", "three", "four"])
        ft = self.FilterTester(np.timedelta64(1, "h"), drop_features=["one"])
        self.assertListEqual(ft.drop_features, ["one"])
        res = ft.model.fit_transform(df)
        np.testing.assert_array_equal(
            res, self.X[:, 1:]
        )  # note output is still np.ndarray

    def test_drop_features__change_features(self):
        # create concrete class with one feature to drop
        ft = self.FilterTester(np.timedelta64(1, "h"), drop_features=[2])
        self.assertListEqual(ft.drop_features, [2])
        with self.assertRaises(AttributeError):
            ft.drop_features = [0, 1]  # can't change drop_features before initial fit
        res = ft.model.fit_transform(self.X)
        expected = self.X.copy()
        expected = expected[:, np.r_[0, 1, 3]]
        np.testing.assert_array_equal(res, expected)

        # now change which features should be dropped
        ft.drop_features = [0, 1]
        self.assertListEqual(ft.drop_features, [0, 1])
        res = ft.model.transform(self.X)
        expected = self.X.copy()
        expected = expected[:, 2:]
        np.testing.assert_array_equal(res, expected)

    def test_drop_features_integrated__empty_noop(self):
        ift = self.IntegratedFilterTester(np.timedelta64(1, "h"))
        self.assertEqual(ift.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ift.drop_features, [])

        ift.model.fit(self.X, self.y)
        ift.model.predict(self.X)

        ift.model.named_steps["estimator"].fit.assert_called_once()
        ift.model.named_steps["estimator"].predict.assert_called_once()

        # Mock assertions aren't well equipped to compare Numpy arrays so do it manually
        # evaluate fit call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(len(ift.model.named_steps["estimator"].fit.call_args.args), 2)
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][0], self.X
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][1], self.y
        )

        # evaluate predict call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args.args), 1
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args[0][0], self.X
        )

    def test_drop_features_integrated__two_null(self):
        ift = self.IntegratedFilterTester(np.timedelta64(1, "h"), drop_features=[1, 2])
        self.assertEqual(ift.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ift.drop_features, [1, 2])

        ift.model.fit(self.X, self.y)
        ift.model.predict(self.X)

        ift.model.named_steps["estimator"].fit.assert_called_once()
        ift.model.named_steps["estimator"].predict.assert_called_once()

        # Mock assertions aren't well equipped to compare Numpy arrays so do it manually
        expected = self.X.copy()
        expected = expected[:, np.r_[0, 3]]

        # evaluate fit call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(len(ift.model.named_steps["estimator"].fit.call_args.args), 2)
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][0], expected
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][1], self.y
        )

        # evaluate predict call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args.args), 1
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args[0][0], expected
        )

    def test_drop_features_integrated__change_features(self):
        ift = self.IntegratedFilterTester(np.timedelta64(1, "h"), drop_features=[1, 2])
        self.assertListEqual(ift.drop_features, [1, 2])

        ift.model.fit(self.X, self.y)
        ift.model.predict(self.X)
        ift.drop_features = [0, 3]
        ift.model.predict(self.X)

        # fit should only have been called once, but predict should have been called twice
        ift.model.named_steps["estimator"].fit.assert_called_once()
        self.assertEqual(ift.model.named_steps["estimator"].predict.call_count, 2)

        # Mock assertions aren't well equipped to compare Numpy arrays so do it manually
        expected_0 = self.X.copy()
        expected_0 = expected_0[:, np.r_[0, 3]]

        expected_1 = self.X.copy()
        expected_1 = expected_1[:, np.r_[1, 2]]

        # evaluate fit call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(len(ift.model.named_steps["estimator"].fit.call_args.args), 2)
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][0], expected_0
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][1], self.y
        )

        # evaluate predict call: should have been called twice, first with expected_0 and second with expected_1
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args_list),
            2,  # one tuple of args per call, for two calls
        )
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args_list[0].args),
            1,  # single arg in first call
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args_list[0].args[0],
            expected_0,  # contents of the first call arg
        )
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args_list[1].args),
            1,  # single arg in second call
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args_list[1].args[0],
            expected_1,  # contents of the second call arg
        )


class NullifyFeaturesTest(unittest.TestCase):
    iris = sklearn.datasets.load_iris()
    X = iris["data"]
    y = iris["target"]
    # check dimensions of test dataset
    assert X.shape == (150, 4)
    assert y.shape == (150,)

    class Tester(ABC):
        def __init__(self, time_step, **kwargs):
            self.time_step = time_step
            self.model = sklearn.pipeline.Pipeline(  # we must have a pipeline to access "named_steps"
                steps=[
                    ("filter", self.filter_features(**kwargs)),
                ]
            )

        @abstractmethod
        def filter_features(self, **kwargs):
            return sklearn.pipeline.Pipeline([("passthrough", None)])

        @property
        def filters(self):
            return self.model.named_steps["filter"]

    def test_tester(self):  # TODO(Michael): extract Tester class (and associated test)
        # confirm the test harness class raises TypeError when abstractmethod isn't overwritten
        with self.assertRaises(TypeError):
            self.Tester(np.timedelta64(1, "h"))

    class IntegrationTester(ABC):
        def __init__(self, time_step, **kwargs):
            self.time_step = time_step
            self.model = sklearn.pipeline.Pipeline(
                steps=[
                    ("filter", self.filter_features(**kwargs)),
                    ("estimator", self.make_estimator()),
                ]
            )

        @abstractmethod
        def filter_features(self, **kwargs):
            return sklearn.pipeline.Pipeline([("passthrough", None)])

        def make_estimator(self, **kwargs):
            return Mock()

        @property
        def filters(self):
            return self.model.named_steps["filter"]

    def test_integration_tester(
        self,
    ):  # TODO(Michael): extract IntegrationTester class (and associated test)
        # confirm the test harness class raises TypeError when abstractmethod isn't overwritten
        with self.assertRaises(TypeError):
            self.IntegrationTester(np.timedelta64(1, "h"))

    class FilterTester(NullifyFeatures, Tester):
        pass

    class IntegratedFilterTester(NullifyFeatures, IntegrationTester):
        pass

    def test_nullify_features__one_null(self):
        ft = self.FilterTester(np.timedelta64(1, "h"), null_features=[1])
        self.assertListEqual(ft.null_features, [1])
        self.assertIsInstance(ft.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(ft.filters, sklearn.pipeline.Pipeline)
        self.assertIsInstance(
            ft.filters.named_steps["nullify_features"],
            sklearn.preprocessing.FunctionTransformer,
        )
        res = ft.model.fit_transform(self.X)
        self.assertTupleEqual(res.shape, self.X.shape)
        expected = self.X.copy()
        expected[:, 1] = np.nan
        np.testing.assert_array_equal(res, expected)

    def test_nullify_features__empty_noop(self):
        # create concrete class with no drop_features -> result should be no-op
        ft = self.FilterTester(np.timedelta64(1, "h"))
        self.assertEqual(ft.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ft.null_features, [])
        self.assertIsInstance(ft.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(ft.filters, sklearn.pipeline.Pipeline)
        self.assertIsInstance(
            ft.filters.named_steps["nullify_features"],
            sklearn.preprocessing.FunctionTransformer,
        )
        res = ft.model.fit_transform(self.X)
        np.testing.assert_array_equal(res, self.X)

    def test_nullify_features__illdefined_raises(self):
        # create concrete class with improperly defined drop_features
        ft = self.FilterTester(np.timedelta64(1, "h"), null_features=["bingo"])
        self.assertEqual(ft.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ft.null_features, ["bingo"])
        self.assertIsInstance(ft.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(ft.filters, sklearn.pipeline.Pipeline)
        self.assertIsInstance(
            ft.filters.named_steps["nullify_features"],
            sklearn.preprocessing.FunctionTransformer,
        )
        # string specification of columns is only supported for pandas DataFrame
        with self.assertRaises(IndexError):
            ft.model.fit_transform(self.X)

    def test_nullify_features__multiple_features(self):
        # null three columns
        ft = self.FilterTester(np.timedelta64(1, "h"), null_features=[0, 1, 3])
        self.assertListEqual(ft.null_features, [0, 1, 3])
        res = ft.model.fit_transform(self.X)
        expected = self.X.copy()
        expected[:, np.r_[0, 1, 3]] = np.nan
        np.testing.assert_array_equal(res, expected)

    def test_nullify_features__pandas_input(self):
        # use pandas DataFrame input
        df = pd.DataFrame(data=self.X, columns=["one", "two", "three", "four"])
        ft = self.FilterTester(np.timedelta64(1, "h"), null_features=["one"])
        self.assertListEqual(ft.null_features, ["one"])

        # TODO(Michael H): do we want to allow string column specification?
        with self.assertRaises(IndexError):
            ft.model.fit_transform(df)

    def test_nullify_features__change_features(self):
        # create concrete class with one feature to drop
        ft = self.FilterTester(np.timedelta64(1, "h"), null_features=[2])
        self.assertListEqual(ft.null_features, [2])
        res = ft.model.fit_transform(self.X)
        expected = self.X.copy()
        expected[:, 2] = np.nan
        np.testing.assert_array_equal(res, expected)

        # now change which features should be dropped
        ft.null_features = [0, 1]
        self.assertListEqual(ft.null_features, [0, 1])
        res = ft.model.transform(self.X)
        expected = self.X.copy()
        expected[:, :2] = np.nan
        np.testing.assert_array_equal(res, expected)

    def test_nullify_features_integrated__empty_noop(self):
        ift = self.IntegratedFilterTester(np.timedelta64(1, "h"))
        self.assertEqual(ift.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ift.null_features, [])

        ift.model.fit(self.X, self.y)
        ift.model.predict(self.X)

        ift.model.named_steps["estimator"].fit.assert_called_once()
        ift.model.named_steps["estimator"].predict.assert_called_once()

        # Mock assertions aren't well equipped to compare Numpy arrays so do it manually
        # evaluate fit call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(len(ift.model.named_steps["estimator"].fit.call_args.args), 2)
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][0], self.X
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][1], self.y
        )

        # evaluate predict call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args.args), 1
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args[0][0], self.X
        )

    def test_nullify_features_integrated__two_null(self):
        ift = self.IntegratedFilterTester(np.timedelta64(1, "h"), null_features=[1, 2])
        self.assertEqual(ift.time_step, np.timedelta64(1, "h"))
        self.assertListEqual(ift.null_features, [1, 2])

        ift.model.fit(self.X, self.y)
        ift.model.predict(self.X)

        ift.model.named_steps["estimator"].fit.assert_called_once()
        ift.model.named_steps["estimator"].predict.assert_called_once()

        # Mock assertions aren't well equipped to compare Numpy arrays so do it manually
        expected = self.X.copy()
        expected[:, np.r_[1, 2]] = np.nan

        # evaluate fit call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(len(ift.model.named_steps["estimator"].fit.call_args.args), 2)
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][0], expected
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][1], self.y
        )

        # evaluate predict call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args.args), 1
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args[0][0], expected
        )

    def test_nullify_features_integrated__change_features(self):
        ift = self.IntegratedFilterTester(np.timedelta64(1, "h"), null_features=[1, 2])
        self.assertListEqual(ift.null_features, [1, 2])

        ift.model.fit(self.X, self.y)
        ift.model.predict(self.X)
        ift.null_features = [0, 3]
        ift.model.predict(self.X)

        # fit should only have been called once, but predict should have been called twice
        ift.model.named_steps["estimator"].fit.assert_called_once()
        self.assertEqual(ift.model.named_steps["estimator"].predict.call_count, 2)

        # Mock assertions aren't well equipped to compare Numpy arrays so do it manually
        expected_0 = self.X.copy()
        expected_0[:, np.r_[1, 2]] = np.nan

        expected_1 = self.X.copy()
        expected_1[:, np.r_[0, 3]] = np.nan

        # evaluate fit call: args passed to Mock estimator are the output of filter mixin in preceding step
        self.assertEqual(len(ift.model.named_steps["estimator"].fit.call_args.args), 2)
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][0], expected_0
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].fit.call_args[0][1], self.y
        )

        # evaluate predict call: should have been called twice, first with expected_0 and second with expected_1
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args_list),
            2,  # one tuple of args per call, for two calls
        )
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args_list[0].args),
            1,  # single arg in first call
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args_list[0].args[0],
            expected_0,  # contents of the first call arg
        )
        self.assertEqual(
            len(ift.model.named_steps["estimator"].predict.call_args_list[1].args),
            1,  # single arg in second call
        )
        np.testing.assert_array_equal(
            ift.model.named_steps["estimator"].predict.call_args_list[1].args[0],
            expected_1,  # contents of the second call arg
        )


if __name__ == "__main__":
    unittest.main()
