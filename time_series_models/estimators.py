import inspect
import logging
import sys
from abc import ABC, abstractmethod

import sklearn
import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import keras.wrappers.scikit_learn

import numpy as np
import xgboost as xgb

from time_series_models.decorators import remove_custom_kwargs
from time_series_models.filters import StandardScaler
from time_series_models.time_series_models import Mixin
from time_series_models.sklearn import sklearn_monkey_patch

# Make sure patches run in joblib workers!
sklearn_monkey_patch.apply_patches()

logger = logging.getLogger(__name__)

DEFAULT_REGRESSOR_METRICS = [
    sklearn.metrics.explained_variance_score,
    sklearn.metrics.max_error,
    sklearn.metrics.mean_absolute_error,
    sklearn.metrics.mean_squared_error,
    sklearn.metrics.mean_squared_log_error,
    sklearn.metrics.median_absolute_error,
    sklearn.metrics.mean_absolute_percentage_error,
    sklearn.metrics.r2_score,
    sklearn.metrics.mean_poisson_deviance,
    sklearn.metrics.mean_gamma_deviance,
    sklearn.metrics.mean_tweedie_deviance,
]

DEFAULT_CLASSIFIER_METRICS = [
    sklearn.metrics.accuracy_score,
    sklearn.metrics.average_precision_score,
    sklearn.metrics.balanced_accuracy_score,
    sklearn.metrics.brier_score_loss,
    sklearn.metrics.classification_report,
    sklearn.metrics.cohen_kappa_score,
    sklearn.metrics.confusion_matrix,
    sklearn.metrics.dcg_score,
    sklearn.metrics.det_curve,
    sklearn.metrics.f1_score,
    sklearn.metrics.hamming_loss,
    sklearn.metrics.hinge_loss,
    sklearn.metrics.jaccard_score,
    sklearn.metrics.log_loss,
    sklearn.metrics.matthews_corrcoef,
    sklearn.metrics.precision_recall_fscore_support,
    sklearn.metrics.precision_score,
    sklearn.metrics.recall_score,
]

DEFAULT_CLASSIFIER_METRICS_BY_NAME = {
    func.__name__: func for func in DEFAULT_CLASSIFIER_METRICS
}

DEFAULT_REGRESSOR_METRICS_BY_NAME = {
    func.__name__: func for func in DEFAULT_REGRESSOR_METRICS
}


class Estimator(Mixin):
    """
    Generic estimator mixin class
    """

    @abstractmethod
    def make_estimator(self, **kwargs):
        """
        Defines how to make a scikit-learn estimator
        :param kwargs: model configuration arguments
        :return: the estimator instance
        """


class EstimatorWrapper(Estimator, ABC):
    """
    Generic estimator wrapper encapsulates the estimator in another sklearn object for feature selection or hyperparameter tuning
    """

    @property
    @abstractmethod
    def estimator(self):
        """
        Property getter for the actual estimator - interacts with the Estimator mixin and RegularTimeSeriesModels!
        :return: the actual estimator object
        """
        # Must call super as it overrides functionality from RegularTimeSeriesModels
        return super().estimator


class Regressor(Estimator, ABC):
    """
    Base class for mixin's which implement a regression estimator
    """

    METRICS = DEFAULT_REGRESSOR_METRICS


class Classifier(Estimator, ABC):
    """
    Base class for mixin's which implement a classifier estimator
    """

    METRICS = DEFAULT_CLASSIFIER_METRICS


class IdentityEstimator(sklearn.preprocessing.FunctionTransformer):
    """
    This class can be used to estimate the value of a dependent variable as equal to the value of the independent
    variable. The features array must be (n, 1).

    This class is a "small e" estimator rather than a "big E" Estimator. To use it with RegularTimeSeriesModels, define
    a Mixin that returns IdentityEstimator for its `make_estimator` method (e.g., the 'IdentityRegressor' class, below).
    TODO: Repair this class to assert sanity about the input. Hacked to work with new location count column.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self._check_input_shape(X)
        return super().fit(X, y)

    def predict(self, X):
        self._check_input_shape(X)
        return self.transform(X)

    @staticmethod
    def _check_input_shape(x):
        try:
            assert x.shape[1] == 1
        except IndexError:
            logger.exception("Feature data must be 2-dimensional")
            raise
        except AssertionError:
            logger.exception(
                "Feature data must consist of a single column! Found %s.", x.shape[1]
            )
            raise


class IdentityRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self):
        """
        Estimator takes a single column of feature data and predicts the same values.
        """
        return IdentityEstimator()


class LinearRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for details
        """
        return sklearn.linear_model.LinearRegression(**kwargs)


class LassoRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html for details
        """
        return sklearn.linear_model.Lasso(**kwargs)


class RidgeRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for details
        """
        return sklearn.linear_model.Ridge(**kwargs)


class ElasticNetRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for details
        """
        return sklearn.linear_model.ElasticNet(**kwargs)


class RidgeClassifier(Classifier):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for details
        """
        return sklearn.linear_model.RidgeClassifier(**kwargs)


class XgbClassifier(Classifier):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst for details
        """
        return xgb.XGBClassifier(use_label_encoder=False, **kwargs)


class XgbRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst for details
        """
        return xgb.XGBRegressor(**kwargs)


class MaskedXGBRegressor(xgb.XGBRegressor):
    """
    Extends xgb.XGBRegressor, dropping feature and target rows during training where the target value is missing.
    If target is not a masked array, there will be no difference from the parent class except for some logging.
    """

    def fit(self, x, y=None, **kwargs):
        logger.debug(
            "MaskedXGBRegressor 'fit' with %s features, %s labels", x.shape, y.shape
        )
        logger.debug("labels nan sum: %s", np.isnan(y).sum())
        if type(y) is np.ma.MaskedArray:
            logger.debug("MaskedXGBRegressor 'fit' detected masked array of labels")
            logger.debug("before: X = %s, y = %s", x.shape, y.shape)
            logger.info(
                "MaskedXGBRegressor dropping %s masked rows from X and y prior to 'fit'!",
                y.mask.sum(),
            )
            x = x[~y.mask.reshape(-1), :].copy()
            y = y[~y.mask].copy()
            logger.debug("after: X is %s, y is %s", x.shape, y.shape)
        logger.debug(
            "MaskedXGBRegressor calling XGBRegressor 'fit' on X %s, y %s",
            x.shape,
            y.shape,
        )
        return super().fit(x, y=y, **kwargs)


class MaskingXGBRegressor(Regressor):
    """Mixin to use MaskedXGBRegressor as the estimator for RegularTimeSeriesModels"""

    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst for details
        """
        return MaskedXGBRegressor(**kwargs)


class MLPClassifier(Classifier):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://scikit-learn.org/stable/modules/neural_networks_supervised.html for details
        """
        return sklearn.pipeline.Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", sklearn.neural_network.MLPClassifier(**kwargs)),
            ]
        )


class MLPRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://scikit-learn.org/stable/modules/neural_networks_supervised.html for details
        # TODO Does the standard scalar pipeline belong here? It is a shortcut for a required behavior
        """
        return sklearn.pipeline.Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", sklearn.neural_network.MLPRegressor(**kwargs)),
            ]
        )


class KerasClassifier(Classifier):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#fit for details
        # TODO Does the standard scalar pipeline belong here? It is a shortcut for a required behavior
        """
        return sklearn.pipeline.Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", keras.wrappers.scikit_learn.KerasClassifier(**kwargs)),
            ]
        )


class KerasRegressor(Regressor):
    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        """
        See https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#fit for details
        # TODO Does the standard scalar pipeline belong here? It is a shortcut for a required behavior
        """
        return sklearn.pipeline.Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", keras.wrappers.scikit_learn.KerasRegressor(**kwargs)),
            ]
        )


class SequentialFeatureSelection(EstimatorWrapper):
    """
    Mixin to use sequential feature selection to reduce over fitting
    # TODO not really an estimator? Make a new base for estimator wrapper?
    """

    # @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        sfs_args = {
            k: kwargs.pop(k)
            for k in ("n_features_to_select", "direction", "scoring", "cv", "n_jobs")
            if k in kwargs
        }
        return sklearn.pipeline.Pipeline(
            [
                (
                    "SFS",
                    sklearn.feature_selection.SequentialFeatureSelector(
                        estimator=super().make_estimator(**kwargs),
                        **sfs_args,
                    ),
                ),
                ("sfs_estimator", super().make_estimator(**kwargs)),
            ]
        )

    @property
    def estimator(self):
        return super().estimator["sfs_estimator"]

    @property
    def selected_features(self):
        return np.array(self.get_feature_names())[
            super().estimator["SFS"].get_support()
        ]

    @property
    def rejected_features(self):
        return np.array(self.get_feature_names())[
            ~super().estimator["SFS"].get_support()
        ]


class TransformedTargetRegressor(EstimatorWrapper):
    """
    Mixin to Transform Targets before Regression
    Example:
    class Model(BalancingAreaHourly, TransformedTargetRegressor, XgbRegressor, RegularTimeSeriesModel):
        pass
    """

    @remove_custom_kwargs
    def make_estimator(self, **kwargs):
        transformer = kwargs.pop("target_transformer", StandardScaler())
        return sklearn.compose.TransformedTargetRegressor(
            regressor=super().make_estimator(**kwargs),
            transformer=transformer,
        )

    @property
    def estimator(self):
        return getattr(super().estimator, "regressor_", None)


class RandomizedSearch(EstimatorWrapper):
    """
    Mixin to add Grid Search for parameter space
    Example:
    class Model(BalancingAreaHourly, RandomizedSearch, TransformedTargetRegressor, XgbRegressor, RegularTimeSeriesModel):
        pass

    make_estimator defaults to a scikit-learn RandomizedSearchCV with a rolling (time series) cross-validation strategy.
    The default splitter is documented at
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    """

    @remove_custom_kwargs
    def make_estimator(
        self, param_distributions=None, cv_splitter="DefaultTimeSeriesSplit", **kwargs
    ):
        # TODO incomplete - set params and sort out parallel execution

        if not param_distributions:
            param_distributions = {"regressor__n_estimators": [1, 10, 50]}

        if cv_splitter == "DefaultTimeSeriesSplit":
            logger.warning(
                "Using default time series split for hyperparameter tuning. "
                "A custom split scheme might be more appropriate!"
            )
            cv_splitter = sklearn.model_selection.TimeSeriesSplit()

        logger.info(
            "RandomizedSearchCV using cross validation splitter: %s", cv_splitter
        )

        return sklearn.model_selection.RandomizedSearchCV(
            estimator=super().make_estimator(**kwargs),
            param_distributions=param_distributions,
            # TODO(Michael H): allow control over n_iter or other params?
            n_iter=10,
            scoring=None,
            n_jobs=kwargs.get("n_jobs", None),
            refit=True,
            cv=cv_splitter,
            verbose=0,
            pre_dispatch="2*n_jobs",
            random_state=None,
            error_score=np.nan,
            return_train_score=False,
        )

    @property
    def estimator(self):
        return getattr(super().estimator, "best_estimator_", None)

    @property
    def best_params(self):
        return super().estimator.best_params_

    @property
    def best_score(self):
        return super().estimator.best_score_

    @property
    def cv_results(self):
        return super().estimator.cv_results_


REGISTERED_CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}
