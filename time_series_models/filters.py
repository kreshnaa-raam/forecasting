import inspect
import logging
import sys
from abc import abstractmethod

import sklearn.compose
import sklearn.preprocessing

import numpy as np

from time_series_models.time_series_models import Mixin
from time_series_models.transformers import (
    interpolate_array_by_group,
    interpolate_array_interior,
    nullify_cols,
    revise_pipeline,
    RowFilteringFunctionTransformer,
)
from time_series_models.sklearn import sklearn_monkey_patch

# Make sure patches run in joblib workers!
sklearn_monkey_patch.apply_patches()

logger = logging.getLogger(__name__)


class Filter(Mixin):
    @abstractmethod
    def filter_features(self, **kwargs):
        """
        Adds a composable filter process between the preprocessor (feature builder) and the estimator.
        :param kwargs: model configuration
        :return: sklearn pipeline step e.g. StandardScaler
        """
        return super().filter_features(**kwargs)


class DropFeatures(Filter):
    """
    Introduce a pipeline step to eliminate a specified set of features.
    :param drop_features: a list of features to drop
    """

    # TODO(Michael H): make this mixin compatible with the new filter pipeline format
    def filter_features(self, **kwargs):
        return sklearn.compose.ColumnTransformer(
            transformers=[
                ("feature_dropper", "drop", kwargs.get("drop_features", [])),
            ],
            remainder="passthrough",
        )

    @property
    def drop_features(self):
        return self.filters.transformers[0][2]

    @drop_features.setter
    def drop_features(self, drop_features: list[int]):
        passthrough_cols = self.filters.transformers_[1][2]
        cols = (
            self.drop_features + passthrough_cols
        )  # this should only work after model fit call
        [cols.remove(col) for col in drop_features]
        self.filters.transformers[0] = ("feature_dropper", "drop", drop_features)
        self.filters.transformers_[0] = ("feature_dropper", "drop", drop_features)
        self.filters.transformers_[1] = ("remainder", "passthrough", cols)


class InterpolateLinear(Filter):
    """
    Applies linear interpolation to the interior of each time series (i.e., between first and last valid indices),
    grouped by domain location.
    Assumes autoregressive features (lags) are leftmost in the feature array (following the location count column).
    """

    def filter_features(self, interpolate_all: bool = True, **kwargs):
        n_col = -1
        if not interpolate_all:
            lags = kwargs.get("lags")
            if lags is None:
                logger.warning(
                    "Could not deduce number of columns to interpolate from a lags arg,"
                    " so defaulting to interpolating all columns!"
                )
            else:
                n_col = len(lags)
                logger.info(
                    "Interpolation configured for the first %i feature columns", n_col
                )

        return revise_pipeline(
            super().filter_features(**kwargs),
            (
                "linear_interpolation",
                sklearn.preprocessing.FunctionTransformer(
                    func=interpolate_array_by_group,
                    kw_args=dict(interp_func=interpolate_array_interior, n_col=n_col),
                ),
            ),
        )


class InterpolateExtrapolateLinear(Filter):
    """
    Applies linear interpolation and constant forward extrapolation to each time series, grouped by domain location.
    Assumes autoregressive features (lags) are leftmost in the feature array (following the location count column).
    """

    def filter_features(self, interpolate_all: bool = True, **kwargs):
        # TODO(Michael H): D.R.Y. this section iff we repeat it again
        n_col = -1
        if not interpolate_all:
            lags = kwargs.get("lags")
            if lags is None:
                logger.warning(
                    "Could not deduce number of columns to interpolate from a lags arg,"
                    " so defaulting to interpolating all columns!"
                )
            else:
                n_col = len(lags)
                logger.info(
                    "Interpolation configured for the first %i feature columns", n_col
                )

        return revise_pipeline(
            super().filter_features(**kwargs),
            (
                "linear_interpolation",
                sklearn.preprocessing.FunctionTransformer(
                    func=interpolate_array_by_group,
                    kw_args=dict(n_col=n_col),
                ),
            ),
        )


class NullifyFeatures(Filter):
    """
    Modify pipeline "filters" step to nullify a specified set of features.
    :param null_features: a list of features to nullify, passed as a kwarg
    """

    def filter_features(self, **kwargs):
        return revise_pipeline(
            super().filter_features(**kwargs),
            (
                "nullify_features",
                sklearn.preprocessing.FunctionTransformer(
                    func=nullify_cols,
                    kw_args=dict(cols=kwargs.get("null_features", [])),
                ),
            ),
        )

    @property
    def null_features(self):
        return self.filters.named_steps["nullify_features"].kw_args["cols"]

    @null_features.setter
    def null_features(self, null_features: list[int]):
        self.filters.named_steps["nullify_features"].kw_args["cols"] = null_features


class RowFilter(Filter):
    """
    This class identifies samples (rows) that are missing target labels (i.e., where fit_range is Null). It removes the
    missing observations fit_range prior to calling pipeline.fit, and removes the corresponding rows from the feature
    array (constructed by the preprocessor) during pipeline.fit, so that the supervised learning estimator is trained on
    a completely labeled target array. This class does not modify the feature array during pipeline.predict.
    """

    def filter_features(self, **kwargs):
        return revise_pipeline(
            super().filter_features(**kwargs),
            ("row_filter", RowFilteringFunctionTransformer()),
        )

    @classmethod
    def _fit(cls, model, fit_domain, fit_range, **kwargs):
        """
        Overrides default estimator _fit. Note, this filter passes a modified fit_range into the start of the pipeline!
        :param model: a scikit-learn like model that can be fit on X=fit_domain and y=fit_range.
        :param fit_domain: the domain array from which the features will be constructed for model fitting.
        :param fit_range: the array of target values that will be the label set for supervised learning.
        :param kwargs: special arguments to the fit method
        TODO pass kwargs to xgb model per https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py
        """
        logger.debug("RowFilter got range with length %i", len(fit_range))
        range_nan = np.isnan(fit_range)
        # filter nan values from fit_range!
        fit_range = fit_range[~range_nan].copy()
        # TODO (Michael H): monitor the original length and the number missing
        logger.warning(
            "RowFilter removed %i missings from fit_range of length %i, "
            "leaving %i observations in fit_range prior to sklearn pipeline fit",
            range_nan.sum(),
            len(range_nan),
            len(fit_range),
        )
        super()._fit(
            model,
            fit_domain,
            fit_range,
            feature_filter__row_filter__range_nan=range_nan,
            **kwargs,
        )


class StandardScaler(Filter):
    """
    Add a standard scaler transform as a filter
    """

    def filter_features(self, **kwargs):
        return revise_pipeline(
            super().filter_features(**kwargs), ("standard_scaler", StandardScaler())
        )


REGISTERED_CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}
