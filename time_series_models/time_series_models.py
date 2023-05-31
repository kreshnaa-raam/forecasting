import io
import logging
import inspect
import sys
import datetime
from abc import ABC, abstractmethod
from pathlib import PurePosixPath

import fsspec
import gcsfs
import google.auth
import numpy as np
import pandas as pd
import cloudpickle as pickle

import sklearn
import sklearn.compose
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

from time_series_models import shap_viz
from time_series_models.constants import LOCATION, DATE_TIME
from time_series_models.decorators import (
    debug,
    domain_wrangler,
)

from time_series_models.sklearn import sklearn_monkey_patch

# Make sure patches run in joblib workers!
sklearn_monkey_patch.apply_patches()

from time_series_models.data_monitor import ForecastDataMonitor
from time_series_models.transformers import (
    count_domain_locs_into_column,
    drop_first_column,
    make_domain,
    make_domain_type,
    multiindex_from_domain,
)
from time_series_models.time_unit import TimeUnitEnum

logger = logging.getLogger(__name__)


class RegularTimeSeriesModel(ABC):
    """
    Defines behavior for regular time series regression models. Implementations may use either regression or classifier
    models. Scikit learn regressors are expected but any implementation of BaseEstimator should be fine. Data are expected to have a
    fixed time step to allow for auto correlated features.
    """

    CUSTOM_KWARGS = []

    @debug(logger)
    def __init__(self, time_step, **kwargs):
        """
        Define the timestep for the model. Other kwargs are passed to the make_model method which defines the
        process being modeled and the features used by the estimator.
        :param time_step:
        :param kwargs:
        """
        # TODO time_step is usually a property of the process (daily, hourly, etc) mixin.
        #  Set it in the mixin and drop it from the api?
        self._time_step = time_step
        self._fit_range = None
        self._fit_domain_args = None
        self._model = self.make_model(**kwargs)

    def slow_copy(self):
        # TODO: replace with proper __deepcopy__ method
        with io.BytesIO() as bbuf:
            self.dump(bbuf)
            bbuf.seek(0)
            return self.load(bbuf)

    def domain(self, start, end, *locations):
        """
        Create the closed time range inclusive of end or upto end + tstep.
        The domain is agnostic to timezone. It can be expressed through the location for transformers that need it
        :param start: the starting datetime
        :param end: the ending datetime
        :param locations: a list of locations
        :return: the time and location domain for the model
        """
        if len(pd.unique(locations)) < len(locations):
            raise ValueError("Duplicate locations not allowed!")
        return make_domain(start, end, self.tstep, *locations)

    @property
    def monitor_unfiltered_fit_stats(self):
        # Prettify the data_monitor's fit_stats into a DataFrame. Singleton values show up as constant columns!
        return pd.DataFrame(
            self.data_monitor_unfiltered.fit_stats,
            # add a "temp" index entry for the domain location count,
            # since ForecastDataMonitor sees that feature column,
            # but don't return stats on the location count column
            index=["temp"] + self.get_feature_names(),
        ).iloc[1:, :]

    @property
    def monitor_unfiltered_fit_stats_by_loc(self):
        # Prettify the data_monitor_pre's fit_stats_by_loc into a DataFrame
        return self._prettify_stats_by_loc(
            self.data_monitor_unfiltered.fit_stats_by_loc
        )

    @property
    def monitor_filtered_fit_stats(self):
        # Prettify the data_monitor's fit_stats into a DataFrame. Singleton values show up as constant columns!
        return pd.DataFrame(
            self.data_monitor_filtered.fit_stats,
            # TODO (Michael H): make this nicer
            index=["temp"] + self.get_feature_names(),
        ).iloc[1:, :]

    @property
    def monitor_filtered_fit_stats_by_loc(self):
        # Prettify the data_monitor_pre's fit_stats_by_loc into a DataFrame
        return self._prettify_stats_by_loc(self.data_monitor_filtered.fit_stats_by_loc)

    @property
    def fit_range_stats(self):
        # get metrics on the raw fit_range
        locs = pd.unique(self.fit_domain[LOCATION].reshape(-1))
        n_locs = len(locs)
        domain_end = self.fit_domain[DATE_TIME].max()
        range_nan = np.isnan(self.fit_range)
        masked_domain = np.where(
            range_nan, np.datetime64("NaT"), self.fit_domain[DATE_TIME]
        )
        most_recent_non_null = pd.DataFrame(
            masked_domain.reshape(-n_locs, n_locs, order="F"),
            columns=locs,
        ).max()
        range_by_loc = self.fit_range.reshape(-n_locs, n_locs, order="F")

        stats_by_loc = {
            "most_recent_non_null": most_recent_non_null,
            "freshness": (domain_end - most_recent_non_null).dt.total_seconds(),
            "fraction_missing": (
                np.isnan(range_by_loc).sum(axis=0) / len(range_by_loc)
            ).reshape(-1, order="F"),
            "mean": np.nanmean(range_by_loc, axis=0).reshape(-1, order="F"),
            "min": np.nanmin(range_by_loc, axis=0).reshape(-1, order="F"),
            "q0.25": np.nanquantile(range_by_loc, 0.25, axis=0).reshape(-1, order="F"),
            "q0.75": np.nanquantile(range_by_loc, 0.75, axis=0).reshape(-1, order="F"),
            "max": np.nanmax(range_by_loc, axis=0).reshape(-1, order="F"),
            "n_stats": len(range_by_loc),
            "n_missing": np.isnan(range_by_loc).sum(axis=0),
            "n_present": len(range_by_loc) - np.isnan(range_by_loc).sum(axis=0),
        }
        stats_by_loc = pd.DataFrame(stats_by_loc, index=pd.Index(locs, name=LOCATION))

        stats_global = {
            "most_recent_non_null": stats_by_loc["most_recent_non_null"].max(),
            "freshness": stats_by_loc["freshness"].min(),
            "fraction_missing": range_nan.sum() / len(self.fit_range),
            "mean": np.nanmean(self.fit_range),
            "min": np.nanmin(self.fit_range),
            "q0.25": np.nanquantile(self.fit_range, 0.25),
            "q0.75": np.nanquantile(self.fit_range, 0.75),
            "max": np.nanmax(self.fit_range),
            "n_stats": len(self.fit_range),
            "n_missing": range_nan.sum(),
            "n_present": len(self.fit_range) - range_nan.sum(),
        }
        stats_global = pd.Series(stats_global).rename("all").to_frame().T

        stats = pd.concat(
            [stats_global, stats_by_loc],
            keys=["all", "by_location"],
            names=["grouping", LOCATION],
        )
        stats.iloc[:, 1:] = stats.iloc[:, 1:].astype(float)
        return stats

    @property
    def monitor_unfiltered_predict_stats(self):
        # Prettify the data_monitor's transform_stats into a DataFrame. Singleton values show up as constant columns!
        return pd.DataFrame(
            self.data_monitor_unfiltered.transform_stats,
            index=["temp"] + self.get_feature_names(),
        ).iloc[1:, :]

    @property
    def monitor_unfiltered_predict_stats_by_loc(self):
        # Prettify the data_monitor_pre's transform_stats_by_loc into a DataFrame.
        return self._prettify_stats_by_loc(
            self.data_monitor_unfiltered.transform_stats_by_loc
        )

    @property
    def monitor_filtered_predict_stats(self):
        # Prettify the data_monitor's transform_stats into a DataFrame. Singleton values show up as constant columns!
        return pd.DataFrame(
            self.data_monitor_filtered.transform_stats,
            # TODO (Michael H): make this nicer
            index=["temp"] + self.get_feature_names(),
        ).iloc[1:, :]

    @property
    def monitor_filtered_predict_stats_by_loc(self):
        # Prettify the data_monitor_pre's transform_stats_by_loc into a DataFrame.
        return self._prettify_stats_by_loc(
            self.data_monitor_filtered.transform_stats_by_loc
        )

    @property
    def domain_type(self):
        """
        Define the numpy data type used in the model transforms.
        The dtype should specify a location identifier and a date time.
        It must be dynamic based on the time step units.
        :return: a numpy dtype
        """
        return make_domain_type(self.tstep)

    @property
    def fit_domain(self):
        """
        The fit domain args (start, end, locations) are persisted for archival purposes by the
        domain wrangler on the fit method. Use it to reconstruct the domain in this helper method.
        :return: the domain used in the fit
        """
        if self._fit_domain_args is not None:
            start, end, locations = self._fit_domain_args
            return self.domain(start, end, *locations)
        else:
            return None

    @property
    def fit_range(self):
        """
        the fit range is persisted for archival purposes
        :return: the range used in the fit
        """
        return self._fit_range

    @abstractmethod
    def get_range(self, domain):
        """
        The model process mixin must implement get_range to get the actual observed values to fit the regression or
        classifier.
        :param domain: the locations and time range
        :return: a vector of observed values used to fit the model or score the prediction
        """
        pass

    @domain_wrangler
    def range(self, domain):
        """
        Get the range using the start, end and locations api
        :param start: the starting datetime for training the model
        :param end: the ending datetime for training the model
        :param *locations: the locations for which to gather data
        """
        return self.get_range(domain)

    @domain_wrangler(stash_domain="_fit_domain_args")
    def fit(self, domain, **kwargs):
        """
        Calls the model fit function with the domain.
        The domain wrangler will set the instance variable fit_domain_args to store the fit domain
        Storing the actual numpy array gets expensive for large domains.
        :param start: the starting datetime for training the model
        :param end: the ending datetime for training the model
        :param *locations: the locations for which to gather data
        """
        # TODO should this method be allowed to be called more than once?

        # TODO: is storing the actual values for the range worth it
        self._fit_range = self.get_range(domain)
        # Persist the fit data - the time period and the observed result for later inspection
        # Unfortunately fetch operations are not idempotent so the range and feature data may change over time
        # The feature data is much too large to store on the model object
        self._fit(self.model, domain, self.fit_range, **kwargs)

    @classmethod
    def _fit(cls, model, fit_domain, fit_range, **kwargs):
        """
        Hook for estimator to override how kwargs are passed to the model.
        :param model: a scikit-learn like model that can be fit on X=fit_domain and y=fit_range.
        :param fit_domain: the domain array from which the features will be constructed for model fitting.
        :param fit_range: the array of target values that will be the label set for supervised learning.
        :param kwargs: special arguments to the fit method
        TODO pass kwargs to xgb model per https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py
        """
        model.fit(fit_domain, fit_range, **kwargs)

    @domain_wrangler
    def predict(self, domain):
        """
        Calls the predict method on the trained model with the domain
        :param start: the starting datetime for training the model
        :param end: the ending datetime for training the model
        :param *locations: the locations for which to gather data
        :return: the predicted result
        """
        return self.model.predict(domain)

    # Drop the domain_wrangler's decorator so that we can call other helper methods with the start, stop, *locations api
    def predict_dataframe(
        self, start, stop, *locations, range: bool = False, feature_score: bool = False
    ):
        """
        Call the predict method on the trained model and returns the result as a DataFrame indexed with the
        specified domain
        :param start: the starting datetime for the model
        :param stop: the ending datetime for the model
        :param locations: the locations for which to gather data
        :param range: optional kwarg to include the actual values where available as column name 'true'
        :param feature_score: optional kwarg to calculate normalized weighted feature score associated with each prediction
        :return: Indexed dataframe of the predicted values for the given domain, optionally including the true values
        as well.
        """
        domain = self.domain(start, stop, *locations)
        df_data = {"predicted": self.model.predict(domain).reshape(-1)}
        if range:
            df_data["true"] = self.get_range(domain).reshape(-1)
        if feature_score:
            raw_features = self.features_dataframe(
                start, stop, *locations, with_filters=False
            )
            try:
                # Try to use Shap to weight missing features
                shap = shap_viz.ShapViz(self, start, stop, *locations)
                shap_df = shap.values_dataframe()
                df_data["feature_score"] = (
                    (((~raw_features.isna()).astype(int) * shap_df.abs()))
                    .divide(shap_df.abs().sum(axis=1), axis=0)
                    .sum(axis=1)
                )
            except TypeError:
                logger.info("Shap TypeError - falling back to simple score")
                # For estimators that don't shap (PV physics models), any missing feature -> zero score
                df_data["feature_score"] = (~(raw_features.isna().any(axis=1))).astype(
                    float
                )

        return pd.DataFrame(df_data, index=multiindex_from_domain(domain))

    @domain_wrangler
    def score(self, domain):
        """
        Calls the score method on the trained model with the domain.
        The score is the objective value of the estimator.
        :param start: the starting datetime for training the model
        :param end: the ending datetime for training the model
        :param *locations: the locations for which to gather data
        :return: the objective value score
        """
        return self.model.score(domain, self.get_range(domain))

    @domain_wrangler
    def metrics(self, domain):
        """
        Evaluate the model range and predicted values for the given domain.
        Scores the results using multiple methods defined for regressors and classifiers.
        :param start: the starting datetime for training the model
        :param end: the ending datetime for training the model
        :param *locations: the locations for which to gather data
        :return: a map of scores
        """
        ytrue = self.get_range(domain)
        ypred = self.model.predict(domain)

        bool_nan = np.isnan(ytrue)

        if np.any(bool_nan):
            logger.warning("Removing %s NaN values in YTrue!", bool_nan.sum())
            ytrue = ytrue[~bool_nan.reshape(-1)]
            ypred = ypred[~bool_nan.reshape(-1)]

        bool_nan = np.isnan(ypred)
        if np.any(bool_nan):
            logger.warning("Removing %s NaN values in YPred!", bool_nan.sum())
            ytrue = ytrue[~bool_nan.reshape(-1)]
            ypred = ypred[~bool_nan.reshape(-1)]

        def safe_metrics(method, y_true, y_pred):
            try:
                return method(y_true, y_pred)
            except ValueError:
                logger.debug(
                    "%s failed for %s on %s to %s",
                    method.__name__,
                    self.__class__.__name__,
                    domain[0],
                    domain[-1],
                )
                return np.nan

        return {f.__name__: safe_metrics(f, ytrue, ypred) for f in self.METRICS}

    def dump(self, file_object):
        """
        Serialize the model using Dill Pickle and write it to the given file_object
        :param file_object: a python IO object
        """
        pickle.dump(self, file_object, protocol=5)

    @classmethod
    def load(cls, file_object):
        """
        Load a pickled object. No security applied here!

        :param file_object: a python IO object
        :return: the deserialized object
        TODO raise if it isn't a RegularTimeSeriesModel ?
        """
        return pickle.load(file_object)

    @classmethod
    def load_from_fs(
        cls,
        model_path: PurePosixPath,
        fs: fsspec.AbstractFileSystem = None,
        project_name: str = None,
    ):
        """
        Load a serialized model instance from a filesystem (e.g. from GCS).
        :param model_path: the path to the model instance
        :param project_name: GCS project from which to load the model instance.
            Default (None) attempts to read from environment.
        :param fs: the filesystem to use; defaults is GCS backend.
        :return: the loaded instance
        """
        if fs is None:
            project_name = project_name or google.auth.default()[1]
            fs = gcsfs.GCSFileSystem(project=project_name)
        with fs.open(model_path, "rb") as model_file:
            return cls.load(model_file)

    @property
    def model(self):
        """
        An sklearn like model object with fit, predict and score methods
        :return: the model
        """
        return self._model

    @debug(logger)
    def make_model(self, **kwargs):
        return sklearn.pipeline.Pipeline(
            [
                ("feature_builder", self.make_featurizer(**kwargs)),
                (
                    "data_monitor_unfiltered",
                    ForecastDataMonitor(use_locs=True, **kwargs),
                ),
                ("feature_filter", self.filter_features(**kwargs)),
                (
                    "data_monitor_filtered",
                    ForecastDataMonitor(use_locs=False, **kwargs),
                ),
                (
                    "drop_location_count",
                    sklearn.preprocessing.FunctionTransformer(func=drop_first_column),
                ),
                ("estimator", self.make_estimator(**kwargs)),
            ]
        )

    @abstractmethod
    def make_preprocessor(self, **kwargs):
        """
        Helper method for make_featurizer, and the target for overwriting with a specific Process implementation.
        Produces a column transformer that converts the domain to the model process features.
        :return: a ColumnTransformer
        """
        return sklearn.compose.ColumnTransformer([])

    def make_featurizer(self, **kwargs):
        """
        Produces a ColumnTransformer that converts the domain to the model features, along with supporting value(s)
        :return: a ColumnTransformer
        """
        pre_proc = self.make_preprocessor(**kwargs)

        if isinstance(pre_proc, sklearn.compose.ColumnTransformer):
            feature_columns = pre_proc.transformers
        elif isinstance(pre_proc, sklearn.pipeline.Pipeline):
            feature_columns = [("feature_columns", pre_proc, [0])]
        else:
            raise ValueError(
                f"Unexpected type {type(pre_proc)} returned by make_preprocessor"
            )

        return sklearn.compose.ColumnTransformer(
            [
                (
                    "count_locs",
                    sklearn.preprocessing.FunctionTransformer(
                        func=count_domain_locs_into_column
                    ),
                    [0],
                )
            ]
            + feature_columns,
        )

    def get_feature_names(self):
        """
        Get the feature names created by the preprocess step
        :return: a list of feature names
        """
        # skip first entry because that is the locations counter that gets dropped
        return self.model.named_steps["feature_builder"].get_feature_names()[1:]

    def filter_features(self, **kwargs):
        """
        Apply a specified filter to features. Provided by a mixin, else all features are retained.
        Takes the form of a Pipeline so that we can chain filters by appending to 'steps'
        :return: a scikit-learn TransformerMixin
        """
        return sklearn.pipeline.Pipeline(steps=[("passthrough", "passthrough")])

    @abstractmethod
    def make_estimator(self, **kwargs):
        """
        Produce an estimator, typically a regressor, classifier or transformed target regressor.
        Provided by an estimator mixin.
        :return: an estimator
        """
        pass

    @property
    def tstep(self):
        """
        tstep must be a np.timedelta64
        :return: the time series step size
        """
        return self._time_step

    @domain_wrangler
    def features(self, domain):
        """
        Build the input features array for the specified domain
        :param start: the starting datetime for training the model
        :param end: the ending datetime for training the model
        :param *locations: the locations for which to gather data
        :return: the feature array
        """
        return self.get_features(domain)

    def get_features(self, domain, with_filters=True):
        """
        Call transform directly on the preprocessor pipeline to get the input features
        :param domain:
        :return:
        """
        features = self.feature_builder.transform(domain)
        if with_filters:
            features = self.filters.transform(features)
        return drop_first_column(features)

    @domain_wrangler
    def features_dataframe(self, domain, with_filters=True):
        """
        Build the input features as an index DataFrame specified domain
        :param start: the starting datetime for the model
        :param end: the ending datetime for the model
        :param *locations: the locations for which to gather data
        :return: the feature array
        """
        return pd.DataFrame(
            self.get_features(domain, with_filters=with_filters),
            columns=self.get_feature_names(),
            index=multiindex_from_domain(domain),
        )

    @property
    def feature_builder(self):
        return self.model["feature_builder"]

    @property
    def filters(self):
        return self.model["feature_filter"]

    @property
    def data_monitor_unfiltered(self):
        return self.model["data_monitor_unfiltered"]

    @property
    def data_monitor_filtered(self):
        return self.model["data_monitor_filtered"]

    @property
    def estimator(self):
        """
        Return the estimator instance. For some wrapped estimators (GridSearchCV, TransformedTargetRegressor) this
        property may be None until the model has been fit.
        :return: the estimator object
        """
        return self.model["estimator"]

    def _prettify_stats_by_loc(
        self, stats_by_loc: dict[str : np.ndarray]
    ) -> pd.DataFrame:
        # Prettify the data_monitor's stats_by_loc dicts into DataFrames.
        locs = pd.unique(self.fit_domain[LOCATION].reshape(-1))
        # add a "temp" column name for the domain location count,
        # since ForecastDataMonitor sees that feature column,
        # but don't return stats on the location count column
        features = ["temp"] + self.get_feature_names()
        return pd.concat(
            {
                k: pd.DataFrame(
                    v,
                    columns=pd.Index(features, name="feature"),
                    index=pd.Index(locs, name=LOCATION),
                ).iloc[:, 1:]
                for k, v in stats_by_loc.items()
            },
            names=["metric"],
        ).unstack(level=0)


class Mixin:
    pass


class XgbEvalSetFit(Mixin):
    @staticmethod
    def _fit(model, fit_domain, fit_range, **kwargs):
        """
        This overrides the _fit method in RegularTimeSeriesModel and skips the pipeline!
        Used after XgbRegressor e.g.
            class NpXgbEvalModel(BalancingAreaHourly, XgbEvalSetFit, XgbRegressor, RegularTimeSeriesModel):
                pass
        TODO Not sure I like this?
          kwargs must not be prefixed with the pipeline path... could emulate that behavior if it helps?
          Seems like this is a diagnostic tool, not something we need in production
        TODO: the current train_test_split is not compatible with time series data. See example implementation in
          back_test.BackTest._split_domain -- maybe extend this to extract specified indices of features?
        """
        # explicitly call fit_transform on the preproessor to get the feature data
        features = model["preprocessor"].fit_transform(fit_domain)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            features, fit_range, test_size=0.3, random_state=0
        )
        # TODO allow passing test_train_split params

        model["estimator"].fit(
            x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], **kwargs
        )


class AggModel:
    """
    Forecast a process as the sum of constituent processes, by specifying the mapping from parent to child nodes.
    TODO(Michael H): generalize the terminology (currently tailored to AMI/transformer relationship)
    """

    def __init__(
        self,
        model: RegularTimeSeriesModel,
        mapping: pd.DataFrame,
    ):
        self._model = model
        self._mapping = mapping
        self._xfr_to_meter_dict = (
            self.mapping.groupby("xfr_id")["meter_number"].unique().to_dict()
        )
        self._meter_to_xfr_dict = dict(
            zip(self.mapping["meter_number"], self.mapping.xfr_id)
        )

    @property
    def domain(self):
        return self.rtsm.domain

    @property
    def estimator(self):
        return self.rtsm.estimator

    @property
    def get_features(self):
        return self.rtsm.get_features

    @property
    def fetcher(self):
        return self.rtsm.labels_fetcher

    @property
    def mapping(self):
        return self._mapping

    @property
    def meter_to_xfr_dict(self):
        return self._meter_to_xfr_dict

    @property
    def model(self):
        return self.rtsm.model

    @property
    def rtsm(self):
        return self._model

    @property
    def tstep(self):
        return self.rtsm.tstep

    @property
    def xfr_to_meter_dict(self):
        return self._xfr_to_meter_dict

    def dump(self, file_object):
        """
        Serialize the model using Dill Pickle and write it to the given file_object
        :param file_object: a python IO object
        """
        pickle.dump(self, file_object, protocol=5)

    def fit(self, start, end, *locations):
        mapped_locations = self._map_xfr_to_meters(locations)
        logger.info(f"AggModel fitting RTSM to {len(mapped_locations)} meters")
        logger.debug(mapped_locations)
        self.rtsm.fit(start, end, *mapped_locations)

    @classmethod
    def load(cls, file_object):
        """
        Load a pickled object. No security applied here!
        :param file_object: a python IO object
        :return: the deserialized object
        """
        return pickle.load(file_object)

    def metrics(self, start, end, *locations):
        pred_df = self.predict_dataframe(start, end, *locations, range=True)
        ytrue = pred_df["true"].to_numpy()
        ypred = pred_df["predicted"].to_numpy()

        bool_nan = np.isnan(ytrue)

        if np.any(bool_nan):
            logger.warning("Removing %s NaN values in YTrue!", bool_nan.sum())
            ytrue = ytrue[~bool_nan.reshape(-1)]
            ypred = ypred[~bool_nan.reshape(-1)]

        bool_nan = np.isnan(ypred)
        if np.any(bool_nan):
            logger.warning("Removing %s NaN values in YPred!", bool_nan.sum())
            ytrue = ytrue[~bool_nan.reshape(-1)]
            ypred = ypred[~bool_nan.reshape(-1)]

        def safe_metrics(method, y_true, y_pred):
            try:
                return method(y_true, y_pred)
            except ValueError:
                logger.debug(
                    "%s failed for %s on %s to %s",
                    method.__name__,
                    self.__class__.__name__,
                    start,
                    end,
                )
                return np.nan

        return {f.__name__: safe_metrics(f, ytrue, ypred) for f in self.rtsm.METRICS}

    def predict(self, start, end, *locations):
        mapped_locations = self._map_xfr_to_meters(locations)
        res = self.rtsm.predict(start, end, *mapped_locations)
        return self._agg_to_xfr(res)

    def features_dataframe(self, start, end, *locations):
        mapped_locations = self._map_xfr_to_meters(locations)
        # TODO(Michael H): prepend xfrmr ID multiindex level
        return self.rtsm.features_dataframe(start, end, *mapped_locations)

    def predict_dataframe(self, start, end, *locations, range=False):
        mapped_locations = self._map_xfr_to_meters(locations)
        logger.debug(mapped_locations)
        res = self.rtsm.predict_dataframe(start, end, *mapped_locations, range=range)
        return self._agg_to_xfr(res)

    def _agg_to_xfr(self, df: pd.DataFrame):
        df = df.reset_index()
        df["xfr"] = [self.meter_to_xfr_dict[loc] for loc in df[LOCATION]]
        df = df.groupby(["xfr", DATE_TIME]).sum(min_count=1)
        return df.rename_axis([LOCATION, DATE_TIME], axis=0)

    def _map_xfr_to_meters(self, locations: tuple):
        mapped_locations = [self.xfr_to_meter_dict[loc] for loc in locations]
        return [item for sublist in mapped_locations for item in sublist]


class MonthSelector(Mixin):
    """
    Builds a domain only including specified months.
    WIP
    TODO generalize to slice out data for different units and ranges

    The current implementation violates assumptions about constant timesteps in the domain
    TODO rewrite the mixin to slice out selected months at the end of the preprocessing step rather than from the domain.
    """

    def __init__(self, *args, **kwargs):
        self.include_only_months = kwargs.pop("include_only_months")
        super().__init__(*args, **kwargs)

    def domain(self, start, end, *locations):
        dom = super().domain(start, end, *locations)

        raw_time = dom[DATE_TIME].astype(datetime.datetime)
        month_of_year = TimeUnitEnum.MONTH.as_unit(raw_time)

        return dom[np.isin(month_of_year, self.include_only_months)].reshape(-1, 1)


REGISTERED_CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}
