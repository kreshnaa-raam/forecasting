import datetime
import logging
import time

import sklearn
import sklearn.model_selection

import cloudpickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Literal

from time_series_models.constants import DATE_TIME, LOCATION
from time_series_models.decorators import domain_wrangler
from time_series_models.transformers import make_domain

logger = logging.getLogger(__name__)


class BackTest:
    """
    # TODO (Michael H): this API is still experimental and subject to change
    Estimate model performance on out-of-sample data by a time series (rolling) cross-validation.

    The cross-validation should be configured to match the intended usage of the model, e.g. a model that will be
    retrained weekly should probably be evaluated with a test window of 7 days. Longer test size will incorporate more
    data for testing but could also allow greater divergence between model and the true data generating process.

    :param model_class: a class of model to backtest
    :param model_config: a config for the model constructor
    :param cv_splitter: a cross-validation splitter object with a `split` method that yields train- and test- indices.
    """

    def __init__(self, model_class, model_config: dict, cv_splitter):
        self.cv_splitter = cv_splitter
        self._metadata = None
        self._model_class = model_class
        self._model_config = model_config
        self._predictions = None
        self._training_metrics = None
        self._tstep = self.construct_model().tstep

    @property
    def metadata(self):
        return self._metadata

    @property
    def model_class(self):
        return self._model_class

    @property
    def model_config(self):
        return self._model_config

    @property
    def predictions(self):
        return self._predictions

    @property
    def predictions_df(self):
        return self._concat_predictions(self.predictions)

    @property
    def training_metrics(self):
        return self._training_metrics

    @property
    def tstep(self):
        return self._tstep

    @classmethod
    def from_default_splitter(
        cls,
        model_class,
        model_config: dict,
        n_splits=5,
        max_train_size=None,
        test_size=None,
    ):
        """
        Instantiate BackTest with a scikit-learn TimeSeriesSplit splitter.

        Configure max_train_size and test_size to match the expected production usage of the model, e.g.,
        test_size should match the interval that a production model would serve before retraining.

        Test windows will be tiled back from the end of the time range. E.g., if test_size is one week,
        and max_train_size is one year, a BackTest run on a two-year interval will have its last fold be tested on the
        final week of the interval, and trained on the preceding one year (from the penultimate week of the first year
        to the penultimate week of the second year). The penultimate fold of the BackTest will be tested on the
        penultimate week of the overall interval, and trained on the preceding one year... and so on. The test folds are
        contiguous and non-overlapping, so with n_splits=5 (the default), the final five weeks of the interval would
        constitute the five test folds. In this example, the BackTest would only use the last five weeks of the first
        year of the available 2-year interval. In contrast, if the BackTest were run on a one-year interval, while the
        test folds and the end points of the training folds would be unchanged, the start points of each training fold
        would not be allowed to extend prior to the beginning of the one-year interval, so we would have differently-
        sized training intervals for each fold of the BackTest.

        :param model_class: a class of model to backtest
        :param model_config: a config for the model constructor
        :param n_splits: the number of train/test windows to evaluate for the backtest. Synonymous with "fold".
        :param max_train_size: maximum number of observations to include in each train window.
        :param test_size: number of observations to include in each test window.
        :return: an instance of BackTest
        """
        return cls(
            model_class,
            model_config,
            sklearn.model_selection.TimeSeriesSplit(
                n_splits=n_splits, test_size=test_size, max_train_size=max_train_size
            ),
        )

    def construct_model(self):
        return self.model_class(**self.model_config)

    def domain(self, start, end, *locations):
        """this method exists to enable domain wrangling for `fit` call"""
        return make_domain(start, end, self.tstep, *locations)

    def get_metrics(
        self,
        how: Literal["fold", "location", "location-fold"] = "fold",
        which: Literal["train", "test"] = "test",
    ) -> pd.DataFrame:
        """
        Interface to calculate (or retrieve) error metrics for a specified aggregation ("how") of the specified target
        ("which").
        :param how: specify the aggregation for calculating metrics
        :param which: specify whether metrics are desired from the test set (default) or the train set
        :return: metrics
        """
        if which == "test":
            if how == "fold":
                return self._metrics_by_fold(self.predictions_df)
            if how == "location":
                return self._metrics_by_location(self.predictions_df)
            return self._metrics_by_location_fold(self.predictions_df)
        if which == "train":
            if how == "fold":
                return self.training_metrics["fold"]
            if how == "location":
                return self.training_metrics["location"]
            return self.training_metrics["location_fold"]
        raise ValueError(
            f"Unrecognized value {which}! Select either 'test' or 'train'."
        )

    def plot_bt_predictions(
        self,
        ids: list | set = None,
        n_max=None,
        n_rows=None,
        n_cols=1,
        figsize=(15, 8),
    ):
        """
        Helper method to make a plot of the backtest predictions. Returns fig, axs results of plt.subplots() call to
        allow additional formatting.
        E.g., add a title with plt.suptitle("text"), save the figure with plt.savefig(path), display with plt.show(),
        and close with plt.close().
        :param ids: a collection of domain locations to include in the plot. If None (default), include all.
        :param n_max: optional limit to how many locations to plot; set either 'n_max' or 'ids', not both.
            Locations are drawn sequentially from self.predictions_df; for finer grained control, use 'ids' instead.
        :param n_rows: number of rows to arrange subplots. If None (default), calculates minimum number of rows needed
            to display all subplots given n_cols.
        :param n_cols: number of columns to arrange subplots.
        :param figsize: size of figure to plot, specified as tuple(numeric, numeric).
        :return: fig, axs results of the call to plt.subplots()
        """
        pred_df = self.predictions_df
        if ids is not None:
            pred_df = pred_df.reindex(pd.Index(ids), level=LOCATION)
        if n_max is not None:
            if ids is not None:
                raise RuntimeError("Set either 'ids' or 'n_max', not both!")
            pred_df = pred_df.loc[
                pred_df.index.get_level_values(LOCATION).isin(
                    pred_df.index.get_level_values(LOCATION).unique()[:n_max]
                )
            ]
        grouped = pred_df.reset_index(level=[0, 1], drop=False).groupby(LOCATION)
        # groupby().groups.keys will not preserve index order, but pd.unique will preserve order
        locs_sorted = pd.unique(pred_df.index.get_level_values(LOCATION))
        n_rows = n_rows or grouped.ngroups // n_cols + (grouped.ngroups % n_cols > 0)

        fig, axs = plt.subplots(
            figsize=figsize,
            nrows=n_rows,
            ncols=n_cols,
            gridspec_kw=dict(hspace=0.4),
        )

        if (n_rows == 1) & (n_cols == 1):
            targets = zip(locs_sorted, [axs])
        else:
            targets = zip(locs_sorted, axs.flatten())
        train_end = [
            fold["train"][1].astype(datetime.datetime)
            for fold in self.metadata.values()
        ]
        fold_edges = (
            train_end.copy()
            + [
                [fold["test"][1].astype(datetime.datetime)]
                for fold in self.metadata.values()
            ][-1]
        )

        try:
            y_label = self.construct_model().labels_fetcher.variables[0]
        except AttributeError:
            logger.warning(
                "Could not deduce variable name from model.labels_fetcher, does this attribute exist?"
            )
            y_label = None
        for i, (key, ax) in enumerate(targets):
            ax.plot(grouped.get_group(key)["true"], label="true")
            ax.plot(grouped.get_group(key)["predicted"], label="predicted")
            ax.set_title(f"id: {key}")
            ax.set_ylabel(y_label)

            # add vertical lines at the end of each training window:
            ax.vlines(
                train_end,
                0,
                grouped.get_group(key).drop(columns=["fold", LOCATION]).max().max(),
                colors="black",
                linestyles=":",
                alpha=0.3,
                label="retrain",
            )
            ax.set_xticks(ticks=fold_edges, labels=fold_edges)
        ax.legend(loc="center right")  # set legend only in final panel
        plt.gcf().autofmt_xdate()
        return fig, axs

    @domain_wrangler
    def run(self, domain, fit_locations: list[str] = None, train_scores=False):
        """
        The primary function of the backtest. For each fold of the back test, fits the model to the training portion and
        scores predictions against the test portion. Saves predictions, global (all-locations) metrics, and metadata for
        each fold of the back test.
        :param start: start of backtest interval, e.g. '2014-01-01'
        :param end: end of backtest interval, e.g. '2021-09-01T12:00:00'
        :param locations: one or more locations, e.g. 'PSCo'
        :param fit_locations: (optional) if the model is to be trained on a set of locations that differs from the
            domain for prediction, pass these locations in a list using this param.
        :param train_scores: if True, calculate training scores. This can result in a very large set of predictions,
            depending on the size of the training set. Useful for assembling a learning curve, but not strictly
            necessary to evaluate out-of-sample prediction performance. Unlike the test metrics, training metrics are
            calculated upfront so that we don't persist a massive training set.
        :return: fitted BackTest instance with derived `cv_scores` attribute
        """
        # TODO(Michael H): establish minimum training window size
        dt_folds, domain_locations = self._split_domain(domain)
        predictions = {}
        training_predictions = {}
        metadata = {}
        for i in range(len(dt_folds)):
            logger.info("BackTest fitting fold %i of %i", i + 1, len(dt_folds))
            train_start = dt_folds[i]["train_start"]
            train_end = dt_folds[i]["train_end"]
            test_start = dt_folds[i]["test_start"]
            test_end = dt_folds[i]["test_end"]
            metadata[i] = {
                "train": (train_start, train_end),
                "test": (test_start, test_end),
                "location": domain_locations,
            }
            if fit_locations is not None:
                # coerce to list, in case we got an object of unknown truthiness (e.g. pd.Series or np.array)
                fit_locations = list(fit_locations)
                metadata[i]["train_locations"] = fit_locations
            model = self.construct_model()
            start = time.time()
            model.fit(train_start, train_end, *(fit_locations or domain_locations))
            metadata[i]["fit_time"] = time.time() - start
            logger.debug("BackTest predicting for fold %i", i + 1)
            start = time.time()
            predictions[i] = model.predict_dataframe(
                test_start,
                test_end,
                *domain_locations,
                range=True,
            )
            metadata[i]["predict_time"] = time.time() - start
            if train_scores:
                logger.debug("BackTest predicting training labels for fold %i", i + 1)
                training_predictions[i] = model.predict_dataframe(
                    train_start,
                    train_end,
                    *(fit_locations or domain_locations),
                    range=True,
                )

        self._metadata = metadata
        self._predictions = predictions
        if train_scores:
            # concat folds into a single Frame
            training_predictions = self._concat_predictions(training_predictions)
            self._set_training_metrics(training_predictions)
        return self

    def dump(self, file_object):
        """
        Serialize the model using CloudPickle and write it to the given file_object
        :param file_object: a python IO object
        """
        pickle.dump(self, file_object, protocol=5)

    @classmethod
    def load(cls, file_object):
        """
        Load a pickled object. No security applied here!

        :param file_object: a python IO object
        :return: the deserialized object, if it is an instance of BackTest
        """
        return pickle.load(file_object)

    @staticmethod
    def _concat_predictions(predictions: dict[any, pd.DataFrame]):
        return pd.concat(
            list(predictions.values()),
            keys=list(predictions.keys()),
            names=["fold", LOCATION, DATE_TIME],
        )[["true", "predicted"]]

    def _metrics_by_fold(self, df):
        return df.groupby("fold").apply(self._metrics_set)

    def _metrics_by_location(self, df):
        return df.groupby(LOCATION).apply(self._metrics_set)

    def _metrics_by_location_fold(self, df):
        return df.groupby([LOCATION, "fold"]).apply(self._metrics_set)

    @staticmethod
    def _metrics_set(x: pd.DataFrame):
        assert ("true" in x.columns) & ("predicted" in x.columns)

        if x["true"].isna().all():
            logger.warning(
                "No y_true values for fold %s, location %s, date_time [%s - %s]",
                x.index.get_level_values("fold").unique().tolist(),
                x.index.get_level_values("location").unique().tolist(),
                x.index.get_level_values("date_time").min(),
                x.index.get_level_values("date_time").max(),
            )
            # TODO(Michael H): link the possible returns so their indices are always identical
            return pd.Series(
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

        if np.isnan(x["true"]).any():
            logger.warning(
                "Excluding %i observations where y_true is NaN!",
                np.isnan(x["true"]).sum(),
            )
            x = x.loc[x["true"].notna(), :]
            logger.debug("Scoring %i remaining observations", len(x))

        def safe_mape(y_true, y_pred):
            if any(y_true == 0):
                return np.nan
            return sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)

        return pd.Series(
            {
                "mean_absolute_percentage_error": safe_mape(x["true"], x["predicted"]),
                "mean_absolute_error": sklearn.metrics.mean_absolute_error(
                    x["true"], x["predicted"]
                ),
                "median_absolute_error": sklearn.metrics.median_absolute_error(
                    x["true"], x["predicted"]
                ),
                "mean_squared_error": sklearn.metrics.mean_squared_error(
                    x["true"], x["predicted"]
                ),
                "root_mean_squared_error": sklearn.metrics.mean_squared_error(
                    x["true"], x["predicted"], squared=False
                ),
                "r2_score": sklearn.metrics.r2_score(x["true"], x["predicted"]),
                "explained_variance_score": sklearn.metrics.explained_variance_score(
                    x["true"], x["predicted"]
                ),
            }
        )

    def _set_training_metrics(self, training_predictions: pd.DataFrame):
        logger.debug("Evaluating and saving training metrics")
        self._training_metrics = dict(
            fold=self._metrics_by_fold(training_predictions),
            location=self._metrics_by_location(training_predictions),
            location_fold=self._metrics_by_location_fold(training_predictions),
        )

    def _split_domain(self, domain):
        domain_locations = pd.unique(domain[LOCATION].reshape(-1))
        start = domain[DATE_TIME].min()
        stop = domain[DATE_TIME].max()
        uni_domain = make_domain(start, stop, self.tstep, domain_locations[0])
        dt_folds = {}
        for i, (train_idx, test_idx) in enumerate(self.cv_splitter.split(uni_domain)):
            dt_folds[i] = dict(
                train_start=uni_domain[DATE_TIME][train_idx].min(),
                train_end=uni_domain[DATE_TIME][train_idx].max(),
                test_start=uni_domain[DATE_TIME][test_idx].min(),
                test_end=uni_domain[DATE_TIME][test_idx].max(),
            )
        return dt_folds, domain_locations
