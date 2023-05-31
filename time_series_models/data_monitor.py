import numpy as np
import scipy.sparse
import sklearn.preprocessing


class ForecastDataMonitor(sklearn.preprocessing.FunctionTransformer):
    """
    Class to capture statistics about the data flowing through a given point in a Scikit-Learn pipeline.
    Wraps a no-op FunctionTransformer, but preserves information on self for subsequent retrieval.

    'transform' stats are recorded during both 'transform' calls and 'fit_transform' calls, which means they will be
    written even during a regular 'fit' call when part of a Scikit-Learn pipeline (i.e., the expected usage). Therefore,
    an API is provided to query the relevant stats: after training, use 'fit_stats', and after predicting, use
    'transform_stats'.

    Note, ForecastDataMonitor will capture statistics on the data passing through it at that *exact* part of the
    pipeline. E.g., if row filtering is performed upstream, that information will not be available inside the pipeline!
    """

    def __init__(
        self,
        quantiles: list[float] = None,
        use_locs: bool = False,
        no_transform_stats: bool = False,
        **kwargs,
    ):
        """
        Configure the DataMonitor, e.g. with custom quantiles
        :param quantiles: a list of one or more quantiles of the training data that we wish to preserve. Default is
            [0.05, 0.25, 0.75, 0.95]; the 0.25 and 0.75 quantiles will always be calculated to aid in outlier detection.
        :param use_locs: whether to calculate stats by location (must have access to domain_n_locations column)
        :param no_transform_stats: flag to disable saving transform_stats. Used to check the adherence of
            ForecastDataMonitor to the sklearn API except for the 'transform' method, which otherwise explicitly
            violates the API by modifying self.
        :param kwargs:
        """
        self.quantiles = quantiles
        self.use_locs = use_locs
        self.no_transform_stats = no_transform_stats
        super().__init__()

    @property
    def fit_stats(self):
        return self.fit_stats_

    @property
    def fit_stats_by_loc(self):
        return self.fit_stats_by_loc_

    @property
    def transform_stats(self):
        return self.transform_stats_

    @property
    def transform_stats_by_loc(self):
        return self.transform_stats_by_loc_

    def fit(self, X, y=None):
        """
        No-op fit that saves stats onto self as a side effect
        :param X:
        :param y:
        :return: self
        """
        if scipy.sparse.issparse(X):
            raise ValueError(
                "ForecastDataMonitor does not support sparse input at this time"
            )
        # create empty dicts, overwriting any preexisting ones
        # (per sklearn convention, attributes_ can be created during fit)
        self.fit_stats_ = {}
        self.fit_stats_by_loc_ = {}
        self.transform_stats_ = {}
        self.transform_stats_by_loc_ = {}
        self.fit_stats_["feature_n_obs"] = len(X)
        self.fit_stats_["feature_n_missing"] = np.isnan(X).sum(axis=0)
        self.fit_stats_["feature_missing"] = np.isnan(X).sum(axis=0) / len(X)
        self.fit_stats_["feature_trailing_n_missing"] = self.count_trailing_nan(X)
        self.fit_stats_["feature_mean"] = np.nanmean(X, axis=0)
        self.fit_stats_["feature_min"] = np.nanmin(X, axis=0)
        self.fit_stats_["feature_max"] = np.nanmax(X, axis=0)
        # feature value outlier detection:
        quantiles = self.get_quantiles()
        q_values = np.nanquantile(X, quantiles, axis=0)
        q_strings = np.array(quantiles).astype(str)
        for q_str, q_val in zip(q_strings, q_values):
            self.fit_stats_[f"feature_q{q_str}"] = q_val

        if self.use_locs:
            n_locs = int(X[0, 0])
            n_cols = X.shape[1]
            features_by_loc = X.reshape(-n_locs * n_cols, n_locs * n_cols, order="F")
            self.fit_stats_by_loc["feature_n_obs"] = len(features_by_loc)
            self.fit_stats_by_loc["feature_n_missing"] = (
                np.isnan(features_by_loc)
                .sum(axis=0)
                .reshape(n_locs, -n_locs, order="F")
            )
            self.fit_stats_by_loc_["feature_missing"] = (
                np.isnan(features_by_loc).sum(axis=0) / len(features_by_loc)
            ).reshape(n_locs, -n_locs, order="F")
            self.fit_stats_by_loc_[
                "feature_trailing_n_missing"
            ] = self.count_trailing_nan(features_by_loc).reshape(
                n_locs, -n_locs, order="F"
            )
            self.fit_stats_by_loc_["feature_mean"] = np.nanmean(
                features_by_loc, axis=0
            ).reshape(n_locs, -n_locs, order="F")
            self.fit_stats_by_loc_["feature_min"] = np.nanmin(
                features_by_loc, axis=0
            ).reshape(n_locs, -n_locs, order="F")
            self.fit_stats_by_loc_["feature_max"] = np.nanmax(
                features_by_loc, axis=0
            ).reshape(n_locs, -n_locs, order="F")
        return super().fit(X, y=y)

    def transform(self, X):
        """
        No-op transform that saves stats onto self as a side effect.
        Note, modifying self via 'transform' violates the sklearn API.
        Since we are not relying on the DataMonitor to do any actual
        transformations, we believe this is okay, but we need to hide
        this behavior when testing other aspects of compatibility with
        the API.
        :param X:
        :return: X
        """
        if self.no_transform_stats:
            # bypass for testing sklearn compatibility...
            # other than the rest of this transform method!
            return super().transform(X)

        # overwrite the dict each time
        self.transform_stats_ = dict(
            feature_n_obs=len(X),
            feature_n_missing=np.isnan(X).sum(axis=0),
            feature_missing=np.isnan(X).sum(axis=0) / len(X),
            feature_trailing_n_missing=self.count_trailing_nan(X),
            # feature value outlier detection:
            feature_min=np.nanmin(X, axis=0),
            feature_max=np.nanmax(X, axis=0),
        )
        if self.use_locs:
            n_locs = int(X[0, 0])
            n_cols = X.shape[1]
            features_by_loc = X.reshape(-n_locs * n_cols, n_locs * n_cols, order="F")
            # overwrite the dict each time
            self.transform_stats_by_loc_ = dict(
                feature_n_obs=len(features_by_loc),
                feature_missing=(
                    np.isnan(features_by_loc).sum(axis=0) / len(features_by_loc)
                ).reshape(n_locs, -n_locs, order="F"),
                feature_n_missing=(np.isnan(features_by_loc).sum(axis=0)).reshape(
                    n_locs, -n_locs, order="F"
                ),
                feature_trailing_n_missing=self.count_trailing_nan(
                    features_by_loc
                ).reshape(n_locs, -n_locs, order="F"),
                # feature value outlier detection:
                feature_min=np.nanmin(features_by_loc, axis=0).reshape(
                    n_locs, -n_locs, order="F"
                ),
                feature_max=np.nanmax(features_by_loc, axis=0).reshape(
                    n_locs, -n_locs, order="F"
                ),
            )
        return super().transform(X)

    def get_quantiles(self) -> list:
        """Return the list of quantiles that this instance will calculate"""
        if self.quantiles is None:
            return self._default_quantiles()
        q_set = set(self.quantiles)
        # ensure we get the 0.25 and 0.75 quantiles
        q_set |= self._required_quantiles()
        # return a sorted list
        return sorted(list(q_set))

    @staticmethod
    def count_trailing_nan(arr: np.ndarray) -> np.ndarray:
        n_obs = len(arr)
        isnan = np.isnan(arr)
        # count trailing nan: find first instance of non-null per column after flipping array vertically
        n_trailing = np.argmax(np.flip(~isnan, axis=0), axis=0)
        # if all values are missing, the column will slip by the previous check, so catch this case next
        all_nan = isnan.sum(axis=0) == n_obs
        n_trailing[all_nan] = n_obs
        return n_trailing

    @classmethod
    def _default_quantiles(cls):
        return [0.05, 0.25, 0.75, 0.95]

    @classmethod
    def _required_quantiles(cls):
        return {0.25, 0.75}
