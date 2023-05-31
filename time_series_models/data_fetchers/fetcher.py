import contextlib
import io
import logging

from abc import ABC, abstractmethod
from copy import deepcopy, copy
from datetime import timedelta, datetime
import typing

import pandas as pd
import numpy as np
import requests

from sklearn.base import TransformerMixin, BaseEstimator
from google.cloud import storage
from prometheus_client import Summary

from time_series_models.constants import (
    DATE_TIME,
    LOCATION,
    MAPPED_LOCATION,
)
from time_series_models.decorators import timer
from time_series_models.exceptions import (
    EmptyFetchException,
    FetcherConfigurationError,
)
from time_series_models.transformers import (
    interpolate_interior,
    is_iterable,
    multiindex_from_domain,
)

logger = logging.getLogger(__name__)

"""
GUIDANCE:
TL;DR:
* For fetcher classes whose source loader can only handle one data source location at a time, inherit from Fetcher.
* For fetcher classes whose source loader can handle multiple data source locations at once, inherit from MultiFetcher.

Full version:
The Fetcher is a class for transforming a forecasting domain into time series data. The Fetcher base classes define
how to conform arbitrary retrieved data to the domain, optionally with some aggregation or transformation applied.
A concrete Fetcher implementation must define the actual data retrieval process.

TODO: define a helper class more narrowly concerned with such transformation?

A domain can include one or more locations, so the Fetcher must determine which data to load for a domain location by
mapping domain location to data source location (as well as extracting the appropriate time slice).

All Fetcher classes implement a 'source_loader' method for reading data into memory from some backend data source.
When reading from a backend multiple times is not meaningfully more expensive than reading one location at a time,
especially if those multiple calls can be parallelized within the source_loader, the Fetcher base class handles all the
transformation logic, and it even provides helper methods for the common use case of reading blobs from GCS.

On the other hand, when reading from a backend multiple times is significantly more expensive than reading one location
at a time, the source_loader method should be written to read all locations and times with a single (and, usually, still
cacheable) call. In those cases, the fetcher class should inherit from the MultiFetcher base class.

So, the primary difference between Fetcher and MultiFetcher is that the former assumes the source_loader will return
data for one domain location at a time, and handle data transformation and assembly accordingly, whereas the latter
assumes that a single call to source_loader will return _all_ relevant data, and therefore uses more complex logic to
conform the full dataset to the full domain at once.
"""


class Fetcher(ABC, TransformerMixin, BaseEstimator):
    """
    The Fetcher is a class for transforming a forecasting domain into time series data. This base class defines how to
    conform arbitrary retrieved data to the domain, optionally with some aggregation or transformation applied. The
    Fetcher base class handles all the transformation logic, and it provides helper methods for the common use case of
    reading blobs from GCS.

    Concrete Fetcher classes must implement a 'source_loader' method for reading data into memory from some backend data
    source. Use the Fetcher as a base class when reading from a backend multiple times is not meaningfully more
    expensive than reading one location at a time, especially if those multiple calls can be parallelized within the
    source_loader.

    In addition to a 'source_loader' method, for compatibility with forecasting you will need to implement
    'variables' and 'get_feature_names'. You must also define 'get_all_data' (or raise NotImplementedError).
    """

    # Use class attributes for now. These are not stored in the pickled model, they are a property of the runtime
    # Staging is now the default bucket value.
    # The bucket can now be set properly at run time using an initializer with the parallel_patch
    # Legacy GCS bucket reader parameter
    # TODO: replace with fsspec.
    GCS_BUCKET = None
    _GCS_BUCKET_CLIENT = None
    _GCS_CLIENT = None

    timer_summary = Summary(
        "fetcher",
        "Time Series Models fetcher summary stats",
        ["klass", "function"],
    )

    def __init__(
        self,
        location_mapper: typing.Callable | typing.Literal["RESOURCE_LOOKUP"],
        selector: typing.Literal["select", "group"],
        selector_args: dict,
        pad_width: int = 0,
    ):
        """
        Initialize Fetcher base class and configure selector for variable selection or groupby operations.
        :param location_mapper: a mapping from domain_location to a dict(latitude={value1}, longitude={value2}).
            Can be the string literal value "RESOURCE_LOOKUP", in which case location_mapper will be constructed by
            connecting to a backend DB and reading from the resource model.
        :param selector: either "select" or "group"
        :param selector_args: dict of arguments to pass to the select or group method, including:
            "method": Literal["aggregate", "transform"] for type of groupby operation, used only if selector is "group";
            "variables": List[str] with variables to retain, used only if not providing a dict of aggregations;
            "grouper": a pd.Grouper used for specifying frequency for grouped aggregation (with selector="group"); and
            "kwargs": dict of additional keyword arguments, including:
                "func": the aggregating or transforming function(s), either as a single function or as a dict of
                    variable_name:List[aggregations], used only if selector is "group"; and
                "engine": the engine to use for groupby computation, Pandas default with engine=None is Cython
                    but can also specify "numba"
        :param pad_width: amount of padding required for proper return of data for requested domain.
        """

        logger.debug(
            "constructing Fetcher(location_mapper=%s, selector=%s, selector_args=%s, pad_width=%s)",
            location_mapper,
            selector,
            selector_args,
            pad_width,
        )

        # Original copy for cloning
        self.location_mapper = location_mapper

        # Mutable copy for modification
        self._location_mapper = location_mapper

        self.selector = selector
        self.selector_args = selector_args
        self.pad_width = pad_width

        self._check_conflicting_variables()

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def get_all_data(self, domain_location):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @abstractmethod
    def source_loader(self, start, end, domain_location):
        """
        Load data for one or more domain_locations from start to end,
        after mapping each domain_location to a data source location.
        :param start: the start of the time range for which to load data
        :param end: the end of the time range for which to load data
        :param domain_location: the domain location for which data is to be identified and loaded
        :return: Pandas DataFrame
        """

    def fit(self, X, y=None):
        """
        No op fit method
        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, X):
        """
        Transform the input data X by calling the get_data method
        :param X: a time series model domain
        :return: an array of values fetched from the data source
        """
        return self.get_data(X)

    @timer(log=logger)
    @timer_summary.labels(klass=__qualname__, function="get_data").time()
    def get_data(self, domain):
        """
        fetch data for the location(s) specified in the domain
        Select the appropriate variables
        :param domain:
        :return: numpy ndarray
        """
        try:
            self._check_domain(domain)

            # TODO(Michael H): does padding work if domain tstep is more than a single frequency unit, e.g. 15 minutes?
            padded_domain = self._pad_domain_dt(domain, pad_width=self.pad_width)
            mask = self._pad_domain_dt(
                domain, pad_value=np.nan, pad_width=self.pad_width
            )

            result = self._load_data(padded_domain)
            select = np.array(~np.isnat(mask[DATE_TIME])).reshape(-1)
            logger.debug("Finishing get_data!")
            return result[select, :]
        except EmptyFetchException:
            logger.exception("Handling empty result and returning nans")
            result = np.empty(
                (len(domain), len(self.get_feature_names())), dtype=np.float64
            )
            result[:] = np.nan
            return result

    def group(self, dataframe, domain) -> pd.DataFrame:
        """
        Apply pandas groupby operation(s) to the fetched data during alignment to the domain.
        Uses a single-location slice of the domain.
        :param dataframe: a pandas DataFrame with a datetime64 index
        :param domain: a single-location slice of the domain
        :return: aggregated or transformed dataframe aligned to domain
        """
        logger.debug("executing Fetcher.group")
        # TODO(Michael H): what happens if not self.variables but we still want a transform with numba?
        if self.variables:
            dataframe = dataframe.reindex(columns=self.variables).astype(np.float64)
        return self._group(dataframe, domain)

    def select(self, dataframe: pd.DataFrame, domain: np.ndarray) -> pd.DataFrame:
        """
        Conform a dataframe to a domain by reindexing the dataframe to the domain tseries and to self.variables.
        Missing entries will be nan-filled.
        :param dataframe: a pandas DataFrame with a datetime64 index
        :param domain:
        :return: pandas DataFrame with a datetime64 index
        """
        tseries = domain[DATE_TIME].reshape(-1)
        dataframe = dataframe.reindex(tseries)
        if self.variables:
            dataframe = dataframe.reindex(columns=self.variables)
        return dataframe.astype(np.float64)

    @staticmethod
    @contextlib.contextmanager
    def file_buffer_loader(path):
        with open(path, "rb") as buf:
            yield buf
        logger.debug("exiting file ctx")

    @classmethod
    @contextlib.contextmanager
    def gcs_buffer_loader(cls, path):
        with io.BytesIO() as buf:
            blob = cls.get_gcs_bucket_client().blob(path)
            blob.download_to_file(buf)
            buf.seek(0)
            yield buf

    @classmethod
    def get_gcs_client(cls):
        if not cls._GCS_CLIENT:
            client = storage.Client()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=128, pool_maxsize=128, max_retries=3, pool_block=True
            )
            client._http.mount("https://", adapter)
            client._http._auth_request.session.mount("https://", adapter)
            cls._GCS_CLIENT = client
        return cls._GCS_CLIENT

    @classmethod
    def get_gcs_bucket_client(cls):
        if not cls._GCS_BUCKET_CLIENT:
            logger.debug("Loading bucket: %s", cls.GCS_BUCKET)
            cls._GCS_BUCKET_CLIENT = cls.get_gcs_client().bucket(cls.GCS_BUCKET)
        return cls._GCS_BUCKET_CLIENT

    @staticmethod
    def monthly_gcs_names(path_format, start, end, **kwargs):
        # Pad the start date by two days
        tstart = start.astype(datetime) - timedelta(days=2)
        # Adjust to the first day of the month
        tstart = datetime(tstart.year, tstart.month, 1)
        # Pad the end date by two days
        tend = end.astype(datetime) + timedelta(days=2)

        return [
            path_format.format(year=date_time.year, month=date_time.month, **kwargs)
            for date_time in pd.date_range(start=tstart, end=tend, freq="MS")
        ]

    def _check_conflicting_variables(self):
        if self.selector_args is not None:
            if self.selector_args.get("method") == "aggregate":
                if self.selector_args.get("variables") is not None:
                    func = self.selector_args.get("kwargs", {}).get("func")
                    if type(func) is dict:
                        raise FetcherConfigurationError(
                            "Encountered both 'variables' and a dict of aggregations. "
                            "If aggregating, don't provide 'variables' separately from aggregations dict."
                        )

    @staticmethod
    def _check_domain(domain):
        # check for interleaving of locations in the domain
        locations = domain[LOCATION]
        new_values = locations != np.roll(
            locations, 1
        )  # True each time a value is different from the preceding one
        new_values[
            0
        ] = True  # np.roll checks this position against new_values[-1] so ensure 1st value is always novel
        if len(new_values[new_values]) > len(np.unique(locations)):
            raise RuntimeError("Encountered non-contiguous locations in domain!")

    @staticmethod
    def _deconstruct_domain(domain, return_all_locs=False) -> tuple:
        """
        Deconstruct (almost) the domain into start datetime, stop datetime, and the first (or all) domain location(s).
        Doesn't quite deconstruct the domain, since we are omitting the timestep.
        Note: with return_all_locs=True we get a result of variable length,
        so you might need to save the resulting tuple and slice out what you need.
        :param domain:
        :param return_all_locs:
        :return: tuple of start, stop, and first or all domain locations
        """
        domain = domain.reshape(-1)
        start = domain[DATE_TIME][0]
        stop = domain[DATE_TIME][-1]
        loc = [domain[LOCATION][0]]
        if return_all_locs:
            loc = pd.unique(domain[LOCATION].reshape(-1))
        return start, stop, *loc

    def _group(self, dataframe: pd.DataFrame, domain_like: np.ndarray) -> pd.DataFrame:
        """
        The core of the Fetcher.group method.
        Allows the numba engine to be used in a mixed context with builtin aggregations on multiple variables.
        Works on single or multiple locations provided the grouper's selector_args are specified properly.
        Pandas groupby with time grouper doesn't play nice with multiindex, so if you wish to _group a DataFrame with a
        multiindex that includes both time and location, reset the location index level to a DataFrame column and ensure
        that it is included in the selector_args["grouper"] list.
        :param dataframe: a pandas DataFrame with a datetime64 index
        :param domain_like: either the entire domain or a single-location slice
        :return:
        """

        """
        Since the Pandas Grouper object is passed in with the constructor it lives on with the fetcher instance as part
        of the trained Time Series Model that is persisted, stored in GCS and reloaded to make forecast predictions.

        Unfortunately, the Grouper object appears to hold the Groups as an instance attribute after calling groupby!
        This results in trained coincident peak models serializing at about 180Mb! By making a local copy of the object
        before using it, we can be sure it will go out of scope and be garbage collected after the group method returns.
        With this change in place the same coincident peak model serializes to about 1Mb.

        Note, we must use deepcopy, because copy.copy is a shallow copy that fails to solve this issue!
        """
        grouper_copy = deepcopy(self.selector_args["grouper"])
        dfg = dataframe.groupby(grouper_copy)

        if ("numba" == self.selector_args.get("kwargs", {}).get("engine", None)) & (
            self.selector_args["method"] == "aggregate"
        ):
            result_list = []
            for var_name, agg_list in self.selector_args["kwargs"]["func"].items():
                serg = dfg[var_name]
                group_method = getattr(  # aggregate or transform
                    serg, self.selector_args["method"]
                )
                for agg in agg_list:
                    logger.debug(
                        "Groupby operation: method %s func %s var %s with %s",
                        self.selector_args["method"],
                        agg,
                        var_name,
                        self.selector_args.get("kwargs", {}),
                    )
                    # allow for a mix of strings and numba optimized functions
                    if isinstance(agg, str):
                        # string functions are not numba optimized so drop the numba engine keyword
                        result_list.append(
                            group_method(
                                func=agg,
                                **{
                                    k: v
                                    for k, v in self.selector_args["kwargs"].items()
                                    if (k != "engine") & (k != "func")
                                },
                            )
                        )
                    else:
                        # proceed with numba
                        result_list.append(
                            group_method(
                                func=agg,
                                **{
                                    k: v
                                    for k, v in self.selector_args["kwargs"].items()
                                    if (k != "func")
                                },
                            )
                        )

            funcd = pd.concat(result_list, axis=1)

        else:
            # TODO(Michael H): allow different aggregations or transforms per variable!
            logger.debug("default grouping pathway")
            """Allow use of the builtin (slow) aggregation or transform method -- though numba for transform
            takes this pathway, too"""
            logger.debug(
                "Groupby operation: method %s with kwargs %s",
                self.selector_args["method"],
                self.selector_args.get("kwargs", {}),
            )
            group_method = getattr(  # aggregate or transform
                dfg, self.selector_args["method"]
            )
            funcd = group_method(**self.selector_args["kwargs"])

        logger.debug("reindexing dataframe to domain (or domain slice)")
        reindexed = self.select(funcd, domain_like)
        # interpolate after reindexing, but only internally (not at start or end)
        # TODO(Michael H): measure how much is missing from each feature at this point, first! (monitoring)
        if self.selector_args.get("interpolate", False):
            # interpolate any (non-leading and non-trailing) missing values that remain after the groupby operation
            reindexed = self._interpolate(reindexed)
        return reindexed.astype(np.float64)

    @staticmethod
    def _interpolate(df):
        return df.loc[: df.last_valid_index()].interpolate(limit_direction="forward")

    def _load_data(self, domain):
        domain_locations = pd.unique(  # pandas unique conserves order
            domain[LOCATION].reshape(-1)
        )

        data_frames = {  # dict preserves insertion order as of python 3.7
            # TODO (Michael H): add test asserting order is conserved?
            loc: self.source_loader(  # source load the domain for one location at a time
                *self._deconstruct_domain(domain[domain[LOCATION] == loc])
            )
            for loc in domain_locations
        }

        selectors = {
            "select": self.select,
            "group": self.group,
        }

        assembled = np.vstack(
            [
                selectors[self.selector](
                    data_frames[loc], domain[domain[LOCATION] == loc]
                ).to_numpy()
                for loc in domain_locations
            ]
        )
        return assembled

    def _location_mapper_is_identity(self, domain):
        # check whether location mapper is effectively an identity mapper for the given domain
        domain_locations = pd.unique(domain[LOCATION].reshape(-1))
        return all([self._location_mapper(loc) == loc for loc in domain_locations])

    @staticmethod
    def _pad_datetime_array(arr, pad_width=1):
        """
        Pad a datetime array by pad_width units. Can also handle multiple arrays as a single ndarray, assuming
        the component arrays are identical (e.g. a stack of identical date_time ranges for multiple locations).
        Return a result with the same number of dimensions as the input (array -> array, ndarray -> ndarray).
        Padding is done symmetrically (same number of units padded to start and to end). Array dtype is conserved.

        Note, padding is of single units of the same dtype as arr, and calculated from the endpoints of arr.
        If arr has non-unit frequency (e.g. three hours for datetime64[h] dtype) then padded frequency will not match.
        """
        n_dim = len(arr.shape)
        if n_dim == 1:
            ndarr = arr.reshape(1, -1)
        else:
            ndarr = arr.copy()
        n_locs = ndarr.shape[0]
        arr_start, arr_end = ndarr[0, 0], ndarr[0, -1]

        pre = arr_start - np.arange(pad_width, 0, -1, dtype=np.timedelta64).reshape(
            1, -1
        )
        ndpre = np.tile(pre, (n_locs, 1))

        post = arr_end + np.arange(1, pad_width + 1, dtype=np.timedelta64).reshape(
            1, -1
        )
        ndpost = np.tile(post, (n_locs, 1))

        out = np.concatenate((ndpre, ndarr, ndpost), axis=1)
        if n_dim > 1:
            return out
        else:
            return out.reshape(
                -1,
            )

    @classmethod
    def _pad_domain_dt(cls, domain, pad_value=None, pad_width=1):
        """
        Extend domain DATE_TIME by pad_width. if provided, pad_value is coerced to a consistent type as DATE_TIME,
        so set it to np.nan to get NaT values for a masking array, or leave at the default None for a simple
        incrementation of the array. DATE_TIME padding is symmetric at start and end for each LOCATION.
        LOCATION is filled in to match the extent of the padded DATE_TIME field.

        Note, padding is in increments of domain[DATE_TIME] dtype, so if domain has non-unit frequency then padding
        frequency will not match.
        """
        n_locs = np.unique(domain[LOCATION]).shape[0]

        # pad date_time with nans
        nd_domain_dt = domain[DATE_TIME].reshape(n_locs, -1)
        if pad_value is not None:
            padded_dt = np.pad(
                nd_domain_dt,
                ((0, 0), (pad_width, pad_width)),
                constant_values=pad_value,
            ).reshape(-1, 1)
        else:
            padded_dt = Fetcher._pad_datetime_array(
                nd_domain_dt, pad_width=pad_width
            ).reshape(-1, 1)

        # extend location to match date_time length
        nd_domain_loc = domain[LOCATION].reshape(n_locs, -1)
        padded_locs = np.pad(
            nd_domain_loc, ((0, 0), (pad_width, pad_width)), mode="edge"
        ).reshape(-1, 1)

        # recombine
        padded_domain = np.empty(padded_dt.shape, dtype=domain.dtype)
        padded_domain[LOCATION] = padded_locs
        padded_domain[DATE_TIME] = padded_dt
        return padded_domain


class MultiFetcher(Fetcher):
    """
    The MultiFetcher is a class for transforming a forecasting domain into time series data. This base class defines how
    to conform arbitrary retrieved data to the domain, optionally with some aggregation or transformation applied. By
    inheriting from Fetcher, the MultiFetcher class likewise handles all the transformation logic and provides helper
    methods for the common use case of reading blobs from GCS. Since the MultiFetcher assumes its source_loader will
    return data for all domain locations and times with a single call, it generalizes the Fetcher's conformation and
    aggregation methods to work for multiple domain locations, accordingly.

    Concrete MultiFetcher classes must implement a 'source_loader' method for reading data into memory from some backend
    data source. Use the MultiFetcher as a base class when reading from a backend multiple times is significantly more
    expensive than reading one location at a time. The source_loader method must be written to read all locations and
    times with a single (and, usually, cacheable) call.

    In addition to a 'source_loader' method, for compatibility with forecasting you will need to implement
    'variables' and 'get_feature_names'. You must also define 'get_all_data' (or raise NotImplementedError).
    """

    # Needed for testing, but storage should be distinct for each concrete implementation
    LOCATION_MAPPING: dict = {}

    def __init__(
        self,
        location_mapper: typing.Callable | typing.Literal["RESOURCE_LOOKUP"],
        selector: typing.Literal["select", "group"],
        selector_args: dict,
        pad_width: int = 0,
        location_mapping: dict | None = None,
        resource_query: str | None = None,
        resource_query_args: dict | None = None,
    ):
        """
        Initialize Fetcher base class and configure selector for variable selection or groupby operations.
        :param location_mapper: a mapping from domain_location to a dict(latitude={value1}, longitude={value2}).
            Can be the string literal value "RESOURCE_LOOKUP", in which case location_mapper will be constructed by
            connecting to a backend DB and reading from the resource model.
        :param selector: either "select" or "group"
        :param selector_args: dict of arguments to pass to the select or group method, including:
            "method": Literal["aggregate", "transform"] for type of groupby operation, used only if selector is "group";
            "variables": List[str] with variables to retain, used only if not providing a dict of aggregations;
            "grouper": a pd.Grouper used for specifying frequency for grouped aggregation (with selector="group"); and
            "kwargs": dict of additional keyword arguments, including:
                "func": the aggregating or transforming function(s), either as a single function or as a dict of
                    variable_name:List[aggregations], used only if selector is "group"; and
                "engine": the engine to use for groupby computation, Pandas default with engine=None is Cython
                    but can also specify "numba"
        :param location_mapping: hook to initialize GriddedDataFetcher with a non-empty mapping
        :param resource_query: query to use for resource mapping in a database
        :param resource_query_args: Arguments for the sql query mogrify operation other than resource_collection
        :param pad_width: amount of padding required for proper return of data for requested domain.
        """
        logger.debug(
            "constructing MultiFetcher(location_mapper=%s, selector=%s, selector_args=%s, location_mapping=%s, "
            "resource_query=%s, resource_query_args=%s, pad_width=%s)",
            location_mapper,
            selector,
            selector_args,
            location_mapping,
            resource_query,
            resource_query_args,
            pad_width,
        )

        if selector == "group":
            if not is_iterable(selector_args["grouper"]):
                selector_args["grouper"] = [selector_args["grouper"]]
            if MAPPED_LOCATION not in selector_args["grouper"]:
                selector_args["grouper"] = [MAPPED_LOCATION, *selector_args["grouper"]]

        # Original copy for cloning
        self.location_mapping = location_mapping
        # Mutable copy for modification
        self._location_mapping = copy(location_mapping or {})

        self._use_resource_lookup = location_mapper == "RESOURCE_LOOKUP"

        # Original copy for cloning
        self.resource_query_args = resource_query_args
        # Mutable copy for modification
        self._resource_query_args = copy(resource_query_args or {})

        self.resource_query = resource_query

        super().__init__(
            location_mapper,
            selector,
            selector_args,
            pad_width=pad_width,
        )

    def _check_conflicting_variables(self):
        super()._check_conflicting_variables()

        if self._use_resource_lookup:
            if self.resource_query is None:
                raise FetcherConfigurationError(
                    "A resource_query must be specified when using resource lookup."
                )

    @property
    def mapper_query(self):
        return self.resource_query

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def get_all_data(self, domain_location):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @abstractmethod
    def source_loader(self, start, end, *domain_locations):
        """
        Load data for one or more domain_locations from start to end,
        after mapping each domain_location to a data source location.

        Source loader does not call update_mapping when using Resource Lookup.
        Use the get_data api or explicitly call update_mapping before using source_loader api

        :param start: the start of the time range for which to load data
        :param end: the end of the time range for which to load data
        :param domain_locations: one or more domain locations for which data is to be identified and loaded
        :return: Pandas DataFrame
        """
        pass

    def group(self, dataframe: pd.DataFrame, domain) -> pd.DataFrame:
        """
        Apply a groupby operation (aggregate or transform) to the fetched data, and algin it to the domain.
        Uses entire domain rather than just single-location subset for grouping/reindexing.
        Location must be included as a grouping variable!
        :param dataframe: a pandas DataFrame with a MultiIndex identifying location and time
        :param domain:
        :return: aggregated or transformed dataframe aligned to domain
        """
        logger.debug("executing MultiFetcher.group")
        # TODO(Michael H): what happens if not self.variables but we still want a transform with numba?
        if self.variables:
            dataframe = dataframe.reindex(columns=self.variables).astype(np.float64)

        # groupby with time grouper doesn't play nice with multiindex, so demote non-time index level to column
        dataframe = dataframe.reset_index(level=MAPPED_LOCATION, drop=False)
        return self._group(dataframe, domain)

    def select(self, dataframe: pd.DataFrame, domain) -> pd.DataFrame:
        """
        Reindex dataframe to domain and to self.variables
        :param dataframe: a pandas DataFrame with a MultiIndex identifying location and time
        :param domain:
        :return: reindexed pandas DataFrame with a MultiIndex consisting of location and time
        """
        logger.debug("executing MultiFetcher.select")
        # take care of columns first
        if self.variables:
            if isinstance(dataframe.columns, pd.MultiIndex):
                dataframe = dataframe.reindex(columns=self.variables, level=0)
            else:
                dataframe = dataframe.reindex(columns=self.variables)

        # now the rows
        if self._location_mapper_is_identity(domain):
            logger.debug("data source location is equivalent to domain location")
            dataframe.index = dataframe.index.rename([LOCATION, DATE_TIME])
            # multiindex version of Fetcher.select
            selected = dataframe.reindex(index=multiindex_from_domain(domain))
        else:
            logger.debug("mapping data source location to domain location")
            selected = pd.DataFrame(
                {
                    DATE_TIME: domain[DATE_TIME].reshape(-1),
                    LOCATION: domain[LOCATION].reshape(-1),
                }
            )
            selected[MAPPED_LOCATION] = [
                self._location_mapper(v) for v in selected[LOCATION]
            ]
            if isinstance(dataframe.columns, pd.MultiIndex):
                # we have already arranged columns in expected order, so now avoid merging frames with varying levels
                dataframe = dataframe.droplevel(1, axis="columns")
            selected = (
                # set index to match that of dataframe for faster merging on MAPPED_LOCATION and DATE_TIME
                selected.set_index([MAPPED_LOCATION, DATE_TIME])
                .merge(dataframe, left_index=True, right_index=True, how="left")
                # don't actually need MAPPED_LOCATION anymore so swap it out of index
                .reset_index(drop=False)
                .set_index([LOCATION, DATE_TIME])
                .drop(columns=[MAPPED_LOCATION])
                .sort_index()  # sort by LOCATION first, DATE_TIME second
            )
        return selected.astype(np.float64)

    @staticmethod
    def _interpolate(df):
        return df.groupby(LOCATION).apply(interpolate_interior)

    def _load_data(self, domain):
        logger.debug("executing MultiFetcher._load_data")

        start, stop, *domain_locations = self._deconstruct_domain(
            domain, return_all_locs=True
        )

        if self._use_resource_lookup:
            # TODO: Consider moving this into source_loader to avoid key errors
            logger.debug("mapping before update: %s", self._location_mapping)
            # update the mapping, and the location mapper, to include any new locations
            self.update_mapping(*domain_locations)
            logger.debug("mapping: %s", self._location_mapping)

        all_data = self.source_loader(start, stop, *domain_locations)
        selectors = {
            "select": self.select,
            "group": self.group,
        }
        return selectors[self.selector](all_data, domain).to_numpy()

    def __reduce__(self):
        """
        Make sure any updates to the class mapping cache get applied to the instance before being pickled
        """
        self._location_mapping |= self.__class__.LOCATION_MAPPING
        return super().__reduce__()

    def update_mapping(self, *locations) -> None:
        """
        Update the instance location mapping by saving any new locations into the existing mapping. Also update the
        class mapping with the new mapping.
        If a requested location does not exist in the backend DB, it will be omitted from the mapping!
        Operation is effectively idempotent.
        :param locations: the locations to include in the mapping
        :return: None
        """
        self._location_mapping |= self.__class__.LOCATION_MAPPING
        self._location_mapping = self._update_mapping(*locations)
        self.__class__.LOCATION_MAPPING |= self._location_mapping
        self._location_mapper = self._update_location_mapper()

    @abstractmethod
    def map_locations(self, *locations) -> dict[str, typing.Any]:
        """
        Get location mapping for a set of locations by reading from a database.
        :param locations: an iterable of one or more locations
        :return: a dict of {location: Any} compatible with implemention of source_loader
        """
        pass

    def _update_location_mapper(self) -> callable:
        logger.debug("constructing new location_mapper!")

        # the latest and greatest
        def location_mapper(x):
            return self._location_mapping[x]

        return location_mapper

    def _update_mapping(self, *locations) -> dict[str, typing.Any]:
        """
        Create a new mapping by combining the existing mapping with any new locations.
        Does not modify the original mapping, but creates a new set of references to a superset
        of the original referrents.
        :param locations: the locations to include in the mapping
        :return: a dict of {location: Any}
        """
        logger.debug("checking for new locations to update mapping")
        # shallow copy, referencing same contents:
        mapping = self._location_mapping.copy()
        unmapped = np.setdiff1d(locations, list(mapping.keys()))
        if len(unmapped) > 0:
            # only lookup mapping for unmapped locations
            new = self.map_locations(*unmapped)
            # add any new locations to the existing mapping
            mapping |= new
        else:
            logger.debug("no-op mapper update!")
        return mapping
