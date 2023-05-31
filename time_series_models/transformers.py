import logging
import typing
from itertools import chain

import sklearn
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing

import numpy as np
import pandas as pd
from scipy import signal

from time_series_models.constants import (
    LOCATION,
    DATE_TIME,
    LOCATION_LENGTH,
)
from time_series_models.decorators import feature_names

logger = logging.getLogger(__name__)


################################
#  time_series_models helpers  #
################################


def decorate_feature_names(func, feature_names_func):
    """
    Helper function to graft a get_feature_names function onto any other function,
    where features are defined at runtime (rather than at import time, as happens with the
    @feature_names decorator). This is useful e.g. when a sklearn FunctionTransformer uses
    an array transformation func that can return distinct feature types depending on how it
    is called.
    Note, the original function gets modified, too!
    :param func: the function to decorate with get_feature_names
    :param feature_names_func: a function that returns feature names
    :return: the decorated function
    """
    # minimal type checking, caveat emptor
    if not is_iterable(feature_names_func()):
        raise RuntimeError("feature_names_func must return an iterable of str")
    func.get_feature_names = feature_names_func
    return func


def get_data_pipeline(fetcher, shift=None, extend=False, **kwargs):
    """
    Create a magical data pipeline transformer... allows for some pretty gnarly composable data transforms
    :param fetcher: an object implementing the DataFetcher API
    :param shift:
    :param kwargs: append additional pipeline transform steps. Feature names are a problem
    :return: the pipeline
    """
    # capture the shifted time context for any later operations.
    transforms = [
        (
            "shift_time",
            sklearn.preprocessing.FunctionTransformer(
                func=shift_time, kw_args=dict(time_delta=shift, extend=extend)
            ),
        ),
        ("fetcher", fetcher),
    ]

    if kwargs:
        for name, transform in kwargs.items():
            # TODO Apply magic here to get feature names from the fetcher step and mangle appropriately.
            # Beware of function transformers - they already have a magic patch, and they might get reused.
            transforms.append((name, transform))

    return sklearn.pipeline.Pipeline(transforms)


def interpolate_array_pandas_default(array: np.ndarray) -> np.ndarray:
    """
    Converts an array to a pandas DataFrame, interpolates with default settings, and returns an array again
    :param array: the array to interpolate
    :return:
    """
    return pd.DataFrame(array).interpolate().to_numpy()


def interpolate_array_interior(array: np.ndarray) -> np.ndarray:
    """
    Wrapper for interpolate_interior to interpolate the interior of a numpy array and return the result as an array.
    Interpolation is linear, and from the first non-missing value to the last non-missing in each column.
    :param array: the array to interpolate
    :return: a numpy ndarray in the same shape as the original
    """
    return interpolate_interior(pd.DataFrame(array)).to_numpy()


def interpolate_interior(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate from the first valid index to the last valid index of a dataframe; returns a copy.
    :param dataframe: a pandas DataFrame
    :return: an interpolated copy of the DataFrame
    """
    df = dataframe.copy()
    for col in df.columns:
        lvi = df[col].last_valid_index()
        df.loc[:lvi, col] = df.loc[:lvi, col].interpolate(limit_direction="forward")
    return df


def is_iterable(maybe_iterable) -> bool:
    # adapted from https://stackoverflow.com/a/1952481
    if type(maybe_iterable) in {str, np.str_}:
        # we don't want to iterate through the characters of a string!
        return False
    try:
        iter(maybe_iterable)
        return True
    except TypeError:
        return False


def uniques_from_sublists(list_of_sublist: list[list]) -> set:
    """
    Identify all unique values across each sublist in a list-of-list, and return as a set.
    Order is not conserved!
    :param list_of_sublist:
    :return:
    """
    return set(chain.from_iterable(list_of_sublist))


def make_domain(start, end, tstep, *locations):
    """
    Create the closed time range inclusive of end or upto end + tstep.
    The domain is agnostic to timezone. It can be expressed through the location for transformers that need it
    :param start: the starting datetime
    :param end: the ending datetime
    :param locations: a list of locations
    :return: the time and location domain for the model
    """
    time_range = np.arange(
        start=np.datetime64(start),
        stop=np.datetime64(end) + tstep,
        step=tstep,
    )
    tsize = time_range.shape[0]
    lsize = len(locations)
    domain_shape = (tsize * lsize, 1)  # sklearn likes (n,1) or (n,m) arrays
    domain = np.empty(domain_shape, dtype=make_domain_type(tstep))

    # Mapping the list and making the integer array is faster in a colab example than making the string array and mapping the array
    location_lengths = np.array(list(map(len, locations)))
    if (location_lengths > 36).any():
        dom_locs = pd.Series(locations)
        raise ValueError(
            f"Invalid locations are too long: {dom_locs.loc[location_lengths > 36].to_list()}"
        )

    dom_loc_np = np.tile(
        np.array(locations, dtype="U36").reshape(-1, 1), (1, tsize)
    ).reshape(-1, 1)
    dom_tsteps = np.tile(time_range.reshape(-1, 1), (lsize, 1))
    domain[LOCATION] = dom_loc_np
    domain[DATE_TIME] = dom_tsteps

    return domain


def make_domain_type(tstep):
    """
    Define the numpy data type used in the model transforms.
    The dtype should specify a location identifier and a date time.
    It must be dynamic based on the time step units.
    :return: a numpy dtype
    """
    # call arange to get the datetime unit with the correct character code (M ~= m)
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.timedelta64
    step_type = np.arange(
        np.datetime64("1970-01-01"), np.datetime64("1970-01-01"), step=tstep
    ).dtype

    return np.dtype([(LOCATION, np.unicode_, LOCATION_LENGTH), (DATE_TIME, step_type)])


def make_lookup(data_source_location):
    """
    Use this function factory to create location mappers for individual transformers in a sklearn ColumnTransformer and
    avoid the problem of mis-specified function scope causing all location mappers to be set to the final value of
    data_source_location (see test cases for illustrative examples).

    https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result

    :param data_source_location: the specific data source to which a domain_location will be mapped
    :return: a location mapper set to the specified data_source_location.
    """

    def lookup(domain_location):
        logger.debug(
            "domain_location %s mapped to %s", domain_location, data_source_location
        )
        return data_source_location

    return lookup


def multiindex_from_domain(domain_like: np.ndarray) -> pd.MultiIndex:
    """
    Convert a domain array to a pandas MultiIndex.
    :param domain_like: an (n, 1) compound numpy ndarray
    :return:
    """
    return pd.MultiIndex.from_frame(pd.DataFrame(domain_like.reshape(-1)))


def domain_from_multiindex(
    domain_like: pd.MultiIndex, tstep: np.timedelta64
) -> np.ndarray:
    """
    Convert a pandas multiindex back to a time series models domain
    # TODO: extract a general domain validation method? Consolidate with fetcher check_domain?
    :param domain_like: the multiindex
    :param tstep: the model timestep
    :return: the numpy array domain
    """
    domain_length = len(domain_like)
    if domain_length == 0:
        raise ValueError("zero length domain!")

    # allow for index using resource instead of location
    if "resource" in domain_like.names:
        domain_like = domain_like.rename(
            names=dict(resource=LOCATION, forecast_valid_time=DATE_TIME)
        )

    if (
        domain_like.dtypes
        != pd.Series([np.dtype("O"), np.dtype("<M8[ns]")], index=[LOCATION, DATE_TIME])
    ).any():
        raise ValueError(
            "Irregular multiindex is not a valid domain because the multiindex dtypes are not correct"
        )

    locations = domain_like.unique(level=LOCATION)
    location_count = len(locations)
    if domain_length % location_count != 0:
        raise ValueError(
            f"Irregular multiindex is not a valid domain because the length {domain_length} isn't divisible by the number of unique locations {location_count}"
        )
    tstep_count = int(domain_length / location_count)

    too_long = domain_like.get_level_values(LOCATION).to_series().transform(len) > 36
    if (too_long).any():
        bad_locations = (
            domain_like.get_level_values(LOCATION).to_series().loc[too_long].to_list()
        )
        raise ValueError(
            f"Irregular multiindex is not a valid domain because the following locations are more than 36 characters: {bad_locations}"
        )

    dom_locs = (
        domain_like.get_level_values(LOCATION)
        .to_numpy(dtype=np.dtype(f"U{LOCATION_LENGTH}"))
        .reshape((location_count, tstep_count))
    )
    dom_tsteps = (
        domain_like.get_level_values(DATE_TIME)
        .to_numpy()
        .reshape((location_count, tstep_count))
    )

    if not np.all(dom_locs == dom_locs[:, 0].reshape((-1, 1))):
        raise ValueError(
            "Irregular multiindex is not a valid domain: locations are not regular"
        )

    if not np.all(dom_tsteps == dom_tsteps[0, :]):
        raise ValueError(
            "Irregular multiindex is not a valid domain: timesteps are not regular"
        )

    if not np.all(np.diff(dom_tsteps, 1) == tstep):
        raise ValueError(
            "Irregular multiindex is not a valid domain: timesteps are not uniform"
        )

    domain = np.empty((domain_length, 1), dtype=make_domain_type(tstep))
    domain[LOCATION] = dom_locs.reshape((-1, 1))
    domain[DATE_TIME] = dom_tsteps.reshape((-1, 1))

    return domain


def revise_pipeline(
    pipe: sklearn.pipeline.Pipeline, new_step: tuple[str, any]
) -> sklearn.pipeline.Pipeline:
    """
    Helper to append a new step to a Pipeline, or replace the current one if it is a single passthrough.
    :param pipe: the Pipeline to update
    :param new_step: the tuple of (name, transformer) defining the new pipeline step
    :return: a new Pipeline
    """
    steps = pipe.steps
    if len(steps) == 1:
        if (steps[0][1] is None) or (steps[0][1] == "passthrough"):
            return sklearn.pipeline.Pipeline([new_step])
    steps.append(new_step)
    return sklearn.pipeline.Pipeline(steps)


def shift_time(x, time_delta, extend=False, tstep=None):
    """
    Subtract an amount of time from the array, shifting the time domain.
    # TODO: make tstep a required argument and update all downstream usages
    Without the tstep argument this method fails if the domain has a single timestep
    :param x: the input time series
    :param time_delta: the np.timedelta64 to shift by
    :param tstep: the np.timedelta64 step size of the domain
    :return: the resulting relative series
    """
    logger.debug(
        "shift_time on array (%s, %s) with time_delta=%s",
        x[0][0][1],
        x[-1][0][1],
        time_delta,
    )
    if time_delta:
        if extend:
            if tstep is None:
                # If you get an index error here passing a domain with a single value, use the tstep kwarg
                tstep = x[DATE_TIME][1, 0] - x[DATE_TIME][0, 0]

            new_start = x[DATE_TIME][0, 0] - time_delta
            new_end = x[DATE_TIME][-1, 0]
            locations, ind = np.unique(x[LOCATION], return_index=True)

            # Catch most obvious cases where the timestep is not uniform
            assert (
                x[DATE_TIME][0, 0] + (x.shape[0] / locations.shape[0] - 1) * tstep
                == new_end
            )

            x = make_domain(new_start, new_end, tstep, *locations[np.argsort(ind)])
        else:
            x = x.copy()
            x[DATE_TIME] = x[DATE_TIME] - time_delta
    logger.debug("shift_time complete, array now (%s, %s)", x[0][0][1], x[-1][0][1])
    return x


##########################
#  bespoke transformers  #
##########################


def convolution_transform(x, domain, window):
    """
    Apply a convolution filter to each location in the series x
    :param x: a [n,m] numpy array of data
    :param domain: the DOMAIN specifying the distinct locations of the series x
    :param window: the filter to apply
    :return:
    """
    if domain.shape[0] != x.shape[0]:
        raise ValueError("Domain length does not match X")

    locations, ind = np.unique(domain[LOCATION], return_index=True)

    if len(window.shape) == 1:
        window = window.reshape(-1, 1)

    res = []
    for loc in locations[np.argsort(ind)]:
        dom_loc = domain[LOCATION] == loc

        conv = signal.convolve2d(x[dom_loc.reshape(-1), :], window, mode="valid")
        res.append(conv)

    return np.vstack(res)


def rank(array, rankings):
    """
    Return an array of rank values for the values of the input array from highest to lowest
    :param array: a [n, 1] numpy array
    :param rankings: an [m>n, 1] numpy array of rank values
    :return: the ranked array [n, 1]
    """
    array = array.reshape(-1)
    rankings = rankings[: array.shape[0]]
    idx = np.argsort(-array)

    result = np.empty_like(rankings)
    result[idx] = rankings
    return result


def ranked_monthly_ranges(time_step, domain, ranks, range_method):
    """
    Transform the raw MW power values to ranked monthly peak classes
    :param domain:
    :return:
    """
    time_domain = domain[DATE_TIME]
    results = []
    for start_date in np.arange(
        time_domain[0, 0],
        time_domain[-1, 0] + time_step,
        step=np.timedelta64(1, "M"),
        dtype="datetime64[M]",  # hardcode to month for now
    ):
        end_date = start_date + np.timedelta64(1, "M")

        selected_index = domain[
            (time_domain >= start_date) & (time_domain < end_date)
        ].reshape(-1, 1)
        raw_mw = range_method(selected_index)
        ranked = rank(raw_mw, ranks)
        results.append(ranked)

    return np.concatenate(results, axis=0).reshape(-1, 1)


def autoregressive_features_pipeline(
    fetcher,
    lags: np.ndarray,
    time_step: np.timedelta64,
    **kwargs,
):
    """
    Autoregressive (lagged) features pipeline that builds featureset from a single fetch from the data source.
    :param fetcher: the configured fetcher instance -- or fetching pipeline! -- to use for fetching data
    :param lags: an array of lags that defines which lagged values will be used as model features.
        Set lags=None to return a single column of np.nan instead of creating any autoregressive features.
    :param time_step: the timestep of the Regular Time Series Model, e.g. np.timedelta64(1, "h") for hourly.
    :param kwargs: optional keyword args to pass to component steps like count_and_fetch, e.g. "n_jobs"
    :return: a scikit-learn Pipeline that will fetch and construct the autoregressive feature set for a
        specified domain.
    """
    if lags is None:
        return sklearn.pipeline.Pipeline(
            [("no_ar_features", sklearn.preprocessing.FunctionTransformer(nan_like))]
        )
    if len(lags) == 0:
        raise ValueError(
            "Improper specification of lags: %s. To specify no lags, use 'lags=None'.",
            lags,
        )
    if isinstance(lags, list):
        # we'd rather it not be, but handle this anyways
        lags = np.array(lags)
    # convert timestep dtype as needed, e.g., when "h" process has "D" lags specified
    lags = lags.astype(time_step.dtype)
    # TODO(Michael H): what if we want a non-unitary time step, e.g. 15 minutes?

    transforms = [
        (
            "extend",
            # pipeline will fetch from beginning of earliest lag to the end of original domain date range
            sklearn.preprocessing.FunctionTransformer(
                func=shift_time,
                kw_args=dict(time_delta=lags.max(), extend=True, tstep=time_step),
            ),
        ),
        (
            "count_and_fetch",
            # transformer to accomplish the fetching, along with encoding a count of locations in the first data column
            sklearn.compose.ColumnTransformer(
                [
                    (
                        "count_domain_locs",
                        sklearn.preprocessing.FunctionTransformer(
                            func=count_domain_locs_into_column
                        ),
                        [0],
                    ),
                    ("fetch", fetcher, [0]),
                ],
                n_jobs=kwargs.get("n_jobs"),
            ),
        ),
        (
            "slice_and_stack",
            # arrange the lagged slices of the fetched data into individual feature columns
            sklearn.preprocessing.FunctionTransformer(
                func=slice_and_hstack(lags=lags),
            ),
        ),
    ]
    logger.debug("Creating AR features pipeline")
    return sklearn.pipeline.Pipeline(transforms)


@feature_names(["domain_locations"])
def count_domain_locs_into_column(domain):
    # hackery to propagate a count of domain locations into a domain-shaped array of float
    n_locs = len(pd.unique(domain[LOCATION].reshape(-1)))
    res = np.ones(len(domain), dtype=float)
    res[:] = n_locs
    logger.debug(
        "count_domain_locs_into found %s locs, returning in shape %s",
        n_locs,
        domain.shape,
    )
    return res.reshape(-1, 1)


def interpolate_array_by_group(
    arr: np.ndarray,
    interp_func: typing.Optional[typing.Callable] = None,
    n_col: int = -1,
) -> np.ndarray:
    """
    Interpolate (possibly some of) the columns of an array after splitting the rows into n groups, where n is the first
    value of the first column. The first 1:(n_col+1) columns will be interpolated if n_col is specified, else all 1:-1
    columns will be interpolated. Column 0 is not modified, since that is just a location count with no missings.
    :param arr: the (m, n) array on which to perform interpolation
    :param interp_func: a callable function that interpolates each column of an (m, n) array, returning an (m, n) array.
        Defaults to linear interpolation with pandas.DataFrame.interpolate.
    :param n_col: how many (non-location) columns to interpolate, counting from the left. Default (-1) automagically
        interpolates all columns in the array
    :return: interpolated (m, n) array
    """
    n_locs = int(arr[0, 0])
    n_col_tot = arr.shape[1] - 1
    interpolated = np.empty_like(arr)
    if n_col == -1:
        n_col = n_col_tot
    elif n_col > n_col_tot:
        raise ValueError(
            f"Cannot interpolate {n_col} columns in an array with {n_col_tot} columns!"
        )
    elif n_col <= 0:
        raise ValueError(
            f"'n_col' must be positive, or -1 to interpolate all columns, but got {n_col}!"
        )
    # don't interpolate past n_col (shifted by 1 to skip over first column of location counts)
    interpolated[:, 0] = arr[:, 0]
    interpolated[:, n_col + 1 :] = arr[:, n_col + 1 :]

    # swing each location up into a new column
    arr = arr[:, 1 : n_col + 1].reshape(-n_locs * n_col, n_locs * n_col, order="F")

    if interp_func is None:
        interp_func = interpolate_array_pandas_default

    # interpolate each column (after wrapping in a DataFrame), extract the array values,
    # and swing back to the original shape... then fill into the array we are returning
    interpolated[:, 1 : n_col + 1] = interp_func(arr).reshape(-n_col, n_col, order="F")
    return interpolated


def overfetched_range_pipeline(
    fetcher,
    lags: np.ndarray,
    time_step: np.timedelta64,
    **kwargs,
):
    """
    This pipeline is intended for use in a given Process's get_range method, in conjunction with
    autoregressive_features_pipeline in make_preprocessor. It pre-caches the entire dataset used for model fitting.

    The pipeline overfetches the range by max(lags) in order to cache the same dataset that will be used by the
    features_fetcher elsewhere in the model. It then returns only the lag-0 (i.e., original requested) slice of
    the fetched data.

    In effect, we are adding some overhead to get_range for the sake of eliminating repeated calls to the data source
    backend by aligning the calls for caching. Note, this makes sense to do ONLY if multiple fetchers can share a common
    cache!

    If we are constructing features of lag 1, 3, and 7 days, then for each row of the domain we need to obtain values
    for 1, 3, and 7 days prior. Therefore if the domain is for 2022-08-10T00 through 2022-08-31T00, then the first row
    will have lag 1 with data from 2022-08-09T00, lag 3 with data from 2022-08-07T00, and lag 7 with data from
    2022-08-03T00, etc. Therefore we really need to fetch data for (start of domain - max(lags) = 2022-08-03T00) through
    (end of domain - min(lags) = 2022-08-30T00).

    When fetching the labels (the "true" values), we need these to be aligned exactly with the domain, from
    (start of domain = 2022-08-10T00) through (end of domain = 2022-08-31T00).

    To make use of caching, we fetch the entire range from (start of domain - max(lags) = 2022-08-03T00) through
    (end of domain = 2022-08-31T00) and then slice out whichever lagged piece we need, since the current values are
    equivalent to lag 0.

    The autoregressive_features_pipeline already takes care of most of the slicing logic; we just supply a lag of 0.
    And all it really needs to fetch the desired extended range is the value of max(lags). Building out the lag 7 column
    for the example above is unnecessary for the model per se, but it allows us to exercise the fetcher on the full
    range. In contrast, building out the lag 3 column, just to slice it out again, would not contribute anything at all.

    :param fetcher: the fetcher -- or fetching pipeline! -- that should be used for getting range data
    :param lags: an array of lags that defines which lagged values will be used as model features.
        Set to None when autoregressive features are not needed in the model.
    :param time_step: the timestep of the Regular Time Series Model, e.g. np.timedelta64(1, "h") for hourly.
    :param kwargs: keyword args to pass to component steps like count_and_fetch, e.g. "n_jobs"
    :return: a pipeline that will return an (n, 1) np.array of the fetched range data
    """
    if lags is None:
        # we still need to fetch the range!
        lags = np.array([0], dtype=time_step.dtype)
    if len(lags) == 0:
        raise ValueError(
            "Improper specification of lags: %s. To specify no lags, use 'lags=None'.",
            lags,
        )
    if isinstance(lags, list):
        lags = np.array(lags)
    lags = lags.astype(time_step.dtype).astype(int)
    """ we don't actually need a full autoregressive feature array... technically we only want the lag-0 data!
    but we force the pipeline in the first step to fetch for the max lag, too, to guarantee we are caching
    exactly what we will need for the features array later """
    lags = np.array([0, max(lags)], dtype=time_step.dtype)
    transforms = [
        (
            "overfetch_and_cache",
            autoregressive_features_pipeline(
                fetcher,
                lags,
                time_step=time_step,
                n_jobs=kwargs.get("n_jobs"),
            ),
        ),
        (
            "just_the_range",
            sklearn.preprocessing.FunctionTransformer(func=first_column_only),
        ),
    ]
    logger.info("Constructing overfetched range pipeline using lags %s", lags)
    return sklearn.pipeline.Pipeline(transforms)


def slice_and_hstack(lags: np.array) -> callable:
    """
    Not a normal function, but a closure that returns slice_and_stack_ configured with lags.
    The new function does the actual slicing and stacking, and also has a get_feature_names method.
    :param lags:
    :return: slice_and_stack function that takes one input, the array to slice and stack
    """
    max_lag = lags.max()

    def slice_and_stack(array: np.array) -> np.array:
        """
        For an (n, 1) array of target series data fetched for a domain that was start-padded by max(lags), slice and
        stack the padded array to create a lagged (autoregressive) feature set. Lags are specified in the outer scope.

        The array represents historical data fetched for each domain location over a padded domain date range:

                (adjusted domain start) = (original domain start - lags.max())

        So this function uses the known set of lags as offsets to slice out the lagged feature set.

        For example, if we have a domain with
        a single location and 24 hourly steps, and the configured lags were [1h, 2h], the input array should span from
        (start - 2h) to end, i.e. 26 hrs total. The 1h lag column should be 24 hourly steps, and consist of the slice
        from (start - 1h) to (end - 1h), whereas the 2h lag column should be (start - 2h) to (end - 2h).
        :param array: target series data fetched for a domain start-padded by max(lags) time steps.
        :return: mxn array of lagged feature data
        """
        logger.debug("slice_and_stack: array shape: %s", array.shape)
        logger.debug("array head: \n%s", array[:3, :])
        n_locs = int(array[0, 0])
        assert (
            array[:, 0] == n_locs
        ).all(), "This function assumes first column is constant valued!"
        assert array.shape[1] == 2, "This function requires an nx2 array."
        # order="F" splits each location to a new column (aligning the timestamps in rows)
        array = array[:, 1].reshape(-n_locs, n_locs, order="F")
        logger.debug("array shape now: %s", array.shape)
        logger.debug("array head now: \n%s", array[:3, :])
        # make a domain-shaped feature column for each lag, and assemble the results into a single 2D array
        logger.debug(
            "slicing using lags %s (max lag %s)", lags.astype(int), max_lag.astype(int)
        )
        res = np.hstack(
            [
                _slice_and_vstack(array, lag.astype(int), max_lag.astype(int))
                for lag in lags
            ]
        )
        logger.debug("slice_and_stack: result head now: \n%s", res[:3, :])
        return res

    def _gfn():
        return [str(lag).replace(" ", "_") for lag in lags]

    def _slice_and_vstack(array_, lag_, max_lag_):
        # here the input array_ has one location per column and one timestamp per row,
        # so all locations can be sliced temporally at once, whereas the output will be
        # a single domain-shaped feature column
        logger.debug("_slice_and_vstack: slicing array to lag %s", lag_)
        if lag_ == 0:
            sliced = array_[max_lag_:]
        else:
            sliced = array_[(max_lag_ - lag_) : -lag_, :]
        # order="F" to swing locations back down into a single column, same order as before
        return sliced.reshape(-1, 1, order="F")

    return decorate_feature_names(slice_and_stack, _gfn)


def mapped_fetch_data(fetcher, mapping: dict) -> callable:
    """
    Not a normal function, but a closure that returns a configured mapped_fetch_data_ function
    that will map from domain_location in arr, and provide a `get_feature_names` method!
    :param fetcher: a fetcher
    :param mapping:
    :return: a function that can transform a domain array into a fetched dataset
    """

    def mapped_fetch_data_(domain: np.array) -> np.array:
        """
        Map locations in domain using mapping from outer scope, and fetches using fetcher from outer scope.
        This is used for a forecast of a hierarchical process, where node A is the parent of node(s) B[, C, D, ...],
        and the process at the parent node is the sum of the child nodes (e.g., xfrmr net-of-btm load is the sum of
        the individual meter net loads, not counting any DER). A forecast in this case would use the location of
        the parent node that was specified in the domain, and map it to the child node locations for actual fetching.

        For example, given a mapping of:
           {
              xfrmr/01:[meter/electrical/01],
              xfrmr/02:[meter/electrical/02, meter/electrical/03, meter/electrical/04]
           },
        a domain of:
           [start=start_date, end=end_date, locations=[xfrmr/01, xfrmr/02]
        would be translated to a new domain of:
           [
              start=start_date,
              end=end_date,
              locations=[
                 meter/electrical/01,
                 meter/electrical/02,
                 meter/electrical/03,
                 meter/electrical/04,
               ]
           ],
        and this new domain is what the fetcher will transform with its get_data method.
        Once the fetcher returns a result, the result is grouped by the original domain
        location, and the values for each subcomponent will be summed up (so xfrmr/01
        will have the same series of values as did meter/electrical/01, whereas the
        time series for xfrmr/02 will be the summation of meter/electrical/02,
        meter/electrical/03, and meter/electrical/04).
        :param domain:
        :return: fetched values for mapped locations
        """
        domain_exploded, domain_loc_original = map_stack_domain(domain, mapping)
        # fetch exploded domain into a DataFrame
        logger.debug("mapped_fetch_data fetching exploded domain")
        logger.debug(domain_exploded.reshape(-1)[:3])
        exploded_fetch = pd.DataFrame(
            data=fetcher.get_data(domain_exploded),
            index=pd.MultiIndex.from_frame(
                pd.DataFrame(domain_exploded.reshape(-1)),
                names=[LOCATION, DATE_TIME],
            ),
        )
        exploded_fetch["location_original"] = domain_loc_original
        # use pandas groupby to sum component series; if all values in a group are nan, sum should be nan
        return (
            exploded_fetch.groupby(["location_original", DATE_TIME])
            .sum(min_count=1)
            .to_numpy()
        )

    def _gfn():
        return fetcher.get_feature_names()

    return decorate_feature_names(mapped_fetch_data_, _gfn)


def map_stack_domain(domain: np.array, mapping: dict) -> tuple[np.array, np.array]:
    """
    Map each domain location using a mapping, and return a new domain of mapped locations
    (and corresponding original domain locations).
    If mapping is one-to-one then the new domain will have the same shape as the original.
    If mapping is one-to-many then the new domain will be longer than the original.
    :param domain:
    :param mapping:
    :return: the mapped domain and the corresponding original domain locations
    """
    logger.debug("executing map_stack_domain")
    domain_locations = pd.unique(domain[LOCATION].reshape(-1))
    # map domain_loc --> list of lists
    mapped_locations = [mapping[loc] for loc in domain_locations]

    # assemble corresponding key of original domain locations
    n_steps = len(
        domain[domain[LOCATION] == domain_locations[0]]
    )  # n time steps per location
    group_len = np.array([len(group) for group in mapped_locations]) * n_steps
    domain_original_key = np.repeat(domain_locations, group_len)

    # flatten list-of-lists after measuring sublist lengths
    mapped_locations = [item for sublist in mapped_locations for item in sublist]
    # construct new domain out of mapped locations
    start = domain[DATE_TIME].min()
    end = domain[DATE_TIME].max()
    tstep = domain[DATE_TIME][1, 0] - domain[DATE_TIME][0, 0]  # domain has shape (n, 1)
    exploded = make_domain(start, end, tstep, *mapped_locations)

    return exploded, domain_original_key


class ColumnTypeTransformer(sklearn.compose.ColumnTransformer):
    """
    This subclass of ColumnTransformer assembles its outputs into a single column with complex dtype, rather than
    returning one or more columns of a single (and necessarily consistent) dtype. This can be useful when a downstream
    transformation requires inputs with more than one dtype.
    This makes the single column something like a pandas dataframe in that each column is names and has a distinct type.
    At some point, it may be possible to actually use a pandas DataFrame instead here.

    The hard requirement of the current implementation is that you must implement get_feature_names for each of the
    column transformers. You can pack additional data into nested datatypes, for instance allowing a transform
    to return an array for each input.
    """

    def _hstack(self, Xs):
        types = []
        slices = []
        for X, (name, transformer, cols) in zip(Xs, self.transformers_):
            feature_names = (
                transformer.get_feature_names()
                if hasattr(transformer, "get_feature_names")
                else [name]
            )

            if len(feature_names) == X.shape[1]:
                # Use the dtype and make one
                if isinstance(X, np.ndarray):
                    for ind, fname in enumerate(feature_names):
                        type_desc = (fname, X.dtype)
                        types.append(type_desc)
                        slices.append(X[:, ind])
                elif isinstance(X, pd.DataFrame):
                    for ind, (fname, dtype) in enumerate(zip(feature_names, X.dtypes)):
                        type_desc = (fname, dtype)
                        types.append(type_desc)
                        slices.append(X.iloc[:, ind])
            else:
                if hasattr(X, "dtype"):
                    dtype = X.dtype
                elif hasattr(X, "dtypes"):
                    dtype = X.dtypes
                else:
                    dtype = f"Unknown X type: {type(X)}"
                raise NotImplementedError(
                    f"Unexpected condition - feature_names: {feature_names}, transformer: {transformer}, cols: {cols}, "
                    f"Xdtype: {dtype}, Xshape: {X.shape}"
                )

        new_type = np.dtype(types)

        data = np.ndarray((X.shape[0], 1), new_type)

        for tp, slc in zip(types, slices):
            logger.info(
                "hstack - type: %s, slice shape: %s, slice type: %s",
                tp,
                slc.shape,
                slc.dtype,
            )
            data[tp[0]][:, 0] = slc

        return data


def one_hot_location(**kw_args):
    """
    Create a pipeline to one hot encode the domain location
    :param kw_args: arguments to OneHotEncoder
    :return: the pipeline to one hot encode the location
    TODO if locations are one hot encoded can the model be called with a location not in the training set?
      Depends on the regressor I think... but not a good idea
    """
    return sklearn.pipeline.Pipeline(
        [
            (
                "split",
                sklearn.preprocessing.FunctionTransformer(
                    func=split_domain, kw_args={"key": LOCATION}
                ),
            ),
            ("one_hot_encode", sklearn.preprocessing.OneHotEncoder(**kw_args)),
        ]
    )


def split_domain(x, key):
    """
    Split a numpy dtype array using a named member
    :param x: the input time series model domain
    :param key: the dtype member name
    :return: the resulting array
    """
    return x[key]


def monitor_fetcher(fetcher, monitor) -> sklearn.pipeline.Pipeline:
    """
    Wrap any Fetcher in a pipeline that directly monitors the output of get_data, by appending a ForecastDataMonitor.
    Use as a model component just as you would for any simple Fetcher.
    Automagically supports location-based stats depending on DataMonitor configuration.
    BYO convenience wrappers for accessing the monitor's `fit_stats` and `transform_stats` attributes.
    :param fetcher: the configured Fetcher to monitor
    :param monitor: a configured ForecastDataMonitor
    :return: a scikit learn Pipeline
    """
    if not monitor.use_locs:
        return sklearn.pipeline.Pipeline(
            [
                ("fetch", fetcher),
                ("monitor", monitor),
            ]
        )
    return sklearn.pipeline.Pipeline(
        [
            (
                "count_and_fetch",
                sklearn.compose.ColumnTransformer(
                    [
                        (
                            "count",
                            # prepend a column of domain location count
                            sklearn.preprocessing.FunctionTransformer(
                                count_domain_locs_into_column
                            ),
                            [0],
                        ),
                        ("fetch", fetcher, [0]),
                    ]
                ),
            ),
            ("monitor", monitor),
            (
                "drop_column",
                # cleanup: eliminate the "count" column
                sklearn.preprocessing.FunctionTransformer(drop_first_column),
            ),
        ]
    )


def net_energy_pipeline(fetcher):
    """
    Since "energy.consumed" and "energy.produced" are actually measurements of net flow in opposite directions, they
    should really be treated as "energy.net" relative to a reference direction.
    TODO: configure to specify reference direction, e.g. if some, but not all, meters are wired backwards
    :param fetcher: a Fetcher configured to fetch "energy.consumed" and "energy.produced".
        If the variables are reversed, the net value will have the opposite polarity.
    :return: a pipeline for fetching net energy flow
    """
    return sklearn.pipeline.Pipeline(
        [
            ("fetch", fetcher),
            (
                "combine",
                sklearn.preprocessing.FunctionTransformer(func=subtract_arrays),
            ),
        ]
    )


###################
#  mask builders  #
###################


def make_domain_mask_from_dict(
    domain: np.array, masking_dict: dict[str, list[tuple]]
) -> np.array:
    """
    Returns a mask in the shape of domain, valued True where the domain value is in masking_dict else False.
    Mask will be aligned to domain (any excess gets truncated).
    If overlapping intervals are specified in the masking_dict:
        * If overlaps are across domain locations, both locations will be masked.
        * If overlaps are within a location, the union of the overlapping regions will be masked.
    :param domain:
    :param masking_dict: a dict of {domain_location: list[tuple[start_time, end_time]]}
    :return: a boolean array to use as a mask on domain
    """
    tstep = domain[DATE_TIME][1, 0] - domain[DATE_TIME][0, 0]  # domain has shape (n, 1)
    masking_domain = masking_domain_from_dict(tstep, masking_dict)
    return np.in1d(domain, masking_domain).astype(np.float64).reshape(-1, 1)


def masking_domain_from_dict(
    tstep: np.timedelta64, masking_dict: dict[str, list[tuple]]
) -> np.array:
    """
    Converts masking_dict into a domain-like array, containing only the elements that should be masked.
    Elements are repeated when overlapping time ranges are specified for a given location in masking_dict.
    :param tstep:
    :param masking_dict:
    :return:
    """
    if len(masking_dict) == 0:
        return np.empty((0, 1), dtype=make_domain_type(tstep))

    arrays_dict = {
        k: np.concatenate(
            [
                np.arange(
                    start=np.datetime64(v[t][0]),
                    stop=np.datetime64(v[t][1]) + tstep,
                    step=tstep,
                )
                for t in range(len(v))
            ]
        )
        for k, v in masking_dict.items()
    }

    # concatenate the expanded date ranges
    times = np.concatenate([v for v in arrays_dict.values()]).reshape(-1, 1)
    # repeat each location (dict key) enough times to match the expanded date ranges
    locations = np.concatenate(
        [np.array([k] * len(v)) for k, v in arrays_dict.items()]
    ).reshape(-1, 1)

    # assemble into a domain-like array
    shape = (len(times), 1)
    to_mask = np.empty(shape, dtype=make_domain_type(tstep))
    to_mask[LOCATION] = locations
    to_mask[DATE_TIME] = times
    return to_mask


def squash_mask(array: np.array, masked_value: float = 0.0) -> np.array:
    """
    Take an (n, 2) array with values of interest in first column and boolean mask in the second,
    and squash into an (n, 1) array. Note, values are retained where mask evaluates to True,
    and converted to masked_value where mask evaluates to False.
    E.g., for masking an array prior to addition or subtraction, use masked_value=0. For masking prior to
    multiplication or division, use masked_value=1.
    :param array: an (n, 2) array
    :param masked_value: the value to assign to the array where mask evaluates to False.
    :return: an (n, 1) array
    """
    mask = array[:, [1]]
    to_mask = array[:, [0]]
    return np.where(mask, to_mask, masked_value)


##################################
#  simple array transformations  #
##################################


@feature_names(["components_sum"])
def add_arrays(arr: np.array) -> np.array:
    """
    Convenience function that decorates np.ndarray.sum with a get_feature_names method (returning ["components_sum"]).
    Summation is across columns (axis=1), but retains dimensionality (keepdims=True), so an (m, n) array sums to an
    (m, 1) output.
    :param arr: an (m, n) numpy array
    :return: the (m, 1) summed numpy array
    """
    return arr.sum(axis=1, keepdims=True)


@feature_names(["components_difference"])
def subtract_arrays(arr: np.array) -> np.array:
    """
    For an (m, n) array where n >= 2, subtract columns 1...n from column 0. Implements a get_feature_names method that
    returns ["components_difference"].
    :param arr: an (m, n) array
    :return: an (m, 1) array
    """
    net = arr[:, [0]]
    to_subtract = arr[:, 1:].sum(axis=1, keepdims=True)
    return np.subtract(net, to_subtract)


def drop_first_column(arr):
    assert len(arr.shape) == 2, "array must be 2-dimensional!"
    return arr[:, 1:]


def first_column_only(arr):
    assert len(arr.shape) == 2, "array must be 2-dimensional!"
    # copy to avoid returning a view, allowing arr to stay scoped
    return arr[:, [0]].copy()


@feature_names(["fill_nan"])
def fill_nan_columns(
    array: np.ndarray, col_from: int = 0, col_to: int = None
) -> np.ndarray:
    """Replace np.nan with 0.0 between specified columns. Modifies array inplace."""
    array[:, col_from:col_to] = np.nan_to_num(array[:, col_from:col_to])
    return array


@feature_names(["nan"])
def nan_like(arr):
    return np.full(arr.shape, np.nan, dtype=float)


def nullify_cols(arr: np.ndarray, cols: list = None):
    """
    Return an array with specified columns set to np.nan. Results will be coerced to float.
    :param arr: a 2D NumPy array
    :param cols: a list of column indices
    :return: numpy ndarray
    """
    if len(arr.shape) != 2:
        raise ValueError("<nullify_cols> operates on 2D NumPy arrays only.")

    if cols is None or not cols:  # include empty list!
        return arr.copy()

    arr = np.copy(arr).astype(dtype=float)
    arr[:, np.r_[cols]] = np.nan
    return arr


# TODO (Michael H): split out to another module, maybe filters?
class RowFilteringFunctionTransformer(sklearn.preprocessing.FunctionTransformer):
    """
    Class to use for filtering feature rows during model.fit for a sklearn supervised learning model, in the scenario
    where the target data have already been filtered prior to calling model.fit.

    In a sklearn supervised learning model, during a 'fit' call all transformers in the pipeline except for the final
    estimator will be sequentially fitted and transform their input prior to passing the result to the next transformer
    (see https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit
    for details). The standard sklearn API does not allow dropping target observations at all.

    Therefore, if we wish to exclude samples from the model training run because the target data were missing, we can
    drop the missing target data (after saving the indices of missing observations as 'range_nan'), construct a full
    feature array (for all rows), and then drop the corresponding feature rows during fit_transform of *this*
    transformer.

    This class wraps a no-op FunctionTransformer (func=None) for the sake of implementing fit, transform, and
    fit_transform. The underlying FunctionTransformer.fit_transform is therefore a no-op, as are the fit and
    transform calls alone, permitting a feature array to pass unmodified through an instance of
    RowFilteringFunctionTransformer during 'transform' as part of a 'model.predict' call.

    Ultimately, the FunctionTransformer is used as a vehicle to slot in a code object to the pipeline that applies row
    filtering to the feature set during pipeline fit, but not during pipeline transform.
    """

    def __init__(self):
        """instantiate FunctionTransformer with default 'func=None' so it is a no-op."""
        super().__init__()

    def fit_transform(self, x: np.array, y: np.array, range_nan: np.array = None):
        """
        Drop feature rows prior to passing features array to next step during model fit_transform. This method should
        only be called during sklearn pipeline fit, so RowFilteringFunctionTransformer.transform remains a no-op.

        :param x: the features array from the preprocessor, as constructed from the fit domain.
        :param y: the label of target values (if including range_nan, y should already be filtered to non-null rows).
        :param range_nan: a bool array indicating whether a given row index corresponds to a missing target value.
        :return: the features array to be passed to the estimator, after filtering rows that lack a corresponding
            target value.
        """
        logger.debug("filtering features")
        if range_nan is not None:
            assert len(x) == len(range_nan), "x and range_nan must have the same shape!"
            logger.debug(
                "filtering %d rows from feature array where target is missing",
                range_nan.sum(),
            )
            x = x[~range_nan.reshape(-1), :].copy()
            assert len(x) == len(y), "ruh-roh! these should match after filtering"
        return super().fit_transform(x, y=y)
