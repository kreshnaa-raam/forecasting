from typing import Any

import numba.core
import numba.typed
import numpy as np
from time_series_models.decorators import function_string_binder


def hour_diff(start, end):
    def _hour_diff(values, index) -> float:
        if values.shape[0] == 24:
            off_set = 0
        elif values.shape[0] == 23:
            # spring ahead
            off_set = -1
        elif values.shape[0] == 25:
            # fall back
            off_set = +1
        else:
            return np.nan
        return values[(start + off_set)] - values[(end + off_set)]

    function_string_binder(_hour_diff, "hour_diff_{}_to_{}".format(start, end))

    # numba can only jit functions, so to enable serialization, define serialization methods and
    # attach them to the function object
    def to_dict() -> dict[str, Any]:
        return {
            "__camus_json_type__": "numba_groupby_hour_diff",
            "__camus_json_data__": [start, end],
        }

    _hour_diff.to_dict = to_dict
    return _hour_diff


def hour_pick(hour):
    def _hour_pick(values, index) -> float:
        if values.shape[0] == 24:
            hour_index = hour
        elif values.shape[0] == 23:
            hour_index = hour - 1
        elif values.shape[0] == 25:
            # Assumes hour is after 2am!
            hour_index = hour + 1
        else:
            return np.nan

        return values[hour_index]

    # Play nice with feature names
    function_string_binder(_hour_pick, name="hour_pick_{}".format(hour))

    # numba can only jit functions, so to enable serialization, define serialization methods and
    # attach them to the function object
    def to_dict() -> dict[str, Any]:
        return {
            "__camus_json_type__": "numba_groupby_hour_pick",
            "__camus_json_data__": hour,
        }

    _hour_pick.to_dict = to_dict
    return _hour_pick


def hour_sum(start, end):
    def _hour_sum(values, index) -> float:
        if values.shape[0] == 24:
            off_set = 0
        elif values.shape[0] == 23:
            # spring ahead
            off_set = -1
        elif values.shape[0] == 25:
            # fall back
            off_set = +1
        else:
            return np.nan

        return np.nansum(values[(start + off_set) : (end + off_set + 1)])

    function_string_binder(_hour_sum, "hour_sum_{}_to_{}".format(start, end))

    def to_dict() -> dict[str, Any]:
        return {
            "__camus_json_type__": "numba_groupby_hour_sum",
            "__camus_json_data__": [start, end],
        }

    _hour_sum.to_dict = to_dict

    return _hour_sum


def nanmax_index_diff():
    """
    Function to take the nanmax of the group and subtract the index from the result
    """

    def _nanmax_index_diff(values, index) -> float:
        return np.nanmax(values) - np.arange(values.shape[0]).reshape(values.shape)

    # Play nice with feature names
    function_string_binder(_nanmax_index_diff, "nanmax_hour_diff")
    return _nanmax_index_diff


def abs_nanmax_index_diff():
    """
    Function to take the nanmax of the group and subtract the index from the result
    Then take absolute value, to produce an inflection point
    """

    def _abs_nanmax_index_diff(values, index) -> float:
        return np.abs(
            np.nanmax(values) - np.arange(values.shape[0]).reshape(values.shape)
        )

    # Play nice with feature names
    function_string_binder(_abs_nanmax_index_diff, "abs_nanmax_hour_diff")
    return _abs_nanmax_index_diff


def nanmode():
    """
    Function to take the mode of the group, ignoring any NaN values.
    """

    def _nanmode(values: np.array, index) -> float:
        arr = values[~np.isnan(values)]  # exclude the null elements
        if len(arr) == 0:  # shortcut out
            return np.nan

        d = numba.typed.Dict.empty(
            key_type=numba.core.types.float64,
            value_type=numba.core.types.int64,
        )
        # enumerate unique values
        for val in arr:
            if val in d:
                d[val] += 1
            else:
                d[val] = 1

        # find the most common value
        max_count = 0
        for k, v in d.items():
            # in case of tie, this selects for first occurrence; to select for last, use >=
            if v > max_count:
                max_count = v
                mode = k

        return mode

    # Play nice with feature names
    function_string_binder(_nanmode, "nanmode")
    return _nanmode


def impute_sum(n_expected: int = None, all_nan_fillval=np.nan):
    """
    Summing groupby operation. If one or more values on an interval has been observed and the rest are missing,
    mean-impute the missings before calculating the sum.
    :param n_expected: the number of observations expected per grouping interval. Leave as None (default) if this
    quantity can be inferred from the data (e.g. a data source that returns np.nan instead of None).
    :param all_nan_fillval: value to return/fill when all input values are nan
    """

    def _impute_sum(values, index) -> float:
        n_expected_ = n_expected or len(values)
        return np.nanmean(values) * n_expected_

    # Play nice with feature names
    function_string_binder(_impute_sum, "impute_sum")

    def _impute_sum_filled(values, index) -> float:
        n_expected_ = n_expected or len(values)
        result = np.nanmean(values) * n_expected_
        if np.isnan(result):
            return all_nan_fillval
        else:
            return result

    # Play nice with feature names
    function_string_binder(_impute_sum_filled, "impute_sum_filled")

    if np.isnan(all_nan_fillval):
        return _impute_sum
    else:
        return _impute_sum_filled
