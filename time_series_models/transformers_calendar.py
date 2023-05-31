import holidays
import sklearn

import numpy as np

from time_series_models.constants import DATE_TIME
from time_series_models.decorators import feature_names
from time_series_models.transformers import split_domain
from time_series_models.time_unit import TimeUnitEnum


def delta_value(x, zero_value=None):
    """
    Subtract a zero value of a numpy array to convert to a series relative value.
    For a time series this will change the type from numpy.timedelta64 to np.double.
    The units (M, D, h, m, s) will be preserved.
    If zero value is not provided, the first value of the array will be used.
    The zero value must be provided for use with harmonic transform!
    :param x: the input time series
    :param zero_value: The value to subtract from the array.
    :return: the resulting relative series
    """
    if zero_value:
        return x - zero_value
    else:
        return x - x[0]


@feature_names(["sin", "cos"])
def harmonic(x, period):
    """
    For a given [n,m] array calculate the sin and cos of the array divided by the period
    # TODO feature names will not work unless m is 1 in the [n,m] array
    :param x: the time domain array
    :param period: a numpy timedelta64
    :return: the harmonic sin and cos series
    """
    return np.concatenate(
        (np.sin(x * 2 * np.pi / period), np.cos(x * 2 * np.pi / period)), 1
    )


def harmonic_transform_pipeline(period, zero_value):
    """
    Create a pipeline to create a circularized harmonic series
    :param period: the period
    :param zero_value: the zero value for the time domain harmonic fit
    :return: the pipeline to circularize the data
    """
    return sklearn.pipeline.Pipeline(
        [
            (
                "split",
                sklearn.preprocessing.FunctionTransformer(
                    func=split_domain, kw_args=dict(key=DATE_TIME)
                ),
            ),
            (
                "delta_value",
                sklearn.preprocessing.FunctionTransformer(
                    func=delta_value, kw_args=dict(zero_value=zero_value)
                ),
            ),
            (
                "harmonic",
                sklearn.preprocessing.FunctionTransformer(
                    func=harmonic, kw_args=dict(period=period)
                ),
            ),
        ]
    )


@feature_names(["business_day"])
def is_business_day(x):
    """
    Convert a datetime object to a day of week string
    :param x: the date time array
    :return: integer array: 1 is business day, 0 is weekend or holiday
    """
    us_holidays = holidays.UnitedStates()
    bdd = np.busdaycalendar(
        weekmask="1111100",
        holidays=us_holidays[
            np.datetime_as_string(x[0, 0]) : np.datetime_as_string(x[-1, 0])
        ],
    )
    # TODO Convert only if hours, minutes or seconds, not weeks, months or years?
    x_days = x.astype(np.datetime64(1, "D"))
    return np.is_busday(x_days, busdaycal=bdd).astype(int)


def is_business_day_pipeline(**kw_args):
    """
    Create a pipeline for integer (0,1) business day feature for the domain
    :param kw_args: na
    :return: the pipeline
    """
    return sklearn.pipeline.Pipeline(
        [
            (
                "split",
                sklearn.preprocessing.FunctionTransformer(
                    func=split_domain, kw_args={"key": DATE_TIME}
                ),
            ),
            (
                "business_day",
                sklearn.preprocessing.FunctionTransformer(func=is_business_day),
            ),
        ]
    )


def one_hot_encode_day_of_week_pipeline(**kw_args):
    """
    Create a pipeline to one hot encode the day of week for the given domain
    :param kw_args: arguments to pass to the OneHotEncoder
    :return: the pipeline
    """
    return sklearn.pipeline.Pipeline(
        [
            (
                "split",
                sklearn.preprocessing.FunctionTransformer(
                    func=split_domain, kw_args={"key": DATE_TIME}
                ),
            ),
            (
                "to_day_of_week",
                sklearn.preprocessing.FunctionTransformer(
                    func=TimeUnitEnum.DAY_OF_WEEK.as_unit
                ),
            ),
            ("one_hot_encode", sklearn.preprocessing.OneHotEncoder(**kw_args)),
        ]
    )
