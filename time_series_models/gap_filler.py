import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def backfill_singleton(df: pd.DataFrame):
    """
    Backfill null values in df by linear interpolation where there is only one missing value at a time.
    This function does not interpolate trailing nulls. If that is the desired outcome,
    just use `pd.DataFrame.interpolate()`
    :param df: the pandas DataFrame to be backfilled
    :return: a pandas DataFrame with singleton nulls interpolated, and only runs of two or more nulls remaining
    """
    no_fill = df.isna() & df.shift(1).isna()
    no_fill = no_fill | no_fill.shift(-1)
    full_interpolate = df.interpolate(limit_direction="backward", limit=1)
    df = df.where(no_fill, full_interpolate)
    df.iloc[:2] = df.iloc[:2].interpolate(limit_direction="backward", limit=1)
    return df


def get_tstep(array):
    """
    Identify the interval between the first two observations in an array.
    :param array:
    :return:
    """
    if len(array.shape) > 1:
        raise ValueError
    tstep = array[1] - array[0]
    # TODO: the length check chokes on numpy datetimes! consider expanding compatibility
    check = pd.date_range(start=array[0], end=array[-1], freq=tstep)
    if len(array) != len(check):
        warnings.warn("detected an irregularly spaced datetime array!", RuntimeWarning)
    return tstep


def return_last_observation(arr, **kwargs):
    """return the last non-null value seen in the array. A.k.a. "naive" or "persistence" algorithm.
    param kwargs is not used but needed for call signature compatibility"""
    if len(arr.shape) > 1:
        raise RuntimeError(
            f"'return_last_observation' method is only implemented for 1D arrays, but got shape {arr.shape}"
        )
    if np.isnan(arr).all():
        return np.nan
    else:
        return arr[~np.isnan(arr)][-1]


class GapFiller:
    def __init__(self, data_with_gaps: pd.DataFrame):
        self._original = data_with_gaps
        self.models = {}

    @property
    def original(self):
        return self._original.copy()

    @classmethod
    def fill_gaps_rolling_1d(
        cls,
        time_series: pd.Series,
        fill_method: callable = return_last_observation,
        fit_len: int = 1,
        **fit_kwargs,
    ) -> pd.Series:
        """
        Roll fill_method through the time_series, filling gaps as they are encountered. Potentially reuses predictions
        as input for subsequent calls to fill_method, depending on fit_len
        :param time_series: the time series with gaps to fill
        :param fill_method: any function that can be called on an array
        :param fit_len: the size of the rolling window of time_series that is passed to fill_method
        :param fit_kwargs: keyword arguments for fill_method
        :return: a copy of time_series with all gaps filled using fill_method
        """
        series = time_series.copy()
        tstep = get_tstep(series.index)
        if fit_kwargs is not None:
            fit_kwargs["fit_len"] = fit_len
        else:
            fit_kwargs = {"fit_len": fit_len}
        logger.debug("filling gaps using <%s>", fill_method)
        for idx in series.loc[series.isna()].index:
            # predict gaps with the specified fill method
            logger.debug("filling gap at <%s>", idx)
            fit_start = max(idx - tstep * (fit_len + 1), series.index[0])
            series.loc[idx] = fill_method(
                series.loc[fit_start : idx - tstep],
                **fit_kwargs,
            )
        return series

    @classmethod
    def pre_interpolated(cls, dataframe: pd.DataFrame):
        return GapFiller(backfill_singleton(dataframe))

    def fill_with_interpolation(self, limit_direction="both", **kwargs):
        model_str = "interpolation"
        logger.info("filling with %s, with kwargs %s", model_str, kwargs)
        return self.original.interpolate(limit_direction=limit_direction, **kwargs)

    def fill_with_rolling_series_method(
        self,
        method: callable,
        fit_len: int,
        **kwargs,
    ) -> pd.DataFrame:
        model_str = f"{str(method)} with fit_len={fit_len}"
        logger.info("filling with %s, with kwargs %s", model_str, kwargs)

        predictions = []
        for col in self.original.columns:
            predictions.append(
                GapFiller.fill_gaps_rolling_1d(
                    self.original[col], method, fit_len=fit_len, **kwargs
                )
            )
        return pd.concat(predictions, axis=1)
