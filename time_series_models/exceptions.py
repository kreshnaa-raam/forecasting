class TimeSeriesModelsException(Exception):
    """
    Base exception for all Time Series Models exceptions
    """


class EmptyFetchException(TimeSeriesModelsException):
    """
    Exception for handling completely empty result set in the fetcher
    This should be caught in the get_data method where an array of nans is returned instead.
    Missing data is handled (detected) at the highest level by monitoring. The forecast should never fail.
    """


class FetcherConfigurationError(TimeSeriesModelsException):
    """
    Exception for flagging a misconfigured Fetcher.
    """
