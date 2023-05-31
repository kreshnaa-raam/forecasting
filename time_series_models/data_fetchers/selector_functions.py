import numpy as np


def diff(series):
    """
    Take the difference of the series dropping the first (nan) value in the result set
    :param series: a pandas Series
    :return: difference series without nan values
    """
    res = series.diff().dropna()
    if res.empty:
        return np.nan
    return res


class GroupHours:
    """
    A callable object used as a pandas groupby operation to select data for particular hour values.

    Super flexible - unbelievably slow!!!
    Use cases:
    * group data by day and get the difference between the value at 2pm and 8pm in each day-group
    * group data by week and get the mean value 4pm in each week-group
    Also provides a nice str name.
    """

    def __init__(self, hours, method):
        self.hours = hours
        self.method = method

    def __call__(self, data):
        selection = data.loc[data.index.hour.isin(self.hours)]
        return self.method(selection)

    def __str__(self):
        return "{}({}__{})".format(
            self.__class__.__name__,
            "_".join(str(x) for x in self.hours),
            self.method.__name__,
        )
