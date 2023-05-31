import inspect
import sklearn
import sys

import numpy as np

from time_series_models.constants import DATE_TIME
from time_series_models.data_fetchers.eia_balancing_area_fetcher import (
    EiaBalancingAreaFetcher,
)
from time_series_models.estimators import Regressor
from time_series_models.processes import Process
from time_series_models.time_series_models import (
    RegularTimeSeriesModel,
)


class DummyDataFrameEstimator(sklearn.base.BaseEstimator, Regressor):
    """
    No-op fit estimator which implements predict by returning values from an existing dataframe.
    """

    def __init__(self, customer_data, forecast_column):
        self.customer_data = customer_data
        self.forecast_column = forecast_column

    def fit(self, X, y, **kwargs):
        """
        No-op fit method
        :param X:
        :param y:
        :return:
        """
        return self

    def predict(self, X):
        return (
            self.customer_data.reindex(
                index=X[DATE_TIME].reshape(-1), columns=[self.forecast_column]
            )
            .to_numpy()
            .astype("float")
        )


class DummyDataModel(Process, Regressor, RegularTimeSeriesModel):
    """
    Model like object backed by external estimates for use with metrics and visualization
    """

    ACTUAL_COLUMN = None
    FORECAST_COLUMN = None

    def __init__(self, time_step, data_frame):
        self.data_frame = data_frame
        super(DummyDataModel, self).__init__(time_step=time_step)

    def make_model(self):
        return DummyDataFrameEstimator(
            customer_data=self.data_frame, forecast_column=self.FORECAST_COLUMN
        )

    def get_range(self, domain):
        return (
            self.data_frame.reindex(
                index=domain[DATE_TIME].reshape(-1), columns=[self.ACTUAL_COLUMN]
            )
            .to_numpy()
            .astype("float")
        )

    def make_preprocessor(self, **kwargs):
        pass

    def make_estimator(self, **kwargs):
        pass


class EiaBalancingAreaModel(DummyDataModel):
    """
    Model object backed by the EIA 1 day balancing area prediction excel file
    """

    ACTUAL_COLUMN = "Adjusted D"
    FORECAST_COLUMN = "DF"

    def __init__(self, domain_location):
        df = EiaBalancingAreaFetcher(
            location_mapper=lambda x: x,
        ).source_loader(None, None, domain_location)
        df.loc[df[self.ACTUAL_COLUMN] == 0] = np.nan
        super().__init__(
            np.timedelta64(1, "h"),
            data_frame=df,
        )


REGISTERED_CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}
