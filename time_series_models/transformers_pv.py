import logging
import os
import typing
from abc import abstractmethod

import numpy as np
import pandas as pd
import sklearn
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing

from time_series_models.config import ConfigHandler
from time_series_models.constants import LOCATION, DATE_TIME
from time_series_models.data_fetchers.hrrr_fetcher import (
    HrrrFetcher,
)
from time_series_models.pv_physical_model import (
    PVPhysicalModel,
)
from time_series_models.transformers import (
    ColumnTypeTransformer,
)

logger = logging.getLogger(__name__)


class PvModelBuilder:
    def __init__(self):
        self.resource_data = None

    @abstractmethod
    def get_resources(self, locations: typing.List[str]):
        """
        Get the resource data for all location
        :param locations: a list of locations to get resource data from
        :return:None
        """
        pass

    @abstractmethod
    def __call__(self, location: str) -> typing.Tuple[PVPhysicalModel, ...]:
        """
        Builds the PVPhyscialModels for a given location
        Concrete implementation use a data source, e.g. static config files to populate
        a particular PVPhysicalModel.
        :param location: the string domain location for which to build a PvModel
        :return: tuple of PVPhysicalModel objects for the given location (solar resource)
        """
        pass


CF_MAPPING = {
    "dswrf": "ghi",
    "vbdsf": "dni",
    "vddsf": "dhi",
    "t": "temp_air",
    "gust": "wind_speed",
    # TODO: replace gust with u/v component and calculate wind speed & direction. Add snow, albedo.
    # See authoritative mapping from SolarArbiter
    # https://github.com/SolarArbiter/solarforecastarbiter-core/blob/master/solarforecastarbiter/io/nwp.py#L17-L24
}


def pv_function(X, pv_model_builder: PvModelBuilder):
    """
    Takes a Numpy Custom data type array of shape [n, 1] and extracts fields for use in PV forecasting
    Field names are hard coded for now, mapping specific variables from HRRR to PvPhysicalModel parameters
    This could be extended with customizable mapping tool
    :param X: a numpy array to transform containing HRRR data.
    :param pv_model_builder: An instance of PvModelBuilder which when called returns a tuple of PvPhysicalModels
    :return: the PV production estimate for each location and time
    """
    logger.debug("Type: %s, Shape: %s", X.dtype, X.shape)
    # Use order preserving pandas unique!
    locations = pd.unique(X["domain"][LOCATION].reshape(-1))

    # Cache any data we need from the db
    pv_model_builder.get_resources(locations.tolist())

    results = []

    for location in locations:
        # Parallelize this?
        local_data = X[X["domain"][LOCATION] == location].reshape(-1, 1)

        data = np.hstack([local_data[key] for key in CF_MAPPING.keys()])
        index = local_data["domain"][DATE_TIME].squeeze()

        dataframe = pd.DataFrame(
            data=data,
            columns=[val for val in CF_MAPPING.values()],
            index=index,
        )
        dataframe.index.name = "date_time"

        logger.info("Feature DF: %s", dataframe)

        pv_systems = pv_model_builder(location)
        if len(pv_systems) > 1:
            # Make a copy - pvsystem may do bad things to the dataframe
            result_list = [
                pv_system.forecast(dataframe.copy()) for pv_system in pv_systems
            ]
            # stack & sum the results for all sites
            site_results = np.hstack(result_list).sum(axis=1, keepdims=True)
        elif len(pv_systems) == 1:
            site_results = pv_systems[0].forecast(dataframe)

        else:
            logger.warning(
                "PV Systems Builder %s returned no systems for location %s - returning nan forecast!",
                pv_model_builder,
                location,
            )

            site_results = np.full(local_data.shape, np.nan)

        results.append(site_results)

    # Resulting Vstack should align with the domain from the input.
    return np.vstack(results)


pv_function.get_feature_names = lambda: [
    "pv_physics",
]


def make_pv_pipeline(
    hrrr_fetcher: HrrrFetcher,
    pv_model_builder: PvModelBuilder,
) -> sklearn.pipeline.Pipeline:
    """
    Convenience method for PV Physical modeling pipeline.
    Use it where it is helpful, but if you want to use the HRRR data in additional contexts, just build your own pipeline.
    """

    transforms = [
        (
            "hrrr_domain",
            ColumnTypeTransformer(
                [("domain", "passthrough", [0]), ("hrrr_data", hrrr_fetcher, [0])]
            ),
        ),
        (
            "pv_forecast",
            sklearn.preprocessing.FunctionTransformer(
                func=pv_function, kw_args=dict(pv_model_builder=pv_model_builder)
            ),
        ),
    ]

    return sklearn.pipeline.Pipeline(transforms)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class ConfigBuilder(PvModelBuilder):
    def __init__(self, config_mapping: dict[str, list[str]]):
        super().__init__()
        self.config_mapping = config_mapping

    def get_resources(self, locations: typing.List[str]):
        # No op for the config builder. Caching is not needed.
        pass

    def load_model(self, path: str):
        abs_path = os.path.join(THIS_DIR, "../..", path)
        logger.info("Trying to load from: %s", abs_path)
        with open(abs_path, "r") as file:
            return ConfigHandler.decode(file.read())

    def __call__(self, location: str) -> typing.Tuple[PVPhysicalModel, ...]:
        return tuple(self.load_model(path) for path in self.config_mapping[location])
