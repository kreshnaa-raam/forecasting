import logging
import typing

from abc import abstractmethod

import pandas as pd
import numpy as np
import xarray as xr

from time_series_models.data_fetchers.fetcher import MultiFetcher
from time_series_models.exceptions import FetcherConfigurationError
from time_series_models.transformers import multiindex_from_domain

logger = logging.getLogger(__name__)


class GriddedDataFetcher(MultiFetcher):
    """
    Fetcher intended to be generic, supporting any gridded data source.

    This could become a common base class or actually take the model path structure as an argument... decide when we
    have enough gridded data products to look for good patterns.
    """

    def __init__(
        self,
        location_mapper: typing.Callable | typing.Literal["RESOURCE_LOOKUP"],
        selector: typing.Literal["select", "group"],
        selector_args: dict,
        location_mapping: dict | None = None,
        resource_type: str = "meter/electrical",
        resource_query: str | None = None,
    ):
        self.resource_type = resource_type
        if location_mapper == "RESOURCE_LOOKUP" and resource_type is None:
            raise FetcherConfigurationError("A single resource type must be specified")

        super().__init__(
            location_mapper,
            selector,
            selector_args,
            location_mapping=location_mapping,
            resource_query=resource_query,
        )

    def map_locations(self, *locations) -> dict[str, dict[str, float]]:
        """
        Get location mapping for a set of locations by reading from a database.
        :param locations: an iterable of one or more locations
        :return: a dict of {location:dict(latitude={value1}, longitude={value2})}
        TODO: Replace the inner dict of lat/lon with a named tuple
        """
        query_result = self._query_db(*locations)
        mapping = pd.DataFrame(
            query_result, columns=["resource", "latitude", "longitude"]
        )
        mapping = mapping.set_index("resource").to_dict(orient="index")
        not_found = np.setdiff1d(locations, list(mapping.keys()))
        if len(not_found) > 0:
            logger.warning(
                "No records found for %i locations: %s %s",
                len(not_found),
                not_found[:5],
                "... (truncated)" * (len(not_found) > 5),
            )
        return mapping

    @abstractmethod
    def select_blob(self, start: np.datetime64, end: np.datetime64):
        pass

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def _load_from_source(
        self, blobs, latitude: xr.DataArray, longitude: xr.DataArray, start, end
    ) -> pd.DataFrame:
        pass

    def source_loader(self, start, end, *domain_locations):
        """
        Load data for one or more domain_location from start to end.
        The domain_location is first mapped to a lat/lon in the grid.

        Source loader does not call update_mapping when using Resource Lookup.
        Use the get_data api or explicitly call update_mapping before using source_loader api

        :param start: a numpy datetime64
        :param end: a numpy datetime64
        :param domain_location: the TSM location
        :return: a dataframe timeseries
        """
        missing = []

        def _location_mapper(x):
            # records missing values in the list from the outer scope
            try:
                return self._location_mapper(x)
            except KeyError:
                missing.append(x)
                return dict(latitude=np.nan, longitude=np.nan)

        # TODO: consider filtering at ColumnTransformer/Pipeline stage rather than directly in the Fetcher
        geo_locations = pd.Series(domain_locations, index=domain_locations).apply(
            _location_mapper
        )
        if missing:
            logger.warning(
                "Missing geo mapping for one or more domain locations: %s", missing
            )

        geo_locations = pd.DataFrame(
            geo_locations.tolist(),
            columns=["latitude", "longitude"],
            index=geo_locations.index,
        ).dropna(how="all", axis=0)
        # next step is a no-op unless _shift_longitude is implemented in a subclass
        self._shift_longitude(geo_locations)

        # todo: need to handle missing files ...
        blobs = self.select_blob(start, end)
        logger.debug("loading xarray from %s", blobs)

        # Hack to get all the hours of the end day for a groupby operation
        time_unit, time_step = np.datetime_data(start)
        if time_unit == "D":
            end = end + np.timedelta64(1, "D")

        # convert series to xarrays to utilize automatic array broadcasting
        lat_xr = xr.DataArray(
            geo_locations["latitude"], [("location", geo_locations.index)]
        )
        lon_xr = xr.DataArray(
            geo_locations["longitude"], [("location", geo_locations.index)]
        )

        return self._load_from_source(blobs, lat_xr, lon_xr, start, end)

    def select(self, dataframe, domain):
        """
        Reindex dataframe to tseries and to self.variables
        Override the MultiFetcher.select method because the data source location is a dict of lat/lon values,
        which blows up the identity mapper check.
        :param dataframe:
        :param tseries:
        :return: np.ndarray
        """
        logger.debug("executing GriddedDataFetcher.select")
        # multiindex version of Fetcher.select
        return dataframe.reindex(
            index=multiindex_from_domain(domain),
            columns=self.variables,
        ).astype(np.float64)

    def group(self, dataframe, tseries):
        # TODO(Michael H): reimplement this!
        raise NotImplementedError(
            "method 'group' not yet implemented for GriddedDataFetcher!"
        )

    def get_all_data(self, domain_location):
        raise NotImplementedError(
            "All data is not well defined for a gridded data product so this method has not been implemented"
        )

    def get_feature_names(self):
        """
        Based on the configuration build a list of feature names.
        TODO DRY this up
        :return: the feature names
        """
        selector_variables = {
            "select": self.selector_args.get("variables"),
            "group": [
                "{}_{}".format(key, agg.__str__())
                for key, aggregations in self.selector_args.get("kwargs", {})
                .get("func", {})
                .items()
                if key != "engine"
                for agg in aggregations or {}
            ],
        }

        return selector_variables[self.selector]

    @staticmethod
    def _shift_longitude(geo_locations: pd.DataFrame) -> None:
        """
        Hook to allow shifting longitude from a (0, 360) range to a (-180, 180) range.
        If implemented, this method should modify geo_locations inplace.
        :param geo_locations: the dataframe containing a column for longitude
        :return: None
        """
        pass

    def _query_db(self, *locations) -> list[typing.Tuple, ...]:
        raise NotImplementedError("DB-specific implementation not provided!")
