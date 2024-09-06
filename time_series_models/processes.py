import inspect
import logging
import sys

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Dict

import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing

import numpy as np
import pandas as pd
import typing

from time_series_models.constants import (
    DATE_TIME,
    LOCATION,
)
from time_series_models.data_fetchers.ami_fetcher import AmiFetcher
from time_series_models.data_fetchers.eia_balancing_area_fetcher import (
    EiaBalancingAreaFetcher,
)
from time_series_models.data_fetchers.hrrr_fetcher import (
    HrrrFetcher,
)
from time_series_models.data_fetchers.nan_fetcher import NanFetcher
from time_series_models.data_fetchers.numba_groupby_functions import (
    impute_sum,
)
from time_series_models.exceptions import FetcherConfigurationError
from time_series_models.time_series_models import Mixin
from time_series_models.transformers import (
    autoregressive_features_pipeline,
    get_data_pipeline,
    one_hot_location,
    overfetched_range_pipeline,
    net_energy_pipeline,
)
from time_series_models.transformers_calendar import (
    harmonic_transform_pipeline,
)
from time_series_models.transformers_pv import (
    make_pv_pipeline,
    ConfigBuilder,
)

from time_series_models.transformers_shortcuts import (
    append_business_day_transformer,
    append_day_of_week_transformer,
    append_harmonic_transformers,
)

from time_series_models.sklearn import sklearn_monkey_patch

# Make sure patches run in joblib workers!
sklearn_monkey_patch.apply_patches()

logger = logging.getLogger(__name__)


class Process(Mixin, ABC):
    """
    A process model which can be composed with estimators and filters to implement a timeseries model for the process.
    """

    @abstractmethod
    def get_range(self, domain):
        """

        :param domain: a TSM numpy domain vector of timestamp and location
        :return: a numpy vector of process values for the given domain
        """

    @abstractmethod
    def make_preprocessor(self, **kwargs):
        """
        Builds or wraps an sklearn step to create feature data for a given time domain
        :param kwargs: configuration arguments to parse and apply to build the model features
        :return: the sklearn pipeline step that creates feature data from the time domain, usually a column transformer
        """


class ExampleProcess(Process):
    def get_range(self, domain):
        rng = np.random.default_rng(seed=10_007)

        tdiff = (domain[DATE_TIME] - np.datetime64("1970-01-01")).astype("float")
        loc_map = {
            key: rng.standard_normal() for key in np.sort(np.unique(domain[LOCATION]))
        }

        location_scale = np.array([loc_map[k] for k in domain[LOCATION][:, 0]]).reshape(
            -1, 1
        )

        return location_scale * (
            rng.standard_normal() * np.sin(tdiff * 2.0 * np.pi / 24.0 + np.pi / 3.0)
            + rng.standard_normal() * np.cos(tdiff * 2 * np.pi / 168)
            + rng.standard_normal() * rng.standard_normal(tdiff.shape)
        )

    def make_preprocessor(self, **kwargs):
        return sklearn.compose.ColumnTransformer(
            transformers=[
                ("location", one_hot_location(handle_unknown="error"), [0]),
                (
                    "24hour",
                    harmonic_transform_pipeline(
                        np.timedelta64(24, "h"), np.datetime64("2000-01-01")
                    ),
                    [0],
                ),
                (
                    "168hour",
                    harmonic_transform_pipeline(
                        np.timedelta64(168, "h"), np.datetime64("2000-01-01")
                    ),
                    [0],
                ),
            ]
        )


class AmiHourlyForecast(Process):
    """
    Forecast AMI series directly when AMI ID is in the domain location
    """

    CUSTOM_KWARGS = [
        "buck_weather",  # TODO(Michael H): delete once we have a general weather source!
        "lags",
        "mapping",
        "met_vars",
        "met_horizon",
        "day_of_week",
        "business_day",
        "harmonics",
    ]

    def __init__(self, lags: list | np.ndarray = None, **kwargs):
        """
        :param lags: optional list or array of numpy timedelta64 defining the autoregressive features, if any.
        :param kwargs: optional keyword arguments for the model constructor
        """
        time_step = np.timedelta64(1, "h")

        # we don't want to interpolate missing labels -- these should be skipped during training and evaluation!
        self._energy_fetcher = AmiFetcher(
            location_mapper=lambda x: x,
            units="energy",
        )
        self._net_energy_pipeline = net_energy_pipeline(self.energy_fetcher)
        self._overfetched_range_pipeline = overfetched_range_pipeline(
            self.net_energy_pipeline,
            lags,
            time_step,
            **kwargs,
        )
        self._weather_fetcher_name = None

        logger.info("Instantiating RegularTimeSeriesModels with kwargs %s", kwargs)
        super().__init__(time_step, lags=lags, **kwargs)

    @property
    def get_range_pipeline(self):
        return self._overfetched_range_pipeline

    @property
    def energy_fetcher(self):
        return self._energy_fetcher

    @property
    def _weather_fetcher(self):
        """
        Use this accessor only if you really need to access the 'transformers' attribute of ColumnTransformer rather
        than the 'named_transformers_' attribute!
        :return: the unfitted weather transformer in the feature_builder ColumnTransformer.
        """
        # look for the *unfitted* weather transformer by name
        for transformer in self.feature_builder.transformers:
            # each "transformer" is actually a tuple of (name, transformer, columns_list)
            if transformer[0] == self._weather_fetcher_name:
                # we found a match! return the actual transformer
                return transformer[1]
        # otherwise, it doesn't exist
        return None

    @property
    def net_energy_pipeline(self):
        return self._net_energy_pipeline

    def domain(self, start, stop, *locations):
        if self._weather_fetcher is not None:
            # get the previous mapping from the fetcher, update with any new locations, and push back to fetcher;
            # this ensures the fetcher will have the latest possible mapping prior to any parallel processes being
            # spawned, and still make use of the fetcher's memoization of the location mapping.
            logger.debug(
                "updating %s weather_fetcher location mapping and pushing any changes to the weather fetcher!",
                self,
            )
            self._weather_fetcher.update_mapping(*locations)
            logger.debug(
                "HrrrFetcher location mapping now %s",
                self._weather_fetcher._location_mapping,
            )
        return super().domain(start, stop, *locations)

    def get_range(self, domain):
        logger.debug("AmiHourlyForecast.get_range calling get_range_pipeline")
        return self.get_range_pipeline.fit_transform(domain)

    def make_preprocessor(
        self,
        lags: list | np.ndarray = None,
        met_vars: list = None,  # hrrr vars: t, 2r, etc.
        met_horizon: str = None,
        day_of_week: bool = False,
        business_day: bool = False,
        harmonics: list | np.ndarray = None,
        # mapping: typing.Callable | None = None,
        mapping: dict | None = None,
        **kwargs,
    ):
        logger.info("creating lags %s", lags)
        transformers = [
            (
                "correlated_load",
                autoregressive_features_pipeline(
                    self.net_energy_pipeline,
                    lags,
                    self.tstep,
                    **kwargs,
                ),
                [0],
            )
        ]

        if met_vars:
            self._weather_fetcher_name = "weather"
            # guess fetcher horizon from minimum lag --> use same forecast horizon
            if met_horizon is None:
                met_horizon = guess_horizon_from_lags(lags)
            transformers.append(
                (
                    self._weather_fetcher_name,
                    HrrrFetcher(
                        location_mapper="RESOURCE_LOOKUP",
                        location_mapping=mapping,
                        selector="select",
                        selector_args={"variables": met_vars},
                        source_mode=f"{met_horizon}_hour_horizon",
                        resource_type="meter/electrical",
                        resource_query="No-Op",
                    ),
                    [0],
                )
            )

        if day_of_week:
            append_day_of_week_transformer(transformers)

        if business_day:
            append_business_day_transformer(transformers)

        if harmonics is not None:
            append_harmonic_transformers(transformers, harmonics)

        logger.info("building preprocessor ColumnTransformer")
        return sklearn.compose.ColumnTransformer(
            transformers=transformers,
            n_jobs=kwargs.get("n_jobs"),
        )


class BalancingAreaHourly(Process):
    """
    The EIA data is UTC Time - not local time!

    Model for EIA balancing area data
    """

    CUSTOM_KWARGS = [
        "lags",
        "met_cities",
        "met_vars",
        "met_kwargs",
        "met_tz_shift",
        "published_native_forecast",
        "balancing_area_forecast",
        "day_of_week",
        "business_day",
        "harmonics",
    ]

    def __init__(self, time_step, **kwargs):
        # TODO drop the time_step argument?
        self._eia_balancing_area_fetcher = EiaBalancingAreaFetcher(
            location_mapper=lambda x: x,
            # Take just the Adjusted Demand variable for now...
            variables=["Adjusted D"],
        )
        super().__init__(time_step, **kwargs)

    @property
    def eia_balancing_area_fetcher(self):
        return self._eia_balancing_area_fetcher

    def get_range(self, domain):
        return self.eia_balancing_area_fetcher.get_data(domain)

    # @debug
    def make_preprocessor(
        self,
        lags=None,
        balancing_area_forecast=False,
        met_cities=None,
        met_vars=None,
        met_tz_shift=None,
        day_of_week=False,
        business_day=True,
        harmonics=None,
        **kwargs,
    ):
        # TODO how to share code with other models?
        if lags is None:
            lags = [np.timedelta64(24, "h"), np.timedelta64(168, "h")]

        transformers = [
            (
                "{}_correlated_load".format(lag),
                get_data_pipeline(self.eia_balancing_area_fetcher, shift=lag),
                [0],
            )
            for lag in lags
        ]

        if balancing_area_forecast:
            transformers.append(
                (
                    "published_balancing_area_forecast",
                    get_data_pipeline(
                        EiaBalancingAreaFetcher(
                            location_mapper=lambda x: x,
                            variables=["DF"],
                        ),
                        shift=None,
                    ),
                    [0],
                )
            )

        if day_of_week:
            append_day_of_week_transformer(transformers)

        if business_day:
            append_business_day_transformer(transformers)

        if harmonics is not None:
            append_harmonic_transformers(transformers, harmonics)

        return sklearn.compose.ColumnTransformer(transformers=transformers)


class PVForecast(Process):
    """
    TODO remove resource model and BT fetcher - follow the site_meter_mapping is None path
    Forecast process mixin for a set of PV Systems.
    The process can be used directly as a PV Physical model, wrapping the physics model as a single feature
    with the IdentityEstimator. It can also be used for ML with lagged correlations (and other exogenous data)
    in addition to the direct physics forecasts.

    Locations in the domain should be solar_distributed or solar_farm resource ids.

    Process output: energy produced in kWh. Negative values are production to the grid, positive values are consumption
    at the meter site
    See details of the PySam model output
    https://nrel-pysam.readthedocs.io/en/master/modules/Pvsamv1.html#PySAM.Pvsamv1.Pvsamv1.Outputs.gen

    TODO: Figure out how to be clear and concise about units and signs.
    This process can be used with AMI energy or Inverter Power - must align with time interval and unit of the pv model
    """

    # static constants used to define how the domain location relates to resources for range, location and
    # pv configuration
    AMI_METER_RESOURCE_LOOKUP = "AMI_METER_RESOURCE_LOOKUP"
    RESOURCE_SOLAR_DISTRIBUTED = "solar/distributed"
    RESOURCE_SOLAR_FARM = "solar/farm"

    CUSTOM_KWARGS = [
        "lags",
        "site_config_mapping",
        "site_meter_mapping",
        "site_latlong_mapping",
        "met_vars",
        "met_horizon",
    ]

    # TODO add day_of_week business_day harmonics

    def __init__(
        self,
        lags: list | np.ndarray,
        site_config_mapping: dict[str, list[str]],
        site_latlong_mapping: Dict[str, Dict[str, float]],
        source_mode="48_hour_forecast",
        **kwargs,
    ):
        """
        The domain location should be a solar resource ID.

        The location is used at transform time (during fit/predict) to build pysam models. Use a static config
        pointing to one or more json config files.

        PySam models take HRRR data as an input.


        The additional argument determine (if) how to read the telemetry
        :param lags: array of numpy timedelta lagged correlations to apply for ML
        :param site_config_mapping: controls the way PvPhysical Models are constructed for each loaction
        :param site_latlong_mapping: controls how the hrrr fetcher gets the lat/lon for each location
        :param source_mode: the HRRR fetcher mode one of 18_hour_forecast, 48_hour_forecast, xx_hour_horizon or xx-yy_hour_horizon
        :param kwargs: additional args passed to the estimator
        """

        # TODO: hard code the time step for now. How to implement fetcher group select for 15 min tstep?
        time_step = np.timedelta64(1, "h")

        # Determine how to fetch the range based on init args
        # Optimization hack - don't bother trying to fetch the range when there is no mapping
        # Need to revisit once we have a mapping for some but not most PV systems
        # Set all the properties to point to the nan fetcher.
        self._range_fetcher = NanFetcher(None, None, None)
        self._overfetched_range_pipeline = (
            self._range_fetcher  # used in get_range and friends
        )
        self._site_config_mapping = site_config_mapping
        self._pv_builder = ConfigBuilder(config_mapping=site_config_mapping)

        self._site_latlong_mapping = site_latlong_mapping
        self._hrrr_fetcher = HrrrFetcher(
            # Pass an explicit mapping from the domain location to the lat lon of interest
            location_mapper=lambda site: site_latlong_mapping[site],
            selector="select",
            selector_args=dict(
                variables=["valid_time", "dswrf", "vbdsf", "vddsf", "t", "gust"],
            ),
            source_mode=source_mode,
        )

        super().__init__(np.timedelta64(1, "h"), lags=lags, **kwargs)

    def domain(self, start, stop, *locations):
        """
        Override domain to force update the fetcher location mapping...
        Get the previous mapping from the fetcher, update with any new locations, and push back to fetcher;
        this ensures the fetcher will have the latest possible mapping prior to any parallel processes being
        spawned, and still make use of the fetcher's memoization of the location mapping.

        :param start: Numpy datetime64 start datetime
        :param stop: Numpy datetime64 stop datetime
        :param locations: list of locations to forecast
        :return: the time series model domain a numpy custom type
        """
        if self.hrrr_fetcher._use_resource_lookup:
            logger.debug(
                "updating %s hrrr_fetcher location mapping and pushing any changes to the hrrr fetcher!",
                self,
            )
            self.hrrr_fetcher.update_mapping(*locations)
            logger.debug(
                "HrrrFetcher location mapping now %s",
                self.hrrr_fetcher._location_mapping,
            )
        return super().domain(start, stop, *locations)

    @property
    def get_range_pipeline(self):
        return self._overfetched_range_pipeline

    @property
    def site_config_mapping(self):
        return self._site_config_mapping

    @property
    def site_latlong_mapping(self):
        return self._site_latlong_mapping

    @property
    def hrrr_fetcher(self):
        return self._hrrr_fetcher

    @property
    def range_fetcher(self):
        return self._range_fetcher

    def get_range(self, domain):
        logger.debug("PVForecast.get_range calling get_range_pipeline")
        return self.get_range_pipeline.fit_transform(domain)

    def make_preprocessor(self, lags=None, **kwargs):
        # TODO: Fix feature monitor for HRRR input data
        # AttributeError: Function Transformer FunctionTransformer(func=<function drop_first_column at 0x7f08970cbbe0>) or its function <function drop_first_column at 0x7f08970cbbe0> must implement get_feature_names
        # monitored_fetcher = monitor_fetcher(
        #    self.hrrr_fetcher, ForecastDataMonitor(use_locs=True)
        # )

        # TODO: move the hrrr fetcher out so it can be shared... using hrrr data as a direct feature
        transformers = [
            (
                "pv_physical",
                # make_pv_pipeline(monitored_fetcher, self._pv_builder),
                make_pv_pipeline(self.hrrr_fetcher, self._pv_builder),
                [0],
            ),
            # TODO: add lagged correlation features for ML
            # (
            #     "correlated_generation",
            #     autoregressive_features_pipeline(
            #         self.range_fetcher,
            #         lags,
            #         self.tstep,
            #         **kwargs,
            #     ),
            #     [0],
            # ),
        ]

        # TODO add harmonics and other features for ML

        return sklearn.compose.ColumnTransformer(
            transformers=transformers,
            n_jobs=kwargs.get("n_jobs"),
        )


def guess_horizon_from_lags(lags):
    # use forecast horizon equivalent to min(lags)
    min_lag = min(lags).astype("timedelta64[h]").astype(int)
    assert min_lag <= 48
    if min_lag <= 18:
        return f"{min_lag:02d}"
    elif 19 <= min_lag <= 24:
        return "19-24"
    elif 25 <= min_lag <= 30:
        return "25-30"
    elif 31 <= min_lag <= 36:
        return "31-36"
    elif 37 <= min_lag <= 42:
        return "37-42"
    elif 43 <= min_lag <= 48:
        return "43-48"


REGISTERED_CLASSES = {
    name: cls
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}
