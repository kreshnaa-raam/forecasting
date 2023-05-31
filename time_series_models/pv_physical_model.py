import logging
import numpy as np
from abc import ABC, abstractmethod

# pvlib imports
from pvlib import pvsystem
import pvlib.location
import pvlib.modelchain

# pysam imports
import PySAM.Pvsamv1
import PySAM.Pvwattsv8

logger = logging.getLogger(__name__)


class PVPhysicalModel(ABC):
    def __init__(self, name, system_model, latitude, longitude):
        """
        Physical model for a PV System
        :param name: system name which can be referenced from within any pipeline
        :param system_model: a model which is able to forecast pv production based on weather time series input.
        :param latitude: the latitude in decimal degrees
        :param longitude: the longitude in decimal degrees [-180 to 180]
        """
        self._name = name
        self._system_model = system_model
        self._latitude = latitude
        self._longitude = longitude

    @property
    def name(self):
        return self._name

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def system_model(self):
        return self._system_model

    @abstractmethod
    def forecast(self, dataframe):
        """
        Forecasts pv based on meteorological data and other kwargs
        :param met_data: dataframe with columns ["dni", "dhi", "ghi", "temp_air", "wind_speed"] and datatime index
        :return: nx1 np.array with forecasted pv value, with units [W]
        """
        pass

    def as_json(self):
        # Order matters - must be init arg ordering
        return dict(
            name=self.name,
            exported=self.system_model.export(),
            latitude=self._latitude,
            longitude=self._longitude,
        )


class PvLibPhysicalModel(PVPhysicalModel):
    def __init__(self, name, system_model, latitude, longitude):
        super().__init__(name, system_model, latitude, longitude)
        if not isinstance(system_model, pvsystem.PVSystem):
            raise Exception(
                f"Expect system_model of type pvsystem.PVSystem. Got {type(system_model)}",
            )

    def forecast(self, dataframe):
        """
        Forecasts pv based on meteorological data
        :param  dataframe: dataframe with columns ["dni", "dhi", "ghi", "temp_air", "wind_speed"] and datatime index
        :return: pv forecast as mx1 array, with units [W]
        Generation appears negative in PvLib, consistent with Camus Canonical form!
        """
        logger.debug("%s forecasting pv_system: %s ... ", self, self.name)
        model_geo_location = pvlib.location.Location(
            latitude=self._latitude, longitude=self._longitude
        )
        pv_model_chain = pvlib.modelchain.ModelChain(
            self.system_model,
            model_geo_location,
            aoi_model="no_loss",
            spectral_model="no_loss",
        )

        # TODO: handle missing null inputs

        pv_model_chain.run_model(dataframe)

        # Convert kW -> W
        generated = np.array([i * 1000 for i in pv_model_chain.results.ac.values])
        return generated.reshape(-1, 1)


class PySamPhysicalModel(PVPhysicalModel, ABC):
    """
    Interface to NREL's PySAM modeling framework
    https://nrel-pysam.readthedocs.io/
    """

    def __init__(
        self, name, system_model, latitude, longitude, elevation=0, albedo=0.4
    ):
        """
        Initialize model interface adding two pysam particular parameters
        :param name: system name which can be referenced from within any pipeline
        :param system_model: a model which is able to forecast pv production based on weather time series input.
        :param latitude: the latitude in decimal degrees
        :param longitude: the longitude in decimal degrees [-180 to 180]
        :param elevation: the altitude in meters
        :param albedo: the surface albedo [0..1]
        """
        super().__init__(name, system_model, latitude, longitude)
        self._elevation = elevation
        self._albedo = albedo

    def as_json(self):
        # Order matters - must be init arg ordering
        return super().as_json() | dict(elevation=self._elevation, albedo=self._albedo)

    @classmethod
    @abstractmethod
    def create(cls, name, config, latitude, longitude, elevation=0, albedo=0.4):
        """
        From a configuration (not the same as the exported form) create an instance with the configured values
        :param name:
        :param config:
        :return:
        """
        pass

    @classmethod
    @abstractmethod
    def create_new_pysam(cls):
        """
        Create an instance of the PySam model object with default values
        :return:
        """
        pass

    @classmethod
    def from_exported(
        cls, name, exported, latitude, longitude, elevation=0, albedo=0.4
    ):
        """
        Reconstructs an instance from a config created by the 'export' function
        :param exported: a dictionary created by the export function
        """
        model = cls.create_new_pysam()
        model.assign(exported)
        return cls(name, model, latitude, longitude, elevation, albedo)

    def __reduce__(self):
        return (
            # See link in doc string for api details
            self.from_exported,
            tuple(self.as_json().values()),
        )

    @abstractmethod
    def build_solar_resource_data(self, dataframe) -> dict:
        """
        Builds a dictionary of parameters including the time series as lists
        :param dataframe:
        :return:
        """

    def forecast(self, dataframe):
        """
        Forecasts pv generation based on meteorological data and other kwargs.
        Generation is negative, per Camus canonical standards.
        :param dataframe: dataframe with columns ["dni", "dhi", "ghi", "temp_air", "wind_speed"] and datatime index
        :return: pv forecast as mx1 array, with units [W]
        """

        logger.debug("forecasting pv_system: %s ... ", self.name)

        # some pre-processing
        dataframe.loc[dataframe.dni < 0, "dni"] = 0.0
        # todo: pysam doesn't do well with high dni. Estimate of 1200 is based on
        #  trial and error. Can we generalize?
        dataframe.loc[dataframe.dni > 1200, "dni"] = 1200.0
        dataframe.loc[dataframe.dhi < 0, "dhi"] = 0.0
        dataframe.loc[dataframe.ghi < 0, "ghi"] = 0.0
        dataframe.temp_air -= 273.15  # Convert to Kelvin

        all_valid = dataframe.notna().all(axis=1)

        inputs = self.build_solar_resource_data(dataframe)

        try:  # solve!
            self.system_model.SolarResource.assign({"solar_resource_data": inputs})
            self.system_model.execute()
            # Convert kW -> W https://nrel-pysam.readthedocs.io/en/master/modules/Pvsamv1.html#PySAM.Pvsamv1.Pvsamv1.Outputs.gen
            # Converts generation to be negative, consistent with Camus Canonical model
            generated = np.array(self.system_model.Outputs.gen) * -1000.0
            # if any input to the physical model was missing, replace the output with nan
            return np.where(all_valid, generated, np.nan).reshape(-1, 1)
        except Exception as e:
            logger.error(
                "Exception while solving for pv system %s: %s",
                self.name,
                str(e).replace("\n", " "),
            )
            return np.full((len(dataframe), 1), np.nan, dtype=float)


class PvSamV1PhysicalModel(PySamPhysicalModel):
    def __init__(
        self, name, system_model, latitude, longitude, elevation=0, albedo=0.4
    ):
        """
        Built on pysam's pvsamv1 modeling framework. Link to documentation:
        https://nrel-pysam.readthedocs.io/en/master/modules/Pvsamv1.html
        """
        super().__init__(name, system_model, latitude, longitude, elevation, albedo)
        if not isinstance(system_model, PySAM.Pvsamv1.Pvsamv1):
            raise ValueError(f"Unexpected type: {type(system_model)}")

    @classmethod
    def create(cls, name, config, latitude, longitude, elevation=0, albedo=0.4):
        """
        Makes instance from the config
        :param config: a dictionary used to construct a model.
        """
        model = cls.create_new_pysam()
        for k, v in config.items():
            model.value(k, v)
        return cls(name, model, latitude, longitude, elevation, albedo)

    @classmethod
    def create_new_pysam(cls):
        return PySAM.Pvsamv1.new()

    def build_solar_resource_data(self, dataframe) -> dict:
        """
        Builds a dictionary of parameters including the time series as lists
        :param dataframe:
        :return:
        """

        # assign values to the system
        # https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html#PySAM.Pvsamv1.Pvsamv1.SolarResource.solar_resource_data
        # https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html#PySAM.Pvwattsv8.Pvwattsv8.SolarResource.solar_resource_data
        return {
            "dn": dataframe.dni.fillna(0.0).to_list(),
            "df": dataframe.dhi.fillna(0.0).to_list(),
            "gh": dataframe.ghi.fillna(0.0).to_list(),
            "tdry": dataframe.temp_air.fillna(273.15).to_list(),
            "wspd": dataframe.wind_speed.fillna(0.0).to_list(),
            "lat": self._latitude,
            "lon": self._longitude,
            "tz": 0,  # pysam loves GMT?
            "elev": self._elevation,
            "Year": dataframe.index.year.tolist(),
            "Month": dataframe.index.month.tolist(),
            "Day": dataframe.index.day.tolist(),
            "Hour": dataframe.index.hour.tolist(),
            "Minute": dataframe.index.minute.tolist(),
            "albedo": self._albedo,
        }


class PvWattsV8PhysicalModel(PySamPhysicalModel):
    def __init__(
        self, name, system_model, latitude, longitude, elevation=0, albedo=0.4
    ):
        """
        Built on pysam's Pvwattsv8 modeling framework. Link to documentation:
        https://nrel-pysam.readthedocs.io/en/master/modules/Pvwattsv8.html
        """
        super().__init__(name, system_model, latitude, longitude, elevation, albedo)
        if not isinstance(system_model, PySAM.Pvwattsv8.Pvwattsv8):
            raise ValueError(f"Unexpected type: {type(system_model)}")

    @classmethod
    def create(cls, name, config, latitude, longitude, elevation=0, albedo=0.4):
        """
        Makes instance from the config
        :param config: a dictionary used to construct a model.
        """
        # start hack
        # Pvwattsv8 segfaults the bazel unit test when we try to execute Pvwattsv8.default("PVWattsNone")
        # so here is the dumped config from an instance.export() in colab:
        default_config = {
            "SolarResource": {"albedo": (0.2,), "use_wf_albedo": 1.0},
            "Lifetime": {},
            "SystemDesign": {
                "array_type": 1.0,
                "azimuth": 180.0,
                "batt_simple_enable": 0.0,
                "bifaciality": 0.0,
                "dc_ac_ratio": 1.15,
                "en_snowloss": 0.0,
                "gcr": 0.3,
                "inv_eff": 96.0,
                "losses": 14.0757,
                "module_type": 0.0,
                "soiling": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "system_capacity": 6.4,
                "tilt": 20.0,
            },
            "AdjustmentFactors": {"constant": 0.0},
            "Outputs": {},
        }
        model = cls.create_new_pysam()
        # mash the default_config into the new instance... piece by piece
        model.SolarResource.assign(default_config["SolarResource"])
        [model.value(k, v) for k, v in default_config["SystemDesign"].items()]
        model.AdjustmentFactors.assign(default_config["AdjustmentFactors"])
        # end hack

        for k, v in config.items():
            model.value(k, v)
        return cls(name, model, latitude, longitude, elevation, albedo)

    @classmethod
    def create_new_pysam(cls):
        return PySAM.Pvwattsv8.new()

    def build_solar_resource_data(self, dataframe) -> dict:
        """
        Builds a dictionary of parameters including the time series as lists
        :param dataframe:
        :return:
        """

        # assign values to the system
        # https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html#PySAM.Pvwattsv8.Pvwattsv8.SolarResource.solar_resource_data
        return {
            "dn": dataframe.dni.fillna(0.0).to_list(),
            "df": dataframe.dhi.fillna(0.0).to_list(),
            "tdry": dataframe.temp_air.fillna(273.15).to_list(),
            "wspd": dataframe.wind_speed.fillna(0.0).to_list(),
            "lat": self._latitude,
            "lon": self._longitude,
            "tz": 0,  # pysam loves GMT?
            "elev": self._elevation,
            "Year": dataframe.index.year.tolist(),
            "Month": dataframe.index.month.tolist(),
            "Day": dataframe.index.day.tolist(),
            "Hour": dataframe.index.hour.tolist(),
            "Minute": dataframe.index.minute.tolist(),
        }
