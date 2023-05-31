import json
import os
import typing
import unittest
import uuid
from pathlib import PurePosixPath

from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import pvlib
import xarray as xr

from time_series_models.data_fetchers.hrrr_fetcher import (
    HrrrFetcher,
    memory,
)
from time_series_models.pv_physical_model import (
    PvLibPhysicalModel,
    PvSamV1PhysicalModel,
    PvWattsV8PhysicalModel,
)
from time_series_models.transformers import (
    make_domain,
    multiindex_from_domain,
)
from time_series_models.transformers_pv import (
    make_pv_pipeline,
    PvModelBuilder,
    ConfigBuilder,
    CF_MAPPING,
)

LATITUDE = 39.41655
LONGITUDE = -107.10143
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_xarray_dataset():
    valid_times = np.arange(
        np.datetime64("2022-07-01T00:00:00"),
        np.datetime64("2022-07-02T00:00:00"),
        step=np.timedelta64(1, "h"),
    )
    ghi = np.random.randn(2, 2, 24)
    dni = np.random.rand(2, 2, 24)
    dhi = np.random.rand(2, 2, 24)
    temperature = np.random.rand(2, 2, 24)
    wind_speed = np.zeros(shape=(2, 2, 24))

    for i in range(24):
        contribution = np.sin(np.pi * i / 24.0)
        temperature[:, :, i] = 288 + 10 * contribution
        dni[:, :, i] = 300 * contribution
        dhi[:, :, i] = 100 * contribution
        ghi[:, :, i] = 50 * contribution

    lats, lons = np.meshgrid(range(39, 41), range(-108, -106), indexing="ij")

    ds = xr.Dataset(
        data_vars=dict(
            dswrf=(["y", "x", "valid_time"], dni),
            vbdsf=(["y", "x", "valid_time"], dhi),
            vddsf=(["y", "x", "valid_time"], ghi),
            t=(["y", "x", "valid_time"], temperature),
            gust=(["y", "x", "valid_time"], wind_speed),
        ),
        coords={
            "valid_time": ("valid_time", valid_times),
            "latitude": (("y", "x"), lats),
            "longitude": (("y", "x"), lons),
        },
    )

    return ds


def _make_hrrr_fetcher(site_latlong_mapping):
    # the instantiation args are irrelevant, since dataset is mocked
    return HrrrFetcher(
        location_mapper=lambda site: site_latlong_mapping[site],
        selector="select",
        selector_args=dict(
            variables=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        ),
        source_mode="18_hour_forecast",
    )


def _make_domain(*locations):
    return make_domain(
        np.datetime64("2022-07-01 15:00:00"),
        np.datetime64("2022-07-01 18:00:00"),
        np.timedelta64(1, "h"),
        *locations,
    )


class TestModelBuilder(PvModelBuilder):
    def __init__(self, n_models: int, make_system):
        super().__init__()
        self._n_models = n_models
        self._make_system = make_system

    def get_resources(self, locations: typing.List[str]):
        pass

    def __call__(self, location: str):
        return tuple(
            self._make_system(f"{location}_{i}") for i in range(self._n_models)
        )


@patch.object(xr, "open_dataset", return_value=_make_xarray_dataset())
@patch("time_series_models.data_fetchers.hrrr_fetcher.fsspec")
class TestTransformersPvLib(unittest.TestCase):
    def setUp(self) -> None:
        memory.clear()

    @staticmethod
    def _make_system(location):
        mount = pvlib.pvsystem.SingleAxisTrackerMount()
        module = pvlib.pvsystem.retrieve_sam("SandiaMod")[
            "Hanwha_HSL60P6_PA_4_250T__2013_"
        ]
        inverter = pvlib.pvsystem.retrieve_sam("CECInverter")[
            "Yaskawa_Solectria_Solar__SGI_500XTM__380V_"
        ]
        temp_model = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
            "open_rack_glass_polymer"
        ]
        array = pvlib.pvsystem.Array(
            module_parameters=module,
            temperature_model_parameters=temp_model,
            mount=mount,
            modules_per_string=18,
            strings=108,
        )
        system = pvlib.pvsystem.PVSystem(
            arrays=[array], inverter_parameters=inverter, name=str(uuid.uuid4())
        )
        return PvLibPhysicalModel(
            system.name + location,
            system_model=system,
            latitude=LATITUDE,
            longitude=LONGITUDE,
        )

    def test_forecast_pvlib(self, mock_fsspec, mock_open_database):
        # Test asserts that no exceptions are raised with forecast_pv
        # NOTE: PV forecasts are not asserted because the system configuration
        # in _make_pv_system is erroneous

        pipeline = make_pv_pipeline(
            hrrr_fetcher=_make_hrrr_fetcher(
                dict(zeno=dict(latitude=40, longitude=-107))
            ),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        domain = _make_domain("zeno")
        res = pipeline.fit_transform(domain)

        # Nonsense value - need a read PV config for correctness
        np.testing.assert_array_equal(res, np.full((4, 1), -152100.0, dtype=np.float32))

    def test_site_pipeline_nans(self, mock_fsspec, mock_open_database):
        # Test asserts that NaNs are handled correctly
        # Test is setup such that the domain is partially outside the time range
        # for which data is available: 2022-07-01 22:00 and 23:00 have available
        # data, while times on 2022-07-02 do not.
        # NOTE: PV forecasts are not asserted because the system configuration
        # in _make_pv_system is erroneous

        test_site = "Zeno"
        domain = make_domain(
            np.datetime64("2022-07-01 22:00:00"),
            np.datetime64("2022-07-02 02:00:00"),
            np.timedelta64(1, "h"),
            test_site,
        )

        site_latlong_mapping = {test_site: dict(latitude=LATITUDE, longitude=LONGITUDE)}

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher(site_latlong_mapping),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )
        result = pipe.fit_transform(domain)

        np.testing.assert_array_equal(
            result,
            np.asarray(
                [-152100.0, -152100.0, np.nan, np.nan, np.nan], dtype=np.float32
            ).reshape(-1, 1),
        )


@patch("time_series_models.data_fetchers.hrrr_fetcher.load_hrrr_data")
@patch("time_series_models.data_fetchers.hrrr_fetcher.fsspec")
class TestPvSamV1(unittest.TestCase):
    @staticmethod
    def _make_system(_):
        path = os.path.join(THIS_DIR, "fixtures", "pysam_physical_model.json")
        with open(path, "r") as file:
            config = json.load(file)
        return PvSamV1PhysicalModel.create(
            str(uuid.uuid4()), config=config, latitude=LATITUDE, longitude=LONGITUDE
        )

    def test_forecast_pysam(self, mock_fspec, mock_load_data):
        # Test asserts that (a) no exceptions are raised with forecast_pv,
        # and (b) pv forecasts are realistic
        # Test is setup with reasonable weather forecasts
        # timestamps correspond to 2022-06-01 18:00:00 -> 2022-06-01 22:00:00 MT
        # and as the sun sets ~ 830 pm, DNI, DHI and GHI all drop to 0s.

        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 05:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [816.0, 78.0, 440.0, 298.0, 5.5],
                [659.0, 59.8, 230.0, 293.0, 5.5],
                [269.0, 24.8, 44.8, 289.0, 5.3],
                [0.0, 0.0, 0.0, 286.0, 4.7],
                [0.0, 0.0, 0.0, 284.0, 5.6],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (5, 1))

        # assert that pv forecast < -100,000.0 at 6 pm MT, and then decreases to
        # greater than 0 (inverter power) past 8 pm.
        self.assertTrue(res[0, 0] < -1000000.0)  # 6 pm MT
        self.assertTrue(res[1, 0] < res[0, 0])  # 7 pm MT
        self.assertTrue(res[2, 0] > res[1, 0])  # 8 pm MT
        self.assertTrue(res[3, 0] > 0.0)  # 9 pm MT
        self.assertTrue(res[4, 0] > 0.0)  # 10 pm MT

    def test_pv_function_mapping(self, mock_fspec, mock_load_data):
        # Test asserts HRRR columns are correctly mapped to the PV model names

        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 05:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        hrrr_df = pd.DataFrame(
            [
                [816.0, 78.0, 440.0, 298.0, 5.5],
                [659.0, 59.8, 230.0, 293.0, 5.5],
                [269.0, 24.8, 44.8, 289.0, 5.3],
                [0.0, 0.0, 0.0, 286.0, 4.7],
                [0.0, 0.0, 0.0, 284.0, 5.6],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )
        mock_load_data.return_value = hrrr_df

        mock_system = Mock()

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=lambda _: mock_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        pv_df = (
            hrrr_df.reset_index(drop=False)
            .set_index("date_time")
            .drop(labels=["location"], axis=1)
            .rename(columns=CF_MAPPING)
        )
        mock_system.forecast.assert_called_once()
        args = mock_system.forecast.call_args.args
        pd.testing.assert_frame_equal(pv_df, args[0])

    def test_forecast_pysam__missing_all(self, mock_fsspec, mock_load_data):
        # Test asserts that (a) no exceptions are raised with forecast_pv,
        # (b) results are returned as nan when any inputs are missing

        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 04:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )
        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (4, 1))
        [self.assertTrue(np.isnan(element)) for element in res]

    def test_forecast_pysam__missing_some(self, mock_fsspec, mock_load_data):
        # Test asserts that (a) no exceptions are raised with forecast_pv,
        # (b) results are returned as nan when any inputs are missing, and
        # (c) pv forecasts are provided wherever possible
        # Test is setup with reasonable weather forecasts
        # timestamps correspond to 2022-06-01 18:00:00 -> 2022-06-01 22:00:00 MT
        # and as the sun sets ~ 830 pm, DNI, DHI and GHI all drop to 0s.

        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 05:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [816.0, 78.0, 440.0, 298.0, 5.5],
                [659.0, 59.8, np.nan, 293.0, 5.5],
                [269.0, 24.8, 44.8, 289.0, 5.3],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 0.0, 0.0, 284.0, 5.6],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (5, 1))

        # assert that pv forecast < -100,000.0 at 6 pm MT, and then decreases to
        # greater than 0 (inverter power) past 8 pm... but is np.nan when datetime or weather is missing
        self.assertTrue(res[0, 0] < -1000000.0)  # 6 pm MT
        self.assertTrue(np.isnan(res[1, 0]))  # 7 pm MT
        self.assertTrue(res[2, 0] > res[0, 0])  # 8 pm MT
        self.assertTrue(np.isnan(res[3, 0]))  # 9 pm MT
        self.assertTrue(np.isnan(res[4, 0]))  # 10 pm MT

    def test_forecast_pysam__handle_pysam_exception(self, mock_fsspec, mock_load_data):
        # Test asserts that a pysam exception is handled gracefully,
        # here due to bad input data (row 0 values are out of allowed bounds),
        # and returns all NaN values

        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 05:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [9999.0, 9999.0, 9999.0, 298.0, 5.5],
                [659.0, 59.8, 230.0, 293.0, 5.5],
                [269.0, 24.8, 44.8, 289.0, 5.3],
                [0.0, 0.0, 0.0, 286.0, 4.7],
                [0.0, 0.0, 0.0, 284.0, 5.6],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (5, 1))
        np.testing.assert_array_equal(res, np.full(res.shape, np.nan))


@patch("time_series_models.data_fetchers.hrrr_fetcher.load_hrrr_data")
@patch("time_series_models.data_fetchers.hrrr_fetcher.fsspec")
class TestPvWattsV8(unittest.TestCase):
    # TODO(Michael H): DRY up the overlap btw Pvsamv1 and Pvwattsv8

    @staticmethod
    def _make_system(location) -> PvWattsV8PhysicalModel:
        config = {
            "albedo": [0.2],
            "use_wf_albedo": 1,
            "system_capacity": 100,  # in kW
            "module_type": 0,
            "dc_ac_ratio": 1.15,
            "bifaciality": 0,
            "array_type": 0,
            "tilt": 20,
            "azimuth": 180,
            "gcr": 0.3,
            "soiling": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "losses": 14.075660688264469,
            "en_snowloss": 0,
            "inv_eff": 96,
            "batt_simple_enable": 0,
            "constant": 0,
        }
        return PvWattsV8PhysicalModel.create(
            location, config=config, latitude=LATITUDE, longitude=LONGITUDE
        )

    def test_forecast_pysam(self, mock_fsspec, mock_load_data):
        # Test asserts that (a) no exceptions are raised with forecast_pv,
        # and (b) pv forecasts are realistic
        # Test is setup with reasonable weather forecasts
        # timestamps correspond to 2022-06-01 18:00:00 -> 2022-06-01 22:00:00 MT
        # and as the sun sets ~ 830 pm, DNI, DHI and GHI all drop to 0s.

        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 05:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [816.0, 78.0, 440.0, 298.0, 5.5],
                [659.0, 59.8, 230.0, 293.0, 5.5],
                [269.0, 24.8, 44.8, 289.0, 5.3],
                [0.0, 0.0, 0.0, 286.0, 4.7],
                [0.0, 0.0, 0.0, 284.0, 5.6],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (5, 1))

        # assert that pv forecast < -100,000.0 at 6 pm MT, and then decreases to
        # greater than 0 (inverter power) past 8 pm.
        self.assertTrue(res[0, 0] < -10000.0)  # 6 pm MT
        self.assertTrue(res[1, 0] > res[0, 0])  # 7 pm MT
        self.assertTrue(res[2, 0] > res[1, 0])  # 8 pm MT
        self.assertTrue(res[3, 0] >= 0.0)  # 9 pm MT
        self.assertTrue(res[4, 0] >= 0.0)  # 10 pm MT

    def test_forecast_pysam__missing_all(self, mock_fsspec, mock_load_data):
        # Test asserts that (a) no exceptions are raised with forecast_pv,
        # (b) results are returned as nan when any inputs are missing

        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 04:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (4, 1))
        np.testing.assert_array_equal(res, np.full(res.shape, np.nan))

    def test_forecast_pysam__missing_some(self, mock_fsspec, mock_load_data):
        # Test asserts that (a) no exceptions are raised with forecast_pv,
        # (b) results are returned as nan when any inputs are missing, and
        # (c) pv forecasts are provided wherever possible
        # Test is setup with reasonable weather forecasts
        # timestamps correspond to 2022-06-01 18:00:00 -> 2022-06-01 22:00:00 MT
        # and as the sun sets ~ 830 pm, DNI, DHI and GHI all drop to 0s.
        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 05:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [816.0, 78.0, 440.0, 298.0, 5.5],
                [np.nan, 59.8, 230.0, 293.0, 5.5],
                [269.0, 24.8, 44.8, 289.0, 5.3],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 0.0, 0.0, 284.0, 5.6],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (5, 1))

        # assert that pv forecast < -100,000.0 at 6 pm MT, and then decreases to
        # greater than 0 (inverter power) past 8 pm... but is np.nan when datetime or weather is missing
        self.assertTrue(res[0, 0] < -10000.0)  # 6 pm MT
        self.assertTrue(np.isnan(res[1, 0]))  # 7 pm MT
        self.assertTrue(res[2, 0] > res[0, 0])  # 8 pm MT
        self.assertTrue(np.isnan(res[3, 0]))  # 9 pm MT
        self.assertTrue(np.isnan(res[4, 0]))  # 10 pm MT

    def test_forecast_pysam__handle_pysam_exception(self, mock_fsspec, mock_load_data):
        # Test asserts that a pysam exception is handled gracefully,
        # here due to bad input data (row 0 values are out of allowed bounds),
        # and returns all NaN values
        domain = make_domain(
            np.datetime64("2022-07-01 01:00:00"),
            np.datetime64("2022-07-01 05:00:00"),
            np.timedelta64(1, "h"),
            "zeno",
        )

        mock_load_data.return_value = pd.DataFrame(
            [
                [9999.0, 9999.0, 9999.0, 298.0, 5.5],
                [659.0, 59.8, 230.0, 293.0, 5.5],
                [269.0, 24.8, 44.8, 289.0, 5.3],
                [0.0, 0.0, 0.0, 286.0, 4.7],
                [0.0, 0.0, 0.0, 284.0, 5.6],
            ],
            index=multiindex_from_domain(domain),
            columns=["dswrf", "vbdsf", "vddsf", "t", "gust"],
        )

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher({}),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        # Location isn't used here. The argument is ignored in _make_system and the hrrr fetcher is a mock
        res = pipe.fit_transform(domain)

        self.assertIsInstance(res, np.ndarray)
        self.assertTupleEqual(res.shape, (5, 1))
        np.testing.assert_array_equal(res, np.full(res.shape, np.nan))


@patch.object(xr, "open_dataset", return_value=_make_xarray_dataset())
@patch("time_series_models.data_fetchers.hrrr_fetcher.fsspec")
class TestFullFlow(unittest.TestCase):
    # TODO(Michael H): DRY up the overlap btw Pvsamv1 and Pvwattsv8

    @staticmethod
    def _make_system(location):
        config = {
            "albedo": [0.2],
            "use_wf_albedo": 1,
            "system_capacity": 100,  # in kW
            "module_type": 0,
            "dc_ac_ratio": 1.15,
            "bifaciality": 0,
            "array_type": 0,
            "tilt": 20,
            "azimuth": 180,
            "gcr": 0.3,
            "soiling": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "losses": 14.075660688264469,
            "en_snowloss": 0,
            "inv_eff": 96,
            "batt_simple_enable": 0,
            "constant": 0,
        }
        return PvWattsV8PhysicalModel.create(
            location, config=config, latitude=LATITUDE, longitude=LONGITUDE
        )

    def test_site_pipeline(self, mock_fsspec, mock_open_database):
        # Test asserts for integration of make_site_pipeline for a single site
        # Case 1: single system: assert for no exceptions
        # Case 2: two systems (identical to the system in case 1): assert that production = 2x

        test_site = "Zeno"
        site_latlong_mapping = {test_site: dict(latitude=LATITUDE, longitude=LONGITUDE)}

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher(site_latlong_mapping),
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )

        single_system_result = pipe.fit_transform(_make_domain(test_site))

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher(site_latlong_mapping),
            pv_model_builder=TestModelBuilder(
                n_models=2, make_system=self._make_system
            ),
        )
        multi_system_result = pipe.fit_transform(_make_domain(test_site))

        # Both single_system_result and multi_system_result have shapes 4x1
        # Assert that multi system result = 2x single system result at all timestamps
        for i, res in enumerate(single_system_result):
            self.assertAlmostEqual(multi_system_result[i][0], 2 * res[0])

    def test_all_sites_column_transformer(self, mock_fsspec, mock_open_database):
        # Test asserts that multiple locations are handled correctly
        # Sits have identical systems but different locations, which results
        # in different forecasts. These are useful for debugging,
        # and not asserted at the moment.

        dx = 1.0  # random perturbation in location
        site_latlong_mapping = {
            "Zeno": dict(latitude=LATITUDE, longitude=LONGITUDE),
            "Zone": dict(latitude=LATITUDE + dx, longitude=LONGITUDE - dx),
        }

        pipe = make_pv_pipeline(
            _make_hrrr_fetcher(site_latlong_mapping),
            # Use the same system definition at each site (name will be a different uuid)
            pv_model_builder=TestModelBuilder(
                n_models=1, make_system=self._make_system
            ),
        )
        result = pipe.fit_transform(_make_domain("Zeno", "Zone"))
        self.assertTupleEqual(result.shape, (8, 1))


class TestPvModelBuilder(unittest.TestCase):
    def test_config_builder(self):
        mapping = dict(
            foobar=[
                "bando/time_series_models/tests/fixtures/example_pv_physical_config.json",
                "bando/time_series_models/tests/fixtures/example_pv_physical_config.json",
            ]
        )
        instance = ConfigBuilder(mapping)

        models = instance("foobar")
        # Name and all parameter of the PvPhysicalModel come from the config
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "ExampleArray")
        self.assertIsInstance(models[0], PvSamV1PhysicalModel)
        self.assertEqual(models[1].name, "ExampleArray")
        self.assertIsInstance(models[1], PvSamV1PhysicalModel)


if __name__ == "__main__":
    unittest.main()
