import json
import os
import unittest

import numpy as np
import pvlib
import uuid

from time_series_models.config import ConfigHandler
from time_series_models import pv_physical_model
from time_series_models.data_fetchers.numba_groupby_functions import (
    hour_pick,
    hour_diff,
    hour_sum,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class ConfigHandlerTest(unittest.TestCase):
    def test_decode_ndarray(self):
        np.testing.assert_array_equal(
            ConfigHandler.decode(
                '{"__camus_json_type__": "numpy.ndarray", "__camus_json_data__": ['
                '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "24", "units": "h"}}, '
                '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "25", "units": "h"}}, '
                '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "36", "units": "h"}}]}'
            ),
            np.array([24, 25, 36], dtype=np.timedelta64(1, "h")),
        )
        np.testing.assert_array_equal(
            ConfigHandler.decode(
                '{"__camus_json_type__": "numpy.ndarray", "__camus_json_data__": [24, 48, 168]}'
            ),
            np.array([24, 48, 168]),
        )

    def test_encode_ndarray(self):
        self.assertEqual(
            ConfigHandler.encode(np.array([24, 25, 36], dtype=np.timedelta64(1, "h"))),
            '{"__camus_json_type__": "numpy.ndarray", "__camus_json_data__": ['
            '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "24", "units": "h"}}, '
            '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "25", "units": "h"}}, '
            '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "36", "units": "h"}}]}',
        )
        with self.assertRaises(ValueError):
            ConfigHandler.encode(np.array([24, 25, 36]))

    def test_decode_np_timedelta(self):
        self.assertEqual(
            ConfigHandler.decode(
                '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "27", "units": "D"}}'
            ),
            np.timedelta64(27, "D"),
        )

        self.assertEqual(
            ConfigHandler.decode(
                '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "3", "units": "M"}}'
            ),
            np.timedelta64(3, "M"),
        )

        with self.assertRaises(TypeError):
            ConfigHandler.decode(
                '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "3", "units": "Q"}}'
            )

    def test_encode_np_timedelta(self):
        self.assertEqual(
            ConfigHandler.encode(np.timedelta64(27, "D")),
            '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "27", "units": "D"}}',
        )

        self.assertEqual(
            ConfigHandler.encode(np.timedelta64(3, "M")),
            '{"__camus_json_type__": "numpy.timedelta64", "__camus_json_data__": {"duration": "3", "units": "M"}}',
        )

    def test_encode_numba_groupby_fns(self):
        self.assertEqual(
            ConfigHandler.encode(hour_pick(12)),
            '{"__camus_json_type__": "numba_groupby_hour_pick", "__camus_json_data__": 12}',
        )
        self.assertEqual(
            ConfigHandler.encode(hour_diff(4, 12)),
            '{"__camus_json_type__": "numba_groupby_hour_diff", "__camus_json_data__": [4, 12]}',
        )
        self.assertEqual(
            ConfigHandler.encode(hour_sum(7, 13)),
            '{"__camus_json_type__": "numba_groupby_hour_sum", "__camus_json_data__": [7, 13]}',
        )

    def test_decode_numba_groupby_fns(self):
        encoded = ConfigHandler.encode(hour_pick(12))
        decoded = ConfigHandler.decode(encoded)
        # To make sure we decoded correctly, check that the __str__() method we set on the function
        # returns what we expect
        self.assertEqual(decoded.__str__(), "hour_pick_12")

        encoded = ConfigHandler.encode(hour_sum(4, 8))
        decoded = ConfigHandler.decode(encoded)
        self.assertEqual(decoded.__str__(), "hour_sum_4_to_8")

        encoded = ConfigHandler.encode(hour_diff(7, 13))
        decoded = ConfigHandler.decode(encoded)
        self.assertEqual(decoded.__str__(), "hour_diff_7_to_13")

    def test_full_config(self):
        expected = dict(
            time_step=np.timedelta64(1, "D"),
            lags=[
                np.timedelta64(1, "D"),
                np.timedelta64(3, "D"),
                np.timedelta64(6, "D"),
            ],
            # note, tuple will serialize to a list, and decode back as a list...
            # so for this test to work as written, we must specify the resource_config using lists not tuples
            published_native_forecast=False,
            balancing_area_forecast=False,
            met_cities=[
                "Denver",
                "Fort_Collins",
                "Colorado_Springs",
            ],
            met_tz_shift="America/Denver",
            met_aggregations={
                "feels_like": [
                    "min",
                    "max",
                ],  # What to do with functions that can't be json? pickle or crazy encoder for numba funcs?
                "snow_1h": ["sum"],
                "humidity": ["max"],
                "pressure": ["median"],
                "wind_speed": ["median"],
            },
            day_of_week=True,
            business_day=True,
        )

        self.assertDictEqual(
            ConfigHandler.decode(ConfigHandler.encode(expected)), expected
        )

    def test_encode_decode_pv_physical_model__pvsamv1(self):
        path = os.path.join(THIS_DIR, "fixtures", "pysam_physical_model.json")
        with open(path, "r") as file:
            config = json.load(file)
        obj = pv_physical_model.PvSamV1PhysicalModel.create(
            "test_system", config=config, latitude=40.0, longitude=-100.0
        )

        encoded = ConfigHandler.encode(obj)
        decoded = ConfigHandler.decode(encoded)
        self.assertEqual(decoded.name, "test_system")
        self.assertAlmostEqual(
            decoded.system_model.value("system_capacity"), 1790.739, 2
        )

    def test_encode_decode_pv_physical_model__pwattsv8(self):
        config = {
            "albedo": [0.2],
            "use_wf_albedo": 1,
            "system_capacity": 100,
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
        obj = pv_physical_model.PvWattsV8PhysicalModel.create(
            "capybara", config, latitude=40.0, longitude=-100.0
        )
        encoded = ConfigHandler.encode(obj)
        decoded = ConfigHandler.decode(encoded)
        self.assertEqual(decoded.name, "capybara")
        self.assertEqual(decoded.system_model.value("system_capacity"), 100.0)

    @unittest.skip("Not implemented")
    def test_encode_decode_pv_physical_model__pvlib(self):
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
        obj = pv_physical_model.PvLibPhysicalModel(
            system.name, system_model=system, latitude=40.0, longitude=-100.0
        )
        encoded = ConfigHandler.encode(obj)
        decoded = ConfigHandler.decode(encoded)
        self.assertEqual(decoded.name, system.name)
        self.assertEqual(decoded.system_model.value("system_capacity"), 100.0)


if __name__ == "__main__":
    unittest.main()
