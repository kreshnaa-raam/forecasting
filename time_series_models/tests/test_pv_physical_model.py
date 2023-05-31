import os.path
import unittest
import logging
import json
import io
import copy
from cloudpickle import pickle
import uuid
import pvlib

# Adds pickling support for pysam when imported
from time_series_models.pv_physical_model import (
    PvSamV1PhysicalModel,
    PvWattsV8PhysicalModel,
    PvLibPhysicalModel,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


class PvLibPhysicalModelTests(unittest.TestCase):
    def create_model(self):
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
            system.name, system_model=system, latitude=40.1, longitude=-102.3
        )

    def test_pickle_pysam(self):
        pvlib_model = self.create_model()

        with io.BytesIO() as bio:
            pickle.dump(pvlib_model, bio, protocol=5)
            bio.seek(0)
            result = pickle.load(bio)

        self.assertEqual(pvlib_model.name, result.name)

        self.assertEqual(
            pvlib_model.system_model.modules_per_string,
            result.system_model.modules_per_string,
        )

    def test_deepcopy_pysam(self):
        pvlib_model = self.create_model()
        result = copy.deepcopy(pvlib_model)
        self.assertEqual(pvlib_model.name, result.name)

        self.assertEqual(
            pvlib_model.system_model.modules_per_string,
            result.system_model.modules_per_string,
        )


class PvSamV1PhysicalModelTests(unittest.TestCase):
    def create_model(self):
        path = os.path.join(THIS_DIR, "fixtures", "pysam_physical_model.json")
        with open(path, "r") as file:
            config = json.load(file)
        return PvSamV1PhysicalModel.create(
            "test_system", config=config, latitude=40.1, longitude=-102.3
        )

    def test_pickle_pysam(self):
        pysam_model = self.create_model()

        with io.BytesIO() as bio:
            pickle.dump(pysam_model, bio, protocol=5)
            bio.seek(0)
            result = pickle.load(bio)

        self.assertEqual(pysam_model.name, result.name)
        self.assertAlmostEqual(
            result.system_model.value("system_capacity"), 1790.739, 2
        )

    def test_deepcopy_pysam(self):
        pysam_model = self.create_model()
        result = copy.deepcopy(pysam_model)
        self.assertEqual(pysam_model.name, result.name)
        self.assertAlmostEqual(
            result.system_model.value("system_capacity"), 1790.739, 2
        )


class PvWattsV8PhysicalModelTests(unittest.TestCase):
    def create_model(self):
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
            "test_system", config=config, latitude=40.1, longitude=-102.3
        )

    def test_pickle_pysam(self):
        pvwattsv8_model = self.create_model()

        with io.BytesIO() as bio:
            pickle.dump(pvwattsv8_model, bio, protocol=5)
            bio.seek(0)
            result = pickle.load(bio)

        self.assertEqual(pvwattsv8_model.name, result.name)
        self.assertAlmostEqual(result.system_model.value("system_capacity"), 100.0)

    def test_deepcopy_pysam(self):
        pvwattsv8_model = self.create_model()

        result = copy.deepcopy(pvwattsv8_model)

        self.assertEqual(pvwattsv8_model.name, result.name)
        self.assertAlmostEqual(result.system_model.value("system_capacity"), 100.0)


if __name__ == "__main__":
    unittest.main()
