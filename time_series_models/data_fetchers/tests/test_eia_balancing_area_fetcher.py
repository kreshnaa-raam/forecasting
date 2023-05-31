import os
import unittest
from unittest.mock import patch
import io

import numpy as np

import cloudpickle as pickle
from time_series_models.constants import LOCATION, DATE_TIME
from time_series_models.data_fetchers.eia_balancing_area_fetcher import (
    EiaBalancingAreaFetcher,
    memory,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@patch.multiple(
    EiaBalancingAreaFetcher,
    source_loader=EiaBalancingAreaFetcher.file_loader,
    URIS={
        "hobbes": os.path.join(THIS_DIR, "fixtures", "PSCO_EIA.xlsx"),
        "locke": os.path.join(THIS_DIR, "fixtures", "PNM_EIA.xlsx"),
    },
)
class EiaBalancingAreaFetcherTests(unittest.TestCase):
    def setup(self):
        # Clear cache for all data. Will affect other processes running locally
        memory.clear()

    def test_get_data(self):
        eia_fetcher = EiaBalancingAreaFetcher(lambda x: x, variables=["D", "NG", "TI"])

        dtype = np.dtype([(LOCATION, np.unicode_, 36), (DATE_TIME, np.dtype("<M8[h]"))])
        domain = np.empty([6, 1], dtype=dtype)

        domain[LOCATION] = np.array(
            ("locke",) * 3 + ("hobbes",) * 3, dtype="U36"
        ).reshape(-1, 1)
        domain[DATE_TIME] = np.arange(
            np.datetime64("2015-07-01T07:00:00"),
            np.datetime64("2015-07-01T13:00:00"),
            step=np.timedelta64(1, "h"),
        ).reshape(-1, 1)

        result = eia_fetcher.get_data(domain)

        expected = np.array(
            [
                [np.nan, np.nan, np.nan],
                [1579.0, 1090.0, -489.0],
                [1514.0, 1027.0, -487.0],
                [4344.0, 3775.0, -569.0],
                [4374.0, 3768.0, -606.0],
                [4505.0, 3963.0, -542.0],
            ]
        )
        np.testing.assert_equal(result, expected)

        # assert pickleable...
        with io.BytesIO() as file_object:
            pickle.dump(eia_fetcher, file_object, protocol=5)
            file_object.seek(0)
            zombie = pickle.load(file_object)
        result = zombie.get_data(domain)
        np.testing.assert_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
