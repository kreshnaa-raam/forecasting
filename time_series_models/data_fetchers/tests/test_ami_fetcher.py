import os
import unittest
from typing import Generator

import numpy as np
import pandas as pd

from unittest.mock import patch
from google.cloud.exceptions import NotFound
from time_series_models.constants import DATE_TIME

from time_series_models.data_fetchers.ami_fetcher import (
    AmiFetcher,
    load_ami_data,
    memory,
)
from time_series_models.transformers import make_domain

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@patch.multiple(
    AmiFetcher,
    source_loader=AmiFetcher.file_loader,
    FILE_SYSTEM_URIS={
        100: "ami100_watts.csv.gz",
        200: "ami200_watts.csv.gz",
        300: "ami1_empty.csv.gz",
    },
    FILE_PATH=os.path.join(THIS_DIR, "fixtures"),
)
class MyTestCase(unittest.TestCase):
    def setUp(self):
        with self.assertLogs(level="WARNING"):  # silence joblib logger warning
            # Clear cache for all data. Will affect other processes running locally
            memory.clear()

    def test_split_every(self):
        my_list = [0, 1, 2, 3, 4]

        res = AmiFetcher.split_every(2, my_list)
        self.assertIsInstance(res, Generator)

        res = list(AmiFetcher.split_every(1, my_list))
        self.assertListEqual(res, [[0], [1], [2], [3], [4]])

        res = list(AmiFetcher.split_every(2, my_list))
        self.assertListEqual(res, [[0, 1], [2, 3], [4]])

        res = list(AmiFetcher.split_every(5, my_list))
        self.assertListEqual(res, [[0, 1, 2, 3, 4]])

        res = list(AmiFetcher.split_every(10, my_list))
        self.assertListEqual(res, [[0, 1, 2, 3, 4]])

        # assert original list is unchanged
        self.assertListEqual(my_list, [0, 1, 2, 3, 4])

    def test_load_ami_data__no_date_parse(self):
        path = os.path.join(THIS_DIR, "fixtures", "ami100_watts.csv.gz")
        result = load_ami_data(path, AmiFetcher.file_buffer_loader)

        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_series_equal(result.dtypes, pd.Series({"100": float}))
        pd.testing.assert_index_equal(result.columns, pd.Index(["100"]))
        self.assertEqual(result.index.name, DATE_TIME)

        self.assertEqual(result.index.dtype, object)

    def test_load_ami_data__parse_dates(self):
        path = os.path.join(THIS_DIR, "fixtures", "ami100_watts.csv.gz")
        result = load_ami_data(path, AmiFetcher.file_buffer_loader, parse_dates=True)

        # same as for no_date_parse
        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_series_equal(result.dtypes, pd.Series({"100": float}))
        pd.testing.assert_index_equal(result.columns, pd.Index(["100"]))
        self.assertEqual(result.index.name, DATE_TIME)

        # different from no_date_parse
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_load_ami_data__notfound(self):
        path = os.path.join(THIS_DIR, "fixtures", "ami1_empty.csv.gz")
        with patch("pandas.read_csv", side_effect=NotFound("oops!")):
            with self.assertLogs(level="WARNING") as logger_warnings:
                result = load_ami_data(path, AmiFetcher.file_buffer_loader)
                records = logger_warnings.records
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].getMessage(), f"Missing AMI data: {path}")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertTupleEqual(result.shape, (0, 1))
            self.assertTrue(result.empty)
            pd.testing.assert_index_equal(result.columns, pd.Index([path], dtype=str))
            pd.testing.assert_index_equal(
                result.index, pd.Index([], name=DATE_TIME, dtype=str)
            )

    def test_load_ami_data__empty(self):
        result = load_ami_data(
            os.path.join(THIS_DIR, "fixtures", "ami1_empty.csv.gz"),
            AmiFetcher.file_buffer_loader,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTupleEqual(result.shape, (0, 1))
        self.assertTrue(result.empty)
        pd.testing.assert_index_equal(result.columns, pd.Index([1], dtype=str))
        pd.testing.assert_index_equal(
            result.index, pd.Index([], name=DATE_TIME, dtype=str)
        )

    def test_build_mids__unary_mapper(self):
        a = AmiFetcher(location_mapper=lambda x: x)
        domain_location = 1
        mids = a._build_mids(domain_location)
        self.assertListEqual(mids, [1])

        domain_location = "betelgeuse"
        mids = a._build_mids(domain_location)
        self.assertListEqual(mids, ["betelgeuse"])

    def test_build_mids__multi_mapper(self):
        a = AmiFetcher(location_mapper=lambda x: ["1", "2", "c"])
        domain_location = "betelgeuse"
        mids = a._build_mids(domain_location)
        self.assertListEqual(mids, ["1", "2", "c"])

    def test_build_uris(self):
        a = AmiFetcher(
            uri_formatter="meter: {meter:s}, units: {units:s}",
            units="energy",
        )
        uris = a._build_uris(None, None, ["betelgeuse"])
        self.assertListEqual(uris, ["meter: betelgeuse, units: energy"])

        uris = a._build_uris(None, None, ["1", "2", "c"])
        self.assertListEqual(
            uris,
            [
                "meter: 1, units: energy",
                "meter: 2, units: energy",
                "meter: c, units: energy",
            ],
        )

    def test_postprocess(self):
        mid = [1, 2]
        df = pd.DataFrame(
            data={"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
            index=pd.DatetimeIndex(
                [
                    pd.to_datetime("2022-04-20 16:00:00"),
                    pd.to_datetime("2022-04-20 17:00:00"),
                    pd.to_datetime("2022-04-20 18:00:00"),
                    pd.to_datetime("2022-04-20 20:00:00"),
                ]
            ),
        )

        a = AmiFetcher(freq="H")
        result = a._postprocess(
            df, mid, start=pd.to_datetime("2022-04-20 17:00:00"), end=None
        )

        expected = pd.DataFrame(
            data={1: [2, 3, np.nan, 4], 2: [6, 7, np.nan, 8]},
            index=pd.DatetimeIndex(
                [
                    pd.to_datetime("2022-04-20 17:00:00"),
                    pd.to_datetime("2022-04-20 18:00:00"),
                    pd.to_datetime("2022-04-20 19:00:00"),
                    pd.to_datetime("2022-04-20 20:00:00"),
                ],
                freq="H",
            ),
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_get_data(self):
        ami_fetcher = AmiFetcher(
            location_mapper=lambda x: [100, 200, 300],
        )
        domain = make_domain(
            "2019-11-20",
            "2019-11-21",
            np.timedelta64(1, "h"),
            "Arcturus",
        )
        result = ami_fetcher.get_data(domain)
        self.assertListEqual(ami_fetcher.variables, [100, 200, 300])

        result_100_200 = result[:, :2]
        result_300 = result[:, 2]

        # mid 300 should be empty (maps to ami1_empty.csv.gz)
        expected = np.ones(len(result))
        expected[:] = np.nan
        self.assertTrue(np.array_equal(result_300, expected, equal_nan=True))

        path100 = os.path.join(THIS_DIR, "fixtures", "ami100_watts.csv.gz")
        path200 = os.path.join(THIS_DIR, "fixtures", "ami200_watts.csv.gz")
        ground_truth = pd.concat(
            [
                pd.read_csv(path, index_col=0, parse_dates=True)
                for path in [path100, path200]
            ],
            axis=1,
        )
        ground_truth["300"] = np.nan

        # ground_truth data have 15 minute increments
        expected = pd.date_range(
            start="2019-11-19T21 Z",
            end="2019-11-20T20:45:00 Z",
            freq="15min",
            name=DATE_TIME,
        )
        pd.testing.assert_index_equal(ground_truth.index[: 24 * 4], expected)

        # ground_truth data are positive float values (with some nan sprinkled in)
        self.assertEqual(ground_truth["100"].dtype, float)
        self.assertEqual(ground_truth["200"].dtype, float)
        self.assertGreater(ground_truth.min().min(), 0)

        # the output of get_data is a simple reindex of the loaded data against the domain
        index = pd.date_range(start="2019-11-20T00 Z", end="2019-11-21T00 Z", freq="H")
        expected = ground_truth[["100", "200"]].reindex(index).to_numpy()
        np.testing.assert_array_equal(result_100_200, expected)

        # now set freq for AmiFetcher
        ami_fetcher = AmiFetcher(location_mapper=lambda x: [100, 200, 300], freq="H")
        new_result = ami_fetcher.get_data(domain)
        np.testing.assert_array_equal(result.shape, new_result.shape)
        # now, data are first resampled prior to reindexing to domain
        expected = ground_truth.resample("H").mean().reindex(index).to_numpy()
        np.testing.assert_array_equal(expected, new_result)


if __name__ == "__main__":
    unittest.main()
