import io
import os
import tempfile
import unittest
import logging
from unittest.mock import patch, DEFAULT, call, MagicMock

import cloudpickle as pickle
import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da

import fsspec
import dask

import time_series_models
from time_series_models.transformers import (
    make_domain,
)
from time_series_models.data_fetchers.hrrr_fetcher import (
    HrrrFetcher,
    load_hrrr_data,
    memory,
    rpath_mapper,
)


def make_xarray_dataset():
    lats, lons = np.meshgrid(range(21, 47), range(237, 299), indexing="ij")
    ds = xr.Dataset(
        coords={
            "valid_time": (
                "valid_time",
                np.arange(
                    np.datetime64("2022-07-01T00:00:00"),
                    np.datetime64("2022-07-04T00:00:00"),
                    step=np.timedelta64(1, "h"),
                ),
            ),
            "latitude": (("y", "x"), lats),
            "longitude": (("y", "x"), lons),
        },
    )

    da1, da2, da3 = da.meshgrid(range(72), range(26), range(62), indexing="ij")

    return ds.assign(
        time_like=xr.DataArray(da1, ds.coords),
        lat_like=xr.DataArray(da2, ds.coords),
        lon_like=xr.DataArray(da3, ds.coords),
    )


@patch.object(xr, "open_dataset", return_value=make_xarray_dataset())
@patch("time_series_models.data_fetchers.hrrr_fetcher.fsspec")
class TestHrrrFetcher(unittest.TestCase):
    def tearDown(self) -> None:
        HrrrFetcher.LOCATION_MAPPING.clear()

    def test_location_mapper_callable(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper=lambda x: x,
            selector="select",
            selector_args={},
        )
        result = [instance.location_mapper(x) for x in [1.0, "Capybara", None]]
        self.assertListEqual(result, [1.0, "Capybara", None])

    def test_location_mapper_resource_lookup(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args={},
            resource_type="meter/electrical",
            resource_query="",
        )
        self.assertEqual(instance.location_mapper, "RESOURCE_LOOKUP")
        with self.assertRaises(TypeError):
            instance.location_mapper("Aristotle")

    @unittest.skip("Not implemented yet")
    def test_location_mapper_prefix(self, mock_fsspec, mock_open_dataset):
        pass

    def test_select_blob_12_hour_horizon(self, mock_fsspec, mock_open_dataset):
        base = "gcs://gcp-public-data-weather/high-resolution-rapid-refresh/version_2"
        instance = HrrrFetcher(
            location_mapper=lambda x: x,
            selector="select",
            selector_args={},
            source_mode="12_hour_horizon",
        )

        blobs = instance.select_blob(
            np.datetime64("2022-07-01 00:00:00"), np.datetime64("2022-12-01 06:00:00")
        )
        self.assertIsInstance(blobs, list)
        expected = [
            f"{base}/monthly_horizon/conus/hrrr.202206/hrrr.wrfsfcf.12_hour_horizon.zarr",
            f"{base}/monthly_horizon/conus/hrrr.202207/hrrr.wrfsfcf.12_hour_horizon.zarr",
            f"{base}/monthly_horizon/conus/hrrr.202208/hrrr.wrfsfcf.12_hour_horizon.zarr",
            f"{base}/monthly_horizon/conus/hrrr.202209/hrrr.wrfsfcf.12_hour_horizon.zarr",
            f"{base}/monthly_horizon/conus/hrrr.202210/hrrr.wrfsfcf.12_hour_horizon.zarr",
            f"{base}/monthly_horizon/conus/hrrr.202211/hrrr.wrfsfcf.12_hour_horizon.zarr",
            f"{base}/monthly_horizon/conus/hrrr.202212/hrrr.wrfsfcf.12_hour_horizon.zarr",
            f"{base}/monthly_horizon/conus/hrrr.202301/hrrr.wrfsfcf.12_hour_horizon.zarr",
        ]
        self.assertListEqual(blobs, expected)

        mock_fsspec.filesystem.assert_called_once_with("gcs", token=None)
        mock_exists = mock_fsspec.filesystem.return_value.exists

        mock_exists.assert_has_calls([call(blob) for blob in expected], any_order=True)

        # Show it filtering a single blob
        mock_exists.side_effect = lambda blob: True if "202301" not in blob else False
        blobs = instance.select_blob(
            np.datetime64("2022-07-01 00:00:00"), np.datetime64("2022-12-01 06:00:00")
        )
        self.assertIsInstance(blobs, list)
        self.assertListEqual(blobs, expected[:-1])

        # Assert raises for all not found
        mock_exists.side_effect = lambda _: False
        with self.assertRaises(FileNotFoundError):
            instance.select_blob(
                np.datetime64("2022-07-01 00:00:00"),
                np.datetime64("2022-12-01 06:00:00"),
            )

    def test_select_blob_18_hour_forecast(self, mock_fsspec, mock_open_dataset):
        base = "gcs://gcp-public-data-weather/high-resolution-rapid-refresh/version_2"
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args={},
            source_mode="18_hour_forecast",
            resource_type="meter/electrical",
            resource_query="",
        )

        blob_name = instance.select_blob(
            np.datetime64("2022-07-01 00:00:00"), np.datetime64("2022-07-01 06:00:00")
        )
        self.assertEqual(
            blob_name,
            f"{base}/forecast_run/conus/hrrr.20220701/hrrr.t00z.wrfsfcf.18_hour_forecast.zarr",
        )

        with self.assertRaises(RuntimeError):
            instance.select_blob(
                np.datetime64("2022-07-01 00:00:00"),
                np.datetime64("2022-07-03 06:00:00"),
            )

    def test_select_blob_48_hour_forecast(self, mock_fsspec, mock_open_dataset):
        base = "gcs://gcp-public-data-weather/high-resolution-rapid-refresh/version_2"
        instance = HrrrFetcher(
            location_mapper=lambda x: x,
            selector="select",
            selector_args={},
            source_mode="48_hour_forecast",
        )

        # At multiples of 6 hours, we should select blob corresponding to the start time
        blob_name = instance.select_blob(
            np.datetime64("2022-07-01 06:00:00"), np.datetime64("2022-07-01 07:00:00")
        )
        self.assertEqual(
            blob_name,
            f"{base}/forecast_run/conus/hrrr.20220701/hrrr.t06z.wrfsfcf.48_hour_forecast.zarr",
        )

        # Otherwise, we should select blob corresponding to the last multiple of 6 hrs
        blob_name = instance.select_blob(
            np.datetime64("2022-07-01 01:00:00"), np.datetime64("2022-07-01 06:00:00")
        )
        self.assertEqual(
            blob_name,
            f"{base}/forecast_run/conus/hrrr.20220701/hrrr.t00z.wrfsfcf.48_hour_forecast.zarr",
        )

        blob_name = instance.select_blob(
            np.datetime64("2022-07-01 08:00:00"), np.datetime64("2022-07-01 12:00:00")
        )
        self.assertEqual(
            blob_name,
            f"{base}/forecast_run/conus/hrrr.20220701/hrrr.t06z.wrfsfcf.48_hour_forecast.zarr",
        )

        blob_name = instance.select_blob(
            np.datetime64("2022-06-30 23:00:00"), np.datetime64("2022-07-01 12:00:00")
        )
        self.assertEqual(
            blob_name,
            f"{base}/forecast_run/conus/hrrr.20220630/hrrr.t18z.wrfsfcf.48_hour_forecast.zarr",
        )

        with self.assertRaises(RuntimeError):
            instance.select_blob(
                np.datetime64("2022-07-01 00:00:00"),
                np.datetime64("2022-07-08 06:00:00"),
            )

    def test_update_mapping(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        self.assertEqual(instance.location_mapping, None)
        self.assertDictEqual(instance._location_mapping, {})
        self.assertDictEqual(instance.LOCATION_MAPPING, {})
        self.assertEqual(instance.location_mapper, "RESOURCE_LOOKUP")

        new_mapping = {"boo": "hoo"}

        def mapper(x):
            return new_mapping[x]

        with patch.multiple(
            "time_series_models.data_fetchers.hrrr_fetcher.HrrrFetcher",
            _update_location_mapper=DEFAULT,
            _update_mapping=DEFAULT,
        ) as mocks:
            mocks["_update_location_mapper"].return_value = mapper
            mocks["_update_mapping"].return_value = new_mapping

            instance.update_mapping("boo", "yay")
            mocks["_update_location_mapper"].assert_called_once()
            mocks["_update_mapping"].assert_called_with("boo", "yay")

        self.assertDictEqual(instance._location_mapping, new_mapping)
        self.assertDictEqual(instance.LOCATION_MAPPING, new_mapping)
        result = instance._location_mapper("boo")
        self.assertEqual(result, "hoo")
        with self.assertRaises(KeyError):
            instance._location_mapper("yay")

    def test_update_mapping__resynchronize(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        instance._location_mapping = {"new": "value"}
        self.assertDictEqual(instance._location_mapping, {"new": "value"})
        self.assertDictEqual(instance.LOCATION_MAPPING, {})
        self.assertEqual(instance.location_mapper, "RESOURCE_LOOKUP")

        instance.update_mapping()
        self.assertDictEqual(instance._location_mapping, {"new": "value"})
        self.assertDictEqual(instance.LOCATION_MAPPING, {"new": "value"})

        HrrrFetcher.LOCATION_MAPPING = {"reset": "class attribute"}
        instance.update_mapping()
        expected = {
            "new": "value",
            "reset": "class attribute",
        }
        self.assertDictEqual(instance._location_mapping, expected)
        self.assertDictEqual(instance.LOCATION_MAPPING, expected)

    def test_map_locations(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args={},
            resource_type="meter/electrical",
            resource_query="",
        )
        query_return = [
            ("meter/electrical/Zeno", 39.41655, 252.8293611),
            ("meter/electrical/Kant", 31.631, 258.821),
        ]
        with self.assertLogs(level="WARNING") as logger_warnings:
            with patch.object(
                HrrrFetcher, "_query_db", return_value=query_return
            ) as mock_query_db:
                result = instance.map_locations(
                    "meter/electrical/Zeno",
                    "meter/electrical/Kant",
                    "meter/electrical/Hipparchia",
                )
                mock_query_db.assert_called_with(
                    "meter/electrical/Zeno",
                    "meter/electrical/Kant",
                    "meter/electrical/Hipparchia",
                )

            self.assertEqual(len(logger_warnings.records), 1)
            self.assertEqual(
                logger_warnings.records[0].getMessage(),
                f"No records found for 1 locations: ['meter/electrical/Hipparchia'] ",
            )

        expected = {
            "meter/electrical/Zeno": {"latitude": 39.41655, "longitude": 252.8293611},
            "meter/electrical/Kant": {"latitude": 31.631, "longitude": 258.821},
            # no trace of Hipparchia...
        }
        self.assertDictEqual(result, expected)

    def test__update_location_mapper(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        with self.assertRaises(TypeError):
            # location_mapper is initially just a str literal
            instance.location_mapper("boo")

        # update mapper without updating mapping
        instance.location_mapper = instance._update_location_mapper()
        with self.assertRaises(KeyError):
            # now it's a callable, but with an empty mapping!
            instance.location_mapper("boo")

        # update mapper after updating mapping
        instance._location_mapping = {"boo": "hoo"}
        instance.location_mapper = instance._update_location_mapper()
        self.assertEqual(instance.location_mapper("boo"), "hoo")

    def test__update_mapping(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        self.assertEqual(instance.location_mapping, None)
        self.assertDictEqual(instance._location_mapping, {})

        mapping = {"boo": "hoo"}
        with patch.object(
            HrrrFetcher, "map_locations", return_value=mapping
        ) as mock_map_locations:
            # OK to call with no locations!
            result1 = instance._update_mapping()
            # expect no lookup
            mock_map_locations.assert_not_called()
            # call again with two locations
            result2 = instance._update_mapping("boo", "yay")
            # expect lookup of the two locations
            mock_map_locations.assert_called_with("boo", "yay")
        # no-op update
        self.assertDictEqual(result1, {})
        self.assertDictEqual(result2, mapping)
        # without actually assigning to instance.location_mapping, we haven't changed it!
        self.assertDictEqual(instance._location_mapping, {})

    def test__update_mapping__superset(self, mock_fsspec, mock_open_dataset):
        # start with a non-empty mapping
        mapping = {"boo": "hoo"}
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            location_mapping=mapping,
            resource_type="meter/electrical",
            resource_query="",
        )
        self.assertDictEqual(instance.location_mapping, mapping)

        new_mapping = {"hi": "there"}
        with patch.object(
            HrrrFetcher, "map_locations", return_value=new_mapping
        ) as mock_map_locations:
            result = instance._update_mapping("hi", "boo")
            # "boo" key is already present, so only "hi" requires a new mapping
            mock_map_locations.assert_called_with("hi")
        expected = {"boo": "hoo", "hi": "there"}
        self.assertDictEqual(result, expected)
        # without actually assigning to instance.location_mapping, we haven't changed it!
        # so the location_mapping is the original mapping
        self.assertDictEqual(instance.location_mapping, mapping)

    def test__update_mapping__subset(self, mock_fsspec, mock_open_dataset):
        mapping = {"boo": "hoo", "hi": "there"}
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            location_mapping=mapping,
            resource_type="meter/electrical",
            resource_query="",
        )
        self.assertDictEqual(instance.location_mapping, mapping)
        with patch.object(HrrrFetcher, "map_locations") as mock_map_locations:
            result = instance._update_mapping("hi", "boo")
            # "hi" and "boo" keys are already present, so _update_mapping should not call map_locations
            mock_map_locations.assert_not_called()
        # the return value of _update_mapping is the original mapping, unchanged
        self.assertDictEqual(result, mapping)

    def test_select(self, mock_fsspec, mock_open_dataset):
        locations = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
            "Kant": dict(latitude=31.631, longitude=258.821),
        }
        locations_z = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
        }
        locations_k = {
            "Kant": dict(latitude=31.631, longitude=258.821),
        }

        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )

        domain = make_domain(
            np.datetime64("2022-07-01 06:00:00"),
            np.datetime64("2022-07-01 10:00:00"),
            np.timedelta64(1, "h"),
            "Zeno",
        )

        with patch.object(
            HrrrFetcher, "map_locations", return_value=locations_z
        ) as mock_map_locations:
            result = instance.get_data(domain)
            mock_map_locations.assert_called_with("Zeno")

        expected = np.asarray(
            [
                [6.0, 18.0],
                [7.0, 18.0],
                [8.0, 18.0],
                [9.0, 18.0],
                [10.0, 18.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        # Add a second expected location
        # TODO: make this a separate test
        domain = make_domain(
            np.datetime64("2022-07-01 06:00:00"),
            np.datetime64("2022-07-01 10:00:00"),
            np.timedelta64(1, "h"),
            "Zeno",
            "Kant",
        )

        with patch.object(
            HrrrFetcher, "map_locations", return_value=locations_k
        ) as mock_map_locations:
            result = instance.get_data(domain)
            mock_map_locations.assert_called_with("Kant")

        kant_expected = np.asarray(
            [
                [6.0, 11.0],
                [7.0, 11.0],
                [8.0, 11.0],
                [9.0, 11.0],
                [10.0, 11.0],
            ]
        )
        np.testing.assert_array_equal(result, np.vstack((expected, kant_expected)))

        self.assertDictEqual(instance._location_mapping, locations)

    def test_select__missing_mapping(self, mock_fsspec, mock_open_dataset):
        locations = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
            "Kant": dict(latitude=31.631, longitude=258.821),
        }
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        domain = make_domain(
            np.datetime64("2022-07-01 06:00:00"),
            np.datetime64("2022-07-01 10:00:00"),
            np.timedelta64(1, "h"),
            "Zeno",
            "Plato",
            "Kant",
        )

        with self.assertLogs(level="WARNING") as logger_warnings:
            with patch.object(
                HrrrFetcher, "map_locations", return_value=locations
            ) as mock_map_locations:
                result = instance.get_data(domain)
                mock_map_locations.assert_called_with("Kant", "Plato", "Zeno")

            self.assertEqual(len(logger_warnings.records), 1)
            self.assertEqual(
                logger_warnings.records[0].getMessage(),
                f"Missing geo mapping for one or more domain locations: ['Plato']",
            )

        self.assertDictEqual(instance._location_mapping, locations)

        zeno_expected = np.asarray(
            [
                [6.0, 18.0],
                [7.0, 18.0],
                [8.0, 18.0],
                [9.0, 18.0],
                [10.0, 18.0],
            ]
        )
        plato_expected = np.zeros_like(zeno_expected)
        plato_expected[:] = np.nan
        kant_expected = np.asarray(
            [
                [6.0, 11.0],
                [7.0, 11.0],
                [8.0, 11.0],
                [9.0, 11.0],
                [10.0, 11.0],
            ]
        )
        np.testing.assert_array_equal(
            result, np.vstack((zeno_expected, plato_expected, kant_expected))
        )

        self.assertEqual(instance.get_feature_names(), ["time_like", "lat_like"])

    def test_pickle_instance_mapping(self, mock_fsspec, mock_open_dataset):
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        instance.location_mapping = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
            "Kant": dict(latitude=31.631, longitude=258.821),
        }

        with io.BytesIO() as bio:
            pickle.dump(instance, bio, protocol=5)
            bio.seek(0)

            pickled_instance = pickle.load(bio)

        self.assertDictEqual(
            instance.location_mapping, pickled_instance.location_mapping
        )

        # Each instance has its own instance mapping mapping
        self.assertIsNot(instance.location_mapping, pickled_instance.location_mapping)

        # But the class mapping is the same object
        self.assertIs(instance.LOCATION_MAPPING, pickled_instance.LOCATION_MAPPING)

    def test_pickle_class_mapping(self, mock_fsspec, mock_open_dataset):
        mapping = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
            "Kant": dict(latitude=31.631, longitude=258.821),
        }

        HrrrFetcher.LOCATION_MAPPING |= mapping

        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        self.assertDictEqual(instance.LOCATION_MAPPING, mapping)

        with io.BytesIO() as bio:
            pickle.dump(instance, bio, protocol=5)
            bio.seek(0)

            # Clear the class attribute (as though we are in a different process
            instance.LOCATION_MAPPING.clear()

            pickled_instance = pickle.load(bio)

        # The unpickled object doesn't get the class attribute state from when it was pickeled
        self.assertDictEqual(pickled_instance.LOCATION_MAPPING, {})

        # But we unioned it onto the instance mapping before pickling!
        self.assertDictEqual(pickled_instance._location_mapping, mapping)

    @unittest.skip("TODO: reimplement group for HrrrFetcher!")
    def test_group(self, mock_fsspec, mock_open_dataset):
        locations = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
            "Kant": dict(latitude=31.631, longitude=258.821),
        }
        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="group",
            selector_args=dict(
                grouper=pd.Grouper(freq="1D"),
                method="aggregate",
                kwargs=dict(
                    func={
                        "time_like": ["min", "max", "mean"],
                        "lon_like": ["mean"],
                    },
                ),
            ),
            resource_type="meter/electrical",
            resource_query="",
        )

        domain = make_domain(
            np.datetime64("2022-07-01 00:00:00"),
            np.datetime64("2022-07-03 00:00:00"),
            np.timedelta64(1, "D"),
            "Zeno",
        )

        with patch.object(
            HrrrFetcher, "map_locations", return_value=locations
        ) as mock_map_locations:
            result = instance.get_data(domain)
            mock_map_locations.assert_called_with("Zeno", "Kant")

        expected = np.asarray(
            [
                [0.0, 23.0, 11.5, 16.0],
                [24.0, 47.0, 35.5, 16.0],
                [48.0, 71.0, 59.5, 16.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        # Add an expected friend
        domain = make_domain(
            np.datetime64("2022-07-01 00:00:00"),
            np.datetime64("2022-07-03 00:00:00"),
            np.timedelta64(1, "D"),
            "Zeno",
            "Kant",
        )

        result = instance.get_data(domain)

        kant_expected = np.asarray(
            [
                [0.0, 23.0, 11.5, 22.0],
                [24.0, 47.0, 35.5, 22.0],
                [48.0, 71.0, 59.5, 22.0],
            ]
        )

        np.testing.assert_array_equal(result, np.vstack((expected, kant_expected)))

        # Add an interloper
        domain = make_domain(
            np.datetime64("2022-07-01 00:00:00"),
            np.datetime64("2022-07-03 00:00:00"),
            np.timedelta64(1, "D"),
            "Camus",
            "Zeno",
            "Kant",
        )

        result = instance.get_data(domain)
        camus_expected = np.zeros_like(expected)
        camus_expected[:] = np.nan
        np.testing.assert_array_equal(
            result, np.vstack((camus_expected, expected, kant_expected))
        )

        self.assertEqual(
            instance.get_feature_names(),
            ["time_like_min", "time_like_max", "time_like_mean", "lon_like_mean"],
        )

    @unittest.skip("TODO: reimplement group for HrrrFetcher!")
    def test_group_numba(self, mock_fsspec, mock_open_dataset):
        def numba_func(values, index) -> float:
            if len(values) > 5:
                return values[2:5].mean()
            else:
                return np.nan

        numba_func.__str__ = lambda: "some_numba_func"

        instance = HrrrFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="group",
            selector_args=dict(
                grouper=pd.Grouper(freq="1D"),
                method="aggregate",
                kwargs=dict(
                    engine="numba",
                    func={
                        "time_like": ["min", "max", numba_func],
                        "lat_like": ["median"],
                    },
                ),
            ),
            resource_type="meter/electrical",
            resource_query="",
        )

        domain = make_domain(
            np.datetime64("2022-07-01 00:00:00"),
            np.datetime64("2022-07-03 00:00:00"),
            np.timedelta64(1, "D"),
            "Plato",
        )

        new = {"Plato": dict(latitude=39.41655, longitude=252.8293611)}
        with patch.object(
            HrrrFetcher, "map_locations", return_value=new
        ) as mock_map_locations:
            result = instance.get_data(domain)
            mock_map_locations.assert_called_with("Plato")

        expected = np.asarray(
            [
                [0.0, 23.0, 3.0, 18.0],
                [24.0, 47.0, 27.0, 18.0],
                [48.0, 71.0, 51.0, 18.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

        self.assertEqual(
            instance.get_feature_names(),
            [
                "time_like_min",
                "time_like_max",
                "time_like_some_numba_func",
                "lat_like_median",
            ],
        )

    def test_shift_longitude(self, mock_fsspec, mock_open_dataset):
        df = pd.DataFrame(
            {
                "longitude": [10, -10, 0],
                "latitude": [0, 20, -20],
            },
            index=["a", "b", "c"],
        )
        HrrrFetcher._shift_longitude(df)

        expected = pd.DataFrame(
            {
                "longitude": [10, 350, 0],
                "latitude": [0, 20, -20],
            },
            index=["a", "b", "c"],
        )
        pd.testing.assert_frame_equal(df, expected)


"""
If these tests cause segfaults, you can run them under gdb and valgrind to try and find issues
bazel test --test_output=all --cache_test_results=no //forecasting/time_series_models/data_fetchers/tests:test_hrrr_fetcher
bazel test --run_under "valgrind --trace-children=yes --vgdb=full" --test_output=all  --cache_test_results=no //forecasting/time_series_models/data_fetchers/tests:test_hrrr_fetcher
bazel test --run_under "gdb --eval-command=run --args python3"  --test_output=all --cache_test_results=no //forecasting/time_series_models/data_fetchers/tests:test_hrrr_fetcher
"""


def make_mapper(rpath):
    return rpath
    # fs = fsspec.filesystem("file")
    # return fs.get_mapper(rpath).root


@unittest.skip("not properly implemented yet with local test fixture")
class TestHrrrFetcherIntegration(unittest.TestCase):
    """
    This class tests the `load_hrr_data` method twice with the same setup, except that
    once the test is run without caching, and the second time it is run with caching.
    """

    def setUp(self) -> None:
        memory.clear()

    def run(
        self, result: unittest.result.TestResult | None = ...
    ) -> unittest.result.TestResult | None:
        # Control the type of dask scheduler to debug kerchunk
        # with dask.config.set(scheduler="single-threaded"):
        with dask.config.set(scheduler="threading"):
            return super().run(result)

    def test_without_cache(self):
        # This method is patched by the class decorator
        self._test_method()

    @patch(
        "time_series_models.data_fetchers.hrrr_fetcher.rpath_mapper",
        make_mapper,
    )
    def _test_method(self):
        # This method is not patch by the class decorator
        lat_xr = xr.DataArray(
            [
                41.813466664122394,
                37.77114229257195,
            ],
            [("location", ["david", "camus"])],
        )

        lon_xr = xr.DataArray(
            [360 - 71.43792877881117, 360 - 122.4193933237225],
            [("location", ["david", "camus"])],
        )

        data = load_hrrr_data(
            "data/high-resolution-rapid-refresh_version_2_monthly_horizon_conus_hrrr.202304_hrrr.wrfsfcf.06_hour_horizon.zarr",
            ["t", "dswrf"],
            lat_xr,
            lon_xr,
            np.datetime64("2023-04-02T00"),
            np.datetime64("2023-04-02T01"),
        )
        print(data)

        expected = pd.DataFrame.from_records(
            [
                (
                    pd.Timestamp("2023-04-02 00:00:00"),
                    "david",
                    285.492645,
                    0.0,
                    753,
                    1611,
                ),
                (
                    pd.Timestamp("2023-04-02 00:00:00"),
                    "camus",
                    287.492645,
                    463.2,
                    600,
                    178,
                ),
                (
                    pd.Timestamp("2023-04-02 01:00:00"),
                    "david",
                    283.577209,
                    0.0,
                    753,
                    1611,
                ),
                (
                    pd.Timestamp("2023-04-02 01:00:00"),
                    "camus",
                    285.639709,
                    253.6,
                    600,
                    178,
                ),
            ],
            columns=["valid_time", "location", "t", "dswrf", "x", "y"],
            index=["valid_time", "location"],
        )

        pd.testing.assert_frame_equal(data, expected)

    def test_with_cache(self):
        # This method is patched by the class decorator
        with (
            tempfile.TemporaryDirectory() as cache_path,
            patch.dict(os.environ, {"GRID_FETCHER_CACHE_DIR": cache_path})
            # Patch again and add the cache dir
        ):
            self._test_method()
            self.assertTrue(
                os.path.exists(cache_path + "/1067085501"),
                "Unexpected hash of gcs checksums or caching just didn't work",
            )


if __name__ == "__main__":
    # Use detailed logging with process and thread names for debugging kerchunk tests
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ P:%(processName)s T:%(threadName)s %(levelname)s:%(filename)s:%(funcName)s:%(message)s",
    )
    unittest.main()
