import io
import unittest
import logging
from unittest.mock import patch, DEFAULT, PropertyMock

import cloudpickle as pickle
import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da

from time_series_models.transformers import (
    make_domain,
    multiindex_from_domain,
)
from time_series_models.data_fetchers.gridded_data_fetcher import (
    GriddedDataFetcher,
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
@patch.multiple(GriddedDataFetcher, __abstractmethods__=set())
class TestGriddedDataFetcher(unittest.TestCase):
    def tearDown(self) -> None:
        GriddedDataFetcher.LOCATION_MAPPING.clear()

    def test_location_mapper_callable(self, mock_open_dataset):
        instance = GriddedDataFetcher(
            location_mapper=lambda x: x,
            selector="select",
            selector_args={},
        )
        result = [instance.location_mapper(x) for x in [1.0, "Capybara", None]]
        self.assertListEqual(result, [1.0, "Capybara", None])

    def test_location_mapper_resource_lookup(self, mock_open_dataset):
        instance = GriddedDataFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args={},
            resource_type="my/type",
            resource_query="",
        )
        self.assertEqual(instance.location_mapper, "RESOURCE_LOOKUP")
        with self.assertRaises(TypeError):
            instance.location_mapper("Aristotle")

    @unittest.skip("Not implemented yet")
    def test_location_mapper_prefix(self, mock_open_dataset):
        pass

    def test_update_mapping(self, mock_open_dataset):
        instance = GriddedDataFetcher(
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
            "time_series_models.data_fetchers.gridded_data_fetcher.GriddedDataFetcher",
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

    def test_update_mapping__resynchronize(self, mock_open_dataset):
        instance = GriddedDataFetcher(
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

        GriddedDataFetcher.LOCATION_MAPPING = {"reset": "class attribute"}
        instance.update_mapping()
        expected = {
            "new": "value",
            "reset": "class attribute",
        }
        self.assertDictEqual(instance._location_mapping, expected)
        self.assertDictEqual(instance.LOCATION_MAPPING, expected)

    def test_map_locations(self, mock_open_dataset):
        instance = GriddedDataFetcher(
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
                GriddedDataFetcher, "_query_db", return_value=query_return
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

    @unittest.skip("TODO")
    def test_query_db(self, mock_open_dataset):
        pass

    def test__update_location_mapper(self, mock_open_dataset):
        instance = GriddedDataFetcher(
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

    def test__update_mapping_noop(self, mock_open_dataset):
        instance = GriddedDataFetcher(
            location_mapper="RESOURCE_LOOKUP",
            selector="select",
            selector_args=dict(
                variables=["time_like", "lat_like"],
            ),
            resource_type="meter/electrical",
            resource_query="",
        )
        self.assertDictEqual(instance._location_mapping, {})

        mapping = {"boo": "hoo"}
        with patch.object(
            GriddedDataFetcher, "map_locations", return_value=mapping
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
        self.assertEqual(instance.location_mapping, None)
        self.assertDictEqual(instance._location_mapping, {})

    def test__update_mapping__superset(self, mock_open_dataset):
        # start with a non-empty mapping
        mapping = {"boo": "hoo"}
        instance = GriddedDataFetcher(
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
            GriddedDataFetcher, "map_locations", return_value=new_mapping
        ) as mock_map_locations:
            result = instance._update_mapping("hi", "boo")
            # "boo" key is already present, so only "hi" requires a new mapping
            mock_map_locations.assert_called_with("hi")
        expected = {"boo": "hoo", "hi": "there"}
        self.assertDictEqual(result, expected)
        # without actually assigning to instance.location_mapping, we haven't changed it!
        # so the location_mapping is the original mapping
        self.assertDictEqual(instance.location_mapping, mapping)

    def test__update_mapping__subset(self, mock_open_dataset):
        mapping = {"boo": "hoo", "hi": "there"}
        instance = GriddedDataFetcher(
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
        with patch.object(GriddedDataFetcher, "map_locations") as mock_map_locations:
            result = instance._update_mapping("hi", "boo")
            # "hi" and "boo" keys are already present, so _update_mapping should not call map_locations
            mock_map_locations.assert_not_called()
        # the return value of _update_mapping is the original mapping, unchanged
        self.assertDictEqual(result, mapping)

    def test_select(self, mock_open_dataset):
        domain = make_domain(
            np.datetime64("2021-05-17"),
            np.datetime64("2021-05-21"),
            np.timedelta64(1, "D"),
            "Zeno",
            "Socrates",
        )
        dataframe = pd.DataFrame(
            data={
                "temperature": [1.1, 2.2, 3.3, 4.4],
                "disposition": [2.1, 3.2, 4.3, 5.4],
                "velocity": [3.1, 4.2, 5.3, 6.4],
            },
            index=pd.MultiIndex.from_product(
                [
                    ["Zeno"],
                    pd.date_range(start="2021-05-17", periods=4, freq="D"),
                ]
            ),
        )
        location_mapping = {
            "Zeno": dict(latitude=39, longitude=253),
            "Kant": dict(latitude=32, longitude=259),
        }
        with patch.object(
            GriddedDataFetcher,
            "variables",
            new_callable=PropertyMock(return_value=["temperature", "disposition"]),
        ):
            instance = GriddedDataFetcher(
                location_mapper=lambda x: location_mapping.get(
                    x, dict(latitude=np.nan, longitude=np.nan)
                ),
                selector="select",
                selector_args=dict(
                    variables=["temperature", "disposition"],
                ),
            )
            result = instance.select(dataframe, domain)

        expected = pd.DataFrame(
            data={
                "temperature": [1.1, 2.2, 3.3, 4.4, np.nan] + 5 * [np.nan],
                "disposition": [2.1, 3.2, 4.3, 5.4, np.nan] + 5 * [np.nan],
            },
            index=multiindex_from_domain(domain),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_pickle_instance_mapping(self, mock_open_dataset):
        instance = GriddedDataFetcher(
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

    def test_pickle_class_mapping(self, mock_open_dataset):
        mapping = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
            "Kant": dict(latitude=31.631, longitude=258.821),
        }

        GriddedDataFetcher.LOCATION_MAPPING |= mapping

        instance = GriddedDataFetcher(
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

    @unittest.skip("TODO: reimplement group for GriddedDataFetcher!")
    def test_group(self, mock_open_dataset):
        locations = {
            "Zeno": dict(latitude=39.41655, longitude=252.8293611),
            "Kant": dict(latitude=31.631, longitude=258.821),
        }
        instance = GriddedDataFetcher(
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
            GriddedDataFetcher, "map_locations", return_value=locations
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

    @unittest.skip("TODO: reimplement group for GriddedDataFetcher!")
    def test_group_numba(self, mock_open_dataset):
        def numba_func(values, index) -> float:
            if len(values) > 5:
                return values[2:5].mean()
            else:
                return np.nan

        numba_func.__str__ = lambda: "some_numba_func"

        instance = GriddedDataFetcher(
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
            GriddedDataFetcher, "map_locations", return_value=new
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

    def test_shift_longitude(self, mock_open_dataset):
        df_original = pd.DataFrame(
            {
                "longitude": [10, -10, 0],
                "latitude": [0, 20, -20],
            },
            index=["a", "b", "c"],
        )
        df_copy = df_original.copy(deep=True)
        GriddedDataFetcher._shift_longitude(df_copy)
        pd.testing.assert_frame_equal(df_original, df_copy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
