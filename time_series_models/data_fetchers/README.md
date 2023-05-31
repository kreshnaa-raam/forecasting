# Data Fetchers
This directory contains fetchers and associated tests for the `time_series_models` library.

Currently implemented fetchers include:
* `fetcher.py`: Fetcher abstract base class that defines the core API and internal behavior; includes MultiFetcher,
 the base class for fetchers that can read data for multiple domain locations simultaneously.
* `ami_fetcher.py`: AmiFetcher for fetching AMI data from GCS.
* `eia_balancing_area_fetcher.py`: EiaBalancingAreaFetcher reads EIA Balancing Area data from Excel spreadsheets.
* `gridded_data_fetcher.py`: GriddedDataFetcher base class for retrieving gridded data products like HRRR.
* `hrrr_fetcher.py`: HrrrFetcher is an implementation of GriddedDataFetcher for retrieving NOAA's HRRR forecasts.

Scripts with additional helper functions:
* `numba_groupby_functions.py`: functions for use with `pandas.DataFrame.groupby` operations with `engine="numba"`.
* `selector_functions.py`: GroupHours class for creating a custom pandas time grouper and a compatible function.

Fetcher exists primarily to bring feature data into memory for RegularTimeSeriesModel classes.
Since RTSM is built from a sklearn Pipeline, the component pieces (including Fetcher)
must implement `fit` and `transform` methods. For Fetcher, `fit` is a no-op that returns
self, whereas `transform` simply calls `get_data`.

Fetcher's `get_data` method can handle any padding that is needed for proper aggregation, with some of the internals
handled by the `_load_data` method, including loading data from source and routing through the `select` or `group`
method with which the Fetcher instance was configured. `source_loader` accomplishes the actual fetching of data from
the desired source. `select` ensures that only the requested features are returned,
and does a simple time series reindex of the data that was loaded from source: non-selected time stamps are excluded
whereas time stamps with no corresponding source data are inserted with null values. `group` uses a dict of source
variable to list of aggregations/transformations that is supplied in `selector_args` during Fetcher configuration, and
applies the specified functions to the specified variables in sequence (optionally using the `numba` engine for better
[performance](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#numba-jit-compilation) via
just-in-time compilation).

The base class Fetcher supports using ResourceLookup to map domain location to the identity or geo location used
in the source loader. There is some very clever caching to avoid hitting the DB too often.
TODO: consolidate testing for ResourceLookup behavior while still ensuring correctness in subclasses