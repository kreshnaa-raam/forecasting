"""
A fetcher for Kerchunk aggregations of gridded data using fsspec reference file system.
This fetcher design assumes interpolation to a point/points.

To cache aggregations and reference data locally, set the env var "GRID_FETCHER_CACHE_DIR"
The Simple Cache assumes data is static. When using caching, new HRRR data will not be available.
The aggregated data will be pinned at the time of the first read.
"""
import logging
import math
import os
import typing
import zlib
import dask
import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from joblib import Memory
from kerchunk.combine import MultiZarrToZarr

from time_series_models.constants import LOCATION, DATE_TIME
from time_series_models.data_fetchers.gridded_data_fetcher import (
    GriddedDataFetcher,
)
from time_series_models.exceptions import FetcherConfigurationError
from time_series_models.time_unit import TimeUnitEnum
from time_series_models.transformers import (
    is_iterable,
    multiindex_from_domain,
)

logger = logging.getLogger(__name__)

# configure to /home/builder/time_series_models_cache to persist across VM start/stop
cache_path = os.environ.get(
    "CACHE_PATH", os.path.join(os.path.sep, "tmp", "time_series_models_cache")
)
cache_verbosity = int(os.environ.get("CACHE_VERBOSITY", 0))
memory = Memory(cache_path, verbose=cache_verbosity)


@memory.cache()
def load_hrrr_data(
    blob_path: str | typing.Iterable[str],
    variables: list,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    start: np.datetime64,
    end: np.datetime64,
):
    try:
        logger.info("HRRR load request for %s to %s", start, end)
        logger.info(
            "Latitudes: count - %s, min - %s, max - %s, mean - %s",
            latitude.count().values,
            latitude.min().values,
            latitude.max().values,
            latitude.mean().values,
        )
        logger.info(
            "Longitude: count - %s, min - %s, max - %s, mean - %s",
            longitude.count().values,
            longitude.min().values,
            longitude.max().values,
            longitude.mean().values,
        )
    except Exception:
        logger.exception("Failed on logging!")

    # Force dask to use threads
    with dask.config.set(scheduler="threading"):
        ds = xr.open_dataset(
            rpath_mapper(blob_path),
            drop_variables=["heightAboveGround"],  # Drop magic broken variable
            engine="zarr",
            backend_kwargs=dict(consolidated=False),
            chunks={"valid_time": 1},
        )

        # Find the grid location closest to geo_location by minimizing the squared error of x and y coordinates
        dist = (ds.latitude - latitude) ** 2 + (ds.longitude - longitude) ** 2
        res = dist.argmin(dim=["x", "y"])

        ds_data = ds[variables].loc[dict(valid_time=pd.IndexSlice[start:end])]

        if logger.level == logging.DEBUG:
            try:
                for name, da_res in res.items():
                    logger.info(
                        "RES %s: count - %s, min - %s, max - %s, mean - %s",
                        name,
                        da_res.count().values,
                        da_res.min().values,
                        da_res.max().values,
                        da_res.mean().values,
                    )

                logger.info("Data count for time range: %s", ds_data.count().compute())
                logger.info("Data min for time range: %s", ds_data.min().compute())
                logger.info("Data max for time range: %s", ds_data.max().compute())
            except Exception:
                logger.exception("Logging failed!")

        # TODO: Log distance to nearest point. Eventually add interpolation and sanity checks.
        result = ds_data.interp(res).to_dataframe()

    logger.info("HRRR loaded from source - result description: %s", result.describe())

    return result


# TODO: Add doc string.
# Very opaque magic from fsspec here https://nbviewer.org/gist/rsignell-usgs/f2021d29079aac029b3787d5578a6fea
def rpath_mapper(rpath: str | typing.Iterable[str]):
    r_opts = {"anon": True}

    if is_iterable(rpath):
        logger.debug("detected multiple paths, performing runtime aggregation")
        # compute a monthly aggregation at runtime
        s_opts = None
        mzz = MultiZarrToZarr(
            rpath,  # Rpaths should start with gcs://
            remote_protocol="gcs",
            remote_options=dict(token=None),
            concat_dims=["valid_time"],
            identical_dims=["latitude", "longitude", "step"],
        )
        # For multiple paths, combine the references into an in memory dictionary to the reference filesystem
        fo = mzz.translate()
        blob_paths = rpath
    else:
        s_opts = r_opts
        # For a single path just pass the path to the reference filesystem
        fo = rpath
        blob_paths = [rpath]

    fs = fsspec.filesystem(
        protocol="reference",
        fo=fo,  # The target zarr json blob
        ref_storage_args=s_opts,
        remote_protocol="gcs",
        remote_options=r_opts,
    )
    cache_dir = os.getenv("GRID_FETCHER_CACHE_DIR")
    if cache_dir is None:
        logger.info(
            "Not using fsspec simplecache because GRID_FETCHER_CACHE_DIR is not set"
        )
        return fs.get_mapper("")
    else:
        # Use the hash of the monthly zarr reference file checksums to invalidate the fsspec cache
        # fsspec doesn't like it when the references mutate, even when strictly adding refs
        # This will lead to disk space leakage which will need to be cleaned up next.
        gcfs = fsspec.filesystem("gcs")
        file_checksums = ",".join(map(str, sorted(map(gcfs.checksum, blob_paths))))

        rpath_suffix = str(zlib.crc32(bytes(file_checksums, "UTF8")))
        cache_path = os.path.join(cache_dir, rpath_suffix)
        logger.info(
            "Detected GRID_FETCHER_CACHE_DIR=%s -> using fsspec simplecache",
            cache_path,
        )
        return fsspec.filesystem(
            "simplecache", cache_storage=cache_path, fs=fs
        ).get_mapper("")


# Source Mode names match suffix in GCS blobs
# todo: for horizon mode, resolve underlying file formats and expose client to
#  00horizon - 48horizon only?
SOURCE_MODES = (
    [f"{val:02}_hour_horizon" for val in range(0, 19)]
    + [f"{val:02}-{val+5:02}_hour_horizon" for val in range(19, 44, 6)]
    + ["18_hour_forecast", "48_hour_forecast"]
)


class HrrrFetcher(GriddedDataFetcher):
    """
    GriddedDataFetcher configured for the HRRR model in the form extracted by our NOAA-NWP service
    The following source modes are available:
    1. Aggregations fixing forecast time horizon (corresponding to various diagonals
        in [this](https://www.unidata.ucar.edu/software/tds/current/tutorial/files/FmrcPoster.pdf) chart).
        These are denoted by the **horizon source modes.
    2. Aggregations over the next **hour at any forecast run time (corresponding to
        various verticals in [this](https://www.unidata.ucar.edu/software/tds/current/tutorial/files/FmrcPoster.pdf)
        chart. These are denoted by the 18hour (published hourly) and
        48hour (published every 6 hours) source modes.

    If your domain crosses a month boundary you will get NAN values.

    This could become a common base class or actually take the model path structure as an argument... decide when we
    have enough gridded data products to look for good patterns.
    """

    HRRR_GCS_BUCKET = "gcp-public-data-weather"
    # Must define storage in this class for location mappings
    LOCATION_MAPPING: dict = {}

    def __init__(
        self,
        location_mapper: typing.Callable | typing.Literal["RESOURCE_LOOKUP"],
        selector: typing.Literal["select", "group"],
        selector_args: dict,
        location_mapping: dict | None = None,
        resource_type: str | None = None,
        resource_query: str | None = None,
        source_mode: str = "48_hour_forecast",
    ):
        """
        Initialize Fetcher base class and configure selector for variable selection or groupby operations.
        :param location_mapper: a mapping from domain_location to a dict(latitude={value1}, longitude={value2}).
            Can be the string literal value "RESOURCE_LOOKUP", in which case location_mapper will be constructed by
            connecting to a backend DB and reading from the resource model.
        :param selector: either "select" or "group"
        :param selector_args: dict of arguments to pass to the select or group method, including:
            "method": Literal["aggregate", "transform"] for type of groupby operation, used only if selector is "group";
            "variables": List[str] with variables to retain, used only if not providing a dict of aggregations;
            "grouper": a pd.Grouper used for specifying frequency for grouped aggregation (with selector="group"); and
            "kwargs": dict of additional keyword arguments, including:
                "func": the aggregating or transforming function(s), either as a single function or as a dict of
                    variable_name:List[aggregations], used only if selector is "group"; and
                "engine": the engine to use for groupby computation, Pandas default with engine=None is Cython
                    but can also specify "numba"
        :param location_mapping: hook to initialize HrrrFetcher with a non-empty mapping
        :param resource_type: The camus canonical resource model type to query in RESOURCE LOOKUP location mapping
        :param source_mode: string value used to determine the slice of the gridded forecast to load
        """
        logger.debug("constructing HrrrFetcher with source_mode=%s", source_mode)

        # Hard code the model name for now
        # See https://www.nco.ncep.noaa.gov/pmb/products/hrrr/ for HRRR product details
        self.model = "wrfsfcf"

        if source_mode in SOURCE_MODES:
            self.source_mode = source_mode
        else:
            raise FetcherConfigurationError(
                f"Unknown source_mode argument {source_mode} not in {SOURCE_MODES}"
            )
        super().__init__(
            location_mapper,
            selector,
            selector_args,
            location_mapping=location_mapping,
            resource_type=resource_type,
            resource_query=resource_query,
        )

    @property
    def variables(self):
        match self.selector:
            case "select":
                return self.selector_args.get("variables")
            case "group":
                return self.selector_args.get("kwargs", {}).get("func", {}).keys()

    def select_blob(self, start: np.datetime64, end: np.datetime64):
        """
        Give a start time and an end time and a source_mode, try to guess what hrrr blob we want to load.
        This method may be too smart... it tries to do something reasonable based on the requested forecast time.
        For the horizon selection, it is pretty reasonable.

        The 18hour and 48hour modes are a bit wonky though. It would break if you wanted to get the last few hours of
        an 18 or 48 hour file - it would assume you want the beginning of a more recent blob. We simplify this to assume
        the start input is the same as the hrrr blob forecast time. That is likely to be the mode we operate in
        initially for HCE.
        We should avoid doing a GCS glob if possible.
        :param start:
        :param end:
        :return:
        """
        # extract the start time components to help build the blob name
        year = TimeUnitEnum.YEAR.as_unit(start)
        month = TimeUnitEnum.MONTH.as_unit(start)
        day = TimeUnitEnum.DAY.as_unit(start)
        hour = TimeUnitEnum.HOUR.as_unit(start)

        logger.debug(
            "Selecting blob based on start datetime: %s, %s, %s, %s",
            year,
            month,
            day,
            hour,
        )

        base = f"gcs://{self.HRRR_GCS_BUCKET}/high-resolution-rapid-refresh/version_2"
        match self.source_mode:
            case "18_hour_forecast":
                if end - start > np.timedelta64(18, "h"):
                    raise RuntimeError(
                        f"Grid Point Fetcher configured with 18 hour model, "
                        f"but requested range is {end - start}"
                    )

                return f"{base}/forecast_run/conus/hrrr.{year}{month:02}{day:02}/hrrr.t{hour:02}z.{self.model}.{self.source_mode}.zarr"

            case "48_hour_forecast":
                if end - start > np.timedelta64(48, "h"):
                    raise RuntimeError(
                        f"Grid Point Fetcher configured with 48 hour model, "
                        f"but requested range is {end - start}"
                    )

                return f"{base}/forecast_run/conus/hrrr.{year}{month:02}{day:02}/hrrr.t{math.floor(hour/6) * 6:02}z.{self.model}.{self.source_mode}.zarr"

            case _:
                start_padded = start.astype("datetime64[M]") - np.timedelta64(1, "M")
                end_padded = end.astype("datetime64[M]") + np.timedelta64(2, "M")
                dates = np.arange(start_padded, end_padded, np.timedelta64(1, "M"))
                dates = zip(
                    TimeUnitEnum("Y").as_unit(dates),
                    TimeUnitEnum("M").as_unit(dates),
                )
                blobs = []
                fs = fsspec.filesystem("gcs", token=None)
                for year, month in dates:
                    blob = f"{base}/monthly_horizon/conus/hrrr.{year}{month:02}/hrrr.{self.model}.{self.source_mode}.zarr"
                    if fs.exists(blob):
                        blobs.append(blob)
                    else:
                        logger.debug("Data for month %s doesn't exist", blob)

                if not blobs:
                    raise FileNotFoundError(
                        f"None of the requested monthly HRRR blobs could be found in {base}/monthly_horizon/conus/"
                    )

                logger.debug("Loading hrrr for: %s", blobs)
                return blobs

    def source_loader(self, start, end, *domain_locations):
        """
        This method implicitly assumes the HRRR dataset is hourly.
        If you pass a start/end with units of days, it will pad out the request. This may lead to nan values.
        Beware using this with daily aggregation for the 18hour and 48 hour slices.
        """
        return super().source_loader(start, end, *domain_locations)

    def _load_from_source(
        self, blob_path, latitude: xr.DataArray, longitude: xr.DataArray, start, end
    ):
        hrrr_data = load_hrrr_data(
            blob_path=blob_path,
            variables=self.variables,
            latitude=latitude,
            longitude=longitude,
            start=start,
            end=end,
        )
        return (
            hrrr_data
            # standardize index
            .rename_axis(index={"location": LOCATION, "valid_time": DATE_TIME})
            .reorder_levels([LOCATION, DATE_TIME])
            .sort_index()
        )

    @staticmethod
    def _shift_longitude(geo_locations: pd.DataFrame) -> None:
        # HRRR data is on a lambert-conformal projection and longitude spans
        # 0 to 360 degrees, as opposed to the conventional -180 to 180 degrees.
        geo_locations["longitude"].loc[geo_locations.longitude < 0] += 360
