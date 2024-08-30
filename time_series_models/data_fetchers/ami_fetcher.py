import datetime
import logging
import os
import multiprocessing
import typing

from typing import (
    Literal,
    Generator,
    List,
)
from itertools import islice

import numpy as np
import pandas as pd

from joblib import (
    Memory,
    Parallel,
    delayed,
)
from google.cloud.exceptions import NotFound

from time_series_models.constants import DATE_TIME
from time_series_models.data_fetchers.fetcher import Fetcher
from time_series_models.transformers import is_iterable

logger = logging.getLogger(__name__)

memory = Memory(os.path.join(os.path.sep, "tmp", "time_series_models_cache"), verbose=0)

# TODO: Deprecate this Fetcher?


@memory.cache(ignore=["buffer_loader"])
def load_ami_data(
    uri,
    buffer_loader,
    parse_dates=False,
    compression="gzip",
) -> pd.DataFrame:
    read_csv_dtypes = {
        "power (W)": float,
        "volts (V)": float,
        "energy (Wh)": float,
    }
    try:
        with buffer_loader(uri) as buf:
            dataframe = pd.read_csv(
                buf,
                index_col=0,
                compression=compression,
                dtype=read_csv_dtypes,
                skip_blank_lines=False,
            ).rename_axis(DATE_TIME)
        if parse_dates:
            # convert to UTC and then drop the tz offset
            dataframe.index = pd.to_datetime(
                dataframe.index,
                utc=True,
            ).tz_localize(tz=None)
            # TODO(Michael H): make consistent with other Fetchers by returning consistently spaced data
    except NotFound:
        # note the missing URI and return an empty DataFrame
        logger.warning("Missing AMI data: %s", uri)
        # data colname will be overwritten later but let's identify with uri for now:
        dataframe = pd.DataFrame(columns=[DATE_TIME, uri])
        # depending on whether we are parsing dates, index is expected to be datetime or str
        index_dtype_selector = {
            True: np.dtype("datetime64[ns]"),
            False: str,
        }
        dataframe[DATE_TIME] = dataframe[DATE_TIME].astype(
            index_dtype_selector[parse_dates]
        )
        dataframe[uri] = dataframe[uri].astype(float)
        dataframe = dataframe.set_index(DATE_TIME)

    if dataframe.empty:
        logger.info("Returning empty DataFrame for AMI file: %s", uri)

    return dataframe


class AmiFetcher(Fetcher):
    """
    Experimental AMI data fetcher for R&D; API subject to change!
    """

    # TODO: update GCS_BUCKET and GCS_PATH_FORMAT depending on where prod data land
    GCS_BUCKET = "seto2243-forecasting"
    GCS_PATH_FORMAT = "ami_validation/{meter:s}_{units:s}.csv.gz"

    # for testing only # TODO(Michael H): extract to Fetcher class?
    FILE_SYSTEM_URIS = {}
    FILE_PATH = ""

    def __init__(
        self,
        location_mapper=lambda x: x,
        freq=None,
        uri_formatter: str = None,
        units: Literal["energy", "volts", "watts"] = "watts",
    ):
        """
        Initialize the AmiFetcher for AMI data.
        :param location_mapper: a mapping from domain_location to AMI meter ID(s)
        :param freq: the desired frequency of AMI data
        :param tz_to: the timezone to which datetimes will be converted prior to dropping TZ offsets
        :param uri_format: optionally override the uri format string, e.g. for loading from a file
        :param units: select whether to retrieve "energy", "volts", or "watts" data
        """
        super().__init__(
            location_mapper,
            selector="select",
            selector_args=dict(variables=None),
        )
        self.freq = freq
        self.uri_formatter = uri_formatter or self.GCS_PATH_FORMAT
        if units.lower() not in ["energy", "volts", "watts"]:
            raise ValueError("'units' must be one of ['energy', 'volts', 'watts']")
        self.units = units

    @property
    def variables(self):
        return self.selector_args.get("variables")

    @variables.setter
    def variables(self, variables):
        self.selector_args["variables"] = variables

    @staticmethod
    def split_every(n, iterable) -> Generator[List, None, None]:
        """
        Split an iterable into smaller groups.
        # TODO(Michael H): extract this to a general utils script
        # from https://stackoverflow.com/a/1915307
        :param n: desired size of new groups
        :param iterable: the iterable to be split
        :return: a generator of lists
        """
        i = iter(iterable)
        piece = list(islice(i, n))
        while piece:
            yield piece
            piece = list(islice(i, n))

    def source_loader(self, start, end, domain_location) -> pd.DataFrame:
        """
        This method maps a specified domain_location to a set of meter IDs,
        and loads data for that location from start to end.
        :param start: the start of the time range for which to load data
        :param end: the end of the time range for which to load data
        :param domain_location: a domain location for which data is to be identified and loaded
        :return: Pandas DataFrame
        """
        mid = self._build_mids(domain_location)  # get meter IDs for a domain_location
        uris = self._build_uris(start, end, mid)

        # Should this be a thread based parallel backend?
        # When the result is memoized starting the process backend with popen is slow
        # When actually making requests to GCS, we do want to make the requests asynchronously
        logger.debug("Loading blobs: %s", uris)
        data_frames = Parallel(backend="threading", n_jobs=multiprocessing.cpu_count())(
            delayed(load_ami_data)(
                uri,
                self.gcs_buffer_loader,
                parse_dates=False,
            )
            for uri in uris
        )

        result = pd.concat(
            [df for df in data_frames if df is not None],
            axis=1,
        )
        # since date parsing is expensive for long files, do it after concatenation
        result.index = pd.to_datetime(result.index, utc=True).tz_localize(tz=None)
        return self._postprocess(result, mid=mid, start=start, end=end)

    def get_all_data(self, domain_location):
        """
        Not implemented for AmiFetcher. There is a lot of data.
        TODO: implement anyways
        :param domain_location:
        :return:
        """
        raise NotImplementedError

    def get_feature_names(self):
        return self.variables

    def file_loader(self, start, end, domain_location):
        """
        Load all data from a file (or set of files) that can be mapped to `domain_location`.
        Patch this method in to replace source_loader for testing
        :param start: not used, but necessary for matching call signature of source_loader
        :param end: not used, but necessary for matching call signature of source_loader
        :param domain_location: the location for which to load data
        :return: DataFrame
        """
        mid = self._build_mids(domain_location)  # get meter IDs for a domain_location
        uris = [
            os.path.join(self.FILE_PATH, self.FILE_SYSTEM_URIS[m]) for m in mid
        ]  # replaces `_build_uris`
        data_frames = [
            load_ami_data(
                uri,
                self.file_buffer_loader,
                parse_dates=True,
            )
            for uri in uris
        ]
        data_frames = pd.concat([df for df in data_frames], axis=1)
        # no need for pd.to_datetime(index) since index dates are already parsed
        return self._postprocess(data_frames, mid)

    def _build_mids(self, domain_location):
        """
        Use the fetcher's location mapper to get all meter IDs associated with a domain_location.
        :param domain_location:
        :return: an iterable of meter ID
        """
        mid = self.location_mapper(domain_location)
        if not is_iterable(mid):
            mid = [mid]
        return mid

    def _build_uris(self, start, end, mid):
        # TODO(Michael H): align with other Fetcher uri formatters (e.g. if data are split into monthly chunks?)
        return [self.uri_formatter.format(meter=m, units=self.units) for m in mid]

    def _postprocess(self, dataframe, mid, start=None, end=None):
        dataframe.columns = mid
        self.variables = mid
        dataframe = dataframe.sort_index()

        # TODO(Michael H): start and end should limit what data is fetched rather than slicing post hoc
        if start is not None:
            dataframe = dataframe.loc[dataframe.index >= start]
        if end is not None:
            dataframe = dataframe.loc[dataframe.index <= end]
        if self.freq is not None:
            dataframe = dataframe.resample(self.freq).mean()
        return dataframe
