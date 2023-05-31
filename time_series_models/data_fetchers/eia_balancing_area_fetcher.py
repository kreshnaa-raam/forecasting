import os
import typing

import numpy as np
import pandas as pd
from joblib import Memory

from time_series_models.data_fetchers.fetcher import Fetcher

memory = Memory(os.path.join(os.path.sep, "tmp", "time_series_models_cache"), verbose=1)


@memory.cache(ignore=["buffer_loader"])
def load_eia_data(uri, buffer_loader):
    """
    Parser for EIA Excel files
    For now, just cache what we get.
    TODO set bytes_limit for the cache!

    :param uri: a unique data uri appropriate for the buffer_loader
    :param buffer_loader: a context managed function which manages the resources associated with the buffer
    :return:
    """
    with buffer_loader(uri) as buf:
        with pd.ExcelFile(buf) as xls:
            return pd.read_excel(xls, "Published Hourly Data").set_index("UTC time")


class EiaBalancingAreaFetcher(Fetcher):
    """
    Get observed EIA Balancing Area Load data in UTC Timezone

    Do not cache data state in the instance!
    Use https://joblib.readthedocs.io/en/latest/memory.html for caching with explicit date range only!

    Link for PSCo download from EIA: https://www.eia.gov/electricity/gridmonitor/knownissues/xls/PSCO.xlsx
    """

    PATH = "EiaBalancingArea/v1"
    URIS = {
        "PSCo": "EiaBalancingArea.xlsx",
    }

    def __init__(self, location_mapper, variables, zeros_to_nan=True):
        """
        Fetcher for EIA Balancing Area load time series
        :param variables: the variables from the EIA 'Published Hourly Data' sheet to present
        """
        # TODO(Michael H): move selector and selector_args to constructor interface?
        super().__init__(
            location_mapper, selector="select", selector_args={"variables": variables}
        )
        self.location_mapper = location_mapper
        self.zeros_to_nan = zeros_to_nan

    def get_all_data(self, domain_location):
        return self.source_loader(None, None, domain_location)

    @property
    def variables(self):
        return self.selector_args.get("variables")

    def get_data(self, domain):
        """
        fetch EIA Load for the location(s) specified in the domain
        Select the appropriate variables
        :param domain:
        :return:
        """
        res = super().get_data(domain)
        if self.zeros_to_nan:
            res = np.where(res == 0, np.nan, res)
        return res

    def source_loader(self, start, end, domain_location):
        uri = os.path.join(self.PATH, self.URIS[self.location_mapper(domain_location)])
        return load_eia_data(uri, self.gcs_buffer_loader)

    def file_loader(self, start, end, domain_location):
        """
        patch this method in to replace source_loader for testing
        :param start:
        :param end:
        :param domain_location:
        :return:
        """
        return load_eia_data(
            self.URIS[self.location_mapper(domain_location)], self.file_buffer_loader
        )

    def get_feature_names(self):
        return self.variables
