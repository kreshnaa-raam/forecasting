import logging
import typing

import numpy as np

from time_series_models.data_fetchers.fetcher import Fetcher

logger = logging.getLogger(__name__)


class NanFetcher(Fetcher):
    """
    TODO deprecate or add tests?
    Currently only used as a hack for physical PV Forecasts with no observed range values
    """

    def __init__(self, location_mapper, selector, selector_args, *args, **kwargs):
        """
        Initialize the AmiFetcher for AMI data.
        Allows any args
        """
        super().__init__(
            location_mapper or (lambda x: x),
            selector=selector or "select",
            selector_args=selector_args
            or dict(
                variables=[
                    "nans",
                ]
            ),
        )
        if args or kwargs:
            logger.info("NanFetcher ignoring args: %s, kwargs: %s", args, kwargs)

    @property
    def variables(self):
        return self.selector_args.get("variables")

    @variables.setter
    def variables(self, variables):
        self.selector_args["variables"] = variables

    def get_all_data(self, domain_location):
        raise NotImplementedError

    def get_feature_names(self):
        return self.variables

    def source_loader(self, start, end, domain_location):
        raise NotImplementedError

    def get_data(self, domain):
        return np.full(domain.shape, np.nan)
