import logging

import numpy as np

from time_series_models.transformers_calendar import (
    harmonic_transform_pipeline,
    is_business_day_pipeline,
    one_hot_encode_day_of_week_pipeline,
)

logger = logging.getLogger(__name__)


def append_business_day_transformer(to_list: list) -> None:
    """
    Append a business-day feature transformer to a list of transformers. Modifies the list inplace!
    :param to_list: the list of transformers
    :return: None
    """
    to_list.append(
        (
            "is_",
            is_business_day_pipeline(),
            [0],
        )
    )
    logger.debug("appended business-day transformer")


def append_day_of_week_transformer(to_list: list) -> None:
    """
    Append a day-of-week feature transformer to a list of transformers. Modifies the list inplace!
    :param to_list: the list of transformers
    :return: None
    """
    to_list.append(
        (
            "day_of_week",
            one_hot_encode_day_of_week_pipeline(handle_unknown="error"),
            [0],
        )
    )
    logger.debug("appended day-of-week transformer")


def append_harmonic_transformers(to_list: list, harmonics: list) -> None:
    """
    Append harmonic feature transformers to a list of transformers. Modifies the list inplace!
    :param to_list: the list of transformers
    :param harmonics: the list of harmonics to construct
    :return: None
    """
    # TODO(Michael H): generalize to work with non-hourly data
    for harmonic_val in harmonics:
        to_list.append(
            (
                "harmonic_{}_hr".format(harmonic_val.astype(int)),
                harmonic_transform_pipeline(harmonic_val, np.datetime64("1970-01-01")),
                [0],
            )
        )
    logger.debug("appended harmonic transformers with %s", harmonics)
