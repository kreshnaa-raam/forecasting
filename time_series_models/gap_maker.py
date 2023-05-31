import logging
import random
import warnings

import numpy as np
import pandas as pd

from typing import Literal

logger = logging.getLogger(__name__)


class GapMaker1D:
    """
    Note, self.synth_gap_idx reflects random sampling order, and is not guaranteed to follow temporal order.
    """

    def __init__(self, series: pd.Series):
        self._series = series
        self.series_nan_idx_set = set(series.loc[series.isna()].index)
        self.series_notna_idx = series.loc[series.notna()].index
        self.synth_gap_idx = None

    @property
    def series(self):
        return self._series.copy()

    def make_synth_gaps(
        self,
        method: Literal["random", "block"] = "random",
        n: int = None,
        frac: float = None,
        random_state=32,
        block_size: int = None,
        n_attempts: int = 3,
        **kwargs,
    ) -> pd.Series:
        """
        Return a copy of the input series with the specified number or fraction of synthetic gaps introduced.
        Gaps are introduced only where the input series does not already have missing data. Gaps can be introduced to
        random (non-missing) locations one at a time, or in blocks of a specified size, or following any other callable
        method that takes a series as an input and returns the same series with gaps as an output.

        Block gaps follow the additional constraint that they are not placed contiguously with any missing values,
        real or synthetic. Since block gaps are introduced sequentially, they are unlikely to be placed consistent with
        maximum packing of gaps, so there may be insufficient contiguous observations to place the remaining  blocks.
        This method will make several new attempts before raising a RuntimeError.
        Caution: If you find yourself here, seeking optimal (i.e., maximum) gap packing, the gap distribution might be
        more uniform than random!

        :param method: the method to use for making gaps. Can be "random" for random sampling, or "block" for random
            sampling of blocks of specified size.
        :param n: if random sampling, then the number of observations to convert to gaps (can be contiguous); if block
            sampling, then the number of block gaps to introduce (blocks will not be contiguous)
        :param frac: if random sampling, the fraction of non-missing observations to convert into gaps; if block
            sampling, the fraction of the entire series to attempt to convert into gaps.
        :param random_state: set for reproducibility, else leave as None for strongest randomization
        :param block_size: if block sampling, specify the sizing of synthetic gaps.
        :param n_attempts: if block sampling, how many times to attempt finding a solution.
        :param kwargs: optional keyword arguments to pass to a gap making method. E.g., if random sampling,
            the keyword arguments will be passed to pandas.Series.sample.
        :return: a pandas Series with synthetic gaps
        """

        if method == "random":
            if block_size is not None:
                warnings.warn(
                    "Ignoring arg 'block_size' for random sampling. Did you mean to use 'block' method?",
                    RuntimeWarning,
                )
            return self._make_random_gaps(
                n=n,
                frac=frac,
                random_state=random_state,
                **kwargs,
            )

        if method == "block":
            return self._make_block_gaps(
                n_blocks=n,
                frac=frac,
                random_state=random_state,
                block_size=block_size,
                n_attempts=n_attempts,
            )

        raise NotImplementedError(f"Method '{method}' is not implemented at this time")

    def _make_random_gaps(
        self,
        n: int = None,
        frac: float = None,
        random_state=None,
        **sample_kwargs,
    ) -> pd.Series:
        """
        Helper method to sample a specified subset of non-missing observations in the target time series using
        pandas.Series.sample method, and place synthetic gaps at these locations. Samples are drawn with equal weight,
        without replacement, and can be adjacent to one another or to pre-existing gaps in the series.
        :param n: the number of observations to convert to gaps. Set either 'n' or 'frac', not both.
        :param frac: the fraction of (non-missing) observations to convert to gaps. Set either 'frac' or n', not both.
        :param random_state: set for reproducibility, else leave as None for strongest randomization,
        :param sample_kwargs: optional keyword arguments to pass to pandas.Series.sample
        :return: a pandas Series with synthetic gaps
        """
        self.synth_gap_idx = (
            self.series.loc[self.series_notna_idx]
            .sample(n=n, frac=frac, random_state=random_state, **sample_kwargs)
            .index
        )
        series_with_synth_gaps = self.series
        series_with_synth_gaps.loc[self.synth_gap_idx] = np.nan
        return series_with_synth_gaps

    def _make_block_gaps(
        self,
        block_size: int = 1,
        n_blocks: int = None,
        frac: float = None,
        n_attempts: int = 3,
        random_state=None,
    ) -> pd.Series:
        """
        Helper method to make gaps of a specified size. To guarantee gap size, synthetic gaps are never placed adjacent
        to any other gaps, whether actual or synthetic.
        :param block_size: controls the size of synthetic gaps
        :param n_blocks: how many block gaps to create. Set either n_blocks or frac, but not both.
        :param frac: what fraction of the total series (including actual gaps!) to convert to synthetic gaps.
        :param n_attempts: if the request requires near-maximum packing of synthetic gaps, the first attempt to place
            gaps could run out of space given the constraints. Trying again several times could help find a solution.
            Note that maximum packing results in a uniform distribution of gaps rather than a random distribution.
        :param random_state: set for reproducibility, else leave as None for strongest randomization.
        :return: a pandas Series with synthetic gaps
        """
        # TODO (Michael H): convert to numpy random generator to avoid modifying global random state
        random.seed(random_state)
        if n_blocks is None and frac is not None:
            n_blocks = int(frac * len(self.series) / block_size)
        elif n_blocks is None and frac is None:
            logger.debug("no gaps specified -> no-op -> returning original series!")
            return self.series
        elif n_blocks is not None and frac is not None:
            raise RuntimeError("Set only one of 'n_gaps' or 'frac', not both")

        for n in list(range(n_attempts)):
            # adapted from https://stackoverflow.com/a/69291286
            # construct all possible blocks of length block_size+2 (buffered so gaps are not adjacent)
            blocks = [
                list(range(i, i + block_size + 2))
                for i in range(0, len(self.series) - block_size)
            ]
            # eliminate any blocks that already have missing data
            blocks = [
                b
                for b in blocks
                if not any(element in self.series_nan_idx_set for element in b)
            ]
            nan_idx = list()
            for i in range(n_blocks):
                try:
                    gap_indices = random.choice(blocks)
                    # use the middle elements of the blocks; outer elements are buffers
                    nan_idx += gap_indices[1:-1]
                    # exclude all blocks which include already selected indices so that we don't overlap gaps
                    blocks = [
                        b
                        for b in blocks
                        if not any(element in gap_indices for element in b)
                    ]
                except IndexError:
                    logger.debug(
                        "Ran out of space for new gaps on attempt #%s of %s",
                        n + 1,
                        n_attempts,
                    )
                    break

            if len(nan_idx) == n_blocks * block_size:
                # TODO(Michael H): should this be a DatetimeIndex named DATE_TIME?
                self.synth_gap_idx = pd.Index(nan_idx)
                series_with_synth_gaps = self.series
                series_with_synth_gaps.loc[self.synth_gap_idx] = np.nan
                return series_with_synth_gaps

        raise RuntimeError(f"Failed to find solution after {n_attempts} attempts")


class GapMaker:
    """This class creates synthetic gaps in each constituent Series,
    and keeps track of synthetic gap locations."""

    def __init__(self, dataframe: pd.DataFrame):
        """
        Constructor for DataFrame GapMaker.
        :param dataframe: the DataFrame of time series into which synthetic gaps will be introduced
        """
        self._original = dataframe
        self.synth_gaps = None

    @property
    def original(self):
        return self._original.copy()

    def make_synth_gaps(self, random_state=None, **kwargs) -> pd.DataFrame:
        """
        Run GapMaker against each Series in the DataFrame.
        We cannot use a simple pandas DataFrame.sample() because pre-existing gaps don't align.
        :param random_state: set for reproducibility, or leave as None for stronger randomization
        :param kwargs: keyword args for `GapMaker1D.make_synth_gaps()`, including method, etc.
        :return: DataFrame with synthetic gaps
        """
        series_with_gaps = []
        # we want a different random state for each series GapMaker1D so set random state here, locally
        rng = np.random.default_rng(random_state)
        for col in self.original.columns:
            gap_maker = GapMaker1D(self.original[col])
            series_with_gaps.append(
                gap_maker.make_synth_gaps(
                    random_state=rng.integers(9999999),
                    **kwargs,
                )
            )
        df_with_gaps = pd.concat(series_with_gaps, axis=1)
        self.synth_gaps = self.original.notna() & df_with_gaps.isna()
        return df_with_gaps
