import logging

import pandas as pd

import sklearn.metrics

logger = logging.getLogger(__name__)


def _score(true, predicted):
    # TODO: use an enum to control which scores are returned
    return {
        "RMSE": sklearn.metrics.mean_squared_error(true, predicted, squared=False),
        "median_absolute_error": sklearn.metrics.median_absolute_error(true, predicted),
    }


def score_df_at_locs(
    df_true: pd.DataFrame,
    df_predicted: pd.DataFrame,
    score_at_locs: pd.DataFrame,
    pooled: bool = False,
) -> pd.DataFrame:
    """
    Score the parts of a DataFrame that were gap-filled, either by column (pooled=False) or in aggregate (pooled=True).
    :param df_true: the original DataFrame prior to the introduction of synthetic gaps
    :param df_predicted: the processed DataFrame with all synthetic gaps re-filled
    :param score_at_locs: a boolean mask indicating the locations of synthetic gaps
    :param pooled: if False (default), then score by column; if True, return single score after pooling all series.
    :return: a DataFrame of error metrics by series (if not pooled) or globally for the entire set (if pooled)
    """
    if pooled:
        df_t = df_true.melt()
        df_p = df_predicted.melt()
        sl = score_at_locs.melt()
        scores = _score(df_t[sl], df_p[sl])
    else:
        scores = {}
        for col in df_true.columns:
            scores[col] = _score(
                df_true.loc[score_at_locs[col], col],
                df_predicted.loc[score_at_locs[col], col],
            )
    return pd.DataFrame.from_dict(scores)
