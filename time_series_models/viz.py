import logging
import pandas as pd
import statsmodels.api as sm
from plotly import graph_objects as go
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

COLORS = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


def plot_predictions(predictions_df):
    """
    Plot the result of `predict_dataframe` method for a trained model, with or without range.
    Usage in colab:
        locations = ["meter/electrical/1234", "meter/electrical/7890"]
        plot_predictions(
            model.predict_dataframe("2023-04-01", "2023-04-13", *locations, range=True)
        )

    :param predictions_df: predictions dataframe with domain multi index
    :return: plotly figure
    """
    fig = go.Figure()

    pivot_predictions = predictions_df.reset_index().pivot(
        index="date_time", columns="location"
    )

    for col, color in zip(pivot_predictions["predicted"].columns, COLORS):
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=pivot_predictions.index,
                y=pivot_predictions["predicted", col],
                marker_color=color,
                mode="lines+markers",
                name="forecast",
                legendgroup=col,
            )
        )

        if "true" in pivot_predictions.columns:
            # Add traces
            fig.add_trace(
                go.Scatter(
                    x=pivot_predictions.index,
                    y=pivot_predictions["true", col],
                    marker_color=color,
                    mode="lines+markers",
                    name="actual",
                    marker=dict(symbol="star", size=10),
                    legendgroup=col,
                    legendgrouptitle_text=col,
                )
            )
    return fig


def plot_residual_scatter(
    predictions: pd.DataFrame,
    true_col: str = "true",
    predicted_col: str = "predicted",
    standardize: bool = False,
    show_trend: bool = True,
    n_sample: int = 50_000,
    unit: str = None,
    title: str = None,
    return_fig: bool = False,
):
    """
    Plot the standardized prediction residuals for a model. If residuals are not randomly distributed
    (the "shotgun blast" pattern), the model might be a poor fit for the data.
    :param predictions: pandas DataFrame containing a column of true values and a column of predicted values
    :param true_col: the name of the column of true value
    :param predicted_col: the name of the column of predicted values
    :param standardize: whether to standardize the residuals or leave them on the scale of the data
    :param show_trend: display a LOWESS trendline
    :param n_sample: the maximum number of observations to sample for plotting. Large samples can cause OOM!
    :param unit: the unit of measurment (e.g., 'MW', 'V', etc.)
    :param title: specify a custom title for the figure
    :param return_fig: if False (default), show the figure and return None; if True, return the go.Figure object instead
    :return: None (default) or the go.Figure object if return_fig=True
    """
    if len(predictions) > n_sample:
        logger.info(
            "subsampling to %i observations (out of %i total)",
            n_sample,
            len(predictions),
        )
        predictions = predictions.sample(n=n_sample)

    residuals = predictions[predicted_col] - predictions[true_col]

    if standardize:
        # Standardize the residuals. StandardScaler expects -- and returns -- a 2D array
        _standardized = StandardScaler().fit_transform(
            residuals.to_numpy().reshape(-1, 1)
        )
        # revert to 1D array
        residuals = _standardized.reshape(-1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions[predicted_col],
            y=residuals,
            mode="markers",
            marker=dict(opacity=0.5),
            name="residual",
        )
    )

    if show_trend:
        lowess = sm.nonparametric.lowess(
            residuals, predictions[predicted_col], frac=0.1
        )
        fig.add_trace(
            go.Scatter(
                x=lowess[:, 0],
                y=lowess[:, 1],
                mode="lines",
                line_color="black",
                line=dict(dash="dash"),
                name="LOWESS",
            )
        )
    # TODO(Michael H): maybe plot abs(residuals)

    # if units are provided, add them to x axis label
    _unit = ""
    if unit is not None:
        _unit = f" ({unit})"

    fig.update_layout(
        title=title or "Residual Scatter",
        xaxis_title=f"Predicted{_unit}",
        yaxis_title="Standardized " * standardize + "Residual",
        width=1000,
        height=800,
    )
    if return_fig:
        # you do you
        return fig
    fig.show()
