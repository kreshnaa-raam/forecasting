import math

import numpy as np
from collections import defaultdict

from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from plotly import figure_factory
import sklearn.metrics

from time_series_models.transformers import rank


class MetricsRunner:
    """
    A container for running models and comparing metrics
    Organized around monthly analysis
    """

    def __init__(self, **models):
        self.models = models

    def fit(self, start, end, *locations, **kwargs):
        for model in self.models.values():
            model.fit(start, end, *locations, **kwargs)

    def metrics_history(
        self,
        start,
        end,
        location,
        step=np.timedelta64(1, "M"),
        dtype="datetime64[M]",
        open_step=np.timedelta64(1, "D"),
    ):
        results = defaultdict(dict)
        for start_date in np.arange(start, end, step=step, dtype=dtype):
            end_date = start_date + step - open_step  # don't use closed intervals

            for mname, model in self.models.items():
                results[mname][start_date] = model.metrics(
                    start_date, end_date, location
                )
        return results

    def predict(self, start, end, location):
        return {
            key: {
                "predicted": model.predict(start, end, location),
                "true": model.range(start, end, location),
                "domain": model.domain(start, end, location),
            }
            for key, model in self.models.items()
        }

    def plot_metrics_history(self, start, end, location):
        mresults = self.metrics_history(start, end, location)
        fig = make_subplots(
            rows=7,
            cols=1,
            shared_xaxes="all",
            subplot_titles=[
                "Mean Absolute Percent Error",
                "Mean Square Error",
                "Mean Absolute Error",
                "Mean Squared Log Error",
                "R Squared",
                "Max Error",
                "Explained Variance Score",
            ],
            y_title="Score",
            x_title="Date",
            vertical_spacing=0.03,
            horizontal_spacing=0.03,
        )

        VAR_METRICS = [
            "mean_absolute_percentage_error",
            "mean_squared_error",
            "mean_absolute_error",
            "mean_squared_log_error",
            "r2_score",
            "max_error",
            "explained_variance_score",
        ]

        for row, var in enumerate(VAR_METRICS):
            for idx, (key, data) in enumerate(mresults.items()):
                fig.add_trace(
                    go.Scatter(
                        x=list(data.keys()),
                        y=[v[var] for v in data.values()],
                        marker=dict(
                            color=qualitative.Bold[idx], cmin=0, cmax=len(mresults) + 1
                        ),
                        legendgroup=idx,
                        name=key,
                        showlegend=row == 0,
                        # yaxis="y1",
                        mode="markers",
                    ),
                    row=row + 1,
                    col=1,
                )

        fig.update_layout(
            title="Model Scores", showlegend=True, height=1000, width=1200
        )
        fig.update_yaxes(rangemode="tozero")
        return fig

    def plot_predictions(self, start, end, location):
        predictions = self.predict(start, end, location)
        fig = go.Figure()
        for idx, (key, data) in enumerate(predictions.items()):
            if idx == 0:
                fig.add_trace(
                    go.Scatter(
                        x=data["domain"]["date_time"].reshape(-1),
                        y=data["true"].reshape(-1),
                        mode="lines+markers",
                        marker=dict(color="blue", symbol="diamond-x-open"),
                        name="Actual",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=data["domain"]["date_time"].reshape(-1),
                    y=data["predicted"].reshape(-1),
                    mode="lines+markers",
                    marker=dict(
                        color=qualitative.Bold[idx], cmin=0, cmax=len(predictions) + 1
                    ),
                    name=key,
                )
            )

        fig.update_layout(
            title="Model Results",
            xaxis_title="Date",
            yaxis_title="MW",
            showlegend=True,
            height=1000,
            width=1200,
        )

        fig.show()

    def plot_prediction_peaks(self, start, end, location, k_peaks=5):
        predictions = self.predict(start, end, location)
        fig = go.Figure()
        for idx, (key, data) in enumerate(predictions.items()):
            if idx == 0:
                actual_idx = np.argsort(-data["true"].reshape(-1))[:k_peaks]

                fig.add_trace(
                    go.Scatter(
                        x=data["domain"]["date_time"].reshape(-1)[actual_idx],
                        y=data["true"].reshape(-1)[actual_idx],
                        mode="markers",
                        marker=dict(color="blue", symbol="diamond-x-open", size=15),
                        name="Actual",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=data["domain"]["date_time"].reshape(-1),
                        y=data["true"].reshape(-1),
                        marker=dict(color="blue"),
                        name="  line",
                        showlegend=True,
                        # yaxis="y1",
                        mode="lines",
                    )
                )

            predicted_idx = np.argsort(-data["predicted"].reshape(-1))[:k_peaks]

            correct_idx = np.fromiter(
                set(predicted_idx).intersection(set(actual_idx)), int
            )
            fig.add_trace(
                go.Scatter(
                    x=data["domain"]["date_time"].reshape(-1)[correct_idx],
                    y=data["predicted"].reshape(-1)[correct_idx],
                    marker=dict(
                        color=qualitative.Bold[idx],
                        cmin=0,
                        cmax=len(predictions) + 1,
                        symbol="circle-dot",
                        size=15,
                    ),
                    legendgroup=idx,
                    name=key,
                    showlegend=True,
                    mode="markers",
                )
            )

            incorrect_idx = np.fromiter(set(predicted_idx) - set(actual_idx), int)
            fig.add_trace(
                go.Scatter(
                    x=data["domain"]["date_time"].reshape(-1)[incorrect_idx],
                    y=data["predicted"].reshape(-1)[incorrect_idx],
                    marker=dict(
                        color=qualitative.Bold[idx],
                        cmin=0,
                        cmax=len(predictions) + 1,
                        symbol="x-open",
                        size=15,
                    ),
                    legendgroup=idx,
                    name="  incorrect",
                    showlegend=True,
                    mode="markers",
                )
            )

            missing_idx = np.fromiter(set(actual_idx) - set(predicted_idx), int)
            fig.add_trace(
                go.Scatter(
                    x=data["domain"]["date_time"].reshape(-1)[missing_idx],
                    y=data["predicted"].reshape(-1)[missing_idx],
                    marker=dict(
                        color=qualitative.Bold[idx],
                        cmin=0,
                        cmax=len(predictions) + 1,
                        symbol="cross-open",
                        size=15,
                    ),
                    legendgroup=idx,
                    name="  missing",
                    showlegend=True,
                    mode="markers",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data["domain"]["date_time"].reshape(-1),
                    y=data["predicted"].reshape(-1),
                    marker=dict(
                        color=qualitative.Bold[idx], cmin=0, cmax=len(predictions) + 1
                    ),
                    legendgroup=idx,
                    name="  line",
                    showlegend=True,
                    mode="lines",
                )
            )

        fig.update_layout(
            title="Model Results: top {} peaks".format(k_peaks),
            xaxis_title="Date",
            yaxis_title="MW",
            showlegend=True,
            height=1000,
            width=1200,
            xaxis=dict(range=[start, end]),
        )
        return fig

    def peak_rank(
        self,
        start,
        end,
        location,
        rankings,
        step=np.timedelta64(1, "M"),
        dtype="datetime64[M]",
        open_step=np.timedelta64(1, "D"),
    ):
        results = defaultdict(dict)
        for mname, model in self.models.items():
            actual_peak_rank = []
            predicted_peak_rank = []
            actual_peak = []
            predicted_peak = []
            domain = []
            for start_date in np.arange(start, end, step=step, dtype=dtype):
                end_date = start_date + step - open_step  # don't use closed intervals
                # print(end_date)
                # Calculate Peak Rank per interval step!
                predicted_peak.append(model.predict(start_date, end_date, location))
                predicted_peak_rank.append(
                    rank(model.predict(start_date, end_date, location), rankings)
                )
                actual_peak.append(model.range(start_date, end_date, location))
                actual_peak_rank.append(
                    rank(model.range(start_date, end_date, location), rankings)
                )
                domain.append(model.domain(start_date, end_date, location).reshape(-1))

            results[mname]["predicted_peak"] = np.concatenate(predicted_peak, axis=0)
            results[mname]["predicted_peak_rank"] = np.concatenate(
                predicted_peak_rank, axis=0
            )
            results[mname]["actual_peak"] = np.concatenate(actual_peak, axis=0)
            results[mname]["actual_peak_rank"] = np.concatenate(
                actual_peak_rank, axis=0
            )
            results[mname]["domain"] = np.concatenate(domain, axis=0)

        return results

    def plot_peak_confusion_matrix(
        self,
        start,
        end,
        location,
        rankings=np.array(["peak"] * 5 + ["offpeak"] * 26),
        step=np.timedelta64(1, "M"),
        dtype="datetime64[M]",
        open_step=np.timedelta64(1, "D"),
    ):
        peak_ranks = self.peak_rank(
            start,
            end,
            location,
            rankings=rankings,
            step=step,
            dtype=dtype,
            open_step=open_step,
        )

        for mname, data in peak_ranks.items():
            uniq, idx = np.unique(rankings, return_index=True)
            labels = uniq[np.argsort(idx)].tolist()
            cmatrix = sklearn.metrics.confusion_matrix(
                y_true=data["actual_peak_rank"],
                y_pred=data["predicted_peak_rank"],
                labels=labels,
            )[::-1]

            fig = figure_factory.create_annotated_heatmap(
                cmatrix, x=labels, y=labels[::-1], colorscale="Viridis", showscale=True
            )

            # rough hack for figure size based on number of labels
            height = math.sqrt(len(labels) / 2) * 200 + 200
            fig.update_layout(
                title="Confusion Matrix: {}".format(mname),
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                showlegend=True,
                height=height,
                width=height + 200,
            )
            fig.update_xaxes(side="bottom")
            fig.show()
        return peak_ranks
