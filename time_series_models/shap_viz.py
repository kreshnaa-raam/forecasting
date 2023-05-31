import shap
import pandas as pd

from time_series_models.transformers import multiindex_from_domain


class ShapViz:
    """
    Handy Shap Score wrapper for poorly documented interfaces complicated by time series models structure
    """

    def __init__(self, time_series_model, start, end, *locations):
        self.time_series_model = time_series_model
        self.domain = time_series_model.domain(start, end, *locations)
        self.features = time_series_model.get_features(self.domain)

        estimator = time_series_model.estimator
        self.explainer = shap.Explainer(
            estimator,
            feature_names=self.time_series_model.get_feature_names(),
        )
        self.shap_values = self.explainer(self.features, check_additivity=False)
        """
        TODO: replace this boolean with a percent error tolerance
        shap.utils._exceptions.ExplainerError: Additivity check failed in TreeExplainer!
        Please ensure the data matrix you passed to the explainer is the same shape that the model was trained on.
        If your data shape is correct then please report this on GitHub. Consider retrying with the
        feature_perturbation='interventional' option. This check failed because for one of the samples the sum of
        the SHAP values was 4507.288086, while the model output was 4507.284668. If this difference is acceptable
        you can set check_additivity=False to disable this check.
        """

    def values_dataframe(self) -> pd.DataFrame:
        """
        Create a labeled dataframe for the score associated with each feature value
        :return: the dataframe
        """
        return pd.DataFrame(
            self.shap_values.values,
            columns=self.shap_values.feature_names,
            index=multiindex_from_domain(self.domain),
        )

    def plot_summary_bar(self):
        shap.summary_plot(self.shap_values, self.features, plot_type="bar")

    def plot_summary_beeswarm(self):
        shap.summary_plot(self.shap_values, self.features)

    def plot_scatters(self):
        for fname in self.shap_values.feature_names:
            shap.plots.scatter(self.shap_values[:, fname], color=self.shap_values)

    def plot_prediction_force(self, prediction):
        return shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[prediction].values,
            feature_names=self.time_series_model.get_feature_names(),
        )

    def plot_prediction_waterfall(self, prediction):
        shap.plots.waterfall(self.shap_values[prediction])

    def plot_force(self):
        return shap.force_plot(
            base_value=self.explainer.expected_value,
            shap_values=self.shap_values.values,
            features=self.features,
            plot_cmap="DrDb",
            feature_names=self.time_series_model.get_feature_names(),
        )
