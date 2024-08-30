import logging
import numpy as np

from time_series_models.time_series_models import RegularTimeSeriesModel
from time_series_models.processes import AmiHourlyForecast, PVForecast
from time_series_models.estimators import (
    XgbRegressor,
    IdentityRegressor,
)

logger = logging.getLogger(__name__)


def run_forecast_example():

    logger.info(
        "Starting forecast example for AMI meter forecast with XgBoost estimator!",
    )

    class XgbModel(AmiHourlyForecast, XgbRegressor, RegularTimeSeriesModel):
        pass

    config = dict(
        lags=np.array([24, 48, 168], dtype="timedelta64[h]"),
        day_of_week=True,
        harmonics=np.array([24, 168, 365 * 24], dtype="timedelta64[h]"),
        met_vars=["t", "r2"],
        met_horizon=12,
        mapping=dict(p2ulv18716=dict(latitude=35.0, longitude=-75.0)),
    )
    instance = XgbModel(**config)

    instance.fit("2021-01-15", "2021-01-31", "p2ulv18716")

    logger.info("Trained instance: %s", instance.model)

    features_df = instance.features_dataframe("2021-02-01", "2021-02-05", "p2ulv18716")
    logger.info("Features data: %s", features_df)

    predicted_df = instance.predict_dataframe(
        "2021-01-01", "2021-02-05", "p2ulv18716", range=True
    )

    logger.info("Predicted: %s", predicted_df)

    logger.info(
        "Starting forecast example for PV physical forecast!",
    )

    pv_config = dict(
        lags=None,
        site_config_mapping={
            "capybara": ["/app/pv_site.json"],
        },
        site_latlong_mapping={
            "capybara": dict(
                latitude=40.0,
                longitude=-100.0,
            ),
        },
        site_meter_mapping=None,
        source_mode="12_hour_horizon",
    )

    class PVForecastModel(
        PVForecast,
        IdentityRegressor,
        RegularTimeSeriesModel,
    ):
        pass

    pv_instance = PVForecastModel(**pv_config)
    pv_instance.model

    pv_instance.fit("2021-01-15", "2021-01-16", "capybara")

    pv_hrrr_df = pv_instance.hrrr_fetcher.source_loader(
        np.datetime64("2021-02-01"), np.datetime64("2021-02-05"), "capybara"
    )
    logger.info("PV HRRR Data: %s", pv_hrrr_df)

    pv_df = pv_instance.predict_dataframe("2021-02-01", "2021-02-05", "capybara")
    logger.info("pv predictions: %s", pv_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forecast_example()
    logging.info("All done!")
    exit(0)
