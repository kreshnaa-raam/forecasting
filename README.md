# Time Series Models

A library for composable forecasting models built on top of scikit-learn

The API for fit, predict, and metrics is reduced to specifying a start and end times for a given location.
The model must construct feature data using column transforms. Having done so, forecasting as a service become trivial.

## Installation

requires the eccodes library: `apt-get install libeccodes-dev`

`pip3 install -r requirements.txt`

Verify you can run the unit tests `python -m unittest`

If running in a container, to start a notebook use `jupyter notebook --NotebookApp.ip=0.0.0.0`

## Usage
Models can be composed of mixins for various estimators and forecast processes.

An ordinary least squares model for a Balancing Area Hourly Load
```python
class NpOlsModel(BalancingAreaHourly, LinearRegressor, RegularTimeSeriesModel):
  pass
```
An XGBoost model for Balancing Area Hourly Load
```python
class XgbModel(BalancingAreaHourly, XgbRegressor, RegularTimeSeriesModel):
  pass
```

The initialization arguments control feature construction.


### Trained models should be fully serializable using Cloud Pickle for maximum flexibility


It is easy to work with time series models in a
[Colab Notebook](https://colab.research.google.com/drive/1Tpoxdyf7aN1kyPrPb0L4uaZrmHQnHyB1?usp=sharing)


## Contents

* `data_fetchers/`: data fetchers and associated tests -- see directory-specific README.
* `back_test.py`: BackTest class for estimating model performance on out-of-sample data using a time series (rolling)
 cross-validation.
* `config.py`: ConfigHandler helper class to encode and decode time series models configurations
* `constants.py`: defines constant variables referenced throughout Time Series Models library
* `data_monitor.py`: ForecastDataMonitor is a pipeline element that collects statistics on the data passing through it
* `decorators.py`: functions that can be used as decorators in Time Series Models library
* `dummy_models.py`: classes implementing `DummyDataModel` for using external predictions with Time Series Models
metrics and vizualizations
* `estimators.py`: Estimator and EstimatorWrapper mixins for forecast models (e.g., `XgbRegressor`, `RandomizedSearch`)
* `filters.py`: Filter mixins (e.g., `DropFeatures`, `RowFilter`, `StandardScaler`)
* Gap solving tools:
  * `gap_maker.py`: utility for introducing synthetic gaps into a dataset
  * `gap_filler.py`: utility for filling data gaps with various interpolative or extrapolative methods
  * `gap_runner.py`: holds score_df_at_locs, a helper to evaluate gap-filling performance
* `metrics_runner.py`: a container for running models and comparing metrics, organized around monthly analysis
* `processes.py`: mixins defining the data sources and preprocessing steps for a given process to forecast
* `pv_physical_model.py`: common interface to using pvlib and PySAM physical models
* visualization tools:
  * `shap_viz.py`: wrapper for applying `shap` library to RegularTimeSeriesModel instances
  * `viz.py`: methods for plotting time series of predicted vs actual, and residual scatter plots
* `time_series_models.py`: core `RegularTimeSeriesModel` and `Mixin` class definitions, along with several other model
  helpers and mixins like `AggModel` and `MonthSelector`.
* `time_unit.py`: Enum for calendar conversion of numpy datetime64 arrays
* transformers: a collection of transformers for composing model features
  * `transformers.py`: broad-purpose (or uncategorized) transformers for use throughout Time Series Models
    * core helper functions like `make_domain`
    * some bespoke transformers that haven't been split out into separate modules
    * simple array transformations
  * `transformers_calendar.py`: calendar feature transformers (e.g., business day, day-of-week, & harmonics)
  * `transformers_pv.py`: transformers for PV forecasting
  * `transformers_shortcuts.py`: help avoid copy-pasting the same core features for each process
  (forecast sum of meters & DER, rather than sum of forecasts)
* `version.py`: Time Series Models library version