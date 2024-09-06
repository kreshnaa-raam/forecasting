# Time Series Models

A library for composable forecasting models built on top of [scikit-learn](https://scikit-learn.org/stable/)


This software library is the open source package for the work done for [SETO project 2243](https://www.energy.gov/eere/solar/seto-2020-artificial-intelligence-applications-solar-energy#:~:text=for%20unobservable%20areas.-,Camus%20Energy,-Project%20Name%3A), _Improving Grid Awareness by Empowering Utilities with Machine Learning and Artificial Intelligence_, specifically task 2: _Endpoint data process and Analysis_. The
package includes meter level forecasting and gap filling for customer load as well as photovoltaic (PV) generation 
forecast models for short term and intra-day PV forecasting.

The package provides the tools to construct machine learning models that fill gaps or forecast in the verification
datasets. This includes loading the data and applying the XGBoost estimator as described in section 3.1 of the Final Technical
Review (FTR). The composable model framework described below provides the configurable model inputs required for the 
Neighbor Informed Estimates and Community Analytics. The PV System model in section 3.2 of the FTR is also implemented using
the composable model framework. Example PV model construction is shown below.

<!-- TODO: Get permalink for FTR once published and link it from here! -->


The API for fit, predict, and metrics is reduced to specifying a start and end times for a given location.
The model must construct feature data using column transforms. Having done so, forecasting as a service become trivial.

## Getting started

Users can verify the code works with the `example_forecast.py` script by running `docker compose up --build`
The log output from the container will include feature and weather data as well as predicted values. The script
takes several minutes to run because the weather data is large.

The dockerfile included in the project will run the example_forecast.py which demonstrates both the machine
learning model for AMI meter forecasting and the physics based PV model using PySam. Users can then choose between
their local working environment and containerized environment to extend and experiment with the time series models
library.

The dockerized example is a great place to start for experimentation and further development. 
### Expected output
Once the container layers are running (this will take several minutes) the console log will show
```console
[+] Running 1/1
 ✔ Container seto_forecasting-server-1  Recreated                                                                                                                                    0.7s
Attaching to seto_forecasting-server-1
```

#### AMI Meter Model
The first example is a meter level xgboost estimator
```console
seto_forecasting-server-1  | INFO:__main__:Starting forecast example for AMI meter forecast with XgBoost estimator!
seto_forecasting-server-1  | INFO:time_series_models.transformers:Constructing overfetched range pipeline using lags [  0 168]
seto_forecasting-server-1  | INFO:time_series_models.processes:Instantiating RegularTimeSeriesModels with kwargs {'day_of_week': True, 'harmonics': array([  24,  168, 8760], dtype='timedelta64[h]'), 'met_vars': ['t', 'r2'], 'met_horizon': 12, 'mapping': {'p2ulv18716': {'latitude': 35.0, 'longitude': -75.0}}}
```
There will be a good deal more log messages (and a few minutes to download weather data) for training and prediction
followed by the (truncated) results of the prediction run for the `p2ulv18716` meter.
```console
seto_forecasting-server-1  | INFO:time_series_models.data_fetchers.fetcher:Finished 'HrrrFetcher' 'get_data' in 248.3485 secs
seto_forecasting-server-1  | INFO:time_series_models.data_fetchers.fetcher:Finished 'AmiFetcher' 'get_data' in 0.0837 secs
seto_forecasting-server-1  | INFO:__main__:Predicted:                                   predicted         true
seto_forecasting-server-1  | location   date_time
seto_forecasting-server-1  | p2ulv18716 2021-01-01 00:00:00   785.116394   600.156056
seto_forecasting-server-1  |            2021-01-01 01:00:00   717.555481  2579.214714
seto_forecasting-server-1  |            2021-01-01 02:00:00   817.579041  2720.881345
seto_forecasting-server-1  |            2021-01-01 03:00:00   507.064819  2341.922617
seto_forecasting-server-1  |            2021-01-01 04:00:00   444.800018  2124.941260
seto_forecasting-server-1  | ...                                     ...          ...
seto_forecasting-server-1  |            2021-02-04 20:00:00   513.341370   425.300693
seto_forecasting-server-1  |            2021-02-04 21:00:00   591.890686   459.267964
seto_forecasting-server-1  |            2021-02-04 22:00:00  2320.842773   546.954250
seto_forecasting-server-1  |            2021-02-04 23:00:00  2011.579346   599.035650
seto_forecasting-server-1  |            2021-02-05 00:00:00  2223.689941   652.819004
seto_forecasting-server-1  |
seto_forecasting-server-1  | [841 rows x 2 columns]
```

#### PV Physics Model
The PV physics model will continue immediately with
```console
seto_forecasting-server-1  | INFO:__main__:Starting forecast example for PV physical forecast!
```
Intermediate weather data will be visible during the training and prediction steps, extracted for the time range, 
latitude and longitude of the site.
```console
seto_forecasting-server-1  | INFO:time_series_models.transformers_pv:Feature DF:                        ghi    dni    dhi    temp_air  wind_speed
seto_forecasting-server-1  | date_time
seto_forecasting-server-1  | 2021-02-01 00:00:00    0.0    0.0    0.0  275.616852    2.302358
seto_forecasting-server-1  | 2021-02-01 01:00:00    0.0    0.0    0.0  273.182281    3.805551
seto_forecasting-server-1  | 2021-02-01 02:00:00    0.0    0.0    0.0  271.327850    2.718800
seto_forecasting-server-1  | 2021-02-01 03:00:00    0.0    0.0    0.0  270.273697    4.061844
seto_forecasting-server-1  | 2021-02-01 04:00:00    0.0    0.0    0.0  270.467178    4.645654
seto_forecasting-server-1  | ...                    ...    ...    ...         ...         ...
seto_forecasting-server-1  | 2021-02-04 20:00:00  586.1  969.0   77.0  281.806671   12.966594
seto_forecasting-server-1  | 2021-02-04 21:00:00  485.4  924.0   72.1  282.243988   15.832714
seto_forecasting-server-1  | 2021-02-04 22:00:00  325.8  547.0  147.3  281.664047   16.190084
seto_forecasting-server-1  | 2021-02-04 23:00:00  128.7  236.0   88.2  279.410828   14.847732
seto_forecasting-server-1  | 2021-02-05 00:00:00    0.0    0.0    0.0  276.510498   12.012517
```
Finally, the predicted values for the `capybara` PV demo site will be logged.
```console
seto_forecasting-server-1  | INFO:time_series_models.transformers_pv:Trying to load from: /app/pv_site.json
seto_forecasting-server-1  | INFO:__main__:pv predictions:                                  predicted
seto_forecasting-server-1  | location date_time
seto_forecasting-server-1  | capybara 2021-02-01 00:00:00  2.555532e+02
seto_forecasting-server-1  |          2021-02-01 01:00:00  2.555532e+02
seto_forecasting-server-1  |          2021-02-01 02:00:00  2.555532e+02
seto_forecasting-server-1  |          2021-02-01 03:00:00  2.555532e+02
seto_forecasting-server-1  |          2021-02-01 04:00:00  2.555532e+02
seto_forecasting-server-1  | ...                                    ...
seto_forecasting-server-1  |          2021-02-04 20:00:00 -1.007929e+06
seto_forecasting-server-1  |          2021-02-04 21:00:00 -1.096734e+06
seto_forecasting-server-1  |          2021-02-04 22:00:00 -9.101299e+05
seto_forecasting-server-1  |          2021-02-04 23:00:00 -4.854458e+05
seto_forecasting-server-1  |          2021-02-05 00:00:00  2.555532e+02
seto_forecasting-server-1  |
seto_forecasting-server-1  | [97 rows x 1 columns]
seto_forecasting-server-1  | INFO:root:All done!
```

Some systems may return an error code on exit.
```console
seto_forecasting-server-1  | free(): invalid pointer
seto_forecasting-server-1  | Aborted
seto_forecasting-server-1 exited with code 134
```
This is likely related to an issue with the version of the pysam library but it does not affect the execution of the example.

### Development & Installation

This library is designed for use by technical engineers and data scientists. It takes advantage of the Python
data science ecosystem and therefore requires installation of many third party open source libraries. It has 
been developed and tested in a Linux operating system. Running on a Docker container such as 
the [canonical Ubuntu image](https://hub.docker.com/_/ubuntu) is strongly recommended. The library was 
developed using Ubuntu 22.04 (Jammy) with Python 3.10.6.

The docker file included in the repository only runs the `example_forecast.py` file. The following code snippets
are provided as helpful steps toward building a developer environment where you can run unit tests, forecasting scripts
and jupyter notebooks.

#### Installing system libraries

After installing [Docker](https://docs.docker.com/engine/install/), run the following command to setup a basic Jammy container with this library:

```sh
docker run -it ubuntu:jammy /bin/bash
apt-get update
apt-get install git
git clone https://github.com/SETO2243/forecasting.git
cd /forecasting
```
Reading the grib weather data requires the eccodes library, which is available from the Jammy package repo, installable via shell:

```sh
apt-get install libeccodes-dev
```

### Installing Python libraries

Run the following command to install Python package dependencies:

```sh
pip3 install -r requirements.txt
```

### Validating your environment

Verify that your environment is fully functional by running the automated unit tests:

```sh
python -m unittest
```

This will print "SUCCESS" near the end if the code work correctly in your new environment.

### Running a notebook
To start jupyter notebook run:

```sh
jupyter notebook --NotebookApp.ip=0.0.0.0
```

This will print a URL, which you can open in your browser. Then open the example notebook and execute the cells in the demonstration to get acquainted with the functionality.

## Usage
Models can be composed of mixins for various estimators and forecast processes. These composable
pieces can be put together in different ways to solve many problems. The RegularTimeSeriesModel is the
core that problem specific parts are added to when forecasting or gap filling a particular timeseries.
The [estimator](https://scikit-learn.org/stable/developers/develop.html#estimators) is the next essential building 
block. The estimator can be either a Classifier (a discrete estimator) or a Regressor (a continuous estimator). There 
are many different numerical techniques for [supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
estimators. The process is the last essential component. It defines the timeseries being forecast and the available 
feature data that might have predictive value. Having composed a Model class from these three parts, it is then up to 
the user to create an instance of the class with configuration arguments that tune the model features for the specific
meter load or pv forecast.

### Compose a model class

New models are defined as Python classes, which utilize building blocks provided by this library as base classes. For example, here is the beginning of a model using an Ordinary Least Squares estimator to forecast Balancing Area Hourly Load:

```python
class OlsModel(BalancingAreaHourly, LinearRegressor, RegularTimeSeriesModel):
  ...
```

And this example is model using an [XgBoost](https://xgboost.readthedocs.io/en/stable/) estimator to forecast AMI (smart meter) Hourly Load:

```python
class XgbModel(AmiHourlyForecast, XgbRegressor, RegularTimeSeriesModel):
  pass
```

Additional behaviors including [filters](https://github.com/SETO2243/forecasting/blob/main/time_series_models/filters.py) 
and [hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html) can also be added via composition.
See the notebook demo for examples.

### Configure a model instance
The initialization arguments control the model input feature. The configuration arguments are specific to the
process and the estimator used to compose the model. The process class (e.g. AmiHourlyForecast) implements the set 
of possible input features that can be configured. When the model instance is created the specific set of input
features for the specific model is defined. Additional parameters can also be passed to the estimator or other 
components.

As an example, we can configure XgbModel from above with three types of features: lagged features as described in the
FTR (Section 3.1.2.2.1),
[one hot encoded](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 
day-of-week values, and a harmonic feature that decomposes time into a sine and cosine waves with the specified frequencies.
```python
config = dict(
lags = np.array([24, 48, 168], dtype="timedelta64[h]"),
day_of_week=True,
harmonics = np.array([24, 168, 365*24], dtype="timedelta64[h]")
)
instance = XgbModel(**config)
```
Below is the [sklearn model diagram](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_pipeline_display.html)
for the pipeline of transformations that implement the model.

*Replace this placeholder diagram generated using the camus internal version of the forecast library.
![Meter Model Diagram](https://github.com/SETO2243/forecasting/assets/84335963/4e07aaf5-b164-4aa8-8948-d6bed83c716d)

### Fit & Predict
Once a model instance is created, we must train the machine learning algorithm. The fit method takes
a start date, stop date and a list of one or more identifiers. For process classes that support it, passing
multiple identifiers allows training a single model for a cohort of resources.

```python
instance.fit("2021-01-15", "2021-06-01", "55797646")
```

Once the model instance is trained, we can call the `predict` method (again with a start, stop and one or more
resource identifiers) to generate new predicted values from the input features for the specified date range.
```python
instance.predict_dataframe(
    "2021-03-15",
    "2021-03-15T04",
    "55797646",
    range=True,
)
```
![Predicted & Actual Values](https://github.com/SETO2243/forecasting/assets/84335963/99cdaaf1-d15a-40fd-bbfa-534da6de4b49)

### PV Model
The PV Model as described in the FTR section 3.2 uses the same composable framework define models
using the HRRR weather (see below) as an input to the NREL PySam PV generatin algorithm. For the 
project we used the PySam generation forecast directly using the configration shown below with the
IdentityRegressor. Building additional input features for sites with direct telemetry would allow
using machine learning models like xgboost too.

```python
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
```
*Replace this placeholder diagram generated using the camus internal version of the forecast library.
![PV Model Diagram](https://github.com/SETO2243/forecasting/assets/84335963/91bbd896-b342-4219-9ee5-c6487b1aa363)

### Usage
Engineers and data scientists commonly use an interactive web-based development environment called [Jupyter Notebook](https://jupyter.org/)
(now Jupyter Lab) to explore and visualize data and algorithms in a cell based execution environment. 

An [example notebook](https://github.com/SETO2243/forecasting/blob/main/example.ipynb) is provided in this GitHub
repository which demonstrates the core capabilities of the time series models library developed for the SETO project. 


## Input Data

Sample AMI meter data for 4 meters is available for validation in a public GCS bucket: gs://seto2243-forecasting/ami_validation
Customer data used in the project is available upon completion of an NDA as described in the FTR.

### High Resolution Rapid Refresh weather data
The [High Resolution Rapid Refresh](https://rapidrefresh.noaa.gov/hrrr/) (HRRR) forecast is an operational weather 
forecasting product of the NOAA Center for Environmental Prediction. The HRRR forecast results are publicly available 
on multiple cloud vendor platforms ([AWS](https://registry.opendata.aws/collab/noaa/),
[GCP](https://cloud.google.com/blog/products/data-analytics/noaa-datasets-on-google-cloud-for-environmental-exploration),
[Azure](https://planetarycomputer.microsoft.com/catalog)) via a public private partnership with the
[NOAA Open Data Dissemination](https://www.noaa.gov/information-technology/open-data-dissemination) program.

These cloud providers host the petabyte scale archive of grib2 files created by the hourly HRRR operational forecast system.
As part of the project a set of metadata file that index the archive was created using an open source tool
called [Kerchunk](https://github.com/fsspec/kerchunk). The weather data is a public archive provided by NODD. To use it
the timeseries models developed for the project a kerchunk metadata index must be created. A sample index is provided
in the project [GCS bucket](https://console.cloud.google.com/storage/browser/seto2243-forecasting).

A kerchunk aggregation of select variables from the 12 hour forecast for 2021 is available in a public GCS bucket: gs://seto2243-forecasting/high-resolution-rapid-refresh

## Contents

Library code is organized into a number of subpackages described below, to aid engineers working on writing or debugging code using this library:

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
* Visualization tools:
  * `shap_viz.py`: wrapper for applying `shap` library to RegularTimeSeriesModel instances
  * `viz.py`: methods for plotting time series of predicted vs actual, and residual scatter plots
* `time_series_models.py`: core `RegularTimeSeriesModel` and `Mixin` class definitions, along with several other model
  helpers and mixins like `AggModel` and `MonthSelector`.
* `time_unit.py`: Enum for calendar conversion of numpy datetime64 arrays
* Transformers: a collection of transformers for composing model features
  * `transformers.py`: broad-purpose (or uncategorized) transformers for use throughout Time Series Models
    * core helper functions like `make_domain`
    * some bespoke transformers that haven't been split out into separate modules
    * simple array transformations
  * `transformers_calendar.py`: calendar feature transformers (e.g., business day, day-of-week, & harmonics)
  * `transformers_pv.py`: transformers for PV forecasting
  * `transformers_shortcuts.py`: help avoid copy-pasting the same core features for each process
  (forecast sum of meters & DER, rather than sum of forecasts)
* `version.py`: Time Series Models library version
