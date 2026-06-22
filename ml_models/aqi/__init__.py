"""AQI Prediction package for ClimaSense.

This package implements the from-scratch Air Quality Index (AQI) prediction
system. This module exposes shared configuration constants used across the
data-science pipeline (loading, preprocessing, EDA, feature engineering,
model training) and the serving components.

Public component exports and ``get_aqi_service()`` are added in later tasks.
"""

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
POLLUTANT_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
WEATHER_FEATURES = ["Temperature", "Humidity", "WindSpeed", "Pressure"]
TEMPORAL_FEATURES = ["Hour", "Day", "Month"]
LAG_FEATURES = ["Prev_AQI", "AQI_24h_avg"]

# ---------------------------------------------------------------------------
# Target / horizon configuration
# ---------------------------------------------------------------------------
TARGET_COLUMN = "AQI"
FORECAST_HORIZON = 1  # periods shifted forward (next-period)
FORECAST_STEPS = 5  # hours produced by the multi-step forecaster (t+1..t+5)

# ---------------------------------------------------------------------------
# Path / artifact configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "city_day.csv"
MODEL_ARTIFACT = "ml_models/aqi/aqi_model.joblib"
EDA_DIR = "static/aqi"

# ---------------------------------------------------------------------------
# Public component exports
# ---------------------------------------------------------------------------
# These imports are intentionally placed at the BOTTOM of this module: the
# submodules below import the configuration constants defined above from this
# package, so importing them here (after the constants exist) avoids a circular
# import at package-initialization time.
from ml_models.aqi.data_loader import DataLoader  # noqa: E402
from ml_models.aqi.preprocessor import Preprocessor  # noqa: E402
from ml_models.aqi.eda import EDAAnalyzer  # noqa: E402
from ml_models.aqi.feature_engineer import FeatureEngineer  # noqa: E402
from ml_models.aqi.model_evaluator import ModelEvaluator  # noqa: E402
from ml_models.aqi.classifier import classify, CPCB_BUCKETS  # noqa: E402
from ml_models.aqi.advisor import advise, ADVISORIES  # noqa: E402
from ml_models.aqi.predictor import AQIPredictor  # noqa: E402
from ml_models.aqi.trends import TrendVisualizer, TrendSeries, NoDataForCityError  # noqa: E402
from ml_models.aqi.live_source import LiveAirQualitySource, LiveDataError  # noqa: E402
from ml_models.aqi.service import (  # noqa: E402
    AQIService,
    ModelUnavailableError,
    get_aqi_service,
)
from ml_models.aqi.forecaster import MultiStepForecaster, ForecastPoint  # noqa: E402

__all__ = [
    # Configuration constants
    "POLLUTANT_FEATURES",
    "WEATHER_FEATURES",
    "TEMPORAL_FEATURES",
    "LAG_FEATURES",
    "TARGET_COLUMN",
    "FORECAST_HORIZON",
    "FORECAST_STEPS",
    "DEFAULT_DATASET",
    "MODEL_ARTIFACT",
    "EDA_DIR",
    # Pipeline components
    "DataLoader",
    "Preprocessor",
    "EDAAnalyzer",
    "FeatureEngineer",
    "ModelEvaluator",
    # Serving components
    "classify",
    "CPCB_BUCKETS",
    "advise",
    "ADVISORIES",
    "AQIPredictor",
    "MultiStepForecaster",
    "ForecastPoint",
    "TrendVisualizer",
    "TrendSeries",
    "LiveAirQualitySource",
    "AQIService",
    "get_aqi_service",
    # Exceptions
    "ModelUnavailableError",
    "NoDataForCityError",
    "LiveDataError",
]
