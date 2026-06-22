"""Shared pytest fixtures for AQI package tests.

Provides a small in-memory air-quality dataset and an on-disk CSV variant that
later tasks (data loading, preprocessing, EDA, feature engineering, training,
serving) can reuse.

The fixture frame includes:
  - ``City``        : at least two distinct cities
  - ``timestamp``   : a parseable observation timestamp
  - pollutant cols  : PM2.5, PM10, NO2, SO2, CO, O3
  - ``AQI``         : the target column
A few timestamps are provided per city so temporal/lag/rolling features can be
exercised downstream.
"""

import pandas as pd
import pytest

from ml_models.aqi import POLLUTANT_FEATURES, TARGET_COLUMN


# Rows span 2 cities across 3 hourly timestamps each (2024-01-01 08:00-10:00).
# Columns: City, timestamp, PM2.5, PM10, NO2, SO2, CO, O3, AQI
_SAMPLE_ROWS = [
    ("Delhi", "2024-01-01 08:00:00", 88.2, 142.0, 41.3, 12.1, 1.2, 22.0, 210.0),
    ("Delhi", "2024-01-01 09:00:00", 95.6, 150.5, 44.0, 13.0, 1.4, 19.5, 225.0),
    ("Delhi", "2024-01-01 10:00:00", 80.1, 138.2, 39.8, 11.5, 1.1, 24.3, 198.0),
    ("Mumbai", "2024-01-01 08:00:00", 42.0, 70.3, 22.1, 6.4, 0.7, 31.0, 95.0),
    ("Mumbai", "2024-01-01 09:00:00", 48.5, 76.0, 24.0, 7.0, 0.8, 28.6, 105.0),
    ("Mumbai", "2024-01-01 10:00:00", 39.7, 65.8, 20.5, 5.9, 0.6, 33.2, 88.0),
]

_COLUMNS = ["City", "timestamp", *POLLUTANT_FEATURES, TARGET_COLUMN]


@pytest.fixture
def sample_aqi_df() -> pd.DataFrame:
    """Return a small air-quality DataFrame across 2 cities and 3 timestamps."""
    df = pd.DataFrame(_SAMPLE_ROWS, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def sample_aqi_csv(sample_aqi_df, tmp_path) -> str:
    """Write ``sample_aqi_df`` to a temp CSV and return its path as a string."""
    csv_path = tmp_path / "city_day.csv"
    sample_aqi_df.to_csv(csv_path, index=False)
    return str(csv_path)
