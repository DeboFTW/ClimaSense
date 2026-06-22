"""Data collection and loading for the AQI prediction pipeline.

The :class:`DataLoader` reads historical air-quality records (Pollutant_Features
plus the AQI target) from the Training_Dataset, attaches a ``City`` and a parsed
``timestamp`` to every record, and optionally left-joins Open-Meteo
Weather_Features on ``(city, timestamp)`` when a weather source is available.

This is the first stage of the offline training pipeline:
``DataLoader -> Preprocessor -> FeatureEngineer -> ModelEvaluator``.

Requirements covered:
  - 1.1  Load Pollutant_Features + AQI from the dataset.
  - 1.2  Every loaded record carries ``City`` and a parsed ``timestamp``.
  - 1.3  Join Weather_Features (left-join on city + time) where available;
         absent weather → NaN, later imputed by the Preprocessor.
  - 1.4  Missing/unreadable dataset → ``FileNotFoundError`` naming the path.
  - 1.5  After load, ``metadata`` records ``record_count`` and unique ``cities``.
"""

from __future__ import annotations

import os
from typing import List, Optional, TypedDict

import pandas as pd

from ml_models.aqi import DEFAULT_DATASET, WEATHER_FEATURES

# Candidate column names (case-insensitive) used to locate the observation
# timestamp and the city across CPCB/OpenAQ-style datasets and our fixtures.
_TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "date_time")
_CITY_CANDIDATES = ("city", "location", "station")

# Canonical column names produced by the loader.
_CITY_COLUMN = "City"
_TIMESTAMP_COLUMN = "timestamp"


class DatasetMetadata(TypedDict):
    """Summary of a loaded dataset (populated after :meth:`DataLoader.load`)."""

    record_count: int
    cities: List[str]


class DataLoader:
    """Load raw air-quality records and optionally join weather data.

    Parameters
    ----------
    dataset_path:
        Path to the air-quality CSV (Pollutant_Features + AQI + city + time).
        Defaults to :data:`ml_models.aqi.DEFAULT_DATASET`.
    weather_path:
        Optional path to an Open-Meteo weather CSV. When provided (and present),
        its Weather_Features are left-joined onto the air-quality records by
        ``(City, timestamp)``. When omitted or absent, loading still succeeds and
        the weather columns are simply not added (the Preprocessor imputes any
        weather features that downstream stages expect).
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET,
        weather_path: Optional[str] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.weather_path = weather_path
        self._metadata: Optional[DatasetMetadata] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load the air-quality dataset into a DataFrame.

        The returned frame contains the Pollutant_Features that are present in
        the source, the ``AQI`` target, a ``City`` column, and a parsed
        ``timestamp`` column (a valid datetime). Where an Open-Meteo weather
        source is available, Weather_Features are left-joined on
        ``(City, timestamp)``; missing matches yield NaN.

        Returns
        -------
        pandas.DataFrame
            The loaded records.

        Raises
        ------
        FileNotFoundError
            If the dataset file is missing or cannot be read. The error message
            names the expected dataset path.
        """
        df = self._read_csv(self.dataset_path)

        df = self._normalize_city(df)
        df = self._normalize_timestamp(df)

        weather_df = self._load_weather()
        if weather_df is not None:
            df = self._join_weather(df, weather_df)

        self._metadata = self._build_metadata(df)
        return df

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata populated by the most recent :meth:`load`.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`load` has been called.
        """
        if self._metadata is None:
            raise RuntimeError("metadata is unavailable until load() has been called")
        return self._metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        """Read a CSV, raising a descriptive ``FileNotFoundError`` on failure."""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"AQI dataset not found at: {path}")
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise FileNotFoundError(f"AQI dataset not found at: {path}") from exc

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
        """Return the first column in ``df`` matching ``candidates`` (case-insensitive)."""
        lookup = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate in lookup:
                return lookup[candidate]
        return None

    def _normalize_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a canonical ``City`` column exists on the frame."""
        if _CITY_COLUMN in df.columns:
            return df

        city_col = self._find_column(df, _CITY_CANDIDATES)
        if city_col is None:
            raise FileNotFoundError(
                f"AQI dataset not found at: {self.dataset_path} "
                f"(no recognizable city column among {list(df.columns)})"
            )
        return df.rename(columns={city_col: _CITY_COLUMN})

    def _normalize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the source time column into a canonical ``timestamp`` datetime."""
        ts_col = self._find_column(df, _TIMESTAMP_CANDIDATES)
        if ts_col is None:
            raise FileNotFoundError(
                f"AQI dataset not found at: {self.dataset_path} "
                f"(no recognizable timestamp column among {list(df.columns)})"
            )

        if ts_col != _TIMESTAMP_COLUMN:
            df = df.rename(columns={ts_col: _TIMESTAMP_COLUMN})

        df[_TIMESTAMP_COLUMN] = pd.to_datetime(df[_TIMESTAMP_COLUMN], errors="coerce")
        # Records without a parseable timestamp cannot satisfy Req 1.2.
        df = df.dropna(subset=[_TIMESTAMP_COLUMN]).reset_index(drop=True)
        return df

    def _load_weather(self) -> Optional[pd.DataFrame]:
        """Load the optional Open-Meteo weather source, if configured and present."""
        if not self.weather_path or not os.path.isfile(self.weather_path):
            return None
        try:
            weather = pd.read_csv(self.weather_path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None

        weather = self._normalize_city(weather)
        ts_col = self._find_column(weather, _TIMESTAMP_CANDIDATES)
        if ts_col is None:
            return None
        if ts_col != _TIMESTAMP_COLUMN:
            weather = weather.rename(columns={ts_col: _TIMESTAMP_COLUMN})
        weather[_TIMESTAMP_COLUMN] = pd.to_datetime(
            weather[_TIMESTAMP_COLUMN], errors="coerce"
        )
        weather = weather.dropna(subset=[_TIMESTAMP_COLUMN])

        # Keep only the join keys plus any recognized Weather_Features.
        keep = [_CITY_COLUMN, _TIMESTAMP_COLUMN] + [
            col for col in WEATHER_FEATURES if col in weather.columns
        ]
        return weather[keep] if len(keep) > 2 else None

    @staticmethod
    def _join_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Left-join Weather_Features onto the records by ``(City, timestamp)``."""
        return df.merge(
            weather_df,
            on=[_CITY_COLUMN, _TIMESTAMP_COLUMN],
            how="left",
        )

    @staticmethod
    def _build_metadata(df: pd.DataFrame) -> DatasetMetadata:
        """Construct dataset metadata from the loaded frame."""
        cities = (
            sorted(str(c) for c in df[_CITY_COLUMN].dropna().unique())
            if _CITY_COLUMN in df.columns
            else []
        )
        return DatasetMetadata(record_count=int(len(df)), cities=cities)
