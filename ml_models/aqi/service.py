"""Serving-path orchestrator for the AQI prediction system.

The :class:`AQIService` ties together the trained model and the supporting
serving components so the Flask API layer (``main.py``) can answer a single
request with a complete air-quality payload. It is the online counterpart to
the offline training pipeline.

For a city it:
  - loads the persisted model once at construction via
    :class:`~ml_models.aqi.predictor.AQIPredictor` (Req 6.4);
  - derives the input features and the "current" observation from the city's
    most recent historical record;
  - predicts a future AQI value, attaches its CPCB bucket
    (:func:`~ml_models.aqi.classifier.classify`) and a health advisory
    (:func:`~ml_models.aqi.advisor.advise`, Req 9.4);
  - renders a trend forecast plot
    (:meth:`~ml_models.aqi.trends.TrendVisualizer.render_forecast_plot`) and
    exposes its web URL; and
  - returns daily/weekly/monthly trend series via :meth:`get_trends`
    (Req 10.1, 10.2).

Error surfacing for the API layer
---------------------------------
The service raises typed exceptions that the Flask layer (task 14) maps to HTTP
responses:

  - :class:`ModelUnavailableError` -- the persisted model could not be loaded;
    the API should return **503 model unavailable** (Req 6.4).
  - :class:`~ml_models.aqi.trends.NoDataForCityError` -- the requested city has
    no records in the dataset (or no dataset is available); the API should
    return **404 no data for city** (Req 8.5). This exception is re-exported
    from this module for convenience.

Both ``get_air_quality`` and ``get_trends`` raise these exceptions rather than
returning ``success: False`` dicts, so the route handlers own the HTTP status
mapping in one place.

Requirements covered: 6.4, 8.5, 9.4, 10.1, 10.2.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

from ml_models.aqi import (
    DEFAULT_DATASET,
    POLLUTANT_FEATURES,
    TEMPORAL_FEATURES,
    WEATHER_FEATURES,
)
from ml_models.aqi.advisor import advise
from ml_models.aqi.classifier import classify
from ml_models.aqi.data_loader import DataLoader
from ml_models.aqi.forecaster import MultiStepForecaster
from ml_models.aqi.live_source import LiveAirQualitySource, LiveDataError
from ml_models.aqi.predictor import AQIPredictor
from ml_models.aqi.trends import NoDataForCityError, TrendVisualizer

# Canonical helper columns produced by the DataLoader on the historical frame.
_CITY_COLUMN = "City"
_TIMESTAMP_COLUMN = "timestamp"

# The pollutant column that the response surfaces as the "current" pm25 value.
_PM25_COLUMN = "PM2.5"


class ModelUnavailableError(Exception):
    """Raised when the persisted AQI model could not be loaded.

    The API layer catches this to return a model-unavailable (HTTP 503)
    response (Req 6.4).
    """

    def __init__(self, message: str = "AQI model is unavailable") -> None:
        super().__init__(message)


class AQIService:
    """Orchestrate prediction, classification, advisory, and trends per city.

    On construction the service loads the persisted model (recording whether the
    load succeeded on :pyattr:`model_loaded`). For per-city data it prefers a
    live Open-Meteo air-quality fetch (:class:`LiveAirQualitySource`), which
    supplies both the "current" features and the trend history. A static
    historical CSV (loaded once and cached) is retained as a fallback for when
    the live source is unavailable (e.g. offline). When neither source yields
    data for a city, requests surface :class:`NoDataForCityError`.
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET,
        live_source: Optional[LiveAirQualitySource] = None,
        use_live: bool = True,
    ) -> None:
        self.predictor = AQIPredictor()
        # Req 6.4: record load outcome so requests can surface model-unavailable.
        self.model_loaded: bool = self.predictor.load()
        self.trends = TrendVisualizer()
        self.dataset_path = dataset_path
        # Primary per-city source: live Open-Meteo air-quality data.
        self.use_live = use_live
        self.live_source = live_source or LiveAirQualitySource()
        # Fallback historical frame; None when no dataset is available.
        self._dataset: Optional[pd.DataFrame] = self._load_dataset(dataset_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_air_quality(self, city: str) -> Dict[str, Any]:
        """Predict, classify, advise, and plot for ``city``.

        Steps:
          1. Raise :class:`ModelUnavailableError` if the model failed to load
             (Req 6.4).
          2. Find the city's most recent historical record (the source of input
             features and the "current" display values); raise
             :class:`NoDataForCityError` if the city has no records (Req 8.5).
          3. Build a feature dict from that record and predict the future AQI;
             the predictor substitutes medians for absent features.
          4. Attach the CPCB bucket and the matching health advisory (Req 9.4).
          5. Render the daily trend forecast plot and expose its web URL.
          6. Assemble the response payload (Req 10.1).

        Parameters
        ----------
        city:
            City name to predict air quality for.

        Returns
        -------
        dict
            ``{success, city, predicted_aqi, bucket, advisory,
            data: {current: {time, pm25, category}}, plot_url}``.

        Raises
        ------
        ModelUnavailableError
            If the persisted model could not be loaded (Req 6.4).
        NoDataForCityError
            If ``city`` has no records in the dataset (Req 8.5).
        """
        if not self.model_loaded:
            raise ModelUnavailableError()

        city_frame = self._city_frame(city)  # raises NoDataForCityError if absent
        latest = self._latest_record(city, city_frame)
        # For the "current" display, prefer the most recent row that actually
        # carries a PM2.5 reading (the very latest forecast-edge hour can be
        # null), falling back to the latest row otherwise.
        observed = self._latest_observed_record(city_frame)
        current_record = observed if observed is not None else latest

        features = self._build_features(current_record)
        prediction = self.predictor.predict(features)
        predicted_aqi = round(float(prediction["aqi"]), 2)
        bucket = prediction["bucket"]
        advisory = advise(bucket)  # Req 9.4

        # Render the daily trend forecast plot for the city and expose its URL.
        daily_series = self.trends.daily(city_frame, city)
        plot_path = self.trends.render_forecast_plot(city, daily_series)
        plot_url = self._to_web_url(plot_path)

        return {
            "success": True,
            "city": city,
            "predicted_aqi": predicted_aqi,
            "bucket": bucket,
            "advisory": advisory,
            "data": {"current": self._current_observation(current_record)},
            "plot_url": plot_url,
        }

    def get_trends(self, city: str) -> Dict[str, Any]:
        """Return daily/weekly/monthly average-AQI series for ``city`` (Req 10.2).

        Parameters
        ----------
        city:
            City name to compute trends for.

        Returns
        -------
        dict
            ``{success, city, daily, weekly, monthly}`` where each series is a
            :class:`~ml_models.aqi.trends.TrendSeries`.

        Raises
        ------
        NoDataForCityError
            If ``city`` has no records in the dataset (Req 8.5). The
            TrendVisualizer series methods raise this directly.
        """
        dataset = self._city_frame(city)
        return {
            "success": True,
            "city": city,
            "daily": self.trends.daily(dataset, city),
            "weekly": self.trends.weekly(dataset, city),
            "monthly": self.trends.monthly(dataset, city),
        }

    def get_forecast(self, city: str) -> Dict[str, Any]:
        """Return a five-hour recursive AQI forecast series for ``city`` (Req 12.6).

        Steps:
          1. Raise :class:`ModelUnavailableError` if the model failed to load
             (Req 12.8).
          2. Find the city's most recent historical record; raise
             :class:`NoDataForCityError` if the city has no records (Req 12.7).
          3. Build the base feature dict and derive ``last_aqi`` from the
             record's AQI value.
          4. Delegate the recursion to
             :class:`~ml_models.aqi.forecaster.MultiStepForecaster` and
             assemble the response payload (Req 12.1–12.5).

        Parameters
        ----------
        city:
            City name to produce a forecast for.

        Returns
        -------
        dict
            ``{success, city, forecast: [{hour_offset, aqi, bucket}, ...]}``.

        Raises
        ------
        ModelUnavailableError
            If the persisted model could not be loaded (Req 12.8).
        NoDataForCityError
            If ``city`` has no records in the dataset (Req 12.7).
        """
        if not self.model_loaded:
            raise ModelUnavailableError()

        city_frame = self._city_frame(city)  # raises NoDataForCityError if absent
        latest = self._latest_record(city, city_frame)

        base_features = self._build_features(latest)

        # Derive last_aqi from the most recent record's AQI column.
        last_aqi: float = 0.0
        if "AQI" in latest.index:
            aqi_numeric = pd.to_numeric(latest["AQI"], errors="coerce")
            if pd.notna(aqi_numeric):
                last_aqi = float(aqi_numeric)

        forecaster = MultiStepForecaster(self.predictor)
        forecast_points = forecaster.forecast(base_features, last_aqi)

        return {
            "success": True,
            "city": city,
            "forecast": list(forecast_points),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _city_frame(self, city: str) -> pd.DataFrame:
        """Return the per-city hourly frame, preferring the live source.

        Resolution order:
          1. Live Open-Meteo air-quality fetch (when ``use_live``). Returns a
             frame shaped like the historical loader output (``City``,
             ``timestamp``, pollutants, ``AQI`` + ``Hour``/``Day``/``Month``).
          2. The cached historical CSV, filtered to the city, as a fallback when
             the live fetch fails (offline, geocoding miss, malformed response).

        Raises :class:`NoDataForCityError` when neither source yields rows for
        the city (Req 8.5).
        """
        if self.use_live:
            try:
                return self.live_source.fetch_city(city)
            except LiveDataError:
                # Fall through to the historical CSV fallback below.
                pass

        return self._city_rows_from_dataset(city)

    def _city_rows_from_dataset(self, city: str) -> pd.DataFrame:
        """Return the historical CSV rows for ``city`` (fallback path).

        Raises :class:`NoDataForCityError` when no dataset is available or the
        city has no rows in it (Req 8.5).
        """
        dataset = self._require_dataset(city)
        if _CITY_COLUMN not in dataset.columns:
            raise NoDataForCityError(city)

        cities = dataset[_CITY_COLUMN].astype(str).str.strip().str.casefold()
        mask = cities == str(city).strip().casefold()
        if not mask.any():
            raise NoDataForCityError(city)

        return dataset.loc[mask].copy()

    def _load_dataset(self, dataset_path: str) -> Optional[pd.DataFrame]:
        """Load the historical dataset once, swallowing any load failure.

        Returns ``None`` when the dataset is missing or unreadable so the service
        still constructs; per-city requests then surface
        :class:`NoDataForCityError` (Req 8.5).
        """
        try:
            return DataLoader(dataset_path=dataset_path).load()
        except Exception:
            # Missing/unreadable dataset must not hard-fail service construction.
            return None

    def _require_dataset(self, city: Optional[str] = None) -> pd.DataFrame:
        """Return the cached dataset or raise no-data when unavailable.

        When no dataset could be loaded there is no data for any city, so this
        surfaces :class:`NoDataForCityError` (Req 8.5).
        """
        if self._dataset is None or self._dataset.empty:
            raise NoDataForCityError(city if city is not None else "<unknown>")
        return self._dataset

    @staticmethod
    def _latest_record(city: str, city_frame: pd.DataFrame) -> pd.Series:
        """Return the most recent record from a city's frame.

        Raises :class:`NoDataForCityError` when the frame has no rows (Req 8.5).
        """
        if city_frame is None or city_frame.empty:
            raise NoDataForCityError(city)

        rows = city_frame.copy()
        if _TIMESTAMP_COLUMN in rows.columns:
            rows[_TIMESTAMP_COLUMN] = pd.to_datetime(
                rows[_TIMESTAMP_COLUMN], errors="coerce"
            )
            rows = rows.sort_values(_TIMESTAMP_COLUMN)

        return rows.iloc[-1]

    @staticmethod
    def _latest_observed_record(city_frame: pd.DataFrame) -> Optional[pd.Series]:
        """Return the most recent row that has a non-null PM2.5 reading.

        The newest forecast-edge hour can have null pollutant cells; for the
        "current" display we prefer the latest row with an actual PM2.5 value.
        Returns ``None`` when no row has a PM2.5 reading (caller falls back).
        """
        if city_frame is None or city_frame.empty or _PM25_COLUMN not in city_frame.columns:
            return None

        rows = city_frame.copy()
        if _TIMESTAMP_COLUMN in rows.columns:
            rows[_TIMESTAMP_COLUMN] = pd.to_datetime(
                rows[_TIMESTAMP_COLUMN], errors="coerce"
            )
            rows = rows.sort_values(_TIMESTAMP_COLUMN)

        observed = rows[pd.to_numeric(rows[_PM25_COLUMN], errors="coerce").notna()]
        if observed.empty:
            return None
        return observed.iloc[-1]

    @staticmethod
    def _build_features(record: pd.Series) -> Dict[str, float]:
        """Build the predictor feature dict from a historical record.

        Maps the pollutant, weather, and temporal columns present on the record
        to a numeric dict. Absent or non-numeric values are simply omitted; the
        predictor substitutes the training median for any feature it does not
        receive (Req 6.2).
        """
        features: Dict[str, float] = {}
        for column in POLLUTANT_FEATURES + WEATHER_FEATURES + TEMPORAL_FEATURES:
            if column not in record.index:
                continue
            value = record[column]
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.notna(numeric):
                features[column] = float(numeric)
        return features

    @staticmethod
    def _current_observation(record: pd.Series) -> Dict[str, Any]:
        """Build the ``data.current`` block from the latest record.

        Contains the ISO ``time`` of the observation, the observed ``pm25``
        value, and the CPCB ``category`` of the observed AQI.
        """
        time_value: Optional[str] = None
        if _TIMESTAMP_COLUMN in record.index:
            parsed = pd.to_datetime(record[_TIMESTAMP_COLUMN], errors="coerce")
            if pd.notna(parsed):
                time_value = parsed.isoformat()

        pm25: Optional[float] = None
        if _PM25_COLUMN in record.index:
            pm25_numeric = pd.to_numeric(record[_PM25_COLUMN], errors="coerce")
            if pd.notna(pm25_numeric):
                pm25 = float(pm25_numeric)

        category: Optional[str] = None
        if "AQI" in record.index:
            aqi_numeric = pd.to_numeric(record["AQI"], errors="coerce")
            if pd.notna(aqi_numeric):
                category = classify(float(aqi_numeric))

        return {"time": time_value, "pm25": pm25, "category": category}

    @staticmethod
    def _to_web_url(filesystem_path: str) -> str:
        """Convert a static filesystem path to a leading-slash web URL.

        e.g. ``"static/aqi/Delhi_forecast.png"`` ->
        ``"/static/aqi/Delhi_forecast.png"``. Backslashes (Windows) are
        normalized to forward slashes for the URL.
        """
        normalized = filesystem_path.replace(os.sep, "/")
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return normalized


# ---------------------------------------------------------------------------
# Cached singleton accessor
# ---------------------------------------------------------------------------
# Module-level cache so the service (and its loaded model + dataset) is created
# once and reused across requests (Req 10.1).
_SERVICE_SINGLETON: Optional[AQIService] = None


def get_aqi_service() -> AQIService:
    """Return a process-wide cached :class:`AQIService` instance.

    The first call constructs the service (loading the model and dataset once);
    subsequent calls return the same instance so the API layer does not reload
    the artifact on every request.
    """
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        _SERVICE_SINGLETON = AQIService()
    return _SERVICE_SINGLETON
