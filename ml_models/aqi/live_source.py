"""Live air-quality data source backed by the Open-Meteo Air Quality API.

This module provides :class:`LiveAirQualitySource`, which fetches recent hourly
pollutant data for a city directly from Open-Meteo -- the same provider the
ClimaSense weather features come from -- so the AQI serving path can derive
"current" features and trend series from live data instead of a static CSV.

Two keyless Open-Meteo endpoints are used:

  - Geocoding API (``geocoding-api.open-meteo.com/v1/search``) resolves a city
    name to latitude/longitude.
  - Air Quality API (``air-quality-api.open-meteo.com/v1/air-quality``) returns
    hourly pollutant concentrations (and a reference ``us_aqi``).

The fetched data is normalized into a DataFrame whose columns match the historical
frame produced by :class:`~ml_models.aqi.data_loader.DataLoader`
(``City``, ``timestamp``, the six ``POLLUTANT_FEATURES``, and ``AQI``), plus the
derived temporal columns ``Hour``/``Day``/``Month``. This lets the existing
:class:`~ml_models.aqi.service.AQIService`,
:class:`~ml_models.aqi.predictor.AQIPredictor`, and
:class:`~ml_models.aqi.trends.TrendVisualizer` consume it unchanged.

Note on the AQI value: Open-Meteo computes ``us_aqi`` on the US EPA scale, which
differs from the Indian CPCB buckets the model/classifier use. The pollutant
concentrations are scale-independent and feed the model directly; ``us_aqi`` is
surfaced only as the observed reference AQI for the "current" display and the
trend series.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Open-Meteo endpoints (no API key required), consistent with the weather path.
# ---------------------------------------------------------------------------
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo hourly variable -> our canonical pollutant column name.
_POLLUTANT_VARIABLE_MAP: Dict[str, str] = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "carbon_monoxide": "CO",
    "ozone": "O3",
}
# The reference AQI variable Open-Meteo exposes (US EPA scale).
_AQI_VARIABLE = "us_aqi"

# Ordered list of hourly variables requested from the Air Quality API.
_HOURLY_VARIABLES: List[str] = list(_POLLUTANT_VARIABLE_MAP.keys()) + [_AQI_VARIABLE]

# Canonical output columns produced by the DataLoader (re-declared here to avoid
# importing the loader and to keep this module self-contained).
_CITY_COLUMN = "City"
_TIMESTAMP_COLUMN = "timestamp"

# Defaults for history depth, request timeout, and cache lifetime.
_DEFAULT_HISTORY_DAYS = 30
_DEFAULT_TIMEOUT = 10  # seconds
_DEFAULT_CACHE_TTL = 900  # seconds (15 minutes)


class LiveDataError(Exception):
    """Raised when live air-quality data cannot be fetched or parsed.

    Covers a city that cannot be geocoded, network/HTTP failures, and malformed
    or empty responses. The service layer treats this as "no data for the city".
    """


class LiveAirQualitySource:
    """Fetch recent hourly air-quality data for a city from Open-Meteo.

    Results are cached per city for ``cache_ttl`` seconds so repeated requests
    (e.g. the air-quality and trends endpoints for the same city) reuse a single
    network round trip.

    Parameters
    ----------
    history_days:
        How many days of hourly history to request (drives the trend depth).
    timeout:
        Per-request HTTP timeout in seconds.
    cache_ttl:
        How long a fetched city frame stays valid in the in-memory cache.
    """

    def __init__(
        self,
        history_days: int = _DEFAULT_HISTORY_DAYS,
        timeout: int = _DEFAULT_TIMEOUT,
        cache_ttl: int = _DEFAULT_CACHE_TTL,
    ) -> None:
        self.history_days = history_days
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        # Cache: normalized city -> (fetched_at_epoch, frame).
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_city(self, city: str) -> pd.DataFrame:
        """Return a normalized hourly air-quality frame for ``city``.

        The frame has columns ``City``, ``timestamp``, the six pollutant
        features, ``AQI`` (from Open-Meteo ``us_aqi``), and derived
        ``Hour``/``Day``/``Month``. Rows are ordered by timestamp.

        Parameters
        ----------
        city:
            City name to resolve and fetch.

        Returns
        -------
        pandas.DataFrame
            The normalized hourly frame (at least one row).

        Raises
        ------
        LiveDataError
            If the city cannot be geocoded, the request fails, or the response
            contains no usable rows.
        """
        if not city or not city.strip():
            raise LiveDataError("A non-empty city name is required")

        cached = self._cached_frame(city)
        if cached is not None:
            return cached

        latitude, longitude, resolved_name = self._geocode(city)
        payload = self._request_air_quality(latitude, longitude)
        frame = self._build_frame(payload, city)

        self._cache[self._key(city)] = (time.monotonic(), frame)
        return frame

    # ------------------------------------------------------------------
    # Internal helpers -- caching
    # ------------------------------------------------------------------
    @staticmethod
    def _key(city: str) -> str:
        """Normalize a city name into a cache key."""
        return str(city).strip().casefold()

    def _cached_frame(self, city: str) -> Optional[pd.DataFrame]:
        """Return a fresh cached frame for ``city`` if one exists, else None."""
        entry = self._cache.get(self._key(city))
        if entry is None:
            return None
        fetched_at, frame = entry
        if time.monotonic() - fetched_at > self.cache_ttl:
            # Stale: drop it so the next call refetches.
            self._cache.pop(self._key(city), None)
            return None
        return frame

    # ------------------------------------------------------------------
    # Internal helpers -- network
    # ------------------------------------------------------------------
    def _geocode(self, city: str) -> Tuple[float, float, str]:
        """Resolve ``city`` to (latitude, longitude, resolved_name)."""
        try:
            response = requests.get(
                _GEOCODING_URL,
                params={"name": city, "count": 1, "format": "json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LiveDataError(f"Failed to geocode city '{city}': {exc}") from exc

        results = data.get("results") or []
        if not results:
            raise LiveDataError(f"No location found for city: {city}")

        top = results[0]
        try:
            return (
                float(top["latitude"]),
                float(top["longitude"]),
                str(top.get("name", city)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LiveDataError(
                f"Geocoding response for '{city}' was malformed: {exc}"
            ) from exc

    def _request_air_quality(self, latitude: float, longitude: float) -> dict:
        """Fetch hourly air-quality data for the given coordinates."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(_HOURLY_VARIABLES),
            "past_days": self.history_days,
            "timezone": "auto",
        }
        try:
            response = requests.get(
                _AIR_QUALITY_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LiveDataError(
                f"Failed to fetch air quality from Open-Meteo: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers -- parsing
    # ------------------------------------------------------------------
    def _build_frame(self, payload: dict, city: str) -> pd.DataFrame:
        """Normalize an Open-Meteo air-quality payload into a DataFrame."""
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            raise LiveDataError(f"Open-Meteo returned no hourly data for: {city}")

        frame = pd.DataFrame({_TIMESTAMP_COLUMN: pd.to_datetime(times, errors="coerce")})

        # Map each requested pollutant variable to its canonical column.
        for variable, column in _POLLUTANT_VARIABLE_MAP.items():
            values = hourly.get(variable)
            frame[column] = pd.to_numeric(pd.Series(values), errors="coerce") if values else pd.NA

        # AQI from the US-EPA reference value Open-Meteo provides.
        aqi_values = hourly.get(_AQI_VARIABLE)
        frame["AQI"] = (
            pd.to_numeric(pd.Series(aqi_values), errors="coerce")
            if aqi_values
            else pd.NA
        )

        frame[_CITY_COLUMN] = city

        # Drop rows with no timestamp or no AQI; the model/trends need both.
        frame = frame.dropna(subset=[_TIMESTAMP_COLUMN, "AQI"]).reset_index(drop=True)
        if frame.empty:
            raise LiveDataError(
                f"Open-Meteo data for '{city}' contained no usable hourly rows"
            )

        frame = frame.sort_values(_TIMESTAMP_COLUMN).reset_index(drop=True)

        # Derive temporal features so the predictor receives real Hour/Day/Month.
        ts = frame[_TIMESTAMP_COLUMN]
        frame["Hour"] = ts.dt.hour.astype(int)
        frame["Day"] = ts.dt.day.astype(int)
        frame["Month"] = ts.dt.month.astype(int)

        return frame
