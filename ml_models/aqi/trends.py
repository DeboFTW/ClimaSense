"""Pollution trend visualization for the AQI prediction system.

The :class:`TrendVisualizer` is the serving-path component that turns a city's
historical AQI records into average-AQI trend series at three granularities and
renders a forecast plot the dashboard can display:

  - :meth:`daily`   -- average AQI per calendar day for a selected city (Req 8.1).
  - :meth:`weekly`  -- average AQI per ISO week for a selected city (Req 8.2).
  - :meth:`monthly` -- average AQI per calendar month for a selected city (Req 8.3).
  - :meth:`render_forecast_plot` -- a matplotlib (Agg) line plot of a series,
    persisted as a PNG under the static assets directory (returns its path).

When a requested city has no records in the dataset, the series methods raise
:class:`NoDataForCityError` so the API layer can surface a no-data (404) error
(Req 8.5).

matplotlib uses the non-interactive ``Agg`` backend, which is mandatory in this
server context where no display is available. The backend is selected *before*
``pyplot`` is imported, per the project's hard tech-stack rule.

Requirements covered: 8.1, 8.2, 8.3, 8.5.
"""

from __future__ import annotations

import os
import re
from typing import List, TypedDict

import matplotlib

# Select the non-interactive Agg backend BEFORE importing pyplot. There is no
# display in the server context, so this is a hard requirement (see tech stack).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import pandas as pd  # noqa: E402

from ml_models.aqi import EDA_DIR, TARGET_COLUMN  # noqa: E402

# Column holding the parsed observation timestamp on loaded/historical frames.
_TIMESTAMP_COLUMN = "timestamp"
# Column holding the city name.
_CITY_COLUMN = "City"

# Characters that are unsafe in a filesystem path are collapsed to this token
# when building the per-city plot filename.
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class TrendSeries(TypedDict):
    """A trend series: parallel ``labels`` (period) and ``values`` (mean AQI)."""

    labels: List[str]
    values: List[float]


class NoDataForCityError(Exception):
    """Raised when a city has no records in the dataset.

    The API layer catches this to return a no-data (HTTP 404) response (Req 8.5).
    """

    def __init__(self, city: str):
        self.city = city
        super().__init__(f"No AQI data available for city: {city}")


class TrendVisualizer:
    """Produce daily/weekly/monthly average-AQI series and a forecast plot.

    The visualizer is stateless; every method takes the historical frame as
    input so the same instance can be reused across cities and datasets.
    """

    # ------------------------------------------------------------------
    # Public API -- trend series
    # ------------------------------------------------------------------
    def daily(self, df: pd.DataFrame, city: str) -> TrendSeries:
        """Average AQI per calendar day for ``city`` (Req 8.1).

        Each point is the mean AQI of all records that fall on that day, ordered
        chronologically. Labels are ISO date strings (e.g. ``"2024-01-01"``).

        Raises
        ------
        NoDataForCityError
            If ``city`` has no records in ``df``.
        """
        timestamps = self._city_timestamps(df, city)
        periods = timestamps.dt.strftime("%Y-%m-%d")
        return self._aggregate(df, city, periods)

    def weekly(self, df: pd.DataFrame, city: str) -> TrendSeries:
        """Average AQI per ISO week for ``city`` (Req 8.2).

        Each point is the mean AQI of all records in that ISO year-week, ordered
        chronologically. Labels look like ``"2024-W01"``.

        Raises
        ------
        NoDataForCityError
            If ``city`` has no records in ``df``.
        """
        timestamps = self._city_timestamps(df, city)
        iso = timestamps.dt.isocalendar()
        periods = iso["year"].astype(int).astype(str) + "-W" + (
            iso["week"].astype(int).map(lambda w: f"{w:02d}")
        )
        # Preserve the per-row index so values realign with the source AQI.
        periods.index = timestamps.index
        return self._aggregate(df, city, periods)

    def monthly(self, df: pd.DataFrame, city: str) -> TrendSeries:
        """Average AQI per calendar month for ``city`` (Req 8.3).

        Each point is the mean AQI of all records in that year-month, ordered
        chronologically. Labels look like ``"2024-01"``.

        Raises
        ------
        NoDataForCityError
            If ``city`` has no records in ``df``.
        """
        timestamps = self._city_timestamps(df, city)
        periods = timestamps.dt.strftime("%Y-%m")
        return self._aggregate(df, city, periods)

    # ------------------------------------------------------------------
    # Public API -- plotting
    # ------------------------------------------------------------------
    def render_forecast_plot(
        self, city: str, series: TrendSeries, out_dir: str = EDA_DIR
    ) -> str:
        """Render ``series`` as a line plot and persist it as a PNG.

        The chart is written under ``out_dir`` (``static/aqi`` by default), which
        is created if it does not already exist, so the dashboard can serve the
        image. The filename is derived from ``city`` (sanitized for the
        filesystem), e.g. ``"Delhi_forecast.png"``.

        Parameters
        ----------
        city:
            The city the series belongs to; used to build the filename.
        series:
            A :class:`TrendSeries` with parallel ``labels`` and ``values``.
        out_dir:
            Destination directory for the PNG. Defaults to
            :data:`~ml_models.aqi.EDA_DIR`.

        Returns
        -------
        str
            The path to the written PNG file.
        """
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{self._safe_city(city)}_forecast.png")

        labels = series["labels"]
        values = series["values"]

        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            ax.plot(labels, values, marker="o", color="#2b8cbe")
            ax.set_title(f"AQI Trend Forecast - {city}")
            ax.set_xlabel("Period")
            ax.set_ylabel("Average AQI")
            # Rotate x labels so dense period axes stay readable.
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output_path)
        finally:
            # Always release the figure so repeated renders do not leak memory.
            plt.close(fig)

        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _city_timestamps(self, df: pd.DataFrame, city: str) -> pd.Series:
        """Return parsed timestamps for ``city``'s rows, or raise no-data.

        Matches the city case-insensitively. Raises :class:`NoDataForCityError`
        when the filter yields zero rows (Req 8.5).
        """
        mask = self._city_mask(df, city)
        if not mask.any():
            raise NoDataForCityError(city)
        return pd.to_datetime(df.loc[mask, _TIMESTAMP_COLUMN])

    @staticmethod
    def _city_mask(df: pd.DataFrame, city: str) -> pd.Series:
        """Boolean mask selecting ``city``'s rows (case-insensitive match)."""
        if _CITY_COLUMN not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        cities = df[_CITY_COLUMN].astype(str).str.strip().str.casefold()
        return cities == str(city).strip().casefold()

    def _aggregate(
        self, df: pd.DataFrame, city: str, periods: pd.Series
    ) -> TrendSeries:
        """Mean AQI grouped by ``periods`` for ``city``, ordered chronologically.

        ``periods`` is a per-row label series aligned to ``df``'s index (already
        restricted to the city's rows). Returns plain python floats and strings.
        """
        mask = self._city_mask(df, city)
        aqi = pd.to_numeric(df.loc[mask, TARGET_COLUMN], errors="coerce")

        frame = pd.DataFrame({"period": periods, "aqi": aqi}).dropna(subset=["aqi"])
        # group keys are already chronologically sortable strings.
        grouped = frame.groupby("period")["aqi"].mean().sort_index()

        return TrendSeries(
            labels=[str(label) for label in grouped.index],
            values=[float(value) for value in grouped.to_numpy()],
        )

    @staticmethod
    def _safe_city(city: str) -> str:
        """Sanitize a city name into a filesystem-safe filename token."""
        safe = _SAFE_NAME_PATTERN.sub("_", str(city).strip())
        return safe or "city"
