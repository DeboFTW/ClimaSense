"""Feature engineering for the AQI prediction pipeline.

The :class:`FeatureEngineer` is the third stage of the offline training pipeline
(``DataLoader -> Preprocessor -> FeatureEngineer -> ModelEvaluator``). It derives
the temporal Lag_Features and Rolling_Average that let the regression models
learn temporal patterns, plus the prediction ``target``.

Per city, ordered by timestamp, it computes:
  - ``Prev_AQI``     : the AQI of the immediately preceding period (Req 4.1).
  - ``AQI_24h_avg``  : the trailing 24-hour rolling mean of AQI (Req 4.2).
  - ``target``       : AQI shifted forward by the Forecast_Horizon (Req 4.5).

It also guarantees the Temporal_Features ``Hour``/``Day``/``Month`` are present
(Req 4.3), imputes lag/rolling cells that lack sufficient prior history with the
training-set median for that feature (Req 4.4), and drops rows whose target is
not available so they are excluded from training (Req 4.5).

Input contract
--------------
Unlike the fully-numeric matrix emitted by the
:class:`~ml_models.aqi.preprocessor.Preprocessor`, the FeatureEngineer needs the
``City`` and ``timestamp`` columns to order records and compute per-city
time-based lag/rolling features. In the training pipeline these two columns are
carried through alongside the numeric feature matrix. This component therefore
expects an input frame that includes:

  - ``City``       : city label used to group records (optional; when absent the
                     whole frame is treated as a single city).
  - ``timestamp``  : a parseable datetime used both to order records and to
                     anchor the time-based 24-hour rolling window. Required.
  - ``AQI``        : the source target column used to derive the lag, rolling,
                     and forward-shifted ``target``.
  - any pollutant / weather / temporal columns produced upstream.

Requirements covered: 4.1, 4.2, 4.3, 4.4, 4.5.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml_models.aqi import (
    FORECAST_HORIZON,
    LAG_FEATURES,
    POLLUTANT_FEATURES,
    TARGET_COLUMN,
    TEMPORAL_FEATURES,
    WEATHER_FEATURES,
)

# Canonical helper column names (consistent with DataLoader / Preprocessor).
_CITY_COLUMN = "City"
_TIMESTAMP_COLUMN = "timestamp"
_TARGET_OUTPUT_COLUMN = "target"

# Trailing window for the rolling AQI average (pandas time-offset string).
_ROLLING_WINDOW = "24h"

# Fallback used only when a lag/rolling feature has no observable values at all
# (e.g. a single-record-per-city dataset), so imputation can still produce a
# fully-numeric column.
_EMPTY_MEDIAN_FILL = 0.0


class FeatureEngineer:
    """Derive lag, rolling-average, temporal features and the prediction target.

    Parameters
    ----------
    horizon:
        Number of periods to shift AQI forward to form the prediction target.
        Defaults to :data:`ml_models.aqi.FORECAST_HORIZON` (next-period).
    """

    def __init__(self, horizon: int = FORECAST_HORIZON) -> None:
        self.horizon = horizon
        self._lag_medians: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer lag/rolling/temporal features and the target on ``df``.

        The transformation, in order:

        1. Order records by ``City`` then ``timestamp`` (Req 4.1/4.2 require a
           per-city, timestamp-ordered series).
        2. Ensure Temporal_Features ``Hour``/``Day``/``Month`` are present,
           deriving them from ``timestamp`` when absent (Req 4.3).
        3. Compute ``Prev_AQI`` (per-city preceding period) (Req 4.1).
        4. Compute ``AQI_24h_avg`` (per-city trailing 24-hour rolling mean) on a
           timestamp index, marking the first record of each city as
           insufficient history (Req 4.2/4.4).
        5. Compute ``target`` = AQI shifted forward by ``horizon`` (Req 4.5).
        6. Impute insufficient-history lag/rolling cells with the training-set
           median for that feature (Req 4.4); medians are computed over the
           available (non-null) computed values and retained on
           :pyattr:`lag_medians`.
        7. Drop rows lacking a valid target so they are excluded from training
           (Req 4.5).

        Parameters
        ----------
        df:
            Input frame including ``timestamp`` and ``AQI`` (and ideally
            ``City``); see the module docstring for the full contract.

        Returns
        -------
        pandas.DataFrame
            The engineered frame: the input columns plus ``Prev_AQI``,
            ``AQI_24h_avg``, and ``target``, ordered by ``City``/``timestamp``,
            with rows lacking a valid target removed.

        Raises
        ------
        ValueError
            If the required ``timestamp`` column or the ``AQI`` target column is
            absent from ``df``.
        """
        self._validate_columns(df)

        work = df.copy()
        work[_TIMESTAMP_COLUMN] = pd.to_datetime(
            work[_TIMESTAMP_COLUMN], errors="coerce"
        )
        work = work.dropna(subset=[_TIMESTAMP_COLUMN])

        has_city = _CITY_COLUMN in work.columns
        sort_keys = [_CITY_COLUMN, _TIMESTAMP_COLUMN] if has_city else [_TIMESTAMP_COLUMN]
        work = work.sort_values(sort_keys).reset_index(drop=True)

        work = self._ensure_temporal_features(work)

        # Per-city grouping; when no City column is present the entire frame is a
        # single implicit group.
        grouper = work.groupby(_CITY_COLUMN) if has_city else None

        work["Prev_AQI"] = self._previous_aqi(work, grouper)
        work["AQI_24h_avg"] = self._rolling_24h_average(work, has_city)
        work[_TARGET_OUTPUT_COLUMN] = self._forward_target(work, grouper)

        # Req 4.4: cells lacking prior history (the first record of each city for
        # both the lag and the rolling features) are imputed with the training
        # median, computed over the available non-null values.
        self._lag_medians = self._compute_lag_medians(work)
        for feature in LAG_FEATURES:
            work[feature] = work[feature].fillna(self._lag_medians[feature])

        # Req 4.5: rows without a valid forward target cannot be used for
        # training and are dropped.
        work = work[work[_TARGET_OUTPUT_COLUMN].notna()].reset_index(drop=True)

        return self._order_columns(work)

    @property
    def feature_columns(self) -> List[str]:
        """Model input columns: pollutant + weather + temporal + lag features.

        This is the ordered list of columns used to train the regression models,
        excluding ``target``, ``AQI``, ``City``, and ``timestamp``.
        """
        return list(POLLUTANT_FEATURES + WEATHER_FEATURES + TEMPORAL_FEATURES + LAG_FEATURES)

    @property
    def lag_medians(self) -> Dict[str, float]:
        """Training-set medians used to impute insufficient-history lag cells.

        Populated by the most recent :meth:`transform` call.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`transform` has been called.
        """
        if self._lag_medians is None:
            raise RuntimeError(
                "lag_medians is unavailable until transform() has been called"
            )
        return dict(self._lag_medians)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Ensure the required ``timestamp`` and ``AQI`` columns are present."""
        missing = [
            col for col in (_TIMESTAMP_COLUMN, TARGET_COLUMN) if col not in df.columns
        ]
        if missing:
            raise ValueError(
                "FeatureEngineer requires columns "
                f"{missing} which are absent from the input frame; "
                f"available columns: {list(df.columns)}"
            )

    @staticmethod
    def _ensure_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive any missing ``Hour``/``Day``/``Month`` from ``timestamp`` (Req 4.3)."""
        ts = df[_TIMESTAMP_COLUMN]
        if "Hour" not in df.columns:
            df["Hour"] = ts.dt.hour.astype(int)
        if "Day" not in df.columns:
            df["Day"] = ts.dt.day.astype(int)
        if "Month" not in df.columns:
            df["Month"] = ts.dt.month.astype(int)
        return df

    @staticmethod
    def _previous_aqi(df: pd.DataFrame, grouper) -> pd.Series:
        """Compute ``Prev_AQI`` = AQI of the preceding period per city (Req 4.1).

        The first record of each city has no preceding period and yields ``NaN``
        (later imputed with the training median).
        """
        if grouper is not None:
            return grouper[TARGET_COLUMN].shift(1)
        return df[TARGET_COLUMN].shift(1)

    def _forward_target(self, df: pd.DataFrame, grouper) -> pd.Series:
        """Compute ``target`` = AQI shifted forward by the horizon per city (Req 4.5).

        The final ``horizon`` records of each city have no future AQI to predict
        and yield ``NaN``; those rows are dropped from the training set by the
        caller.
        """
        if grouper is not None:
            return grouper[TARGET_COLUMN].shift(-self.horizon)
        return df[TARGET_COLUMN].shift(-self.horizon)

    def _rolling_24h_average(self, df: pd.DataFrame, has_city: bool) -> pd.Series:
        """Compute the per-city trailing 24-hour rolling mean of AQI (Req 4.2).

        The rolling mean is computed on a timestamp index within each city group,
        then realigned to the frame's positional index. The first record of each
        city has no prior history, so its value is set to ``NaN`` and treated as
        insufficient history (Req 4.4), to be imputed with the training median.
        """
        result = pd.Series(np.nan, index=df.index, dtype=float)

        if has_city:
            groups = df.groupby(_CITY_COLUMN).groups.values()
        else:
            groups = [df.index]

        for idx in groups:
            group = df.loc[idx].sort_values(_TIMESTAMP_COLUMN)
            rolled = (
                group.set_index(_TIMESTAMP_COLUMN)[TARGET_COLUMN]
                .rolling(_ROLLING_WINDOW)
                .mean()
            )
            # Restore the original positional index so values realign on assign.
            rolled.index = group.index
            # Mark the earliest record of the city as insufficient history.
            if len(group) > 0:
                rolled.iloc[0] = np.nan
            result.loc[group.index] = rolled

        return result

    @staticmethod
    def _compute_lag_medians(df: pd.DataFrame) -> Dict[str, float]:
        """Median of each lag feature over its available (non-null) values (Req 4.4)."""
        medians: Dict[str, float] = {}
        for feature in LAG_FEATURES:
            computed = df[feature].median()
            medians[feature] = (
                float(computed) if pd.notna(computed) else _EMPTY_MEDIAN_FILL
            )
        return medians

    def _order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the engineered frame with a stable, readable column order."""
        leading = [
            col for col in (_CITY_COLUMN, _TIMESTAMP_COLUMN) if col in df.columns
        ]
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        trailing = [
            col
            for col in (TARGET_COLUMN, _TARGET_OUTPUT_COLUMN)
            if col in df.columns
        ]

        ordered = leading + feature_cols + trailing
        # Append any remaining columns not explicitly placed, preserving them.
        remaining = [col for col in df.columns if col not in ordered]
        return df[ordered + remaining].reset_index(drop=True)
