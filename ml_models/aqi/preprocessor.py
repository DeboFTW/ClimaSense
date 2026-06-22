"""Data preprocessing for the AQI prediction pipeline.

The :class:`Preprocessor` is the second stage of the offline training pipeline
(``DataLoader -> Preprocessor -> FeatureEngineer -> ModelEvaluator``). It turns
the raw, possibly-incomplete frame produced by the
:class:`~ml_models.aqi.data_loader.DataLoader` into a clean, fully-numeric
feature matrix suitable for feature engineering and model training.

Responsibilities:
  - Drop rows that are missing the ``AQI`` target before anything else, so the
    excluded rows never influence the imputation medians (Req 2.2).
  - Compute the training-set median for each Pollutant_Feature and
    Weather_Feature and impute every missing cell with it; the medians are
    retained on :pyattr:`feature_medians` (Req 2.1).
  - Derive the Temporal_Features ``Hour``, ``Day``, and ``Month`` from each
    record's timestamp (Req 2.3).
  - Return a frame whose every cell is a finite numeric value (Req 2.4).
  - Raise :class:`ValueError` when no usable records remain (Req 2.5).

Per the design data model, the output feature matrix columns are exactly
``POLLUTANT_FEATURES + WEATHER_FEATURES + TEMPORAL_FEATURES + [AQI]`` and every
column is a numeric float. Weather (or pollutant) columns absent from the source
are added and filled so the contract holds even when a source omits them.

Requirements covered: 2.1, 2.2, 2.3, 2.4, 2.5.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ml_models.aqi import (
    POLLUTANT_FEATURES,
    TARGET_COLUMN,
    TEMPORAL_FEATURES,
    WEATHER_FEATURES,
)

# Canonical timestamp column produced by the DataLoader. A few aliases are
# accepted so the Preprocessor is robust to raw frames fed in directly.
_TIMESTAMP_COLUMN = "timestamp"
_TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "date_time")

# Fallback used only when a feature column is entirely absent or has no
# observable values, so a median cannot be computed. Keeps Req 2.4 satisfiable.
_EMPTY_COLUMN_FILL = 0.0


class Preprocessor:
    """Clean, impute, and standardize raw records into a numeric feature matrix.

    The class follows the scikit-learn ``fit_transform`` naming convention: the
    medians used for imputation are learned from the input frame ("fit") and the
    same call returns the transformed feature matrix.
    """

    def __init__(self) -> None:
        self._feature_medians: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize ``df`` into a fully-numeric feature matrix.

        The transformation, in order:

        1. Drop every row missing the ``AQI`` target (Req 2.2) so excluded rows
           do not bias the imputation medians.
        2. Parse the timestamp and drop rows whose timestamp is unparseable,
           since they cannot yield valid Temporal_Features.
        3. Raise :class:`ValueError` if no usable records remain (Req 2.5).
        4. Derive ``Hour``/``Day``/``Month`` from the timestamp (Req 2.3).
        5. Compute the training-set median per pollutant/weather feature and
           impute missing cells with it, recording the medians on
           :pyattr:`feature_medians` (Req 2.1).

        Parameters
        ----------
        df:
            Raw records, typically from :meth:`DataLoader.load`, containing at
            least ``AQI`` and a ``timestamp`` column plus any available
            pollutant/weather columns.

        Returns
        -------
        pandas.DataFrame
            A frame with columns
            ``POLLUTANT_FEATURES + WEATHER_FEATURES + TEMPORAL_FEATURES + [AQI]``
            in which every cell is a finite numeric value (Req 2.4).

        Raises
        ------
        ValueError
            If zero usable records remain after dropping rows missing the AQI
            target or an unparseable timestamp (Req 2.5).
        """
        work = df.copy()

        work = self._drop_missing_target(work)
        work = self._parse_timestamp(work)

        # Req 2.5: nothing left to train on.
        if len(work) == 0:
            raise ValueError("No usable records after preprocessing")

        work = self._add_temporal_features(work)
        medians = self._impute_features(work)
        self._feature_medians = medians

        return self._build_feature_matrix(work)

    @property
    def feature_medians(self) -> Dict[str, float]:
        """Training-set median used to impute each pollutant/weather feature.

        Populated by the most recent :meth:`fit_transform` call.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`fit_transform` has been called.
        """
        if self._feature_medians is None:
            raise RuntimeError(
                "feature_medians is unavailable until fit_transform() has been called"
            )
        return dict(self._feature_medians)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows missing the ``AQI`` target (Req 2.2).

        If the target column is entirely absent, no record is usable, so an
        empty frame is returned and the caller surfaces the Req 2.5 error.
        """
        if TARGET_COLUMN not in df.columns:
            return df.iloc[0:0]
        return df[df[TARGET_COLUMN].notna()].reset_index(drop=True)

    @classmethod
    def _parse_timestamp(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the timestamp column and drop rows that fail to parse.

        Rows with an unparseable timestamp cannot produce valid
        Temporal_Features and would violate the non-null guarantee (Req 2.4), so
        they are excluded here.
        """
        ts_col = cls._find_timestamp_column(df)
        if ts_col is None:
            raise ValueError(
                "No usable records after preprocessing "
                "(no recognizable timestamp column to derive Hour/Day/Month)"
            )

        if ts_col != _TIMESTAMP_COLUMN:
            df = df.rename(columns={ts_col: _TIMESTAMP_COLUMN})

        df = df.copy()
        df[_TIMESTAMP_COLUMN] = pd.to_datetime(df[_TIMESTAMP_COLUMN], errors="coerce")
        return df.dropna(subset=[_TIMESTAMP_COLUMN]).reset_index(drop=True)

    @staticmethod
    def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
        """Return the first column matching a timestamp alias (case-insensitive)."""
        lookup = {col.lower(): col for col in df.columns}
        for candidate in _TIMESTAMP_CANDIDATES:
            if candidate in lookup:
                return lookup[candidate]
        return None

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive ``Hour``/``Day``/``Month`` from the parsed timestamp (Req 2.3)."""
        ts = df[_TIMESTAMP_COLUMN]
        df["Hour"] = ts.dt.hour.astype(int)
        df["Day"] = ts.dt.day.astype(int)
        df["Month"] = ts.dt.month.astype(int)
        return df

    @staticmethod
    def _impute_features(df: pd.DataFrame) -> Dict[str, float]:
        """Impute pollutant/weather cells with the training median (Req 2.1).

        The frame is mutated in place. Columns absent from the source are added
        (per the design data model) and filled. The median for each column is
        computed over its observed (non-null) values; a column with no
        observable values falls back to :data:`_EMPTY_COLUMN_FILL` so the output
        can remain fully numeric (Req 2.4).

        Returns
        -------
        dict[str, float]
            The median used for each pollutant/weather feature.
        """
        medians: Dict[str, float] = {}
        for col in POLLUTANT_FEATURES + WEATHER_FEATURES:
            if col not in df.columns:
                # Absent column: add it so the feature matrix has every expected
                # column, with no observable values to compute a median from.
                df[col] = pd.NA
                median = _EMPTY_COLUMN_FILL
            else:
                # Coerce to numeric so stray non-numeric strings become NaN and
                # are imputed rather than breaking the fully-numeric guarantee.
                df[col] = pd.to_numeric(df[col], errors="coerce")
                computed = df[col].median()
                median = (
                    float(computed) if pd.notna(computed) else _EMPTY_COLUMN_FILL
                )

            medians[col] = median
            df[col] = df[col].fillna(median).astype(float)

        return medians

    @staticmethod
    def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Assemble the ordered, fully-numeric output frame (Req 2.4).

        Columns are exactly
        ``POLLUTANT_FEATURES + WEATHER_FEATURES + TEMPORAL_FEATURES + [AQI]``.
        """
        output_columns = (
            POLLUTANT_FEATURES + WEATHER_FEATURES + TEMPORAL_FEATURES + [TARGET_COLUMN]
        )
        matrix = df[output_columns].copy()
        return matrix.astype(float).reset_index(drop=True)
