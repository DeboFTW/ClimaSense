"""AQI prediction from a persisted model artifact.

The :class:`AQIPredictor` is the serving-path counterpart to the offline
training pipeline. It loads the joblib artifact produced by
:meth:`ml_models.aqi.model_evaluator.ModelEvaluator.save` and answers
single-row prediction requests, substituting training medians for any absent
input features, clamping the result to be non-negative, and attaching the CPCB
bucket via :func:`ml_models.aqi.classifier.classify`.

Persisted artifact format (a joblib-serialized ``dict``)::

    {
        "model": <fitted sklearn/xgboost estimator>,
        "feature_columns": [...],          # ordered training feature names
        "feature_medians": {col: float},   # training-set median per feature
        "metadata": {...},                 # ComparisonReport + dataset metadata
    }

Design contract:
  - ``load(path=MODEL_ARTIFACT) -> bool``: load model + medians + feature
    columns; return ``False`` (never raise) when the artifact is absent or
    cannot be read, so the API layer can surface a model-unavailable error
    (Req 6.4).
  - ``predict(features) -> dict``: build the model input row in the exact order
    of ``feature_columns``, substituting the training median for any feature
    absent from ``features`` (Req 6.2); predict; clamp the result to ``>= 0``
    (Req 6.3); return a finite float ``aqi`` (Req 6.1) plus its CPCB ``bucket``
    (Req 7.3).

Requirements covered: 6.1, 6.2, 6.3, 6.4, 7.3.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from ml_models.aqi import MODEL_ARTIFACT
from ml_models.aqi.classifier import classify

# Defensive fallback used only when a feature has neither a provided value nor a
# captured training median. The training pipeline always records a median for
# every feature column, so this is a guard against malformed artifacts.
_MISSING_FEATURE_DEFAULT = 0.0


class AQIPredictor:
    """Load a persisted AQI model artifact and predict a future AQI value.

    A predictor must be successfully loaded (via :meth:`load`) before
    :meth:`predict` can be called. After a successful load, the fitted model,
    its ordered feature columns, the training-set feature medians, and the
    artifact metadata are available on the instance.
    """

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._feature_columns: Optional[List[str]] = None
        self._feature_medians: Dict[str, float] = {}
        self._metadata: Optional[dict] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, path: str = MODEL_ARTIFACT) -> bool:
        """Load the persisted model bundle from ``path``.

        On success, stores the fitted model, ordered feature columns, training
        feature medians, and metadata on the instance and returns ``True``. If
        the artifact is absent or cannot be read/deserialized, returns ``False``
        without raising, so callers (the API layer) can surface a
        model-unavailable error (Req 6.4).

        Parameters
        ----------
        path:
            Path to the joblib artifact. Defaults to
            :data:`ml_models.aqi.MODEL_ARTIFACT`.

        Returns
        -------
        bool
            ``True`` if the artifact loaded successfully, ``False`` otherwise.
        """
        # Import joblib lazily so module import never fails on a missing dep,
        # and so a broken environment surfaces as a False load rather than an
        # import-time crash.
        try:
            import joblib

            bundle = joblib.load(path)
        except Exception:
            # Absent file, unreadable file, deserialization error, or missing
            # dependency -- all are treated as "model unavailable" (Req 6.4).
            return False

        if not isinstance(bundle, dict):
            return False

        model = bundle.get("model")
        feature_columns = bundle.get("feature_columns")
        if model is None or not feature_columns:
            return False

        feature_medians = bundle.get("feature_medians") or {}

        self._model = model
        self._feature_columns = [str(col) for col in feature_columns]
        self._feature_medians = {
            str(col): float(value) for col, value in feature_medians.items()
        }
        self._metadata = bundle.get("metadata")
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict a future AQI value and classify it into a CPCB bucket.

        Builds a single-row input in the exact order of the training
        ``feature_columns``. For each column the provided value in ``features``
        is used when present; otherwise the training-set median is substituted
        (Req 6.2). The estimator's prediction is clamped to ``>= 0`` (Req 6.3)
        and returned as a finite float (Req 6.1) alongside its CPCB bucket
        (Req 7.3).

        Parameters
        ----------
        features:
            Mapping of feature name to numeric value. Any feature absent from
            this mapping is filled with its training median; unknown keys are
            ignored (only ``feature_columns`` are consumed).

        Returns
        -------
        dict
            ``{"aqi": float, "bucket": str}`` where ``bucket == classify(aqi)``.

        Raises
        ------
        RuntimeError
            If called before a successful :meth:`load`.
        ValueError
            If the model produces a non-finite prediction.
        """
        if self._model is None or self._feature_columns is None:
            raise RuntimeError(
                "AQIPredictor.predict() called before a model was loaded. "
                "Call load() and ensure it returned True first."
            )

        features = features or {}

        # Assemble the row in the exact training column order so the estimator
        # sees the same feature names/positions it was trained on.
        row: Dict[str, float] = {}
        for column in self._feature_columns:
            if column in features and features[column] is not None:
                row[column] = float(features[column])
            elif column in self._feature_medians:
                row[column] = float(self._feature_medians[column])  # Req 6.2
            else:
                row[column] = _MISSING_FEATURE_DEFAULT

        X = pd.DataFrame([row], columns=self._feature_columns)

        raw_prediction = self._model.predict(X)
        predicted = float(raw_prediction[0])

        if not math.isfinite(predicted):
            raise ValueError(
                f"Model produced a non-finite AQI prediction: {predicted!r}"
            )

        aqi = max(0.0, predicted)  # Req 6.3: clamp to >= 0
        return {"aqi": aqi, "bucket": classify(aqi)}  # Req 7.3
