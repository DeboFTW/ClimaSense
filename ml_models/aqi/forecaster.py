"""Iterative five-hour AQI forecaster for the AQI prediction pipeline.

The :class:`MultiStepForecaster` produces a recursive multi-step forecast by
composing an :class:`~ml_models.aqi.predictor.AQIPredictor`. It predicts hour
t+1 from the city's current feature row (with ``Prev_AQI = last_aqi``), then
feeds each predicted AQI back as the ``Prev_AQI`` lag feature to predict the
next hour, through t+5.

The forecaster owns no model of its own — it delegates every single-step
inference to the predictor so the recursion and the single-step inference stay
independently testable.

Design contract:
  - ``forecast(base_features, last_aqi) -> list[ForecastPoint]``: produce
    exactly :data:`~ml_models.aqi.FORECAST_STEPS` (5) forecast points with
    ``hour_offset`` 1..5 (Req 12.1).
  - Step 1 (t+1): predict from ``base_features`` with ``Prev_AQI = last_aqi``
    (Req 12.2).
  - Steps 2..5 (t+2..t+5): copy ``base_features``, overwrite ``Prev_AQI`` with
    the previous step's predicted AQI (Req 12.3).
  - Each predicted AQI is clamped >= 0 by the predictor (Req 12.4) and
    classified into a CPCB bucket via the predictor (Req 12.5).

Requirements covered: 12.1, 12.2, 12.3, 12.4, 12.5.
"""

from __future__ import annotations

from typing import Dict, List

from typing_extensions import TypedDict

from ml_models.aqi import FORECAST_STEPS
from ml_models.aqi.predictor import AQIPredictor

# The lag feature name that the model uses as the previous-period AQI input.
_PREV_AQI_FEATURE = "Prev_AQI"


class ForecastPoint(TypedDict):
    """A single point in a multi-step AQI forecast series.

    Attributes
    ----------
    hour_offset:
        Hours ahead of the current time this point represents (1..5).
    aqi:
        Forecasted AQI value, always >= 0 (clamped by the predictor).
    bucket:
        CPCB category for this AQI value, as returned by
        :func:`~ml_models.aqi.classifier.classify`.
    """

    hour_offset: int  # 1..5  (t+1 .. t+5)
    aqi: float  # forecasted AQI, >= 0
    bucket: str  # classify(aqi)


class MultiStepForecaster:
    """Produce an iterative (recursive) multi-step AQI forecast.

    The forecaster wraps a loaded :class:`~ml_models.aqi.predictor.AQIPredictor`
    and iterates it ``steps`` times, feeding each predicted AQI back as the
    ``Prev_AQI`` lag feature for the next step. Only ``Prev_AQI`` is rewritten
    between steps; the other pollutant/weather/temporal inputs are held at the
    city's current-state values in ``base_features``.

    This matches the model's strong persistence behaviour (Prev_AQI importance
    ≈ 0.98) and keeps the recursion the single moving part, making the
    forecaster easy to unit-test with a stub predictor.

    Parameters
    ----------
    predictor:
        A loaded :class:`~ml_models.aqi.predictor.AQIPredictor` instance.
        The caller is responsible for loading it before passing it here.
    steps:
        Number of forecast points to produce. Defaults to
        :data:`~ml_models.aqi.FORECAST_STEPS` (5).
    """

    def __init__(
        self,
        predictor: AQIPredictor,
        steps: int = FORECAST_STEPS,
    ) -> None:
        self._predictor = predictor
        self._steps = steps

    def forecast(
        self,
        base_features: Dict[str, float],
        last_aqi: float,
    ) -> List[ForecastPoint]:
        """Produce ``steps`` forecast points (t+1..t+steps).

        - Step 1 (t+1): predict from ``base_features`` with
          ``Prev_AQI = last_aqi`` (Req 12.2).
        - Step k (t+2..t+steps): copy ``base_features``, overwrite ``Prev_AQI``
          with the previous step's predicted AQI (Req 12.3), then predict.
        - Each predicted AQI is clamped >= 0 by the predictor (Req 12.4) and
          classified into a CPCB bucket via the predictor result (Req 12.5).

        Returns exactly ``steps`` :class:`ForecastPoint` dicts with
        ``hour_offset`` values 1..steps (Req 12.1).

        Parameters
        ----------
        base_features:
            Feature dict representing the city's current state. Must contain
            the pollutant/weather/temporal columns the model expects. Any absent
            feature is filled with its training median by the predictor.
        last_aqi:
            The city's most recent observed (or previously predicted) AQI value,
            used as ``Prev_AQI`` for the first forecast step.

        Returns
        -------
        list[ForecastPoint]
            Exactly ``steps`` forecast points with ``hour_offset`` 1..steps.
        """
        results: List[ForecastPoint] = []
        prev_aqi = float(last_aqi)

        for step in range(1, self._steps + 1):
            # Build the feature dict for this step: start from base_features,
            # then overwrite Prev_AQI with the appropriate lag value.
            # - Step 1: Prev_AQI = last_aqi  (Req 12.2)
            # - Step k: Prev_AQI = previous step's predicted AQI  (Req 12.3)
            features: Dict[str, float] = dict(base_features)
            features[_PREV_AQI_FEATURE] = prev_aqi

            # Delegate to the predictor; it clamps aqi >= 0 (Req 12.4) and
            # attaches the CPCB bucket via classify() (Req 12.5).
            prediction = self._predictor.predict(features)
            predicted_aqi = float(prediction["aqi"])  # already >= 0
            bucket: str = prediction["bucket"]

            results.append(
                ForecastPoint(
                    hour_offset=step,  # Req 12.1: 1..steps
                    aqi=predicted_aqi,
                    bucket=bucket,
                )
            )

            # Feed this step's predicted AQI back as Prev_AQI for the next step.
            prev_aqi = predicted_aqi

        return results  # exactly self._steps points  (Req 12.1)
