"""CPCB AQI category classification for the AQI prediction pipeline.

This module provides the :data:`CPCB_BUCKETS` ordering and a pure
:func:`classify` function that maps a numeric AQI value to its CPCB
(Central Pollution Control Board, India) category label.

The function is intentionally a side-effect-free pure function so it can be
shared by both the serving path (:class:`~ml_models.aqi.predictor.AQIPredictor`
attaches a bucket to every prediction) and by tests, without any model or I/O
dependencies.

CPCB boundaries (Req 7.1):
  - 0   through 50  -> Good
  - 51  through 100 -> Satisfactory
  - 101 through 200 -> Moderate
  - 201 through 300 -> Poor
  - 301 through 400 -> Very Poor
  - 401 and above   -> Severe

The boundaries are interpreted as inclusive upper bounds so the function is
total for any ``aqi >= 0`` and handles fractional predicted values (for
example ``212.4``). A value of exactly ``50`` is Good; anything strictly
greater than ``50`` (such as ``50.5`` or the integer ``51``) starts the
Satisfactory band, and so on. This yields exactly one bucket for any AQI value
greater than or equal to 0 (Req 7.2).

Requirements covered: 7.1, 7.2.
"""

from __future__ import annotations

# CPCB AQI_Bucket labels ordered from cleanest to most hazardous air quality.
CPCB_BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# Inclusive upper bound for each non-Severe bucket, ordered ascending. Anything
# above the final bound (400) falls into the open-ended "Severe" band (401+).
_UPPER_BOUNDS = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
)


def classify(aqi: float) -> str:
    """Return the CPCB AQI_Bucket for a numeric AQI value.

    The bucket boundaries are inclusive upper bounds: ``aqi <= 50`` is Good,
    ``aqi <= 100`` is Satisfactory, ``aqi <= 200`` is Moderate,
    ``aqi <= 300`` is Poor, ``aqi <= 400`` is Very Poor, and anything greater
    is Severe. This makes the function total for any ``aqi >= 0`` and returns
    exactly one of :data:`CPCB_BUCKETS`.

    Args:
        aqi: A numeric AQI value. The system guarantees ``aqi >= 0``; negative
            inputs are defensively classified as ``Good``.

    Returns:
        One of the six CPCB bucket labels in :data:`CPCB_BUCKETS`.
    """
    for upper, bucket in _UPPER_BOUNDS:
        if aqi <= upper:
            return bucket
    return "Severe"
