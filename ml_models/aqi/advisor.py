"""CPCB AQI health advisory messages for the AQI prediction pipeline.

This module maps each CPCB (Central Pollution Control Board, India) AQI
category bucket to a plain-language health advisory message. It is a pure,
side-effect-free module (no I/O) so it can be shared by the serving path and
by tests.

The :data:`ADVISORIES` mapping provides exactly one non-empty advisory for
each of the six CPCB buckets defined in
:data:`ml_models.aqi.classifier.CPCB_BUCKETS`:

  - ``Good``        (AQI <= 50)  -> air quality is good, no precautions needed.
  - ``Satisfactory``(51-100)     -> minor concern for sensitive individuals.
  - ``Moderate``    (101-200)    -> sensitive groups should limit exertion.
  - ``Poor``        (201-300)    -> reduce prolonged outdoor exertion.
  - ``Very Poor``   (301-400)    -> avoid outdoor activities.
  - ``Severe``      (401+)       -> avoid outdoor activities.

Requirements covered: 9.1, 9.2, 9.3.
"""

from __future__ import annotations

from ml_models.aqi.classifier import CPCB_BUCKETS

# One non-empty health advisory message per CPCB bucket (Req 9.1, Property 21).
ADVISORIES: dict[str, str] = {
    # Req 9.2: Good -> air quality is good, no precautions needed.
    "Good": (
        "Air quality is good. Air pollution poses little or no risk, so no "
        "precautions are needed. Enjoy your usual outdoor activities."
    ),
    "Satisfactory": (
        "Air quality is satisfactory. Air pollution poses little risk for most "
        "people, but unusually sensitive individuals should consider limiting "
        "prolonged outdoor exertion."
    ),
    "Moderate": (
        "Air quality is moderate. Sensitive groups such as children, older "
        "adults, and people with respiratory or heart conditions should reduce "
        "prolonged or heavy outdoor exertion."
    ),
    "Poor": (
        "Air quality is poor. Everyone should reduce prolonged outdoor "
        "exertion, and sensitive groups should avoid strenuous outdoor "
        "activity."
    ),
    # Req 9.3: Very Poor -> avoid outdoor activities.
    "Very Poor": (
        "Air quality is very poor. Avoid outdoor activities and stay indoors "
        "with windows closed where possible. Sensitive groups are at serious "
        "health risk."
    ),
    # Req 9.3: Severe -> avoid outdoor activities.
    "Severe": (
        "Air quality is severe and hazardous. Avoid all outdoor activities and "
        "remain indoors. Everyone may experience serious health effects."
    ),
}

# Fallback advisory for any unrecognized bucket. Kept non-empty and cautious so
# that callers always receive actionable guidance instead of an exception.
_DEFAULT_ADVISORY = (
    "Air quality information is unavailable. As a precaution, monitor local air "
    "quality updates and limit prolonged outdoor exertion if you feel unwell."
)

# Coverage guard: every CPCB bucket must have a specific, non-empty advisory.
assert all(
    bucket in ADVISORIES and ADVISORIES[bucket].strip() for bucket in CPCB_BUCKETS
), "ADVISORIES must define a non-empty message for every CPCB bucket"


def advise(bucket: str) -> str:
    """Return the health advisory message for a CPCB AQI bucket.

    Args:
        bucket: A CPCB AQI category label (one of
            :data:`~ml_models.aqi.classifier.CPCB_BUCKETS`).

    Returns:
        The advisory string for the bucket. For an unrecognized bucket a
        sensible non-empty default advisory is returned rather than raising.
    """
    return ADVISORIES.get(bucket, _DEFAULT_ADVISORY)
