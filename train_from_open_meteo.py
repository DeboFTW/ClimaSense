"""Train the AQI model using historical data pulled live from Open-Meteo.

Instead of relying on a static ``city_day.csv``, this script fetches recent
hourly air-quality history for a set of cities directly from the Open-Meteo
Air Quality API (keyless), assembles a training dataset, and runs the existing
offline pipeline (DataLoader -> Preprocessor -> EDAAnalyzer -> FeatureEngineer
-> ModelEvaluator) to produce and persist the model artifact.

Usage:
    python train_from_open_meteo.py [city1 city2 ...]

If no cities are given, a default spread of cities is used. The assembled data
is written to ``city_day.csv`` (so the trained model and the CSV fallback share
the same source) and the model artifact is saved to ``MODEL_ARTIFACT``.
"""

from __future__ import annotations

import sys

import pandas as pd

import train_aqi_model
from ml_models.aqi import DEFAULT_DATASET
from ml_models.aqi.live_source import LiveAirQualitySource, LiveDataError

# Open-Meteo air-quality history depth (the API supports up to ~92 past days).
_HISTORY_DAYS = 92

# A reasonable default spread of cities (mix of Indian + global) so the model
# sees a range of pollution regimes.
_DEFAULT_CITIES = [
    "Delhi",
    "Mumbai",
    "Kolkata",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "London",
    "Beijing",
    "Los Angeles",
    "Tokyo",
]


def collect_dataset(cities: list[str]) -> pd.DataFrame:
    """Fetch hourly air-quality history for each city and concatenate it."""
    source = LiveAirQualitySource(history_days=_HISTORY_DAYS)
    frames: list[pd.DataFrame] = []

    for city in cities:
        try:
            frame = source.fetch_city(city)
        except LiveDataError as exc:
            print(f"  ! skipping {city}: {exc}")
            continue
        print(f"  + {city}: {len(frame)} hourly records")
        frames.append(frame)

    if not frames:
        raise SystemExit(
            "No data could be fetched from Open-Meteo for any requested city."
        )

    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    cities = args if args else _DEFAULT_CITIES

    print(f"Collecting Open-Meteo air-quality history for {len(cities)} cities ...")
    dataset = collect_dataset(cities)

    # Persist so the trained model and the CSV fallback share the same source.
    dataset.to_csv(DEFAULT_DATASET, index=False)
    print(f"\nWrote {len(dataset)} records to {DEFAULT_DATASET}")

    # Reuse the existing offline pipeline end-to-end.
    print("\nRunning training pipeline ...")
    train_aqi_model.run_pipeline(DEFAULT_DATASET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
