"""
Precompute climate analytics aggregates from popular_cities_weather.csv.

The dataset holds monthly weather observations (2020-2025) for ~100 Indian
cities. Columns:
    date  month timestamp (YYYY-MM-01)
    tavg  average temperature (C)
    tmin  minimum temperature (C)
    tmax  maximum temperature (C)
    prcp  monthly precipitation (mm)
    wspd  wind speed (mostly empty)
    pres  sea-level pressure (hPa)
    tsun  sunshine duration (minutes)
    city  city name

This script aggregates the data down to a compact per-city JSON summary
(``static/climate_cache.json``) that the Flask ``/climate`` route serves.

Usage:
    python precompute_climate.py [path_to_csv]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = "popular_cities_weather.csv"
OUTPUT = Path("static/climate_cache.json")
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ALL_LABEL = "All Cities"


def _round(value, ndigits=1):
    """Round, returning None for NaN so JSON stays clean."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), ndigits)


def build_city_summary(df: pd.DataFrame) -> dict:
    """Build monthly climatology, yearly trend and headline metrics."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # ---- Monthly climatology (mean across all years) ----
    monthly = []
    for m in range(1, 13):
        sub = df[df["month"] == m]
        monthly.append({
            "month": MONTH_NAMES[m - 1],
            "temp": _round(sub["tavg"].mean()),
            "tmin": _round(sub["tmin"].mean()),
            "tmax": _round(sub["tmax"].mean()),
            "precip": _round(sub["prcp"].mean()),
        })

    # ---- Yearly trend ----
    yearly = []
    baseline = None
    for year in sorted(df["year"].dropna().unique()):
        sub = df[df["year"] == year]
        mean_temp = sub["tavg"].mean()
        # annual precipitation = sum of monthly means available that year
        annual_precip = sub["prcp"].sum(min_count=1)
        if np.isnan(mean_temp):
            continue
        if baseline is None:
            baseline = mean_temp
        yearly.append({
            "year": int(year),
            "temp": _round(mean_temp),
            "anomaly": _round(mean_temp - baseline, 2),
            "precip": _round(annual_precip, 0),
        })

    # ---- Headline metrics ----
    temps = [y["temp"] for y in yearly if y["temp"] is not None]
    valid_months = [m for m in monthly if m["temp"] is not None]
    wet_months = [m for m in monthly if m["precip"] is not None]

    hottest = max(valid_months, key=lambda m: m["temp"]) if valid_months else None
    coldest = min(valid_months, key=lambda m: m["temp"]) if valid_months else None
    wettest = max(wet_months, key=lambda m: m["precip"]) if wet_months else None

    warming = _round(temps[-1] - temps[0], 2) if len(temps) > 1 else 0.0
    year_vals = [y["year"] for y in yearly]

    metrics = {
        "mean_temp": _round(df["tavg"].mean()),
        "max_temp": _round(df["tmax"].max()),
        "min_temp": _round(df["tmin"].min()),
        "total_precip": _round(df["prcp"].sum(min_count=1), 0),
        "year_range": f"{year_vals[0]}-{year_vals[-1]}" if year_vals else "N/A",
        "records": int(df["tavg"].notna().sum()),
        "warming": warming,
        "hottest_month": hottest["month"] if hottest else "N/A",
        "coldest_month": coldest["month"] if coldest else "N/A",
        "wettest_month": wettest["month"] if wettest else "N/A",
    }

    return {"monthly": monthly, "yearly": yearly, "metrics": metrics}


def precompute(csv_path: str) -> dict:
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["city"])

    cities = sorted(df["city"].unique())
    print(f"Found {len(cities)} cities, {len(df):,} rows.")

    data = {}
    # Aggregate across every city ("All Cities" view)
    data[ALL_LABEL] = build_city_summary(df)

    for city in cities:
        data[city] = build_city_summary(df[df["city"] == city])

    return {
        "cities": [ALL_LABEL] + cities,
        "default_city": ALL_LABEL,
        "data": data,
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not Path(csv_path).exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    result = precompute(csv_path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Wrote climate aggregates to {OUTPUT}")
    print(f"   Cities: {len(result['cities'])}")
    sample = result["data"][ALL_LABEL]["metrics"]
    print(f"   All-cities mean temp: {sample['mean_temp']} C "
          f"({sample['year_range']})")


if __name__ == "__main__":
    main()
