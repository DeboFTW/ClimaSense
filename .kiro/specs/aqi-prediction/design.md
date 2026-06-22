# Design Document: AQI Prediction System

## Overview

This document describes the design for a from-scratch **Air Quality Index (AQI) Prediction system** integrated into the ClimaSense Flask application. The system replaces the existing `ml_models/aqi_model.py` `AQIPredictor` entirely and implements the full data-science lifecycle: data collection, preprocessing, EDA, feature engineering, model training/comparison (Linear Regression vs Random Forest vs XGBoost on RMSE), forecasting, CPCB category classification, trend visualization, health advisory, and Flask API + frontend dashboard integration.

The implementation language is **Python 3.12**, consistent with the existing codebase (Flask 2.2.3, pandas ≥ 2.0, numpy < 2.0, scikit-learn ≥ 1.3, matplotlib Agg backend, Chart.js via CDN). XGBoost is added as a new pinned dependency.

### Goals

- Train and persist a best-of-three regression model that predicts a future AQI value for a configurable forecast horizon.
- Serve predictions, CPCB categories, health advisories, and daily/weekly/monthly trends through the existing `/api/air-quality/<city>` contract and a new trends endpoint.
- Serve an iterative five-hour AQI forecast (t+1..t+5) for a city through a new `/api/air-quality/<city>/forecast` endpoint, rendered as a distinct Chart.js chart.
- Demonstrate every lifecycle stage with reusable, testable components.

### Non-Goals

- Live pollutant ingestion from third-party APIs at request time (the system serves predictions from a persisted model and the historical training dataset).
- Modifying the existing weather prediction (ARIMA/LSTM) or chatbot subsystems.
- Real-time model retraining on each request (training is an offline script).

## Architecture

The system follows a pipeline-of-components design. An offline **training pipeline** produces a persisted model artifact and EDA assets; an online **serving path** loads the artifact and answers API requests.

```
                         OFFLINE (train_aqi_model.py)
  Training_Dataset ──► DataLoader ──► Preprocessor ──► FeatureEngineer ──► ModelEvaluator
   (city_day.csv)          │              │                  │                  │
                           ▼              ▼                  ▼                  ▼
                       metadata      feature matrix     engineered set    aqi_model.joblib
                                          │                                 + metadata
                                          ▼                                 + comparison report
                                    EDAAnalyzer ──► static/aqi/*.png
                                    TrendVisualizer (precompute per city)

                         ONLINE (main.py Flask routes)
  Browser AQI tab ──► GET /api/air-quality/<city> ──► AQIService
                  ──► GET /api/air-quality/<city>/trends   │
                  ──► GET /api/air-quality/<city>/forecast  │
                                                           ├─► AQIPredictor.predict()  (loaded artifact)
                                                           ├─► MultiStepForecaster.forecast()  (recursive t+1..t+5)
                                                           ├─► AQIClassifier.classify()
                                                           ├─► HealthAdvisor.advise()
                                                           └─► TrendVisualizer.series() + matplotlib plot
```

### Module Layout

All AQI code lives under a dedicated package to keep it isolated from the weather models:

```
ml_models/aqi/
├── __init__.py            # exports public components + get_aqi_service()
├── data_loader.py         # DataLoader
├── preprocessor.py        # Preprocessor
├── eda.py                 # EDAAnalyzer
├── feature_engineer.py    # FeatureEngineer
├── model_evaluator.py     # ModelEvaluator (LR, RF, XGBoost)
├── predictor.py           # AQIPredictor (loads persisted artifact)
├── forecaster.py          # MultiStepForecaster (recursive 5-hour forecast)
├── classifier.py          # AQIClassifier (CPCB buckets)
├── trends.py              # TrendVisualizer
├── advisor.py             # HealthAdvisor
└── service.py             # AQIService (orchestrates serving path) + get_aqi_service()
train_aqi_model.py         # rewritten offline training entry point
```

Rationale: the existing `ml_models/aqi_model.py` mixes loading, training, and prediction in one class. Splitting into single-responsibility modules makes each lifecycle stage independently testable and maps cleanly to the requirements.

## Components and Interfaces

### Configuration

```python
# ml_models/aqi/__init__.py (or a small config object)
POLLUTANT_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
WEATHER_FEATURES   = ["Temperature", "Humidity", "WindSpeed", "Pressure"]
TEMPORAL_FEATURES  = ["Hour", "Day", "Month"]
LAG_FEATURES       = ["Prev_AQI", "AQI_24h_avg"]
TARGET_COLUMN      = "AQI"
FORECAST_HORIZON   = 1          # periods shifted forward (next-period)
FORECAST_STEPS     = 5          # hours produced by the multi-step forecaster (t+1..t+5)
DEFAULT_DATASET    = "city_day.csv"
MODEL_ARTIFACT     = "ml_models/aqi/aqi_model.joblib"
EDA_DIR            = "static/aqi"
```

### DataLoader (`data_loader.py`)

Loads raw air-quality records and optionally joins Open-Meteo weather data.

```python
class DatasetMetadata(TypedDict):
    record_count: int
    cities: list[str]

class DataLoader:
    def __init__(self, dataset_path: str = DEFAULT_DATASET): ...

    def load(self) -> pd.DataFrame:
        """Load records with Pollutant_Features + AQI + City + timestamp.
        Raises FileNotFoundError with the expected path if dataset missing/unreadable.
        Attaches Weather_Features where Open-Meteo data is available for (city, time).
        Sets self.metadata after load."""

    @property
    def metadata(self) -> DatasetMetadata: ...
```

- _Req 1.1_: load Pollutant_Features + AQI.
- _Req 1.2_: every row carries `City` and a parsed `timestamp`.
- _Req 1.3_: join Weather_Features when available (left-join on city+time; absent → NaN, later imputed).
- _Req 1.4_: missing/unreadable file → `FileNotFoundError(f"AQI dataset not found at: {path}")`.
- _Req 1.5_: `metadata` records `record_count` and unique `cities`.

### Preprocessor (`preprocessor.py`)

Cleans, imputes, and standardizes raw records into a numeric feature matrix.

```python
class Preprocessor:
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows missing AQI; impute missing pollutant/weather cells with
        training-set median; parse timestamp into Hour/Day/Month; return a
        fully-numeric frame. Stores self.feature_medians.
        Raises ValueError if zero usable records remain."""

    @property
    def feature_medians(self) -> dict[str, float]: ...
```

- _Req 2.1_: median imputation for pollutant/weather columns; medians captured in `feature_medians`.
- _Req 2.2_: rows with missing AQI dropped before imputation.
- _Req 2.3_: `Hour`, `Day`, `Month` derived from timestamp.
- _Req 2.4_: output frame contains only numeric, non-null cells.
- _Req 2.5_: empty result → `ValueError("No usable records after preprocessing")`.

### EDAAnalyzer (`eda.py`)

Computes summary statistics, correlations, and renders distribution charts.

```python
class EDAAnalyzer:
    def summary_statistics(self, df) -> dict[str, dict]:
        """count, mean, min, max, std for AQI and each Pollutant_Feature."""

    def correlations(self, df) -> dict[str, float]:
        """Pearson correlation of each Pollutant_Feature with AQI."""

    def render_distribution(self, df, out_dir=EDA_DIR) -> str:
        """Render AQI distribution histogram to a PNG under static dir; return path."""
```

- _Req 3.1_: stats dict per metric. _Req 3.2_: correlation per pollutant.
- _Req 3.3 / 3.4_: matplotlib (Agg) histogram persisted to `static/aqi/`.

### FeatureEngineer (`feature_engineer.py`)

Derives lag, rolling-average, temporal features and the prediction target.

```python
class FeatureEngineer:
    def __init__(self, horizon: int = FORECAST_HORIZON): ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per city, ordered by timestamp:
          - Prev_AQI    = AQI.shift(1)
          - AQI_24h_avg = AQI.rolling('24h').mean()
          - target      = AQI.shift(-horizon)
        Include Hour/Day/Month. Impute insufficient-history lag/rolling cells
        with training-set median. Rows with no valid target are dropped for training."""

    @property
    def feature_columns(self) -> list[str]: ...
```

- _Req 4.1_: `Prev_AQI` per-city preceding period. _Req 4.2_: 24h trailing rolling mean per city.
- _Req 4.3_: temporal features included. _Req 4.4_: median imputation for missing history.
- _Req 4.5_: target = AQI shifted forward by horizon.

### ModelEvaluator (`model_evaluator.py`)

Trains and compares the three regressors, selects best by RMSE, persists artifact.

```python
class ComparisonReport(TypedDict):
    models: dict[str, dict]   # name -> {rmse, mae, r2}
    best_model: str
    conclusion: str           # "<best> performed best with RMSE <value>"
    feature_importance: dict[str, float] | None

class ModelEvaluator:
    def train_and_compare(self, X, y) -> ComparisonReport:
        """Train LinearRegression, RandomForestRegressor, XGBRegressor on a
        train/test split; compute RMSE/MAE/R2 per model; pick lowest test RMSE.
        Records feature importance when best is tree-based.
        Raises ImportError with install hint if xgboost missing."""

    def save(self, predictor_bundle, path=MODEL_ARTIFACT) -> str: ...
```

- _Req 5.1_: three models trained. _Req 5.2_: RMSE/MAE/R² on held-out split.
- _Req 5.3_: best = lowest test RMSE. _Req 5.4_: report with metrics, best id, one-line conclusion.
- _Req 5.5_: persist best model + metadata (medians, feature columns, report) via joblib.
- _Req 5.6_: feature importance for tree-based winner.
- _Req 11.2_: `xgboost` import guarded → `ImportError("XGBoost is required... pip install xgboost==<pin>")`.

### AQIClassifier (`classifier.py`)

Pure function mapping numeric AQI to CPCB bucket.

```python
CPCB_BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

def classify(aqi: float) -> str:
    """0-50 Good, 51-100 Satisfactory, 101-200 Moderate,
       201-300 Poor, 301-400 Very Poor, 401+ Severe."""
```

- _Req 7.1_: exact CPCB boundaries. _Req 7.2_: exactly one bucket for any aqi ≥ 0.

### AQIPredictor (`predictor.py`)

Loads the persisted artifact and predicts a future AQI value.

```python
class AQIPredictor:
    def load(self, path=MODEL_ARTIFACT) -> bool:
        """Load model + medians + feature columns. False if artifact absent."""

    def predict(self, features: dict[str, float]) -> dict:
        """Substitute training median for any absent feature; predict; clamp >= 0.
        Returns {'aqi': float, 'bucket': str} (bucket via AQIClassifier)."""
```

- _Req 6.1_: numeric prediction for horizon. _Req 6.2_: median substitution for absent features.
- _Req 6.3_: output clamped to ≥ 0. _Req 6.4_: load failure surfaced so API returns model-unavailable error.
- _Req 7.3_: attaches bucket to result.

### MultiStepForecaster (`forecaster.py`)

Produces an iterative (recursive) five-hour AQI forecast by reusing the loaded
`AQIPredictor`. It predicts hour t+1 from the city's current feature row, then
feeds each predicted AQI back as the `Prev_AQI` lag feature to predict the next
hour, through t+5. It owns no model of its own — it composes the predictor so
the recursion and the single-step inference stay independently testable.

```python
class ForecastPoint(TypedDict):
    hour_offset: int    # 1..5  (t+1 .. t+5)
    aqi: float          # forecasted AQI, >= 0
    bucket: str         # classify(aqi)

class MultiStepForecaster:
    def __init__(self, predictor: AQIPredictor, steps: int = FORECAST_STEPS): ...

    def forecast(self, base_features: dict[str, float], last_aqi: float) -> list[ForecastPoint]:
        """Produce `steps` forecast points (t+1..t+5).
          - Step 1 (t+1): predict from base_features with Prev_AQI = last_aqi (Req 12.2).
          - Step k (t+2..t+5): copy base_features, overwrite Prev_AQI with the
            previous step's predicted AQI (Req 12.3), then predict.
          - Each predicted AQI is clamped >= 0 by the predictor (Req 12.4) and
            classified into a CPCB bucket (Req 12.5).
        Returns exactly `steps` points with hour_offset 1..steps (Req 12.1)."""
```

- _Req 12.1_: returns exactly `FORECAST_STEPS` (5) points with `hour_offset` 1..5.
- _Req 12.2_: step 1 uses the city's most recent available AQI as `Prev_AQI`.
- _Req 12.3_: steps 2..5 use the immediately preceding predicted AQI as `Prev_AQI` (recursive feedback).
- _Req 12.4_: each forecasted AQI is ≥ 0 (delegated to `AQIPredictor.predict`).
- _Req 12.5_: each point's `bucket` is `classify(aqi)` (delegated to the predictor result).

Rationale: only the `Prev_AQI` lag feature is rewritten between steps; the
other pollutant/weather/temporal inputs are held at the city's current-state
values. This matches the model's strong persistence behaviour (Prev_AQI
importance ≈ 0.98) and keeps the recursion the single moving part, isolating
exactly the behaviour Requirement 12 specifies.

### HealthAdvisor (`advisor.py`)

```python
ADVISORIES = { "Good": "...", "Satisfactory": "...", ... }   # one per bucket

def advise(bucket: str) -> str:
    """Return health advisory message for the bucket."""
```

- _Req 9.1_: message per bucket. _Req 9.2_: ≤50 (Good) → "good, no precautions".
- _Req 9.3_: >300 (Very Poor/Severe) → "avoid outdoor activities".

### TrendVisualizer (`trends.py`)

```python
class TrendSeries(TypedDict):
    labels: list[str]
    values: list[float]

class TrendVisualizer:
    def daily(self, df, city) -> TrendSeries:    # avg AQI per day
    def weekly(self, df, city) -> TrendSeries:   # avg AQI per ISO week
    def monthly(self, df, city) -> TrendSeries:  # avg AQI per month
    def render_forecast_plot(self, city, series, out_dir=EDA_DIR) -> str:
        """matplotlib Agg plot saved as PNG; returns path for plot_url."""
```

- _Req 8.1/8.2/8.3_: daily/weekly/monthly average series per city.
- _Req 8.5_: city absent from dataset surfaced so API returns no-data error.

### AQIService (`service.py`) + Flask routes (`main.py`)

`AQIService` orchestrates the serving path; `main.py` exposes the routes.

```python
class AQIService:
    def __init__(self): self.predictor = AQIPredictor(); self.predictor.load()
    def get_air_quality(self, city: str) -> dict   # predict + classify + advise + plot
    def get_trends(self, city: str) -> dict         # daily/weekly/monthly series
    def get_forecast(self, city: str) -> dict       # 5-hour recursive forecast series

def get_aqi_service() -> AQIService:                 # cached singleton
```

Routes in `main.py`:

```python
@app.route('/api/air-quality/<city>')           # Req 10.1
@app.route('/api/air-quality/<city>/trends')    # Req 10.2
@app.route('/api/air-quality/<city>/forecast')  # Req 12.6
```

`get_forecast` reuses the same per-city resolution as `get_air_quality`: it
raises `ModelUnavailableError` when the model is not loaded (Req 12.8) and
`NoDataForCityError` when the city has no records (Req 12.7). It builds the
base feature dict from the city's most recent record, derives `last_aqi` from
that record's AQI, and delegates the recursion to `MultiStepForecaster.forecast`
(Req 12.1–12.5).

Forecast response contract:

```json
{
  "success": true,
  "city": "Delhi",
  "forecast": [
    {"hour_offset": 1, "aqi": 212.4, "bucket": "Poor"},
    {"hour_offset": 2, "aqi": 210.1, "bucket": "Poor"},
    {"hour_offset": 3, "aqi": 208.7, "bucket": "Poor"},
    {"hour_offset": 4, "aqi": 207.0, "bucket": "Poor"},
    {"hour_offset": 5, "aqi": 205.6, "bucket": "Poor"}
  ]
}
```

Response contract (preserves the existing frontend expectation and extends it):

```json
{
  "success": true,
  "city": "Delhi",
  "predicted_aqi": 212.4,
  "bucket": "Poor",
  "advisory": "Reduce prolonged outdoor exertion...",
  "data": { "current": { "time": "2024-01-01T10:00:00", "pm25": 88.2, "category": "Poor" } },
  "plot_url": "/static/aqi/Delhi_forecast.png"
}
```

- _Req 10.1_: success flag, city, predicted AQI, bucket, advisory, plot URL.
- _Req 10.2_: trends endpoint returns daily/weekly/monthly series.
- _Req 10.3_: missing city param → error identifying the parameter.
- _Req 6.4_: predictor not loaded → `{success: false, error: "model unavailable"}` (HTTP 503).
- _Req 8.5_: unknown city → `{success: false, error: "no data for city"}` (HTTP 404).
- _Req 12.6_: forecast endpoint returns success, city, and the 5-point series (hour_offset, aqi, bucket).
- _Req 12.7_: unknown city on forecast → no-data error (HTTP 404).
- _Req 12.8_: predictor not loaded on forecast → model-unavailable error (HTTP 503).

### Frontend (`templates/index.html`)

The existing AQI tab handler already consumes `success`, `city`, `data.current.{time,pm25,category}`, and `plot_url`. The design adds rendering of `bucket`, `advisory`, and three Chart.js trend charts (daily/weekly/monthly) fed by the trends endpoint. It also adds a distinct Chart.js line chart — the **Forecast_Chart** — fed by the `/forecast` endpoint, rendering the five forecast points with hour labels (`+1h`..`+5h`), separate from the historical trend charts.

- _Req 10.4_: display predicted AQI, bucket, advisory, trend charts.
- _Req 10.5_: existing `#aq-loading` indicator shown while requests are in progress.
- _Req 12.9_: render the five forecast points as a labeled Chart.js line chart distinct from trend charts.
- _Req 12.10_: reuse the existing loading indicator while the forecast request is in progress.

#### AQI Dashboard Panel Layout

The existing `#aq-results` container currently stacks four cards vertically (summary, forecast image, forecast chart placeholder, trends). The design reorganizes the contents of `#aq-results` into a single **AQI_Dashboard_Panel** — a CSS grid that presents the summary, advisory, forecast, and trend charts as a cohesive dashboard rather than a flat list. This is a pure markup/CSS/JS reorganization within the existing AQI tab; no new route, template, or endpoint is introduced, and all four existing JSON responses (air-quality, trends, forecast) compose into the one panel.

**DOM structure** (inside the existing `#aq-results`, replacing the flat card stack):

```
#aq-results
└── .aqi-dashboard                      (AQI_Dashboard_Panel — CSS grid)
    ├── .aqi-summary-card #aq-summary    (AQI_Summary_Card — color-coded by bucket)
    │     ├── #aq-city-name, #aq-time
    │     ├── #aq-predicted-value + #aq-bucket   (predicted AQI + CPCB bucket, prominent)
    │     └── #aq-current-pm25 + #aq-current-category  (current PM2.5 context)
    ├── .aqi-advisory-card                (health advisory, adjacent to summary)
    │     └── #aq-advisory
    ├── .aqi-chart-card                   (Forecast_Chart)
    │     └── <canvas id="aq-forecast-chart">
    ├── .aqi-chart-card                   (daily trend)
    │     └── <canvas id="aq-daily-chart">
    ├── .aqi-chart-card                   (weekly trend)
    │     └── <canvas id="aq-weekly-chart">
    └── .aqi-chart-card                   (monthly trend)
          └── <canvas id="aq-monthly-chart">
```

The existing element IDs (`#aq-city-name`, `#aq-time`, `#aq-current-pm25`, `#aq-current-category`, `#aq-predicted-value`, `#aq-bucket`, `#aq-advisory`, the trend canvases, and the forecast canvas added in Task 19) are preserved and simply relocated into the grid, so the existing fetch/render JS continues to populate them unchanged. The legacy server-rendered `#aq-plot-img` may remain inside a chart card or be retained as-is; the dashboard layout does not depend on it.

**Grid layout (CSS):** A new `.aqi-dashboard` rule defines a responsive grid:

```css
#panel-aqi .aqi-dashboard {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 16px;
}
/* Summary card spans the full width and leads the panel */
#panel-aqi .aqi-dashboard .aqi-summary-card { grid-column: 1 / -1; }
@media (max-width: 768px) {
  #panel-aqi .aqi-dashboard { grid-template-columns: 1fr; }  /* reflow to single column */
}
```

These rules are added to **`static/css/panels.css`**, which already owns all `#panel-aqi .aqi-*` styling (summary cards, search form, loading/error states). This follows the project convention of extending the existing per-feature stylesheet rather than creating a new one; `static/css/dashboard.css` is reserved for the weather dashboard and is not reused here to avoid cross-feature coupling.

**CPCB color-coding (CPCB_Color_Map):** The `AQI_Summary_Card` is color-coded by the predicted `bucket`. A JS lookup maps each bucket to its CPCB color and applies it (e.g. as a left border / header accent) when the response is rendered:

```javascript
const CPCB_COLORS = {
  'Good': '#009966', 'Satisfactory': '#84cf33', 'Moderate': '#ffde33',
  'Poor': '#ff9933', 'Very Poor': '#cc0033', 'Severe': '#7e0023'
};
// on successful response: summaryCard.style.borderColor = CPCB_COLORS[response.bucket] || '';
```

A small set of `.aqi-bucket-*` helper classes in `panels.css` (or inline style assignment via the map above) provides the per-bucket accent. The map mirrors the CPCB_Color_Map defined in the glossary.

**Composition of the three responses into one panel:** rendering stays driven by the existing handler flow — the air-quality response populates the summary card and advisory and color-codes the card; `loadTrends(city)` populates the three trend chart cards; the forecast fetch (Task 19) populates the forecast chart card. The existing `#aq-loading` indicator remains shown until all in-flight requests for the panel settle, satisfying the single shared loading indicator requirement.

- _Req 13.1_: summary card, advisory, forecast chart, and daily/weekly/monthly trend charts all rendered within the single `.aqi-dashboard` panel.
- _Req 13.2_: two-column CSS grid that reflows to one column at ≤768px.
- _Req 13.3_: predicted AQI + bucket are the prominent, full-width lead element of the panel.
- _Req 13.4_: summary card color-coded per the CPCB_Color_Map by the response bucket.
- _Req 13.5_: advisory rendered in a card adjacent to the summary card.
- _Req 13.6_: forecast and daily/weekly/monthly trends rendered as distinct chart cards.
- _Req 13.7_: existing `#aq-loading` indicator reused while panel requests are in progress.
- _Req 13.8_: panel lives in the existing AQI tab of `index.html`; no new route or template.

### Dependencies (`requirements.txt`)

- _Req 11.1_: add `xgboost==2.0.3` (version-pinned, compatible with numpy < 2.0 and scikit-learn ≥ 1.3).

## Data Models

**Loaded record** (post-DataLoader): `City: str`, `timestamp: datetime`, pollutant columns (float), optional weather columns (float).

**Feature matrix** (post-Preprocessor): all columns numeric float; columns = POLLUTANT_FEATURES + WEATHER_FEATURES + TEMPORAL_FEATURES + `AQI`.

**Engineered set** (post-FeatureEngineer): feature matrix + `Prev_AQI`, `AQI_24h_avg` + `target`.

**Persisted artifact** (joblib bundle): `{ model, feature_columns, feature_medians, metadata (ComparisonReport + dataset metadata) }`.

## Error Handling

| Condition | Component | Behavior | Requirement |
|---|---|---|---|
| Dataset file missing/unreadable | DataLoader | `FileNotFoundError` with expected path | 1.4 |
| Zero usable records after preprocessing | Preprocessor | `ValueError` descriptive | 2.5 |
| xgboost not installed | ModelEvaluator | `ImportError` with install hint | 11.2 |
| Persisted model cannot load | AQIService/API | JSON error, model unavailable (503) | 6.4, 12.8 |
| City has no records | API | JSON error, no data (404) | 8.5, 12.7 |
| Missing city parameter | API | JSON error naming parameter (400) | 10.3 |

## Testing Strategy

A test framework is not yet configured. The design uses **pytest** plus **Hypothesis** (property-based testing) since the requirements contain many universal invariants (boundary classification, median imputation, lag/rolling correctness, metric formulas). Hypothesis is added as a dev dependency. Tests live under `tests/aqi/`.

- **Unit/example tests**: data loading on a fixture CSV, EDA chart generation, model persistence round-trip, API contract via Flask test client, error paths.
- **Property-based tests**: the correctness properties below.

## Correctness Properties

These universal properties are derived from the testable acceptance criteria identified in prework and MUST be implemented as property-based tests.

**Property 1: Loaded records always carry city and parseable timestamp**
For every record returned by `DataLoader.load()`, `City` is non-null and `timestamp` is a valid datetime.
_Validates: Requirements 1.2_

**Property 2: Dataset metadata is consistent with loaded data**
After `load()`, `metadata.record_count == len(df)` and `set(metadata.cities) == set(df["City"].unique())`.
_Validates: Requirements 1.5_

**Property 3: Missing pollutant/weather cells are imputed to the training median**
For any input frame, every originally-missing pollutant/weather cell equals that column's training median in the output, and no such cell remains null.
_Validates: Requirements 2.1_

**Property 4: No record missing AQI survives preprocessing**
For any input frame, the output contains no row whose source AQI was null.
_Validates: Requirements 2.2_

**Property 5: Temporal features match the source timestamp**
For any valid timestamp, derived `Hour`/`Day`/`Month` equal that timestamp's hour/day/month.
_Validates: Requirements 2.3_

**Property 6: Preprocessed feature matrix is fully numeric and non-null**
For any input frame yielding ≥1 usable record, every cell of the output is a finite number.
_Validates: Requirements 2.4_

**Property 7: Summary statistics match a reference computation**
For arbitrary numeric arrays, EDA count/mean/min/max/std equal the numpy reference values (within float tolerance).
_Validates: Requirements 3.1_

**Property 8: Correlations are valid and match reference**
For arbitrary paired numeric arrays, each correlation is within [-1, 1] and equals the Pearson reference (within tolerance).
_Validates: Requirements 3.2_

**Property 9: Previous-AQI lag equals the preceding period per city**
For any per-city series ordered by timestamp, `Prev_AQI[i] == AQI[i-1]` for all i with prior history.
_Validates: Requirements 4.1_

**Property 10: 24-hour rolling average matches the reference window mean**
For any per-city ordered series, `AQI_24h_avg` equals the trailing-24h mean computed by reference.
_Validates: Requirements 4.2_

**Property 11: Insufficient-history lag/rolling cells are imputed to the median**
For the first record of each city (no prior history), the lag/rolling cells equal the training median.
_Validates: Requirements 4.4_

**Property 12: Target equals AQI shifted forward by the horizon**
For any per-city ordered series, `target[i] == AQI[i + horizon]` for all valid i.
_Validates: Requirements 4.5_

**Property 13: Evaluation metrics are well-formed and match reference**
For arbitrary prediction/actual arrays, computed RMSE/MAE equal reference formulas and RMSE ≥ 0, MAE ≥ 0, R² ≤ 1.
_Validates: Requirements 5.2_

**Property 14: Best model is the one with lowest test RMSE**
For any mapping of model→metrics, the selected best model has the minimum RMSE among candidates.
_Validates: Requirements 5.3_

**Property 15: Prediction returns a finite numeric AQI**
For any valid feature dict, `predict()` returns a finite float for `aqi`.
_Validates: Requirements 6.1_

**Property 16: Absent features are equivalent to supplying the median**
For any feature dict, predicting with a key omitted equals predicting with that key explicitly set to its training median.
_Validates: Requirements 6.2_

**Property 17: Predicted AQI is always non-negative**
For any input, `predict()['aqi'] >= 0`.
_Validates: Requirements 6.3_

**Property 18: Classification respects CPCB boundaries and is total**
For any aqi ≥ 0, `classify(aqi)` returns exactly one of the six CPCB buckets according to the boundaries, including boundary values 50/51/100/101/200/201/300/301/400/401.
_Validates: Requirements 7.1, 7.2_

**Property 19: Prediction result carries the bucket consistent with the classifier**
For any prediction, `result['bucket'] == classify(result['aqi'])`.
_Validates: Requirements 7.3_

**Property 20: Trend series equal the per-period mean of the source data**
For arbitrary per-city dated AQI data, each daily/weekly/monthly point equals the mean AQI of the records in that period (oracle comparison).
_Validates: Requirements 8.1, 8.2, 8.3_

**Property 21: Advisory exists for every bucket**
For each of the six CPCB buckets, `advise(bucket)` returns a non-empty message.
_Validates: Requirements 9.1_

**Property 22: Prediction result with a bucket always carries an advisory**
For any prediction result that includes a bucket, an advisory string is attached and corresponds to that bucket.
_Validates: Requirements 9.4_

**Property 23: Five-hour forecast always has exactly five points with offsets 1..5**
For any base feature dict and last_aqi, `MultiStepForecaster.forecast()` returns exactly `FORECAST_STEPS` (5) points whose `hour_offset` values are 1, 2, 3, 4, 5 in order.
_Validates: Requirements 12.1_

**Property 24: Forecast recursion feeds the prior prediction back as Prev_AQI**
Using a deterministic stub predictor that records its received `Prev_AQI`, for any base features and last_aqi the `Prev_AQI` seen at step k equals the forecasted AQI produced at step k-1, and the `Prev_AQI` at step 1 equals `last_aqi`.
_Validates: Requirements 12.2, 12.3_

**Property 25: Every forecasted AQI is non-negative**
For any base features and last_aqi, every point in the returned series has `aqi >= 0`.
_Validates: Requirements 12.4_

**Property 26: Every forecast point's bucket is consistent with the classifier**
For any returned series, each point satisfies `point["bucket"] == classify(point["aqi"])`.
_Validates: Requirements 12.5_
