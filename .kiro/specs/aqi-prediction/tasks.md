# Implementation Plan: AQI Prediction System

## Overview

This plan implements the AQI Prediction system from scratch in Python, replacing the existing `ml_models/aqi_model.py`. Work proceeds bottom-up through the data-science pipeline (loading → preprocessing → EDA → feature engineering → model training), then the serving components (classifier, predictor, advisor, trends), and finally wires everything into the Flask API and frontend dashboard. Each step builds on the previous one and ends integrated, with no orphaned code.

The implementation language is **Python 3.12** (matching the existing ClimaSense codebase). Property-based tests use **pytest + Hypothesis**; each property test references a property from the design's Correctness Properties section.

## Tasks

- [x] 1. Set up AQI package scaffolding, config, and dependencies
  - Create the `ml_models/aqi/` package with `__init__.py`
  - Define shared config constants: `POLLUTANT_FEATURES`, `WEATHER_FEATURES`, `TEMPORAL_FEATURES`, `LAG_FEATURES`, `TARGET_COLUMN`, `FORECAST_HORIZON`, `DEFAULT_DATASET`, `MODEL_ARTIFACT`, `EDA_DIR`
  - Add `xgboost==2.0.3` as a version-pinned dependency in `requirements.txt`
  - Add `pytest` and `hypothesis` as dev dependencies in `requirements.txt`
  - Create `tests/aqi/` directory with `__init__.py` and a `conftest.py` exposing a small fixture air-quality DataFrame/CSV
  - _Requirements: 11.1_

- [x] 2. Implement data collection and loading
  - [x] 2.1 Implement DataLoader
    - Create `ml_models/aqi/data_loader.py` with `DataLoader` class
    - Load pollutant + AQI records from the dataset; attach `City` and parsed `timestamp` to each record
    - Left-join Open-Meteo Weather_Features by (city, time) where available
    - Raise `FileNotFoundError` naming the expected path when the dataset is missing/unreadable
    - Populate `metadata` with `record_count` and unique `cities` after load
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property test for loaded record integrity
    - **Property 1: Loaded records always carry city and parseable timestamp**
    - **Validates: Requirements 1.2**

  - [ ]* 2.3 Write property test for dataset metadata consistency
    - **Property 2: Dataset metadata is consistent with loaded data**
    - **Validates: Requirements 1.5**

  - [ ]* 2.4 Write unit test for missing-dataset error path
    - Assert `FileNotFoundError` with the expected path is raised
    - _Requirements: 1.4_

- [x] 3. Implement data preprocessing
  - [x] 3.1 Implement Preprocessor
    - Create `ml_models/aqi/preprocessor.py` with `Preprocessor` class
    - Drop rows missing AQI; impute missing pollutant/weather cells with training-set median; capture `feature_medians`
    - Parse timestamp into `Hour`, `Day`, `Month`; return a fully-numeric, non-null feature matrix
    - Raise `ValueError` when zero usable records remain
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 Write property test for median imputation
    - **Property 3: Missing pollutant/weather cells are imputed to the training median**
    - **Validates: Requirements 2.1**

  - [ ]* 3.3 Write property test for AQI-missing exclusion
    - **Property 4: No record missing AQI survives preprocessing**
    - **Validates: Requirements 2.2**

  - [ ]* 3.4 Write property test for temporal feature parsing
    - **Property 5: Temporal features match the source timestamp**
    - **Validates: Requirements 2.3**

  - [ ]* 3.5 Write property test for numeric feature matrix
    - **Property 6: Preprocessed feature matrix is fully numeric and non-null**
    - **Validates: Requirements 2.4**

  - [ ]* 3.6 Write unit test for zero-usable-records error path
    - Assert `ValueError` is raised on an empty/all-missing frame
    - _Requirements: 2.5_

- [x] 4. Implement exploratory data analysis
  - [x] 4.1 Implement EDAAnalyzer
    - Create `ml_models/aqi/eda.py` with `summary_statistics`, `correlations`, and `render_distribution`
    - Compute count/mean/min/max/std for AQI and each pollutant; compute Pearson correlation of each pollutant with AQI
    - Render an AQI distribution histogram via matplotlib (Agg) and persist it under `static/aqi/`
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 4.2 Write property test for summary statistics
    - **Property 7: Summary statistics match a reference computation**
    - **Validates: Requirements 3.1**

  - [ ]* 4.3 Write property test for correlations
    - **Property 8: Correlations are valid and match reference**
    - **Validates: Requirements 3.2**

  - [ ]* 4.4 Write unit test for EDA chart persistence
    - Assert the distribution chart file is created in the static directory
    - _Requirements: 3.3, 3.4_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement feature engineering
  - [x] 6.1 Implement FeatureEngineer
    - Create `ml_models/aqi/feature_engineer.py` with `FeatureEngineer` class
    - Per city ordered by timestamp: compute `Prev_AQI` (preceding period), `AQI_24h_avg` (trailing 24h rolling mean), and `target` (AQI shifted forward by horizon)
    - Include `Hour`/`Day`/`Month`; impute insufficient-history lag/rolling cells with the training median; expose `feature_columns`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 6.2 Write property test for previous-AQI lag
    - **Property 9: Previous-AQI lag equals the preceding period per city**
    - **Validates: Requirements 4.1**

  - [ ]* 6.3 Write property test for 24-hour rolling average
    - **Property 10: 24-hour rolling average matches the reference window mean**
    - **Validates: Requirements 4.2**

  - [ ]* 6.4 Write property test for insufficient-history imputation
    - **Property 11: Insufficient-history lag/rolling cells are imputed to the median**
    - **Validates: Requirements 4.4**

  - [ ]* 6.5 Write property test for prediction target shift
    - **Property 12: Target equals AQI shifted forward by the horizon**
    - **Validates: Requirements 4.5**

- [x] 7. Implement AQI category classification
  - [x] 7.1 Implement AQIClassifier
    - Create `ml_models/aqi/classifier.py` with `CPCB_BUCKETS` and a pure `classify(aqi)` function
    - Map AQI to CPCB buckets per the boundaries (0-50 Good ... 401+ Severe)
    - _Requirements: 7.1, 7.2_

  - [ ]* 7.2 Write property test for CPCB classification
    - **Property 18: Classification respects CPCB boundaries and is total**
    - **Validates: Requirements 7.1, 7.2**

- [x] 8. Implement model training and comparison
  - [x] 8.1 Implement ModelEvaluator
    - Create `ml_models/aqi/model_evaluator.py` with `ModelEvaluator` class
    - Train Linear Regression, Random Forest, and XGBoost on a train/test split; compute RMSE/MAE/R² per model on the held-out split
    - Select the lowest-test-RMSE model as best; build a `ComparisonReport` with per-model metrics, best-model id, a one-line conclusion, and feature importance when best is tree-based
    - Guard the `xgboost` import and raise a descriptive `ImportError` with an install hint when missing
    - Implement `save()` persisting the best model + metadata (medians, feature columns, report) via joblib
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 11.2_

  - [ ]* 8.2 Write property test for evaluation metrics
    - **Property 13: Evaluation metrics are well-formed and match reference**
    - **Validates: Requirements 5.2**

  - [ ]* 8.3 Write property test for best-model selection
    - **Property 14: Best model is the one with lowest test RMSE**
    - **Validates: Requirements 5.3**

  - [ ]* 8.4 Write unit test for comparison report and persistence round-trip
    - Assert report contains all metrics, best id, and conclusion; assert save→load round-trips the model
    - _Requirements: 5.4, 5.5_

- [x] 9. Wire the offline training entry point
  - [x] 9.1 Rewrite train_aqi_model.py
    - Rewrite `train_aqi_model.py` to run the full pipeline: DataLoader → Preprocessor → EDAAnalyzer → FeatureEngineer → ModelEvaluator → save artifact + EDA charts
    - Print the comparison report and best-model conclusion; persist the artifact to `MODEL_ARTIFACT`
    - _Requirements: 5.1, 5.4, 5.5_

  - [ ]* 9.2 Write integration test for the training pipeline
    - Run the pipeline on the fixture dataset and assert an artifact and EDA chart are produced
    - _Requirements: 5.5_

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement prediction and health advisory
  - [x] 11.1 Implement AQIPredictor
    - Create `ml_models/aqi/predictor.py` with `AQIPredictor` class
    - Implement `load()` returning False when the artifact is absent
    - Implement `predict(features)`: substitute training median for absent features, predict, clamp to ≥ 0, attach bucket via `AQIClassifier`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.3_

  - [ ]* 11.2 Write property test for finite numeric prediction
    - **Property 15: Prediction returns a finite numeric AQI**
    - **Validates: Requirements 6.1**

  - [ ]* 11.3 Write property test for absent-feature median equivalence
    - **Property 16: Absent features are equivalent to supplying the median**
    - **Validates: Requirements 6.2**

  - [ ]* 11.4 Write property test for non-negative prediction
    - **Property 17: Predicted AQI is always non-negative**
    - **Validates: Requirements 6.3**

  - [ ]* 11.5 Write property test for bucket consistency on results
    - **Property 19: Prediction result carries the bucket consistent with the classifier**
    - **Validates: Requirements 7.3**

  - [x] 11.6 Implement HealthAdvisor
    - Create `ml_models/aqi/advisor.py` with `ADVISORIES` mapping and `advise(bucket)`
    - Return a message per bucket; ≤50 (Good) → good/no precautions; >300 (Very Poor/Severe) → avoid outdoor activities
    - _Requirements: 9.1, 9.2, 9.3_

  - [ ]* 11.7 Write property test for advisory coverage
    - **Property 21: Advisory exists for every bucket**
    - **Validates: Requirements 9.1**

  - [ ]* 11.8 Write unit test for boundary advisory content
    - Assert Good guidance for AQI ≤ 50 and avoid-outdoor guidance for AQI > 300
    - _Requirements: 9.2, 9.3_

- [x] 12. Implement trend visualization
  - [x] 12.1 Implement TrendVisualizer
    - Create `ml_models/aqi/trends.py` with `daily`, `weekly`, `monthly` series methods and `render_forecast_plot`
    - Compute average AQI per day/week/month for a selected city; render a matplotlib (Agg) forecast plot to `static/aqi/` and return its path
    - Signal when a city has no records so the API can return a no-data error
    - _Requirements: 8.1, 8.2, 8.3, 8.5_

  - [ ]* 12.2 Write property test for trend series means
    - **Property 20: Trend series equal the per-period mean of the source data**
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 13. Implement the AQI service orchestrator
  - [x] 13.1 Implement AQIService and cached accessor
    - Create `ml_models/aqi/service.py` with `AQIService` and `get_aqi_service()`
    - `get_air_quality(city)`: predict → classify → advise → render plot, assembling the response payload
    - `get_trends(city)`: return daily/weekly/monthly series
    - Surface model-unavailable and no-data conditions for the API layer
    - Export public components and `get_aqi_service()` from `ml_models/aqi/__init__.py`
    - _Requirements: 6.4, 8.5, 9.4, 10.1, 10.2_

  - [ ]* 13.2 Write property test for advisory attachment on results
    - **Property 22: Prediction result with a bucket always carries an advisory**
    - **Validates: Requirements 9.4**

- [x] 14. Wire Flask API routes
  - [x] 14.1 Add AQI API routes to main.py
    - Add `GET /api/air-quality/<city>` returning `{ success, city, predicted_aqi, bucket, advisory, data.current, plot_url }` via `get_aqi_service()`
    - Add `GET /api/air-quality/<city>/trends` returning daily/weekly/monthly series
    - Return error responses: missing city param (400, naming the parameter), model unavailable (503), no data for city (404)
    - Remove the legacy `ml_models/aqi_model.py` import path usage; ensure no references to the replaced model remain
    - _Requirements: 6.4, 8.5, 10.1, 10.2, 10.3_

  - [ ]* 14.2 Write integration tests for the API contract via Flask test client
    - Assert success payload shape for a known city; assert error responses for missing city param, unknown city, and unloaded model
    - _Requirements: 10.1, 10.2, 10.3_

- [x] 15. Wire the frontend dashboard
  - [x] 15.1 Update the AQI tab in templates/index.html
    - Render `bucket` and `advisory` from the air-quality response in the result card
    - Fetch the trends endpoint and render daily/weekly/monthly Chart.js line charts
    - Ensure the existing `#aq-loading` indicator shows while requests are in progress
    - _Requirements: 10.4, 10.5_

- [x] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Implement the five-hour multi-step forecaster
  - [x] 17.1 Add FORECAST_STEPS config and implement MultiStepForecaster
    - Add `FORECAST_STEPS = 5` to `ml_models/aqi/__init__.py` config constants and export it
    - Create `ml_models/aqi/forecaster.py` with a `MultiStepForecaster` class that composes an `AQIPredictor`
    - Implement `forecast(base_features, last_aqi)`: step 1 (t+1) predicts with `Prev_AQI = last_aqi`; steps t+2..t+5 copy `base_features` and overwrite `Prev_AQI` with the previous step's predicted AQI
    - Return exactly `FORECAST_STEPS` `ForecastPoint` dicts with `hour_offset` 1..5, each carrying `aqi` (≥ 0, via the predictor) and `bucket` (via the predictor/classifier)
    - Export `MultiStepForecaster` and `ForecastPoint` from `ml_models/aqi/__init__.py`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 17.2 Write property test for forecast series shape
    - **Property 23: Five-hour forecast always has exactly five points with offsets 1..5**
    - **Validates: Requirements 12.1**

  - [ ]* 17.3 Write property test for recursive Prev_AQI feedback
    - **Property 24: Forecast recursion feeds the prior prediction back as Prev_AQI**
    - Use a deterministic stub predictor that records the `Prev_AQI` it receives
    - **Validates: Requirements 12.2, 12.3**

  - [ ]* 17.4 Write property test for non-negative forecast values
    - **Property 25: Every forecasted AQI is non-negative**
    - **Validates: Requirements 12.4**

  - [ ]* 17.5 Write property test for forecast bucket consistency
    - **Property 26: Every forecast point's bucket is consistent with the classifier**
    - **Validates: Requirements 12.5**

- [x] 18. Add the forecast serving path and API route
  - [x] 18.1 Implement AQIService.get_forecast
    - Add `get_forecast(city)` to `ml_models/aqi/service.py`: reuse the existing per-city resolution to find the city's most recent record and derive `base_features` plus `last_aqi` from it
    - Raise `ModelUnavailableError` when the model is not loaded and `NoDataForCityError` when the city has no records
    - Delegate the recursion to `MultiStepForecaster.forecast` and assemble `{ success, city, forecast: [{hour_offset, aqi, bucket}, ...] }`
    - _Requirements: 12.1, 12.6, 12.7, 12.8_

  - [x] 18.2 Add the forecast Flask route to main.py
    - Add `GET /api/air-quality/<city>/forecast` returning the forecast payload via `get_aqi_service()`
    - Map error conditions to HTTP responses: model unavailable (503) and no data for city (404)
    - _Requirements: 12.6, 12.7, 12.8_

  - [ ]* 18.3 Write integration test for the forecast API contract
    - Assert the success payload shape (five points with `hour_offset`/`aqi`/`bucket`) for a known city; assert 404 for an unknown city and 503 for an unloaded model
    - _Requirements: 12.6, 12.7, 12.8_

- [x] 19. Wire the forecast chart into the dashboard
  - [x] 19.1 Add the Forecast_Chart to the AQI tab in templates/index.html
    - Fetch `GET /api/air-quality/<city>/forecast` when a city is selected
    - Render the five forecast points as a distinct Chart.js line chart with hour labels (`+1h`..`+5h`), separate from the daily/weekly/monthly trend charts
    - Reuse the existing `#aq-loading` indicator while the forecast request is in progress
    - _Requirements: 12.9, 12.10_

  - [x] 19.2 Build the AQI dashboard panel layout in the AQI tab
    - Reorganize the contents of the existing `#aq-results` container in `templates/index.html` into a single `.aqi-dashboard` grid panel containing the AQI_Summary_Card, an advisory card, and distinct chart cards for the Forecast_Chart and the daily/weekly/monthly trend charts
    - Preserve and relocate the existing element IDs (`#aq-city-name`, `#aq-time`, `#aq-current-pm25`, `#aq-current-category`, `#aq-predicted-value`, `#aq-bucket`, `#aq-advisory`, the trend canvases, and the forecast canvas) so the existing fetch/render JS populates them unchanged
    - Make the AQI_Summary_Card the prominent, full-width lead element showing the predicted AQI value and its AQI_Bucket, with the advisory card adjacent
    - Apply CPCB_Color_Map color-coding to the AQI_Summary_Card based on the response `bucket` via a JS bucket→color lookup
    - Add `.aqi-dashboard` grid rules to `static/css/panels.css` (two-column grid, summary card full-width, reflow to a single column at ≤768px); reuse the existing `#aq-loading` indicator while panel requests are in progress
    - Keep all work within the existing AQI tab — no new route or template
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8_

- [x] 20. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each implementation task references specific requirements for traceability.
- Property-based test tasks each reference a numbered property from the design's Correctness Properties section and the requirement clause it validates.
- Checkpoints ensure incremental validation as the pipeline is assembled.
- The legacy `ml_models/aqi_model.py` and the old `ml_models/aqi_model.joblib` are replaced by the new `ml_models/aqi/` package and artifact.
