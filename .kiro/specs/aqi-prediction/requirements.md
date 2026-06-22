# Requirements Document

## Introduction

This document specifies a complete, **from-scratch Air Quality Index (AQI) Prediction system** for the ClimaSense Flask application, built as a final-year college project titled *"Air Quality Index Prediction Using Machine Learning and Weather Parameters."* This system is a fresh ground-up design; any previously existing AQI model assets are considered replaced and are not extended by this specification.

The system demonstrates the full data-science lifecycle end to end: data collection, data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning, model evaluation, and visualization on a web dashboard. It delivers four user-facing capabilities:

1. **AQI Prediction** — forecast a future AQI value (next-hour or next-day) from pollutant, weather, and temporal features using engineered lag and rolling-average inputs.
2. **AQI Category Classification** — assign the CPCB bucket: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.
3. **Pollution Trend Visualization** — render daily, weekly, and monthly AQI trends.
4. **Health Advisory** — present health guidance derived from the predicted AQI category.

For evaluation, the system trains and compares three regression models — **Linear Regression, Random Forest, and XGBoost** — reported with **RMSE** (plus MAE and R²) and a written conclusion identifying the best model. XGBoost is added as a pinned project dependency.

The system integrates with the existing single-page Jinja2 frontend. The AQI tab in `templates/index.html` already issues a `fetch('/api/air-quality/<city>')` request and expects a JSON response shaped as `{ success, city, data: { current: { time, pm25, category } }, plot_url }`; this system must implement that endpoint and contract on the Flask backend (`main.py`), using Chart.js for interactive charts and matplotlib (Agg backend) for server-rendered plots.

## Glossary

- **AQI**: Air Quality Index — a numeric scale (0–500) summarizing air pollution severity, computed per the Indian CPCB standard.
- **CPCB**: Central Pollution Control Board (India), whose AQI bucketing scheme is used by this system.
- **AQI_Bucket**: The CPCB category label assigned to a numeric AQI value: Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), Severe (401 and above).
- **Pollutant_Features**: The measured pollutant concentration inputs — PM2.5, PM10, NO2, SO2, CO, O3.
- **Weather_Features**: Temperature, Humidity, Wind Speed, and Pressure.
- **Temporal_Features**: Hour, Day, and Month derived from the observation timestamp.
- **Lag_Features**: Prior-period values used as inputs — Previous AQI (immediately preceding period) and AQI last-24-hour average.
- **Rolling_Average**: A moving average of AQI computed over a fixed trailing window.
- **Forecast_Horizon**: The configured prediction target offset — either next-hour or next-day.
- **Multi_Step_Forecaster**: The component that produces an iterative (recursive) AQI forecast for the next five consecutive hours by predicting hour t+1, then feeding each predicted value back as the Previous AQI Lag_Feature to predict the following hour, through hour t+5.
- **Hourly_Forecast_Series**: The ordered sequence of five forecasted points for a selected city, where each point contains its hour offset (t+1 through t+5), a predicted numeric AQI value, and the corresponding AQI_Bucket.
- **Forecast_Chart**: The Chart.js line chart on the AQI_Dashboard that renders the Hourly_Forecast_Series with hour labels, distinct from the daily, weekly, and monthly historical trend charts.
- **AQI_Predictor**: The machine-learning component that predicts a future AQI value for the Forecast_Horizon.
- **AQI_Classifier**: The component that maps a numeric AQI value to its CPCB AQI_Bucket.
- **Model_Evaluator**: The component that trains and compares the Linear Regression, Random Forest, and XGBoost regressors and reports comparison metrics.
- **Trend_Visualizer**: The component that produces daily, weekly, and monthly AQI trend data and charts.
- **Health_Advisor**: The component that returns health guidance text for a given AQI_Bucket.
- **AQI_API**: The Flask endpoint(s) that serve AQI prediction, classification, trend, and advisory data to the frontend.
- **AQI_Dashboard**: The frontend AQI tab in `templates/index.html` that displays predictions, categories, trends, and advisories.
- **AQI_Dashboard_Panel**: The single cohesive dashboard layout within the AQI_Dashboard that arranges, as a readable grid rather than a flat vertical list, the AQI_Summary_Card, the health advisory, the Forecast_Chart, and the daily, weekly, and monthly trend charts.
- **AQI_Summary_Card**: The prominent summary element of the AQI_Dashboard_Panel that displays the predicted AQI value together with its AQI_Bucket, visually color-coded by the CPCB_Color_Map.
- **CPCB_Color_Map**: The fixed mapping from each AQI_Bucket to a display color — Good to green (#009966), Satisfactory to light green (#84cf33), Moderate to yellow (#ffde33), Poor to orange (#ff9933), Very Poor to red (#cc0033), and Severe to maroon (#7e0023).
- **Training_Dataset**: The historical air-quality dataset used for training — OpenAQ/CPCB air-quality records (for example `city_day.csv`), optionally augmented with Open-Meteo weather data.
- **Data_Loader**: The component that collects and loads raw air-quality and weather data into memory.
- **Preprocessor**: The component that cleans, imputes, and standardizes raw records into a numeric feature matrix.
- **Feature_Engineer**: The component that derives Lag_Features, Rolling_Average, Temporal_Features, and the prediction target.
- **RMSE**: Root Mean Squared Error — the primary regression evaluation metric.
- **MAE**: Mean Absolute Error — a supporting regression metric.
- **R²**: Coefficient of determination — a supporting regression metric.
- **System**: The overall AQI Prediction system within the ClimaSense application.

## Requirements

### Requirement 1: Data Collection and Loading

**User Story:** As a data-science student, I want the system to collect and load historical air-quality and weather data, so that models can be trained on real measurements.

#### Acceptance Criteria

1. THE Data_Loader SHALL load historical air-quality records containing Pollutant_Features and AQI from the Training_Dataset.
2. THE Data_Loader SHALL associate a city name and an observation timestamp with each loaded record.
3. WHERE Open-Meteo weather data is available for a record's city and timestamp, THE Data_Loader SHALL associate Weather_Features with that record.
4. IF the Training_Dataset file is missing or cannot be read, THEN THE Data_Loader SHALL raise a descriptive error identifying the expected dataset path.
5. WHEN loading completes, THE Data_Loader SHALL record the number of loaded records and the list of covered cities as dataset metadata.

### Requirement 2: Data Preprocessing

**User Story:** As a data-science student, I want raw data cleaned and standardized, so that the models receive valid inputs.

#### Acceptance Criteria

1. WHEN a record is missing a value for a Pollutant_Feature or a Weather_Feature, THE Preprocessor SHALL impute that value using the training-set median for that feature.
2. WHEN a record is missing the AQI value, THE Preprocessor SHALL exclude that record from the training set.
3. THE Preprocessor SHALL parse each record's timestamp into Temporal_Features Hour, Day, and Month.
4. WHEN preprocessing completes, THE Preprocessor SHALL produce a feature matrix in which every cell contains a numeric value.
5. IF the dataset contains zero usable records after preprocessing, THEN THE Preprocessor SHALL raise a descriptive error.

### Requirement 3: Exploratory Data Analysis (EDA)

**User Story:** As a data-science student, I want summary statistics and distributions, so that I can understand and present the dataset.

#### Acceptance Criteria

1. THE System SHALL compute summary statistics consisting of count, mean, minimum, maximum, and standard deviation for AQI and each Pollutant_Feature.
2. THE System SHALL compute the correlation coefficient between each Pollutant_Feature and AQI.
3. THE System SHALL generate at least one distribution chart of AQI values across the Training_Dataset.
4. THE System SHALL persist each generated EDA chart as an image file in the static assets directory that the AQI_Dashboard can display.

### Requirement 4: Feature Engineering

**User Story:** As a data-science student, I want engineered lag and rolling-average features, so that the prediction models can learn temporal patterns.

#### Acceptance Criteria

1. THE Feature_Engineer SHALL compute a Previous AQI Lag_Feature equal to the AQI value of the immediately preceding period for each record within the same city, ordered by timestamp.
2. THE Feature_Engineer SHALL compute an AQI last-24-hour-average Rolling_Average over a trailing 24-hour window for each record within the same city, ordered by timestamp.
3. THE Feature_Engineer SHALL include Temporal_Features Hour, Day, and Month in the engineered feature set.
4. WHEN a record lacks sufficient prior history to compute a Lag_Feature or Rolling_Average, THE Feature_Engineer SHALL impute that engineered value using the training-set median for that feature.
5. THE Feature_Engineer SHALL define the prediction target as the AQI value shifted forward by one period according to the configured Forecast_Horizon.

### Requirement 5: Model Training and Comparison

**User Story:** As an evaluator, I want Linear Regression, Random Forest, and XGBoost compared on RMSE, so that I can see which model predicts AQI best.

#### Acceptance Criteria

1. THE Model_Evaluator SHALL train a Linear Regression model, a Random Forest model, and an XGBoost model on the engineered feature set with the prediction target.
2. THE Model_Evaluator SHALL evaluate each model on a held-out test split and compute RMSE, MAE, and R² for each model.
3. THE Model_Evaluator SHALL select the model with the lowest test RMSE as the best model.
4. THE Model_Evaluator SHALL record a comparison report containing each model's RMSE, MAE, and R², the identity of the best model, and a one-line written conclusion naming the best model and its RMSE.
5. WHEN training completes, THE Model_Evaluator SHALL persist the best model and its metadata to disk for reuse without retraining.
6. WHERE the best model is a tree-based model, THE Model_Evaluator SHALL record the relative importance of each input feature.

### Requirement 6: AQI Prediction

**User Story:** As an end user, I want a predicted future AQI for a city, so that I can plan ahead.

#### Acceptance Criteria

1. WHEN a prediction is requested with Pollutant_Features, Weather_Features, Lag_Features, and Temporal_Features, THE AQI_Predictor SHALL return a numeric predicted AQI value for the configured Forecast_Horizon.
2. WHEN an input feature is absent from a prediction request, THE AQI_Predictor SHALL substitute the training-set median for that feature before predicting.
3. THE AQI_Predictor SHALL constrain the returned AQI value to be greater than or equal to 0.
4. IF the persisted best model cannot be loaded, THEN THE AQI_API SHALL return an error response indicating the model is unavailable.

### Requirement 7: AQI Category Classification

**User Story:** As an end user, I want the AQI translated into a category, so that I can quickly understand air quality severity.

#### Acceptance Criteria

1. WHEN given a numeric AQI value, THE AQI_Classifier SHALL return the CPCB AQI_Bucket according to these boundaries: 0 through 50 Good, 51 through 100 Satisfactory, 101 through 200 Moderate, 201 through 300 Poor, 301 through 400 Very Poor, and 401 and above Severe.
2. THE AQI_Classifier SHALL return exactly one AQI_Bucket for any AQI value greater than or equal to 0.
3. WHEN the AQI_Predictor produces a predicted AQI value, THE System SHALL attach the corresponding AQI_Bucket to the prediction result.

### Requirement 8: Pollution Trend Visualization

**User Story:** As an end user, I want to see daily, weekly, and monthly pollution trends, so that I can understand how air quality changes over time.

#### Acceptance Criteria

1. THE Trend_Visualizer SHALL produce an average-AQI-per-day series for a selected city.
2. THE Trend_Visualizer SHALL produce an average-AQI-per-week series for a selected city.
3. THE Trend_Visualizer SHALL produce an average-AQI-per-month series for a selected city.
4. THE AQI_Dashboard SHALL render the daily, weekly, and monthly trend series as charts.
5. IF a selected city has no records in the Training_Dataset, THEN THE AQI_API SHALL return an error response indicating no data is available for that city.

### Requirement 9: Health Advisory

**User Story:** As an end user, I want health guidance based on the AQI, so that I know what precautions to take.

#### Acceptance Criteria

1. WHEN given an AQI_Bucket, THE Health_Advisor SHALL return a health advisory message corresponding to that bucket.
2. WHERE the AQI value is at or below 50, THE Health_Advisor SHALL return guidance indicating air quality is good and no precautions are needed.
3. WHERE the AQI value is above 300, THE Health_Advisor SHALL return guidance advising avoidance of outdoor activities.
4. THE System SHALL attach the health advisory message to each prediction result that includes an AQI_Bucket.

### Requirement 10: AQI API and Dashboard Integration

**User Story:** As an end user, I want the AQI tab to display predictions, categories, trends, and advisories, so that I can use the system through the web app.

#### Acceptance Criteria

1. WHEN the AQI_Dashboard requests AQI data for a city at `/api/air-quality/<city>`, THE AQI_API SHALL return a JSON response containing a success flag, the city name, the predicted AQI value, its AQI_Bucket, the corresponding health advisory, and a plot image URL.
2. WHEN the AQI_Dashboard requests trend data for a city, THE AQI_API SHALL return the daily, weekly, and monthly trend series for that city.
3. IF an AQI_API request omits a required city parameter, THEN THE AQI_API SHALL return an error response identifying the missing parameter.
4. WHEN the AQI_API returns a successful response, THE AQI_Dashboard SHALL display the predicted AQI value, AQI_Bucket, health advisory, and trend charts.
5. WHILE an AQI_API request is in progress, THE AQI_Dashboard SHALL display a loading indicator.

### Requirement 11: XGBoost Dependency

**User Story:** As a developer, I want XGBoost added to the project dependencies, so that the model comparison can run.

#### Acceptance Criteria

1. THE System SHALL declare XGBoost as a version-pinned dependency in the project requirements file.
2. IF XGBoost is not installed at training time, THEN THE Model_Evaluator SHALL raise a descriptive error instructing the developer to install the dependency.

### Requirement 12: Five-Hour Multi-Step AQI Forecast

**User Story:** As an end user, I want a chart of the predicted AQI for the next five hours in a selected city, so that I can anticipate short-term air-quality changes.

#### Acceptance Criteria

1. WHEN a five-hour forecast is requested for a city, THE Multi_Step_Forecaster SHALL produce a Hourly_Forecast_Series of exactly five forecasted points covering hour offsets t+1 through t+5.
2. WHEN forecasting hour offset t+1, THE Multi_Step_Forecaster SHALL use the most recent available AQI value for the city as the Previous AQI Lag_Feature.
3. WHEN forecasting each hour offset from t+2 through t+5, THE Multi_Step_Forecaster SHALL use the predicted AQI value of the immediately preceding hour offset as the Previous AQI Lag_Feature.
4. THE Multi_Step_Forecaster SHALL constrain each forecasted AQI value in the Hourly_Forecast_Series to be greater than or equal to 0.
5. WHEN a forecasted AQI value is produced for an hour offset, THE Multi_Step_Forecaster SHALL attach the corresponding AQI_Bucket to that point using the AQI_Classifier.
6. WHEN the AQI_Dashboard requests a five-hour forecast for a city at `/api/air-quality/<city>/forecast`, THE AQI_API SHALL return a JSON response containing a success flag, the city name, and the Hourly_Forecast_Series with each point's hour offset, predicted AQI value, and AQI_Bucket.
7. IF a selected city has no records in the Training_Dataset, THEN THE AQI_API SHALL return an error response indicating no data is available for that city.
8. IF the persisted best model cannot be loaded, THEN THE AQI_API SHALL return an error response indicating the model is unavailable.
9. WHEN the AQI_API returns a successful five-hour forecast response, THE AQI_Dashboard SHALL render the Hourly_Forecast_Series as a Forecast_Chart with one labeled point per forecasted hour.
10. WHILE a five-hour forecast request is in progress, THE AQI_Dashboard SHALL display a loading indicator.

### Requirement 13: AQI Dashboard Panel Layout

**User Story:** As an end user, I want the predicted AQI, its category, the health advisory, the five-hour forecast, and the historical trends presented together as a single readable dashboard, so that I can understand a city's air quality at a glance instead of scrolling through a flat list.

#### Acceptance Criteria

1. WHEN the AQI_API returns a successful response for a city, THE AQI_Dashboard SHALL present the AQI_Summary_Card, the health advisory, the Forecast_Chart, and the daily, weekly, and monthly trend charts within a single AQI_Dashboard_Panel.
2. THE AQI_Dashboard_Panel SHALL arrange the AQI_Summary_Card, the health advisory, the Forecast_Chart, and the trend charts as a grid layout that reflows to a single column at viewport widths at or below 768 pixels.
3. THE AQI_Summary_Card SHALL display the predicted AQI value and its AQI_Bucket together as the most prominent element of the AQI_Dashboard_Panel.
4. WHEN the AQI_Summary_Card displays an AQI_Bucket, THE AQI_Dashboard SHALL color-code the AQI_Summary_Card using the color assigned to that AQI_Bucket by the CPCB_Color_Map.
5. THE AQI_Dashboard SHALL render the health advisory text within the AQI_Dashboard_Panel adjacent to the AQI_Summary_Card.
6. THE AQI_Dashboard SHALL render the Forecast_Chart and the daily, weekly, and monthly trend charts as distinct charts within the AQI_Dashboard_Panel.
7. WHILE an AQI_API request for the AQI_Dashboard_Panel is in progress, THE AQI_Dashboard SHALL display the existing loading indicator.
8. THE AQI_Dashboard SHALL render the AQI_Dashboard_Panel within the existing AQI tab of `templates/index.html` without introducing a new route or template.
