"""
AQI Prediction Model for ClimaSense
=====================================
Predicts the Air Quality Index (AQI) for Indian cities from pollutant
concentrations using supervised regression.

Data: city_day.csv  (Delhi, Mumbai, Chennai, Kolkata, Bangalore | 2015-2024)
Features: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
Target:   AQI  (numeric)  ->  also mapped to an AQI_Bucket category

The trainer compares several regressors (Random Forest, Gradient Boosting,
Linear Regression) and automatically keeps the best performer by
cross-validated R². The fitted model + metadata are persisted with joblib so
the Flask app can load it instantly without retraining on every request.
"""
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Pollutant columns used as model features (order matters for prediction)
FEATURE_COLUMNS: List[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]
TARGET_COLUMN = "AQI"

# Default persisted-model location (relative to project root)
DEFAULT_MODEL_PATH = os.path.join("ml_models", "aqi_model.joblib")


def aqi_to_bucket(aqi: float) -> str:
    """Map a numeric AQI value to the Indian CPCB AQI category."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


class AQIPredictor:
    """Regression model that predicts AQI from pollutant concentrations."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        self.feature_medians: Dict[str, float] = {}
        self.city_profiles: Dict[str, Dict[str, float]] = {}
        self.metadata: Dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, csv_path: str = "city_day.csv") -> Dict:
        """Train and select the best regressor, then store it on the instance.

        Returns a dictionary describing every candidate model and the winner.
        """
        warnings.filterwarnings("ignore")

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=self.feature_columns + [TARGET_COLUMN])

        X = df[self.feature_columns]
        y = df[TARGET_COLUMN]

        # Per-feature medians let us fill missing pollutant inputs sensibly.
        self.feature_medians = X.median().to_dict()

        # Per-city average pollutant profile -> used to pre-fill the web form.
        if "City" in df.columns:
            self.city_profiles = (
                df.groupby("City")[self.feature_columns]
                .mean()
                .round(2)
                .to_dict(orient="index")
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        candidates = {
            "RandomForest": RandomForestRegressor(
                n_estimators=150, max_depth=None, n_jobs=-1, random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            "LinearRegression": LinearRegression(),
        }

        results: Dict[str, Dict] = {}
        best_name = None
        best_score = -np.inf
        best_model = None

        for name, model in candidates.items():
            # 3-fold CV on the training split for model selection
            cv_r2 = cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2)),
                "test_r2": float(r2_score(y_test, preds)),
                "test_mae": float(mean_absolute_error(y_test, preds)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            }
            results[name] = metrics

            if metrics["cv_r2_mean"] > best_score:
                best_score = metrics["cv_r2_mean"]
                best_name = name
                best_model = model

        self.model = best_model

        # Feature importances (only for tree-based winners)
        feature_importance = {}
        if hasattr(best_model, "feature_importances_"):
            feature_importance = {
                feat: float(imp)
                for feat, imp in zip(self.feature_columns, best_model.feature_importances_)
            }

        self.metadata = {
            "best_model": best_name,
            "candidates": results,
            "n_samples": int(len(df)),
            "n_features": len(self.feature_columns),
            "cities": sorted(self.city_profiles.keys()),
            "feature_importance": feature_importance,
            "test_r2": results[best_name]["test_r2"],
            "test_mae": results[best_name]["test_mae"],
            "test_rmse": results[best_name]["test_rmse"],
        }
        return self.metadata

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> str:
        path = path or self.model_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "feature_medians": self.feature_medians,
                "city_profiles": self.city_profiles,
                "metadata": self.metadata,
            },
            path,
        )
        return path

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.model_path
        if not os.path.exists(path):
            return False
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.feature_medians = bundle["feature_medians"]
        self.city_profiles = bundle["city_profiles"]
        self.metadata = bundle["metadata"]
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, pollutants: Dict[str, float]) -> Dict:
        """Predict AQI from a dict of pollutant values.

        Missing pollutants are filled with the training-set median so the
        model still produces a sensible estimate from partial input.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

        row = []
        for feat in self.feature_columns:
            value = pollutants.get(feat)
            if value is None or value == "":
                value = self.feature_medians.get(feat, 0.0)
            row.append(float(value))

        X = pd.DataFrame([row], columns=self.feature_columns)
        aqi_value = float(self.model.predict(X)[0])
        aqi_value = max(0.0, round(aqi_value, 1))

        return {
            "aqi": aqi_value,
            "bucket": aqi_to_bucket(aqi_value),
        }


# Singleton so the Flask app reuses one loaded model instance
_predictor: Optional[AQIPredictor] = None


def get_aqi_predictor() -> AQIPredictor:
    """Return a shared AQIPredictor, loading the persisted model on first use."""
    global _predictor
    if _predictor is None:
        _predictor = AQIPredictor()
        _predictor.load()
    return _predictor
