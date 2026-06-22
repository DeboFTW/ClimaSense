"""Model training and comparison for the AQI prediction pipeline.

The :class:`ModelEvaluator` is the final stage of the offline training pipeline
(``DataLoader -> Preprocessor -> FeatureEngineer -> ModelEvaluator``). It trains
the three candidate regressors, evaluates each on a held-out test split, selects
the best by RMSE, and persists the winning model alongside its metadata.

Responsibilities:
  - Train a Linear Regression, a Random Forest, and an XGBoost regressor on a
    train/test split of the engineered feature set (Req 5.1).
  - Compute RMSE, MAE, and R-squared for each model on the held-out test split
    (Req 5.2).
  - Select the model with the lowest test RMSE as the best model (Req 5.3).
  - Build a :class:`ComparisonReport` containing each model's metrics, the best
    model identity, a one-line written conclusion, and -- when the winner is a
    tree-based model -- the relative feature importance (Req 5.4, 5.6).
  - Persist the best model plus metadata (medians, feature columns, report) to
    disk via joblib for reuse without retraining (Req 5.5 via :meth:`save`).
  - Guard the optional ``xgboost`` import and raise a descriptive
    :class:`ImportError` with an install hint when it is missing (Req 11.2).

The model selected by :meth:`train_and_compare` is retained on
:pyattr:`best_estimator_` so the training entry point can bundle it for
persistence.

Requirements covered: 5.1, 5.2, 5.3, 5.4, 5.6, 11.2.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml_models.aqi import MODEL_ARTIFACT

# Reproducibility / split configuration.
_RANDOM_STATE = 42
_TEST_SIZE = 0.2

# Model identifiers used as keys in the ComparisonReport.
_LINEAR_REGRESSION = "LinearRegression"
_RANDOM_FOREST = "RandomForest"
_XGBOOST = "XGBoost"

# Models whose winner exposes relative feature importance (Req 5.6).
_TREE_BASED_MODELS = frozenset({_RANDOM_FOREST, _XGBOOST})

# Install hint surfaced when xgboost is unavailable (Req 11.2).
_XGBOOST_INSTALL_HINT = (
    "XGBoost is required for AQI model comparison. "
    "Install it with: pip install xgboost==2.0.3"
)


class ComparisonReport(TypedDict):
    """Structured result of comparing the three candidate regressors.

    Attributes
    ----------
    models:
        Mapping of model name to its held-out metrics ``{"rmse", "mae", "r2"}``.
    best_model:
        The name of the model with the lowest test RMSE.
    conclusion:
        A one-line written conclusion naming the best model and its RMSE.
    feature_importance:
        Mapping of feature name to relative importance when the best model is
        tree-based (Random Forest or XGBoost); ``None`` otherwise.
    """

    models: Dict[str, Dict[str, float]]
    best_model: str
    conclusion: str
    feature_importance: Optional[Dict[str, float]]


def _import_xgboost():
    """Import and return the ``XGBRegressor`` class, guarding the dependency.

    Returns
    -------
    type
        The ``xgboost.XGBRegressor`` class.

    Raises
    ------
    ImportError
        With a descriptive install hint when ``xgboost`` is not installed
        (Req 11.2).
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise ImportError(_XGBOOST_INSTALL_HINT) from exc
    return XGBRegressor


class ModelEvaluator:
    """Train, compare, and persist the best AQI regression model.

    The three candidate regressors are trained on a fixed train/test split for
    reproducibility. After :meth:`train_and_compare`, the winning fitted
    estimator is available on :pyattr:`best_estimator_` and the latest report on
    :pyattr:`report_`.

    Parameters
    ----------
    test_size:
        Fraction of the data held out for evaluation. Defaults to ``0.2``.
    random_state:
        Seed used for the split and the stochastic estimators, ensuring
        reproducible comparisons. Defaults to ``42``.
    """

    def __init__(
        self, test_size: float = _TEST_SIZE, random_state: int = _RANDOM_STATE
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self._best_estimator: Optional[Any] = None
        self._report: Optional[ComparisonReport] = None
        self._feature_columns: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_and_compare(self, X, y) -> ComparisonReport:
        """Train the three regressors and select the best by test RMSE.

        The procedure:

        1. Split ``X``/``y`` into train/test partitions with a fixed
           ``random_state`` for reproducibility.
        2. Train Linear Regression, Random Forest, and XGBoost on the training
           partition (Req 5.1).
        3. Compute RMSE, MAE, and R-squared on the held-out test partition for
           each model (Req 5.2).
        4. Select the model with the lowest test RMSE as best (Req 5.3).
        5. Build a :class:`ComparisonReport` with per-model metrics, the best
           model id, a one-line conclusion, and -- when the winner is
           tree-based -- relative feature importance (Req 5.4, 5.6).

        The winning fitted estimator is retained on :pyattr:`best_estimator_`.

        Parameters
        ----------
        X:
            Feature matrix (``pandas.DataFrame`` with the feature columns, or an
            array-like). Column names, when present, are used as feature labels
            for the importance report.
        y:
            Target series/array aligned with ``X``.

        Returns
        -------
        ComparisonReport
            The structured comparison result.

        Raises
        ------
        ImportError
            If ``xgboost`` is not installed (Req 11.2).
        """
        xgb_regressor_cls = _import_xgboost()

        feature_columns = self._extract_feature_columns(X)
        self._feature_columns = feature_columns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        estimators: Dict[str, Any] = {
            _LINEAR_REGRESSION: LinearRegression(),
            _RANDOM_FOREST: RandomForestRegressor(random_state=self.random_state),
            _XGBOOST: xgb_regressor_cls(random_state=self.random_state),
        }

        models_metrics: Dict[str, Dict[str, float]] = {}
        fitted: Dict[str, Any] = {}
        for name, estimator in estimators.items():
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            models_metrics[name] = self._compute_metrics(y_test, predictions)
            fitted[name] = estimator

        best_model = self._select_best_model(models_metrics)
        best_estimator = fitted[best_model]
        self._best_estimator = best_estimator

        feature_importance = self._feature_importance(
            best_model, best_estimator, feature_columns
        )

        best_rmse = models_metrics[best_model]["rmse"]
        report: ComparisonReport = {
            "models": models_metrics,
            "best_model": best_model,
            "conclusion": f"{best_model} performed best with RMSE {best_rmse:.4f}",
            "feature_importance": feature_importance,
        }
        self._report = report
        return report

    def save(self, predictor_bundle, path: str = MODEL_ARTIFACT) -> str:
        """Persist the predictor bundle to disk via joblib (Req 5.5).

        The bundle is the persisted artifact described in the design data model:
        ``{ model, feature_columns, feature_medians, metadata }`` where
        ``metadata`` carries the :class:`ComparisonReport` plus dataset
        metadata. The caller (the training entry point) assembles the bundle so
        this method stays agnostic to how the pieces are gathered.

        Parameters
        ----------
        predictor_bundle:
            The artifact dictionary to persist.
        path:
            Destination path for the joblib artifact. Defaults to
            :data:`ml_models.aqi.MODEL_ARTIFACT`. Parent directories are created
            when absent.

        Returns
        -------
        str
            The path the artifact was written to.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        joblib.dump(predictor_bundle, path)
        return path

    @property
    def best_estimator_(self) -> Any:
        """The fitted best model selected by the latest comparison.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`train_and_compare` has been called.
        """
        if self._best_estimator is None:
            raise RuntimeError(
                "best_estimator_ is unavailable until train_and_compare() has been called"
            )
        return self._best_estimator

    @property
    def report_(self) -> ComparisonReport:
        """The comparison report from the latest comparison.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`train_and_compare` has been called.
        """
        if self._report is None:
            raise RuntimeError(
                "report_ is unavailable until train_and_compare() has been called"
            )
        return self._report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
        """Compute RMSE, MAE, and R-squared on a prediction/actual pair (Req 5.2).

        RMSE is computed as ``sqrt(mean_squared_error(...))`` rather than relying
        on the version-dependent ``squared=False`` keyword, so the calculation is
        robust across scikit-learn versions.
        """
        mse = mean_squared_error(y_true, y_pred)
        return {
            "rmse": float(math.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def _select_best_model(models_metrics: Dict[str, Dict[str, float]]) -> str:
        """Return the model name with the minimum test RMSE (Req 5.3)."""
        return min(models_metrics, key=lambda name: models_metrics[name]["rmse"])

    @staticmethod
    def _extract_feature_columns(X) -> List[str]:
        """Derive feature labels from ``X`` (column names or positional indices)."""
        if isinstance(X, pd.DataFrame):
            return [str(col) for col in X.columns]
        arr = np.asarray(X)
        n_features = arr.shape[1] if arr.ndim > 1 else 1
        return [f"feature_{i}" for i in range(n_features)]

    @staticmethod
    def _feature_importance(
        best_model: str, estimator: Any, feature_columns: List[str]
    ) -> Optional[Dict[str, float]]:
        """Return relative feature importance when the best model is tree-based.

        Tree-based winners (Random Forest, XGBoost) expose
        ``feature_importances_``; the importances are paired with the feature
        column labels (Req 5.6). When the best model is Linear Regression, no
        importance is recorded and ``None`` is returned.
        """
        if best_model not in _TREE_BASED_MODELS:
            return None

        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            return None

        return {
            str(name): float(value)
            for name, value in zip(feature_columns, importances)
        }
