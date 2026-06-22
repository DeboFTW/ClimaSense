"""Offline training entry point for the AQI prediction system.

Runs the full data-science pipeline end to end and persists a reusable model
artifact plus EDA charts:

    DataLoader -> Preprocessor -> EDAAnalyzer -> FeatureEngineer -> ModelEvaluator
              -> save artifact (MODEL_ARTIFACT) + EDA distribution chart (EDA_DIR)

Usage:
    python train_aqi_model.py [dataset_path]

If ``dataset_path`` is omitted it defaults to ``DEFAULT_DATASET`` (city_day.csv).

The script prints dataset metadata, EDA summary statistics and correlations, the
model comparison report (per-model RMSE/MAE/R2), the best-model conclusion, and
feature importance when available, then saves the artifact to ``MODEL_ARTIFACT``.

Exit codes:
    0  success
    1  dataset missing/unreadable, no usable records, or a missing dependency
"""

from __future__ import annotations

import sys

import pandas as pd

from ml_models.aqi import (
    DEFAULT_DATASET,
    EDA_DIR,
    MODEL_ARTIFACT,
    TARGET_COLUMN,
)
from ml_models.aqi.data_loader import DataLoader
from ml_models.aqi.eda import EDAAnalyzer
from ml_models.aqi.feature_engineer import FeatureEngineer
from ml_models.aqi.model_evaluator import ModelEvaluator
from ml_models.aqi.preprocessor import Preprocessor

# Helper columns that the FeatureEngineer needs but the Preprocessor drops.
_CITY_COLUMN = "City"
_TIMESTAMP_COLUMN = "timestamp"


def _build_engineering_frame(
    raw_df: pd.DataFrame, processed: pd.DataFrame
) -> pd.DataFrame:
    """Re-attach ``City`` and ``timestamp`` to the preprocessed feature matrix.

    The :class:`Preprocessor` returns a fully-numeric matrix that drops the
    ``City`` and ``timestamp`` columns, but the :class:`FeatureEngineer` needs
    both to order records and compute per-city, time-based lag/rolling features.

    The Preprocessor produces its matrix by (1) dropping rows missing the AQI
    target and (2) dropping rows whose timestamp cannot be parsed, resetting the
    index after each step. We replicate that exact filtering on the raw frame so
    the surviving ``City``/``timestamp`` rows align positionally with the
    processed matrix, then horizontally concatenate them.

    A length guard raises a clear error if the row counts diverge, which would
    indicate the filtering logic has drifted out of sync with the Preprocessor.
    """
    aligned = raw_df.copy()

    # Mirror Preprocessor._drop_missing_target: drop rows missing AQI.
    if TARGET_COLUMN in aligned.columns:
        aligned = aligned[aligned[TARGET_COLUMN].notna()].reset_index(drop=True)
    else:
        aligned = aligned.iloc[0:0]

    # Mirror Preprocessor._parse_timestamp: parse timestamp, drop unparseable.
    aligned[_TIMESTAMP_COLUMN] = pd.to_datetime(
        aligned[_TIMESTAMP_COLUMN], errors="coerce"
    )
    aligned = aligned.dropna(subset=[_TIMESTAMP_COLUMN]).reset_index(drop=True)

    if len(aligned) != len(processed):
        raise RuntimeError(
            "Row-count mismatch while re-attaching City/timestamp to the "
            f"preprocessed matrix: raw-aligned has {len(aligned)} rows but the "
            f"processed matrix has {len(processed)}. The training script's "
            "filtering is out of sync with the Preprocessor."
        )

    # processed has a fresh RangeIndex; align the carried columns to it.
    carried = aligned[[_CITY_COLUMN, _TIMESTAMP_COLUMN]].reset_index(drop=True)
    processed = processed.reset_index(drop=True)
    return pd.concat([carried, processed], axis=1)


def _print_metadata(metadata) -> None:
    """Print loaded dataset metadata (record count and covered cities)."""
    cities = metadata["cities"]
    print("\n=== Dataset Metadata ===")
    print(f"Records loaded : {metadata['record_count']}")
    print(f"Cities ({len(cities)}) : {', '.join(cities) if cities else '(none)'}")


def _print_eda(analyzer: EDAAnalyzer, processed: pd.DataFrame) -> None:
    """Print EDA summary statistics and pollutant/AQI correlations."""
    print("\n=== EDA Summary Statistics ===")
    stats = analyzer.summary_statistics(processed)
    header = f"{'metric':10s} {'count':>8s} {'mean':>10s} {'min':>10s} {'max':>10s} {'std':>10s}"
    print(header)
    for metric, values in stats.items():
        print(
            f"{metric:10s} {values['count']:>8.0f} {values['mean']:>10.3f} "
            f"{values['min']:>10.3f} {values['max']:>10.3f} {values['std']:>10.3f}"
        )

    print("\n=== Pollutant / AQI Correlations ===")
    correlations = analyzer.correlations(processed)
    if correlations:
        for pollutant, coefficient in correlations.items():
            print(f"  {pollutant:8s} {coefficient:+.4f}")
    else:
        print("  (no pollutant columns available)")


def _print_report(report) -> None:
    """Print the per-model comparison metrics, conclusion, and importance."""
    print("\n=== Model Comparison (held-out test split) ===")
    print(f"{'model':18s} {'RMSE':>12s} {'MAE':>12s} {'R2':>10s}")
    best = report["best_model"]
    for name, metrics in report["models"].items():
        marker = "  <-- BEST" if name == best else ""
        print(
            f"{name:18s} {metrics['rmse']:>12.4f} {metrics['mae']:>12.4f} "
            f"{metrics['r2']:>10.4f}{marker}"
        )

    importance = report.get("feature_importance")
    if importance:
        print("\n=== Feature Importance (best model) ===")
        ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        for feature, value in ranked:
            print(f"  {feature:12s} {value:.4f}")

    print(f"\nConclusion: {report['conclusion']}")


def run_pipeline(dataset_path: str) -> str:
    """Execute the full training pipeline and persist the model artifact.

    Returns the path the artifact was written to.
    """
    # 1. Load raw records (Pollutant_Features + AQI + City + timestamp).
    loader = DataLoader(dataset_path=dataset_path)
    print(f"Loading AQI dataset from: {dataset_path}")
    raw_df = loader.load()
    _print_metadata(loader.metadata)

    # 2. Preprocess into a clean, fully-numeric feature matrix; capture medians.
    preprocessor = Preprocessor()
    processed = preprocessor.fit_transform(raw_df)

    # 3. Exploratory data analysis: stats, correlations, distribution chart.
    analyzer = EDAAnalyzer()
    _print_eda(analyzer, processed)
    chart_path = analyzer.render_distribution(processed, out_dir=EDA_DIR)
    print(f"\nEDA distribution chart saved to: {chart_path}")

    # 4. Feature engineering needs City + timestamp alongside the numeric matrix.
    engineering_input = _build_engineering_frame(raw_df, processed)
    engineer = FeatureEngineer()
    engineered = engineer.transform(engineering_input)

    # 5. Assemble the model matrix and target.
    X = engineered[engineer.feature_columns]
    y = engineered["target"]

    # 6. Train and compare the three regressors; select best by RMSE.
    print("\nTraining and comparing models (LinearRegression, RandomForest, XGBoost) ...")
    evaluator = ModelEvaluator()
    report = evaluator.train_and_compare(X, y)

    # 7. Print the comparison report and best-model conclusion.
    _print_report(report)

    # 9. Assemble and persist the model artifact bundle.
    bundle = {
        "model": evaluator.best_estimator_,
        "feature_columns": engineer.feature_columns,
        "feature_medians": {
            **preprocessor.feature_medians,
            **engineer.lag_medians,
        },
        "metadata": {
            "report": evaluator.report_,
            "dataset": loader.metadata,
        },
    }
    saved_path = evaluator.save(bundle, MODEL_ARTIFACT)
    print(f"\nSaved model artifact to: {saved_path}")
    return saved_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = sys.argv[1:] if argv is None else argv
    dataset_path = args[0] if args else DEFAULT_DATASET

    try:
        run_pipeline(dataset_path)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Provide a valid dataset path, e.g. "
            "`python train_aqi_model.py path/to/city_day.csv`.",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"\nERROR: no usable records to train on: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"\nERROR: missing dependency: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
