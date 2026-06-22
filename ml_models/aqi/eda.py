"""Exploratory data analysis (EDA) for the AQI prediction pipeline.

The :class:`EDAAnalyzer` is the EDA stage of the offline training pipeline. It
operates on the fully-numeric feature matrix produced by the
:class:`~ml_models.aqi.preprocessor.Preprocessor` and provides the artifacts a
data-science student needs to understand and present the dataset:

  - :meth:`summary_statistics` -- count, mean, min, max, and standard deviation
    for ``AQI`` and each Pollutant_Feature (Req 3.1).
  - :meth:`correlations` -- the Pearson correlation coefficient between each
    Pollutant_Feature and ``AQI`` (Req 3.2).
  - :meth:`render_distribution` -- an AQI distribution histogram persisted as a
    PNG under the static assets directory so the dashboard can display it
    (Req 3.3, 3.4).

matplotlib uses the non-interactive ``Agg`` backend, which is mandatory in this
server context where no display is available. The backend is selected *before*
``pyplot`` is imported, per the project's hard tech-stack rule.

Requirements covered: 3.1, 3.2, 3.3, 3.4.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib

# Select the non-interactive Agg backend BEFORE importing pyplot. There is no
# display in the server context, so this is a hard requirement (see tech stack).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml_models.aqi import (  # noqa: E402
    EDA_DIR,
    POLLUTANT_FEATURES,
    TARGET_COLUMN,
)

# Statistic keys reported per metric, in a stable, presentation-friendly order.
_STAT_KEYS = ("count", "mean", "min", "max", "std")

# Filename of the rendered AQI distribution histogram.
_DISTRIBUTION_FILENAME = "aqi_distribution.png"

# Number of histogram bins for the AQI distribution chart.
_HISTOGRAM_BINS = 30


class EDAAnalyzer:
    """Compute summary statistics, correlations, and distribution charts.

    The analyzer is stateless; every method takes the feature matrix as input so
    the same instance can be reused across datasets.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def summary_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute count/mean/min/max/std for ``AQI`` and each pollutant (Req 3.1).

        Parameters
        ----------
        df:
            A feature matrix containing ``AQI`` and the Pollutant_Features.

        Returns
        -------
        dict[str, dict[str, float]]
            A mapping of metric name (``"AQI"`` plus each present pollutant) to a
            stats dict with keys ``count``, ``mean``, ``min``, ``max``, ``std``.
            Statistics are computed over the observed (non-null) values of each
            column. ``std`` is the sample standard deviation (``ddof=1``) to match
            the numpy/pandas reference (``numpy.std(values, ddof=1)``).
        """
        stats: Dict[str, Dict[str, float]] = {}
        for column in self._metric_columns(df):
            stats[column] = self._column_statistics(df[column])
        return stats

    def correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute the Pearson correlation of each pollutant with ``AQI`` (Req 3.2).

        Parameters
        ----------
        df:
            A feature matrix containing ``AQI`` and the Pollutant_Features.

        Returns
        -------
        dict[str, float]
            A mapping of each present Pollutant_Feature to its Pearson
            correlation coefficient with ``AQI``. Each value lies in ``[-1, 1]``.
            A pollutant whose correlation is undefined (for example, a constant
            column or fewer than two paired observations) maps to ``0.0``.
        """
        if TARGET_COLUMN not in df.columns:
            return {}

        target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        result: Dict[str, float] = {}
        for pollutant in POLLUTANT_FEATURES:
            if pollutant not in df.columns:
                continue
            series = pd.to_numeric(df[pollutant], errors="coerce")
            result[pollutant] = self._pearson(series, target)
        return result

    def render_distribution(self, df: pd.DataFrame, out_dir: str = EDA_DIR) -> str:
        """Render an AQI distribution histogram and persist it as a PNG (Req 3.3/3.4).

        The chart is written under ``out_dir`` (``static/aqi`` by default), which
        is created if it does not already exist, so the AQI dashboard can serve
        the image.

        Parameters
        ----------
        df:
            A feature matrix containing the ``AQI`` column.
        out_dir:
            Destination directory for the PNG. Defaults to
            :data:`~ml_models.aqi.EDA_DIR`.

        Returns
        -------
        str
            The path to the written PNG file.

        Raises
        ------
        ValueError
            If ``df`` has no ``AQI`` column or no observable AQI values to plot.
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Cannot render AQI distribution: missing '{TARGET_COLUMN}' column"
            )

        values = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna()
        if values.empty:
            raise ValueError(
                "Cannot render AQI distribution: no observable AQI values"
            )

        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, _DISTRIBUTION_FILENAME)

        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            ax.hist(values.to_numpy(), bins=_HISTOGRAM_BINS, color="#2b8cbe",
                    edgecolor="white")
            ax.set_title("AQI Distribution")
            ax.set_xlabel("AQI")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            fig.savefig(output_path)
        finally:
            # Always release the figure so repeated renders do not leak memory.
            plt.close(fig)

        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _metric_columns(df: pd.DataFrame) -> list[str]:
        """Return ``AQI`` followed by each present Pollutant_Feature, in order."""
        columns: list[str] = []
        if TARGET_COLUMN in df.columns:
            columns.append(TARGET_COLUMN)
        columns.extend(p for p in POLLUTANT_FEATURES if p in df.columns)
        return columns

    @staticmethod
    def _column_statistics(column: pd.Series) -> Dict[str, float]:
        """Compute count/mean/min/max/std over a column's observed values.

        ``count`` is the number of non-null observations. ``std`` is the sample
        standard deviation (``ddof=1``); it is ``0.0`` when fewer than two values
        are present, matching the convention that a single observation has no
        spread.
        """
        values = pd.to_numeric(column, errors="coerce").dropna().to_numpy()
        count = int(values.size)
        if count == 0:
            return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        std = float(np.std(values, ddof=1)) if count > 1 else 0.0
        return {
            "count": float(count),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": std,
        }

    @staticmethod
    def _pearson(a: pd.Series, b: pd.Series) -> float:
        """Pearson correlation of two series over their shared non-null rows.

        Mirrors ``numpy.corrcoef`` while guarding the degenerate cases (fewer
        than two paired observations or a constant column) that would otherwise
        produce ``NaN``; those map to ``0.0``. The result is clipped to
        ``[-1, 1]`` to absorb floating-point overshoot.
        """
        paired = pd.concat([a, b], axis=1).dropna()
        if len(paired) < 2:
            return 0.0

        x = paired.iloc[:, 0].to_numpy()
        y = paired.iloc[:, 1].to_numpy()
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return 0.0

        coefficient = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(coefficient):
            return 0.0
        return float(np.clip(coefficient, -1.0, 1.0))
