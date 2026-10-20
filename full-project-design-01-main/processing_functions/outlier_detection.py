"""Outlier detection strategies (Z-score, IQR) plus removal, capping and plots."""

import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class OutlierDetectionStrategy(ABC):
    """Common interface for outlier detection strategies."""

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """Return a boolean Series that is True where ``column_name`` is an outlier."""


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    """Flag values more than ``threshold`` standard deviations from the mean."""

    def __init__(self, threshold: float = 3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        logger.info("Detecting outliers using the Z-score method (threshold=%s).", self.threshold)
        column = df[column_name]
        z_scores = np.abs((column - column.mean()) / column.std())
        return z_scores > self.threshold


class IQROutlierDetection(OutlierDetectionStrategy):
    """Flag values outside 1.5 * IQR of the first and third quartiles."""

    def detect_outliers(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        logger.info("Detecting outliers using the IQR method.")
        column = df[column_name]
        q1, q3 = column.quantile(0.25), column.quantile(0.75)
        iqr = q3 - q1
        return (column < (q1 - 1.5 * iqr)) | (column > (q3 + 1.5 * iqr))


class OutlierDetector:
    """Detect and handle outliers using an interchangeable strategy."""

    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy) -> None:
        logger.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        logger.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df, column_name)

    def handle_outliers(
        self, df: pd.DataFrame, column_name: str, method: str = "remove", **kwargs
    ) -> pd.DataFrame:
        """Remove outlier rows (``method='remove'``) or cap ``column_name`` at its
        1st/99th percentiles (``method='cap'``).
        """
        if method == "remove":
            logger.info("Removing outliers from the dataset.")
            return df[~self.detect_outliers(df, column_name)]
        if method == "cap":
            logger.info("Capping outliers in column '%s'.", column_name)
            capped = df.copy()
            capped[column_name] = capped[column_name].clip(
                lower=df[column_name].quantile(0.01), upper=df[column_name].quantile(0.99)
            )
            return capped
        logger.warning("Unknown method '%s'. No outlier handling performed.", method)
        return df

    def visualize_outliers(self, df: pd.DataFrame, features: list[str]) -> None:
        logger.info("Visualizing outliers for features: %s", features)
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
