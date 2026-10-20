"""Missing-value handling strategies (drop or fill)."""

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class MissingValueHandlingStrategy(ABC):
    """Common interface for missing-value handling strategies."""

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with missing values handled."""


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Drop rows or columns containing missing values.

    Parameters:
        axis: 0 to drop rows, 1 to drop columns.
        thresh: Minimum number of non-NA values required to keep a row/column.
    """

    def __init__(self, axis: int = 0, thresh: int | None = None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Dropping missing values with axis=%s and thresh=%s", self.axis, self.thresh)
        return df.dropna(axis=self.axis, thresh=self.thresh)


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """Fill missing values.

    Parameters:
        method: ``mean``, ``median`` (numeric columns), ``mode`` (all columns)
            or ``constant`` (uses ``fill_value``).
        fill_value: Value used when ``method='constant'``.
    """

    def __init__(self, method: str = "mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filling missing values using method: %s", self.method)
        df_cleaned = df.copy()
        if self.method in {"mean", "median"}:
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            statistic = getattr(df[numeric_columns], self.method)()
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(statistic)
        elif self.method == "mode":
            for column in df_cleaned.columns:
                modes = df[column].mode()
                if not modes.empty:
                    df_cleaned[column] = df_cleaned[column].fillna(modes.iloc[0])
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logger.warning("Unknown method '%s'. No missing values handled.", self.method)
        return df_cleaned


class MissingValueHandler:
    """Handle missing values using an interchangeable strategy."""

    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy) -> None:
        logger.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)
