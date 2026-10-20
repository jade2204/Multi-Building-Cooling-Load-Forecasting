"""Basic dataframe inspection strategies (dtypes, summary statistics)."""

from abc import ABC, abstractmethod

import pandas as pd


class DataInspectionStrategy(ABC):
    """Common interface for data inspection strategies."""

    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """Print the results of a specific inspection to the console."""


class DataTypesInspectionStrategy(DataInspectionStrategy):
    """Print column data types and non-null counts."""

    def inspect(self, df: pd.DataFrame) -> None:
        print("\nData Types and Non-null Counts:")
        df.info()


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    """Print summary statistics for numerical and categorical columns."""

    def inspect(self, df: pd.DataFrame) -> None:
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        categorical = df.select_dtypes(include=["object", "category", "string"])
        if not categorical.empty:
            print("\nSummary Statistics (Categorical Features):")
            print(categorical.describe())


class DataInspector:
    """Run a data inspection using an interchangeable strategy."""

    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy) -> None:
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame) -> None:
        self._strategy.inspect(df)
