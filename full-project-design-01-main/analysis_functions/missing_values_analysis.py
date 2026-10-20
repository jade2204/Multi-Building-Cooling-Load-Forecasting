"""Missing-value analysis (counts per column and a heatmap)."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MissingValuesAnalysisTemplate(ABC):
    """Template: identify missing values, then visualize them."""

    def analyze(self, df: pd.DataFrame) -> None:
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """Print the count of missing values for each affected column."""

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """Draw a visualization (e.g. a heatmap) of missing values."""


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
