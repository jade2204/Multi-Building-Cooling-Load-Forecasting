"""Univariate analysis strategies (numerical and categorical distributions)."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class UnivariateAnalysisStrategy(ABC):
    """Common interface for univariate analysis strategies."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """Visualize the distribution of a single column."""


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """Histogram with a KDE curve."""

    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """Bar plot of category frequencies."""

    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue=feature, data=df, palette="muted", legend=False)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


class UnivariateAnalyzer:
    """Run a univariate analysis using an interchangeable strategy."""

    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str) -> None:
        self._strategy.analyze(df, feature)
