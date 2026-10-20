"""Bivariate analysis strategies (scatter plot, box plot, bar chart)."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _finish_plot(feature1: str, feature2: str, rotate_xticks: bool = False) -> None:
    plt.title(f"{feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    if rotate_xticks:
        plt.xticks(rotation=45)
    plt.show()


class BivariateAnalysisStrategy(ABC):
    """Common interface for bivariate analysis strategies."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, option: str) -> None:
        """Visualize the relationship between two columns.

        Parameters:
            df: The dataframe containing the data.
            feature1: Name of the first column (x axis).
            feature2: Name of the second column (y axis).
            option: Which chart to draw; the supported values depend on the strategy.
        """


class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """Relationship between two numerical features. Supported option: ``scatter_plot``."""

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, option: str) -> None:
        if option != "scatter_plot":
            raise ValueError(f"Unsupported option for numerical analysis: {option!r}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        _finish_plot(feature1, feature2)


class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """Relationship between a categorical and a numerical feature.

    Supported options: ``boxplot`` and ``column_chart``.
    """

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, option: str) -> None:
        plotters = {"boxplot": sns.boxplot, "column_chart": sns.barplot}
        if option not in plotters:
            raise ValueError(f"Unsupported option for categorical analysis: {option!r}")
        plt.figure(figsize=(10, 6))
        plotters[option](x=feature1, y=feature2, data=df)
        _finish_plot(feature1, feature2, rotate_xticks=True)


class BivariateAnalyzer:
    """Run a bivariate analysis using an interchangeable strategy."""

    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str, option: str) -> None:
        self._strategy.analyze(df, feature1, feature2, option)
