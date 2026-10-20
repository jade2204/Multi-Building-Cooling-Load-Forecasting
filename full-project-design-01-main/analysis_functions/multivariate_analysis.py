"""Multivariate analysis (correlation heatmap and pair plot)."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultivariateAnalysisTemplate(ABC):
    """Template: draw a correlation heatmap, then a pair plot."""

    def analyze(self, df: pd.DataFrame) -> None:
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Generate and display a heatmap of the correlations between features."""

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame) -> None:
        """Generate and display a pair plot of the features."""


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
        )
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame) -> None:
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()
