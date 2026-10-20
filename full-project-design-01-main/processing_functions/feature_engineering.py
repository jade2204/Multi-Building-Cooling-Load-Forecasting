"""Feature transformation strategies (log, scaling, one-hot encoding)."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineeringStrategy(ABC):
    """Common interface for feature engineering strategies."""

    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a transformed copy of ``df``."""


class LogTransformation(FeatureEngineeringStrategy):
    """Apply ``log1p`` to skewed features (``log1p`` is safe for zero values)."""

    def __init__(self, features: list[str]):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying log transformation to features: %s", self.features)
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        return df_transformed


class StandardScaling(FeatureEngineeringStrategy):
    """Z-score normalization: zero mean and unit variance."""

    def __init__(self, features: list[str]):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying standard scaling to features: %s", self.features)
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        return df_transformed


class MinMaxScaling(FeatureEngineeringStrategy):
    """Scale features to a fixed range (default ``[0, 1]``)."""

    def __init__(self, features: list[str], feature_range: tuple[float, float] = (0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "Applying Min-Max scaling to features: %s with range %s",
            self.features,
            self.scaler.feature_range,
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        return df_transformed


class OneHotEncoding(FeatureEngineeringStrategy):
    """One-hot encode categorical features, dropping the first level of each."""

    def __init__(self, features: list[str]):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying one-hot encoding to features: %s", self.features)
        encoded = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        remaining = df.drop(columns=self.features).reset_index(drop=True)
        return pd.concat([remaining, encoded], axis=1)


class FeatureEngineer:
    """Apply a feature transformation using an interchangeable strategy."""

    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy) -> None:
        logger.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)
