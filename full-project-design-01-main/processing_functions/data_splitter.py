"""Train/test and stratified train/validation/test splitting strategies."""

import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    """Common interface for data splitting strategies."""

    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """Split ``df`` into feature/target subsets (the tuple shape depends on the strategy)."""


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Random train/test split.

    Returns ``X_train, X_test, y_train, y_test``.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        logger.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        result = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        logger.info("Train-test split completed.")
        return result


class StratifiedTrainValidationTestSplitStrategy(DataSplittingStrategy):
    """Stratified train, validation and test split for classification targets.

    The validation set is used for model selection; the test set stays untouched
    until a champion has been selected.

    Returns ``X_train, X_validation, X_test, y_train, y_validation, y_test``.
    """

    def __init__(
        self, validation_size: float = 0.15, test_size: float = 0.15, random_state: int = 42
    ):
        if validation_size <= 0 or test_size <= 0:
            raise ValueError("validation_size and test_size must be positive.")
        if validation_size + test_size >= 1:
            raise ValueError("validation_size + test_size must be less than 1.")
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        logger.info("Performing stratified train-validation-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        holdout_size = self.validation_size + self.test_size
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=holdout_size, random_state=self.random_state, stratify=y
        )
        X_validation, X_test, y_validation, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=self.test_size / holdout_size,
            random_state=self.random_state,
            stratify=y_holdout,
        )
        logger.info("Stratified train-validation-test split completed.")
        return X_train, X_validation, X_test, y_train, y_validation, y_test


class DataSplitter:
    """Split data using an interchangeable strategy."""

    def __init__(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy) -> None:
        logger.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        logger.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df, target_column)
