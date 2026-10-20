"""Regression tests for the reusable strategy modules."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from analysis_functions.bivariate_analysis import (
    BivariateAnalyzer,
    CategoricalVsNumericalAnalysis,
)
from analysis_functions.univariate_analysis import (
    CategoricalUnivariateAnalysis,
    UnivariateAnalyzer,
)
from processing_functions.feature_engineering import FeatureEngineer, OneHotEncoding
from processing_functions.handle_missing_values import (
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from processing_functions.ingest_data import ZipDataIngestor
from processing_functions.outlier_detection import IQROutlierDetection, OutlierDetector


def test_one_hot_encoding_drops_source_column_and_first_level():
    df = pd.DataFrame({"kind": ["a", "b", "c"], "value": [1, 2, 3]})
    result = FeatureEngineer(OneHotEncoding(["kind"])).apply_feature_engineering(df)
    assert list(result.columns) == ["value", "kind_b", "kind_c"]


def test_mode_fill_handles_all_missing_column():
    df = pd.DataFrame({"x": [1.0, 1.0, np.nan], "y": [np.nan] * 3})
    result = MissingValueHandler(FillMissingValuesStrategy("mode")).handle_missing_values(df)
    assert result["x"].tolist() == [1.0, 1.0, 1.0]
    assert result["y"].isna().all()


def test_outlier_cap_only_touches_requested_column():
    df = pd.DataFrame({"a": list(range(100)) + [10_000], "label": ["x"] * 101})
    detector = OutlierDetector(IQROutlierDetection())
    capped = detector.handle_outliers(df, "a", method="cap")
    assert capped["a"].max() < 10_000
    assert capped["label"].eq("x").all()
    assert len(detector.handle_outliers(df, "a", method="remove")) == 100


def test_zip_ingestor_reads_single_csv(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/data.csv", "a,b\n1,2\n")
    assert ZipDataIngestor().ingest(str(archive)).to_dict("records") == [{"a": 1, "b": 2}]
    with pytest.raises(ValueError):
        ZipDataIngestor().ingest(str(tmp_path / "data.txt"))


def test_unsupported_plot_options_raise(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    df = pd.DataFrame({"c": ["a", "b"], "v": [1, 2]})
    with pytest.raises(ValueError):
        BivariateAnalyzer(CategoricalVsNumericalAnalysis()).execute_analysis(df, "c", "v", "nope")
    UnivariateAnalyzer(CategoricalUnivariateAnalysis()).execute_analysis(df, "c")
