"""Data ingestion from compressed archives."""

import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataIngestor(ABC):
    """Common interface for data ingestors."""

    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Read the data in ``file_path`` into a dataframe."""


class ZipDataIngestor(DataIngestor):
    """Read the single CSV file contained in a ``.zip`` archive."""

    def ingest(self, file_path: str) -> pd.DataFrame:
        if not str(file_path).endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract into a throwaway directory so repeated runs never see stale files.
        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(file_path, "r") as archive:
                archive.extractall(extract_dir)
            csv_files = sorted(Path(extract_dir).rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV file found in the extracted data.")
            if len(csv_files) > 1:
                raise ValueError("Multiple CSV files found. Please specify which one to use.")
            return pd.read_csv(csv_files[0])


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Return the ingestor matching a file extension (e.g. ``.zip``)."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        raise ValueError(f"No ingestor available for file extension: {file_extension}")
