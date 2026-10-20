"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    timestamp_column: str
    entity_column: str
    target_column: str
    frequency: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class FeatureConfig:
    forecast_horizon_hours: int
    target_lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    sensor_columns: tuple[str, ...]


@dataclass(frozen=True)
class ClusteringConfig:
    n_clusters: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    data: DataConfig
    features: FeatureConfig
    clustering: ClusteringConfig
    config_path: Path

    @property
    def root(self) -> Path:
        return self.config_path.parent.parent

    def path(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]


def parse_config_sections(
    raw: dict[str, Any],
) -> tuple[DataConfig, FeatureConfig, ClusteringConfig]:
    """Build the typed config sections from a raw config mapping.

    Shared by `load_config` (YAML on disk) and the inference service (config
    snapshot stored inside a model bundle).
    """
    data = raw["data"]
    features = raw["features"]
    clustering = raw["clustering"]
    return (
        DataConfig(
            timestamp_column=data["timestamp_column"],
            entity_column=data["entity_column"],
            target_column=data["target_column"],
            frequency=data["frequency"],
            required_columns=tuple(data["required_columns"]),
        ),
        FeatureConfig(
            forecast_horizon_hours=int(features["forecast_horizon_hours"]),
            target_lags=tuple(int(value) for value in features["target_lags"]),
            rolling_windows=tuple(int(value) for value in features["rolling_windows"]),
            sensor_columns=tuple(features["sensor_columns"]),
        ),
        ClusteringConfig(
            n_clusters=int(clustering["n_clusters"]),
            features=tuple(clustering["features"]),
        ),
    )


def load_config(path: str | Path = "configs/base.yaml") -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    data, features, clustering = parse_config_sections(raw)
    return ProjectConfig(
        raw=raw,
        data=data,
        features=features,
        clustering=clustering,
        config_path=config_path,
    )
