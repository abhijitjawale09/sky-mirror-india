from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.preprocessing import (
    FEATURE_COLUMNS,
    REGION_PROFILES,
    TARGET_COLUMNS,
    RegionProfile,
    add_lag_features,
    build_region_code_map,
    normalize_raw_observations,
    prepare_inference_features,
    prepare_training_features,
)


@dataclass(frozen=True)
class RealtimeFeatureResult:
    features: pd.DataFrame
    region: str
    target_date: pd.Timestamp
    step_ahead: int
    region_code_map: dict[str, int]
    basis: dict[str, str]


class RealtimePreprocessor:
    def __init__(self, region_code_map: dict[str, int] | None = None) -> None:
        self.region_code_map = region_code_map or build_region_code_map(
            [r.name for r in REGION_PROFILES]
        )
        self._history_cache: dict[str, pd.DataFrame] = {}

    def initialize_history(self, observations: pd.DataFrame) -> None:
        normalized = normalize_raw_observations(observations)
        with_lags = add_lag_features(normalized, self.region_code_map)
        core_lag_cols = [c for c in FEATURE_COLUMNS if c not in ("insat_lst_lag_1", "insat_sst_lag_1")]
        with_lags = with_lags.dropna(subset=core_lag_cols).reset_index(drop=True)
        for c in ["insat_lst_lag_1", "insat_sst_lag_1"]:
            if c in with_lags.columns:
                with_lags[c] = with_lags[c].fillna(0.0)

        for region in with_lags["region"].unique():
            region_df = with_lags[with_lags["region"] == region].copy()
            self._history_cache[region] = region_df

    def update_history(self, region: str, new_observation: dict[str, Any]) -> None:
        if region not in self._history_cache:
            raise ValueError(f"Region {region} not initialized. Call initialize_history first.")

        new_row = pd.DataFrame([new_observation])
        new_row["date"] = pd.to_datetime(new_row["date"])
        new_row["region"] = new_row["region"].astype(str)

        required = {"date", "region", "latitude", "longitude", "rainfall_mm", "tmax_c", "tmin_c", "humidity_pct"}
        missing = required.difference(new_row.columns)
        if missing:
            raise ValueError(f"New observation missing required columns: {missing}")

        current_history = self._history_cache[region]
        combined = pd.concat([current_history, new_row], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)

        combined = add_lag_features(combined, self.region_code_map)
        core_lag_cols = [c for c in FEATURE_COLUMNS if c not in ("insat_lst_lag_1", "insat_sst_lag_1")]
        combined = combined.dropna(subset=core_lag_cols).reset_index(drop=True)
        for c in ["insat_lst_lag_1", "insat_sst_lag_1"]:
            if c in combined.columns:
                combined[c] = combined[c].fillna(0.0)

        self._history_cache[region] = combined

    def prepare_features(
        self,
        region: str,
        target_date: pd.Timestamp | None = None,
        step_ahead: int = 1,
    ) -> RealtimeFeatureResult:
        if region not in self._history_cache:
            raise ValueError(f"Region {region} not initialized. Call initialize_history first.")

        history = self._history_cache[region]

        if target_date is None:
            target_date = pd.Timestamp(history.iloc[-1]["date"]) + pd.Timedelta(days=step_ahead)

        features = prepare_inference_features(
            history=history,
            region_code_map=self.region_code_map,
            target_date=target_date,
            step_ahead=step_ahead,
        )

        return RealtimeFeatureResult(
            features=features,
            region=region,
            target_date=target_date,
            step_ahead=step_ahead,
            region_code_map=self.region_code_map.copy(),
            basis={
                "features": "shared_preprocessing_pipeline",
                "lag_features": "computed_from_history_cache",
                "region_code": "consistent_with_training",
            },
        )

    def get_history(self, region: str) -> pd.DataFrame | None:
        return self._history_cache.get(region)

    def get_feature_columns(self) -> list[str]:
        return FEATURE_COLUMNS.copy()

    def get_target_columns(self) -> list[str]:
        return TARGET_COLUMNS.copy()

    def get_region_code_map(self) -> dict[str, int]:
        return self.region_code_map.copy()


def create_preprocessor_from_training_data(csv_path: Path | str) -> RealtimePreprocessor:
    frame = pd.read_csv(csv_path)
    _, region_code_map = prepare_training_features(frame)
    preprocessor = RealtimePreprocessor(region_code_map=region_code_map)
    preprocessor.initialize_history(frame)
    return preprocessor


def prepare_single_observation(
    observation: dict[str, Any],
    region_code_map: dict[str, int],
    history: pd.DataFrame,
    step_ahead: int = 1,
) -> pd.DataFrame:
    required = {"date", "region", "latitude", "longitude", "rainfall_mm", "tmax_c", "tmin_c", "humidity_pct"}
    missing = required.difference(observation.keys())
    if missing:
        raise ValueError(f"Observation missing required columns: {missing}")

    obs_df = pd.DataFrame([observation])
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    obs_df["region"] = obs_df["region"].astype(str)

    normalized = normalize_raw_observations(obs_df)
    combined = pd.concat([history, normalized], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    combined = add_lag_features(combined, region_code_map)
    core_lag_cols = [c for c in FEATURE_COLUMNS if c not in ("insat_lst_lag_1", "insat_sst_lag_1")]
    combined = combined.dropna(subset=core_lag_cols).reset_index(drop=True)
    for c in ["insat_lst_lag_1", "insat_sst_lag_1"]:
        if c in combined.columns:
            combined[c] = combined[c].fillna(0.0)

    target_date = pd.Timestamp(observation["date"]) + pd.Timedelta(days=step_ahead)
    features = prepare_inference_features(
        history=combined,
        region_code_map=region_code_map,
        target_date=target_date,
        step_ahead=step_ahead,
    )
    return features