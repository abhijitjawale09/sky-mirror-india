#!/usr/bin/env python3
"""Test to verify preprocessing parity between training and real-time inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from climate_twin.data.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    normalize_raw_observations,
    prepare_inference_features,
    prepare_training_features,
)
from climate_twin.services.realtime_preprocess import RealtimePreprocessor, prepare_single_observation


def load_training_data() -> pd.DataFrame:
    """Load the actual training data."""
    path = "data/processed/climate_training_data.csv"
    return pd.read_csv(path)


def test_training_feature_structure():
    """Test that training features have the expected structure."""
    print("=" * 60)
    print("TEST 1: Training feature structure")
    print("=" * 60)

    df = load_training_data()
    print(f"Raw data shape: {df.shape}")
    print(f"Raw columns: {list(df.columns)}")

    feature_frame, region_code_map = prepare_training_features(df)

    print(f"\nFeature frame shape: {feature_frame.shape}")
    print(f"Feature columns: {list(feature_frame.columns)}")
    print(f"Expected feature columns: {FEATURE_COLUMNS}")
    print(f"Target columns: {TARGET_COLUMNS}")
    print(f"Region code map: {region_code_map}")

    # Verify all expected feature columns exist
    missing = set(FEATURE_COLUMNS) - set(feature_frame.columns)
    extra = set(feature_frame.columns) - set(FEATURE_COLUMNS) - set(TARGET_COLUMNS)
    assert not missing, f"Missing feature columns: {missing}"
    assert not extra, f"Extra unexpected columns: {extra}"

    # Verify no NaN in features
    nan_count = feature_frame[FEATURE_COLUMNS].isna().sum().sum()
    assert nan_count == 0, f"NaN values in features: {nan_count}"

    print(f"\n✓ Training features valid: {feature_frame.shape[0]} rows, {len(FEATURE_COLUMNS)} features")
    print(f"  Targets: {list(TARGET_COLUMNS)}")
    print(f"  Region codes: {region_code_map}")

    return feature_frame, region_code_map


def test_realtime_preprocessor_parity():
    """Test that realtime preprocessor produces same features as training for historical data."""
    print("\n" + "=" * 60)
    print("TEST 2: Realtime preprocessor parity with training")
    print("=" * 60)

    df = load_training_data()
    feature_frame, region_code_map = prepare_training_features(df)

    # Initialize realtime preprocessor with training data
    preprocessor = RealtimePreprocessor(region_code_map=region_code_map)
    preprocessor.initialize_history(df)

    # Test for each region
    for region in df["region"].unique():
        history = preprocessor.get_history(region)
        if history is None:
            print(f"  ⚠ No history for {region}")
            continue

        # Get the last date in history
        last_date = pd.Timestamp(history.iloc[-1]["date"])

        # Prepare features using realtime preprocessor for the LAST historical date (step_ahead=0)
        # This should match the training features for that date
        result = preprocessor.prepare_features(region=region, target_date=last_date, step_ahead=0)
        rt_features = result.features

        # Get the corresponding training feature row (the last row for this region in training)
        train_region_features = feature_frame[feature_frame["region_code"] == region_code_map[region]].tail(1)[
            FEATURE_COLUMNS
        ]

        print(f"\n  Region: {region}")
        print(f"    Training feature shape: {train_region_features.shape}")
        print(f"    Realtime feature shape: {rt_features.shape}")

        # Compare feature values
        train_vals = train_region_features.iloc[0].values
        rt_vals = rt_features.iloc[0].values

        diff = np.abs(train_vals - rt_vals)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"    Max diff: {max_diff:.6f}")
        print(f"    Mean diff: {mean_diff:.6f}")

        # Allow small numerical differences due to float precision
        assert max_diff < 1e-5, f"Feature mismatch for {region}: max_diff={max_diff}"
        print(f"    ✓ Features match within tolerance")

    print("\n✓ All regions pass parity test")


def test_prepare_single_observation():
    """Test the prepare_single_observation function."""
    print("\n" + "=" * 60)
    print("TEST 3: prepare_single_observation function")
    print("=" * 60)

    df = load_training_data()
    feature_frame, region_code_map = prepare_training_features(df)

    preprocessor = RealtimePreprocessor(region_code_map=region_code_map)
    preprocessor.initialize_history(df)

    region = "Kerala Coast"
    history = preprocessor.get_history(region)

    # Create a synthetic observation for the next day
    last_row = history.iloc[-1]
    next_date = pd.Timestamp(last_row["date"]) + pd.Timedelta(days=1)

    observation = {
        "date": next_date.strftime("%Y-%m-%d"),
        "region": region,
        "latitude": float(last_row["latitude"]),
        "longitude": float(last_row["longitude"]),
        "rainfall_mm": 5.0,
        "tmax_c": 32.0,
        "tmin_c": 25.0,
        "humidity_pct": 80.0,
    }

    features = prepare_single_observation(
        observation=observation,
        region_code_map=region_code_map,
        history=history,
        step_ahead=1,
    )

    print(f"  Input observation: {observation}")
    print(f"  Output features shape: {features.shape}")
    print(f"  Output features columns: {list(features.columns)}")
    print(f"  Feature values: {features.iloc[0].to_dict()}")

    assert features.shape == (1, len(FEATURE_COLUMNS))
    assert list(features.columns) == FEATURE_COLUMNS
    print("  ✓ prepare_single_observation works correctly")


def test_feature_column_order():
    """Verify feature column order is consistent."""
    print("\n" + "=" * 60)
    print("TEST 4: Feature column order consistency")
    print("=" * 60)

    df = load_training_data()
    feature_frame, region_code_map = prepare_training_features(df)

    preprocessor = RealtimePreprocessor(region_code_map=region_code_map)
    preprocessor.initialize_history(df)

    region = "Kerala Coast"
    history = preprocessor.get_history(region)
    last_date = pd.Timestamp(history.iloc[-1]["date"])
    target_date = last_date + pd.Timedelta(days=1)

    result = preprocessor.prepare_features(region=region, target_date=target_date, step_ahead=1)

    train_cols = list(feature_frame[FEATURE_COLUMNS].columns)
    rt_cols = list(result.features.columns)

    print(f"  Training feature order: {train_cols}")
    print(f"  Realtime feature order: {rt_cols}")
    print(f"  Expected order: {FEATURE_COLUMNS}")

    assert train_cols == rt_cols == FEATURE_COLUMNS
    print("  ✓ Feature column order is consistent")


def test_model_input_shape():
    """Verify the model input shape matches expectations."""
    print("\n" + "=" * 60)
    print("TEST 5: Model input shape verification")
    print("=" * 60)

    df = load_training_data()
    feature_frame, region_code_map = prepare_training_features(df)

    print(f"  Training feature matrix shape: {feature_frame[FEATURE_COLUMNS].shape}")
    print(f"  Number of features: {len(FEATURE_COLUMNS)}")
    print(f"  Number of targets: {len(TARGET_COLUMNS)}")
    print(f"  Target columns: {TARGET_COLUMNS}")

    # Test with a single inference
    preprocessor = RealtimePreprocessor(region_code_map=region_code_map)
    preprocessor.initialize_history(df)

    region = "Kerala Coast"
    history = preprocessor.get_history(region)
    last_date = pd.Timestamp(history.iloc[-1]["date"])
    target_date = last_date + pd.Timedelta(days=1)

    result = preprocessor.prepare_features(region=region, target_date=target_date, step_ahead=1)

    print(f"  Single inference feature shape: {result.features.shape}")
    assert result.features.shape == (1, len(FEATURE_COLUMNS))
    print("  ✓ Model input shape is correct")


def main():
    print("\n" + "#" * 60)
    print("# PREPROCESSING PARITY VERIFICATION")
    print("#" * 60)

    test_training_feature_structure()
    test_realtime_preprocessor_parity()
    test_prepare_single_observation()
    test_feature_column_order()
    test_model_input_shape()

    print("\n" + "#" * 60)
    print("# ALL TESTS PASSED ✓")
    print("#" * 60)
    print("\nSummary:")
    print(f"  - Training features: {len(FEATURE_COLUMNS)} columns")
    print(f"  - Target columns: {TARGET_COLUMNS}")
    print(f"  - Feature columns: {FEATURE_COLUMNS}")
    print(f"  - Region code map: {len(REGION_PROFILES)} regions")
    print("  - Shared preprocessing pipeline verified for parity")


if __name__ == "__main__":
    main()