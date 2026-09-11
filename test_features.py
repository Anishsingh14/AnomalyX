"""
test_features.py — AnomalyX
------------------------------
Unit tests for MAIN/features.py. This file lives at the repo root — run with:
    pytest test_features.py -v
    (or simply: pytest -v, from the repo root)
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "MAIN"))
from features import (
    validate_schema, load_and_clean, engineer_features,
    get_feature_columns, risk_tier, REQUIRED_RAW_COLUMNS, SENSOR_COLUMNS
)


def make_dummy_df(n_rows=30, device_id="SRV-TEST"):
    """Build a small, well-formed dummy telemetry dataframe for testing."""
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": timestamps,
        "device_id": device_id,
        "cpu_temp": rng.normal(50, 2, n_rows),
        "ram_usage": rng.normal(45, 3, n_rows),
        "disk_io_errors": rng.integers(0, 2, n_rows),
        "vibration_level": rng.normal(0.3, 0.05, n_rows),
    })


def test_validate_schema_passes_with_all_columns():
    df = make_dummy_df()
    validate_schema(df)  # should not raise


def test_validate_schema_fails_with_missing_column():
    df = make_dummy_df().drop(columns=["vibration_level"])
    with pytest.raises(ValueError):
        validate_schema(df)


def test_engineer_features_adds_expected_columns():
    df = make_dummy_df(n_rows=30)
    result = engineer_features(df, window=12)
    for col in SENSOR_COLUMNS:
        assert f"{col}_roll_mean" in result.columns
        assert f"{col}_roll_std" in result.columns
        assert f"{col}_zscore" in result.columns
    assert "temp_ram_ratio" in result.columns


def test_engineer_features_drops_warmup_rows():
    df = make_dummy_df(n_rows=30)
    window = 12
    result = engineer_features(df, window=window)
    # first (window - 1) rows per device should be dropped due to warm-up
    assert len(result) == 30 - (window - 1)


def test_engineer_features_no_nans_in_output():
    df = make_dummy_df(n_rows=40)
    result = engineer_features(df, window=12)
    feature_cols = get_feature_columns(result)
    assert not result[feature_cols].isna().any().any()


def test_get_feature_columns_count():
    df = make_dummy_df(n_rows=30)
    result = engineer_features(df, window=12)
    cols = get_feature_columns(result)
    # 4 sensors x 6 engineered features each + 1 cross-feature ratio
    assert len(cols) == 4 * 6 + 1


@pytest.mark.parametrize("prob,expected", [
    (0.0, "Normal"),
    (0.29, "Normal"),
    (0.30, "Watch"),
    (0.59, "Watch"),
    (0.60, "Warning"),
    (0.84, "Warning"),
    (0.85, "Critical"),
    (1.0, "Critical"),
])
def test_risk_tier_thresholds(prob, expected):
    assert risk_tier(prob) == expected


def test_load_and_clean_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_clean("this_file_does_not_exist.csv")


def test_load_and_clean_roundtrip(tmp_path):
    df = make_dummy_df(n_rows=20)
    csv_path = tmp_path / "test_input.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_and_clean(str(csv_path))
    assert set(REQUIRED_RAW_COLUMNS).issubset(loaded.columns)
    assert len(loaded) == 20
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
