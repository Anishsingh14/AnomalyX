"""
This module is imported by BOTH train_model.py and predict.py so that the
exact same transformations are applied at training time and at inference
time.
"""

import pandas as pd
import numpy as np

REQUIRED_RAW_COLUMNS = [
    "timestamp", "device_id", "cpu_temp", "ram_usage",
    "disk_io_errors", "vibration_level"
]

SENSOR_COLUMNS = ["cpu_temp", "ram_usage", "disk_io_errors", "vibration_level"]

# Rolling window size, expressed in number of samples.
# Our data is sampled every 5 minutes, so 12 samples = 1 hour.
ROLLING_WINDOW = 12


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a clear error if the uploaded file doesn't have the right columns."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required column(s): {missing}\n"
            f"Required columns are: {REQUIRED_RAW_COLUMNS}"
        )


def load_and_clean(path: str) -> pd.DataFrame:
    """Load a CSV, validate it, sort it, and coerce types safely."""
    df = pd.read_csv(path)
    validate_schema(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some rows have an unparseable 'timestamp' value. "
                          "Expected a standard datetime format, e.g. 2026-08-01 00:05:00")

    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=SENSOR_COLUMNS)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Note: dropped {dropped} row(s) with non-numeric/missing sensor values.")

    df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Compute rolling-window statistics per device:
      - rolling mean, rolling std, rolling max
      - rate of change (diff)
      - z-score relative to the rolling window

    Returns a new dataframe with the engineered columns added.
    NaN rows at the start of each device's window (warm-up period) are dropped.
    """
    df = df.copy()
    grouped = df.groupby("device_id", group_keys=False)

    for col in SENSOR_COLUMNS:
        roll_mean = grouped[col].transform(lambda x: x.rolling(window, min_periods=window).mean())
        roll_std = grouped[col].transform(lambda x: x.rolling(window, min_periods=window).std())
        roll_max = grouped[col].transform(lambda x: x.rolling(window, min_periods=window).max())
        diff = grouped[col].transform(lambda x: x.diff())

        df[f"{col}_roll_mean"] = roll_mean
        df[f"{col}_roll_std"] = roll_std
        df[f"{col}_roll_max"] = roll_max
        df[f"{col}_diff"] = diff
        # avoid divide-by-zero: where std is 0 or NaN, z-score is set to 0
        safe_std = roll_std.replace(0, np.nan)
        df[f"{col}_zscore"] = ((df[col] - roll_mean) / safe_std).fillna(0)

    # a couple of simple cross-feature signals
    df["temp_ram_ratio"] = df["cpu_temp"] / df["ram_usage"].replace(0, np.nan)
    df["temp_ram_ratio"] = df["temp_ram_ratio"].fillna(0)

    before = len(df)
    df = df.dropna(subset=[f"{c}_roll_mean" for c in SENSOR_COLUMNS]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Note: dropped {dropped} row(s) during rolling-window warm-up "
              f"(first {window} readings of each device).")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of engineered feature column names used for modeling."""
    feature_cols = []
    for col in SENSOR_COLUMNS:
        feature_cols += [col, f"{col}_roll_mean", f"{col}_roll_std",
                          f"{col}_roll_max", f"{col}_diff", f"{col}_zscore"]
    feature_cols.append("temp_ram_ratio")
    return feature_cols


def risk_tier(prob: float) -> str:
    """Convert a predicted probability into a human-readable alert tier."""
    if prob < 0.30:
        return "Normal"
    elif prob < 0.60:
        return "Watch"
    elif prob < 0.85:
        return "Warning"
    else:
        return "Critical"
