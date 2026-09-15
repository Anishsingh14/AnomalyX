"""
train_model.py — AnomalyX
---------------------------
Trains the Random Forest downtime-risk classifier on the synthetic telemetry
dataset (synthetic_telemetry_dataset.csv) and saves the trained model
+ feature list to the models/ folder for predict.py to use later.

Run this once before running predict.py:
    python MAIN/train_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import engineer_features, get_feature_columns, load_and_clean

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "synthetic_telemetry_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

RANDOM_STATE = 42


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Split chronologically (not randomly!) so the test set is always
    LATER in time than the training set. Random shuffling would leak
    future information into training and inflate accuracy scores.
    """
    cutoff = df["timestamp"].quantile(1 - test_fraction)
    train_df = df[df["timestamp"] <= cutoff]
    test_df = df[df["timestamp"] > cutoff]
    return train_df, test_df


def main():
    print("=" * 60)
    print("  ANOMALYX — MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Training dataset not found at: {DATA_PATH}")
        sys.exit(1)

    print(f"\n[1/5] Loading dataset from {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    print(f"  Loaded {len(df):,} rows across {df['device_id'].nunique()} devices.")

    if "label" not in df.columns:
        print("ERROR: Training dataset must contain a 'label' column (0/1 downtime risk).")
        sys.exit(1)

    print("\n[2/5] Engineering rolling-window features ...")
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} features engineered. {len(df):,} usable rows remain.")

    print("\n[3/5] Splitting into train/test sets (time-based, not random) ...")
    train_df, test_df = time_based_split(df, test_fraction=0.2)
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]
    print(f"  Train: {len(X_train):,} rows ({y_train.mean()*100:.2f}% positive)")
    print(f"  Test:  {len(X_test):,} rows ({y_test.mean()*100:.2f}% positive)")

    print("\n[4/5] Training Random Forest Classifier ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",   # handles the rare-failure class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("  Training complete.")

    print("\n[5/5] Evaluating on held-out (future) data ...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "At-Risk"], zero_division=0))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {cm}")

    if y_test.nunique() > 1:
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        print(f"\n  ROC-AUC : {roc_auc:.4f}")
        print(f"  PR-AUC  : {pr_auc:.4f}  (more informative than ROC-AUC on imbalanced data)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"\nModel saved to:    {MODEL_PATH}")
    print(f"Feature list saved to: {FEATURES_PATH}")
    print("\nDone. You can now run: python MAIN/predict.py")


if __name__ == "__main__":
    main()
