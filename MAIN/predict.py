"""
predict.py — AnomalyX
----------------------
MAIN EXECUTABLE. Run this file directly from any terminal or IDE:

    python MAIN/predict.py

It will interactively ask you for the path to a telemetry CSV file,
analyze it using the trained Random Forest model, print a risk summary
to the terminal, and save 4 visualization charts to the Sample_Outputs/ folder.

If no trained model is found in models/, it will automatically train one
first using the bundled synthetic dataset.
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe non-interactive backend for any terminal/IDE
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import engineer_features, get_feature_columns, load_and_clean, risk_tier, RISK_THRESHOLDS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.joblib")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_input.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Sample_Outputs")

TIER_COLORS = {"Normal": "#2ecc71", "Watch": "#f1c40f", "Warning": "#e67e22", "Critical": "#e74c3c"}


# --------------------------------------------------------------------------
# STEP 0: Interactive input prompt
# --------------------------------------------------------------------------
def prompt_for_input_file() -> str:
    """Ask the user to provide/upload a CSV file path. Retries on bad input."""
    print("=" * 65)
    print("  ANOMALYX")
    print("  Predictive Equipment Maintenance & Anomaly Alert System")
    print("=" * 65)
    print("\nRequired columns: timestamp, device_id, cpu_temp, ram_usage,")
    print("                  disk_io_errors, vibration_level\n")

    while True:
        default_hint = f" [press Enter to use bundled sample: {SAMPLE_PATH}]" if os.path.exists(SAMPLE_PATH) else ""
        user_path = input(f"Enter the path to your telemetry CSV file{default_hint}: ").strip().strip('"')

        if user_path == "" and os.path.exists(SAMPLE_PATH):
            return SAMPLE_PATH

        if user_path == "":
            print("  No path entered and no sample file available. Please try again.\n")
            continue

        if not os.path.exists(user_path):
            print(f"  File not found at: {user_path}\n  Please check the path and try again.\n")
            continue

        if not user_path.lower().endswith(".csv"):
            print("  Warning: file does not have a .csv extension. Attempting to read it anyway.\n")

        return user_path


# --------------------------------------------------------------------------
# STEP 1: Ensure a trained model exists
# --------------------------------------------------------------------------
def ensure_model_exists():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        return
    print("\nNo trained model found — training one now using the bundled synthetic dataset.")
    print("(This only needs to happen once. Future runs will reuse the saved model.)\n")
    import train_model
    train_model.main()


# --------------------------------------------------------------------------
# STEP 2: Run inference
# --------------------------------------------------------------------------
def run_inference(input_path: str):
    print(f"\n[Step 1/4] Loading and validating input file: {input_path}")
    raw_df = load_and_clean(input_path)
    print(f"  {len(raw_df):,} valid rows across {raw_df['device_id'].nunique()} device(s).")

    print("\n[Step 2/4] Engineering rolling-window features ...")
    feat_df = engineer_features(raw_df)
    if len(feat_df) == 0:
        print("\nERROR: Not enough rows per device to compute rolling-window features.")
        print("Each device needs at least 12 consecutive readings (1 hour of data at 5-min intervals).")
        sys.exit(1)

    print("\n[Step 3/4] Loading trained model and generating predictions ...")
    clf = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print(f"ERROR: engineered features don't match model's expected features: {missing}")
        sys.exit(1)

    X = feat_df[feature_cols]
    feat_df["risk_probability"] = clf.predict_proba(X)[:, 1]
    feat_df["risk_tier"] = feat_df["risk_probability"].apply(risk_tier)

    print("  Done.")
    return feat_df, clf, feature_cols


# --------------------------------------------------------------------------
# STEP 3: Print a human-readable summary to the terminal
# --------------------------------------------------------------------------
def print_summary(feat_df: pd.DataFrame):
    print("\n[Step 4/4] Analysis Summary")
    print("-" * 65)
    tier_counts = feat_df["risk_tier"].value_counts().reindex(
        ["Normal", "Watch", "Warning", "Critical"], fill_value=0)
    for tier, count in tier_counts.items():
        pct = 100 * count / len(feat_df)
        print(f"  {tier:10s}: {count:7,d} readings  ({pct:5.2f}%)")

    alerts = feat_df[feat_df["risk_tier"].isin(["Warning", "Critical"])]
    print(f"\n  Total Warning/Critical alerts: {len(alerts):,}")

    if len(alerts) > 0:
        print("\n  Devices with the highest average risk:")
        top_devices = (alerts.groupby("device_id")["risk_probability"]
                        .mean().sort_values(ascending=False).head(5))
        for dev, avg_prob in top_devices.items():
            print(f"    - {dev}: avg risk probability {avg_prob:.2f}")

        worst_row = feat_df.loc[feat_df["risk_probability"].idxmax()]
        print(f"\n  Highest single risk reading: {worst_row['device_id']} at "
              f"{worst_row['timestamp']} — probability {worst_row['risk_probability']:.2f} "
              f"({worst_row['risk_tier']})")
    else:
        print("\n  No Warning/Critical alerts detected in this file — all readings look normal.")
    print("-" * 65)


# --------------------------------------------------------------------------
# STEP 4: Visualizations (4 charts, saved as PNG files)
# --------------------------------------------------------------------------
def make_visualizations(feat_df: pd.DataFrame, clf, feature_cols: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = []

    # ---- Chart 1: Fleet-wide risk heatmap (devices x time) ----
    try:
        pivot_df = feat_df.copy()
        pivot_df["time_bucket"] = pivot_df["timestamp"].dt.floor("1h")
        heat_data = pivot_df.pivot_table(
            index="device_id", columns="time_bucket", values="risk_probability", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * len(heat_data))))
        im = ax.imshow(heat_data.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_yticks(range(len(heat_data.index)))
        ax.set_yticklabels(heat_data.index)
        n_ticks = min(10, heat_data.shape[1])
        tick_positions = np.linspace(0, heat_data.shape[1] - 1, n_ticks).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([heat_data.columns[i].strftime("%m-%d %H:%M") for i in tick_positions],
                            rotation=45, ha="right")
        ax.set_title("Fleet-Wide Downtime Risk Heatmap (hourly average)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Device")
        fig.colorbar(im, ax=ax, label="Risk Probability")
        fig.tight_layout()
        path1 = os.path.join(OUTPUT_DIR, "1_fleet_risk_heatmap.png")
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        saved.append(path1)
    except Exception as e:
        print(f"  (skipped fleet heatmap: {e})")

    # ---- Chart 2: Risk timeline for the highest-risk device ----
    try:
        riskiest_device = feat_df.groupby("device_id")["risk_probability"].mean().idxmax()
        dev_df = feat_df[feat_df["device_id"] == riskiest_device].sort_values("timestamp")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dev_df["timestamp"], dev_df["risk_probability"], color="#34495e", linewidth=1.2)
        ax.axhspan(RISK_THRESHOLDS["critical"], 1.0, color=TIER_COLORS["Critical"], alpha=0.15, label="Critical")
        ax.axhspan(RISK_THRESHOLDS["warning"], RISK_THRESHOLDS["critical"], color=TIER_COLORS["Warning"], alpha=0.15, label="Warning")
        ax.axhspan(RISK_THRESHOLDS["watch"], RISK_THRESHOLDS["warning"], color=TIER_COLORS["Watch"], alpha=0.15, label="Watch")
        ax.axhspan(0.0, RISK_THRESHOLDS["watch"], color=TIER_COLORS["Normal"], alpha=0.15, label="Normal")
        ax.set_ylim(0, 1)
        ax.set_title(f"Downtime Risk Timeline — {riskiest_device} (highest average risk)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Predicted Risk Probability")
        ax.legend(loc="upper left", ncol=4, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        path2 = os.path.join(OUTPUT_DIR, "2_risk_timeline_top_device.png")
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        saved.append(path2)
    except Exception as e:
        print(f"  (skipped risk timeline: {e})")

    # ---- Chart 3: Feature importance (top contributors driving predictions) ----
    try:
        importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=True)
        top_n = importances.tail(15)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top_n.index, top_n.values, color="#2980b9")
        ax.set_title("Top 15 Feature Importances (Random Forest)")
        ax.set_xlabel("Importance Score")
        fig.tight_layout()
        path3 = os.path.join(OUTPUT_DIR, "3_feature_importance.png")
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        saved.append(path3)
    except Exception as e:
        print(f"  (skipped feature importance chart: {e})")

    # ---- Chart 4: Sensor trend with rolling band + flagged anomalies ----
    try:
        dev_df = feat_df[feat_df["device_id"] == riskiest_device].sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dev_df["timestamp"], dev_df["cpu_temp"], color="#7f8c8d", linewidth=0.9, label="cpu_temp (actual)")
        ax.plot(dev_df["timestamp"], dev_df["cpu_temp_roll_mean"], color="#2c3e50", linewidth=1.5, label="rolling mean")
        upper = dev_df["cpu_temp_roll_mean"] + dev_df["cpu_temp_roll_std"]
        lower = dev_df["cpu_temp_roll_mean"] - dev_df["cpu_temp_roll_std"]
        ax.fill_between(dev_df["timestamp"], lower, upper, color="#3498db", alpha=0.15, label="±1 std band")

        anomalies = dev_df[dev_df["cpu_temp_zscore"].abs() > 2]
        ax.scatter(anomalies["timestamp"], anomalies["cpu_temp"], color="#e74c3c", s=25,
                   zorder=5, label="anomaly (|z| > 2)")

        ax.set_title(f"CPU Temperature Trend & Anomalies — {riskiest_device}")
        ax.set_xlabel("Time")
        ax.set_ylabel("CPU Temperature (°C)")
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        path4 = os.path.join(OUTPUT_DIR, "4_sensor_trend_anomalies.png")
        fig.savefig(path4, dpi=150)
        plt.close(fig)
        saved.append(path4)
    except Exception as e:
        print(f"  (skipped sensor trend chart: {e})")

    return saved


def main():
    input_path = prompt_for_input_file()
    ensure_model_exists()
    feat_df, clf, feature_cols = run_inference(input_path)
    print_summary(feat_df)

    print("\nGenerating visualizations ...")
    saved_charts = make_visualizations(feat_df, clf, feature_cols)

    # also save the full scored dataset as CSV for further analysis
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scored_path = os.path.join(OUTPUT_DIR, "scored_results.csv")
    feat_df[["timestamp", "device_id", "risk_probability", "risk_tier"]].to_csv(scored_path, index=False)

    print(f"\nSaved {len(saved_charts)} chart(s) to: {OUTPUT_DIR}")
    for p in saved_charts:
        print(f"  - {os.path.basename(p)}")
    print(f"\nFull scored results saved to: {scored_path}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
