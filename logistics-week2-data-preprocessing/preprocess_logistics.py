"""
Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Uses a small SIMULATED sample dataset for demonstration. Replace it with an authorized
real dataset before making real-world conclusions.
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "logistics_raw_sample.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Rows loaded: {len(df)}")
    print("Missing values before cleaning:")
    print(df.isna().sum())

    # Convert dates safely; invalid values become NaT.
    for col in ["order_date", "delivery_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric columns to numeric values; invalid text becomes NaN.
    numeric_cols = ["distance_km", "delivery_time_hours", "delivery_cost_inr"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove exact duplicate rows.
    before = len(df)
    df = df.drop_duplicates()
    print(f"Exact duplicate rows removed: {before - len(df)}")

    # Treat impossible negative distance/time/cost as missing for review.
    for col in numeric_cols:
        df.loc[df[col] < 0, col] = pd.NA

    # Flag impossible date order instead of silently deleting the record.
    df["date_order_valid"] = (
        df["order_date"].notna()
        & df["delivery_date"].notna()
        & (df["delivery_date"] >= df["order_date"])
    )

    # Fill missing numeric values with the median, if a median is available.
    # In a real project, confirm whether imputation is appropriate for each field.
    for col in numeric_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    # Flag outliers using the IQR rule; retain rows so they can be reviewed.
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[f"{col}_outlier_flag"] = (df[col] < low) | (df[col] > high)

    # Scale selected numeric features for demonstration; retain original columns too.
    scale_cols = ["distance_km", "delivery_time_hours", "delivery_cost_inr"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    for idx, col in enumerate(scale_cols):
        df[f"{col}_scaled"] = scaled[:, idx]

    # One-hot encode categories in a separate file to show a model-ready representation.
    category_cols = ["vehicle_type", "delivery_status"]
    encoded = pd.get_dummies(df, columns=category_cols, dtype=int)

    cleaned_path = OUTPUT_DIR / "logistics_cleaned.csv"
    encoded_path = OUTPUT_DIR / "logistics_preprocessed_encoded.csv"
    df.to_csv(cleaned_path, index=False, date_format="%Y-%m-%d")
    encoded.to_csv(encoded_path, index=False, date_format="%Y-%m-%d")

    print("\nCleaning summary:")
    print(f"Rows after exact-duplicate removal: {len(df)}")
    print(f"Cleaned dataset saved to: {cleaned_path}")
    print(f"Encoded dataset saved to: {encoded_path}")
    print("\nNote: Review invalid dates and outlier flags before using this data for modelling.")

if __name__ == "__main__":
    main()
