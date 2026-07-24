import pandas as pd
import numpy as np
import os

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv"
)


def load_data(path: str = None) -> pd.DataFrame:
    """Load raw CSV into a DataFrame. Defaults to the bundled dataset regardless of cwd."""
    if path is None:
        path = DEFAULT_DATA_PATH
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges has some blank strings for customers with 0 tenure; fix that
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop customer ID — not predictive, just an identifier
    df = df.drop(columns=["customerID"])

    # Standardize target column to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # SeniorCitizen is already 0/1 but stored as int64 — leave as-is
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few business-relevant derived features."""
    df = df.copy()

    # Tenure buckets — helps capture non-linear churn risk by customer lifecycle stage
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
        include_lowest=True,
    )

    # Average monthly spend vs total — flags customers paying more than their tenure average
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Count of add-on services subscribed to (proxy for engagement/stickiness)
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_addon_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # Flag customers with no internet-dependent add-ons at all (low engagement)
    df["is_low_engagement"] = (df["num_addon_services"] == 0).astype(int)

    return df


def get_processed_data(path: str = None) -> pd.DataFrame:
    """Convenience function: load -> clean -> engineer, in one call."""
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = get_processed_data()
    print(f"Shape after processing: {df.shape}")
    print(df.head())
    print(f"\nChurn rate: {df['Churn'].mean():.2%}")
