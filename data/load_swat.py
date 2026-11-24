
import pandas as pd
from pathlib import Path

# Project paths
DATA_DIR = Path("data/raw/SWaT")

NORMAL_FILE = DATA_DIR /  "SWaT_Dataset_Normal_v1.xlsx"
ATTACK_FILE = DATA_DIR / "SWaT_Dataset_Attack_v0.xlsx"


def load_swat_file(file_path):

    print(f"\nLoading: {file_path.name}")

    # Row 2 (index 1) contains the actual column names
    df = pd.read_excel(
        file_path,
        header=1,
        engine="openpyxl"
    )

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Remove unwanted spaces from text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # Convert sensor and actuator columns to numeric
    excluded = ["Timestamp", "Normal/Attack"]

    feature_columns = [
        col for col in df.columns
        if col not in excluded
    ]

    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("Timestamp errors:", df["Timestamp"].isna().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nLabel distribution:")
    print(df["Normal/Attack"].value_counts(dropna=False))

    print("\nMissing values after conversion:")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\nFirst 5 records:")
    print(df.head())

    return df


def main():

    normal_df = load_swat_file(NORMAL_FILE)
    attack_df = load_swat_file(ATTACK_FILE)

    print("\n" + "=" * 65)
    print("SWaT DATASET LOADING COMPLETED")
    print("=" * 65)

    print("Normal shape:", normal_df.shape)
    print("Attack shape:", attack_df.shape)

    print("\nNormal timestamps:")
    print(normal_df["Timestamp"].min())
    print(normal_df["Timestamp"].max())

    print("\nAttack timestamps:")
    print(attack_df["Timestamp"].min())
    print(attack_df["Timestamp"].max())


if __name__ == "__main__":
    main()
