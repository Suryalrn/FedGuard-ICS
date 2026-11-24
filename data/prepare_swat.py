
import pandas as pd
from pathlib import Path


from load_swat import (
    NORMAL_FILE,
    ATTACK_FILE,
    load_swat_file
)



OUTPUT_DIR = Path("data/interim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_labels(df, dataset_name):

    df = df.copy()

    # Remove leading/trailing spaces and internal whitespace
    original_labels = df["Normal/Attack"].astype(str)

    normalized = (
        original_labels
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.lower()
    )

    label_mapping = {
        "normal": 0,
        "attack": 1
    }

    unknown = sorted(set(normalized.unique()) - set(label_mapping))

    if unknown:
        raise ValueError(
            f"Unknown labels in {dataset_name}: {unknown}"
        )

    df["Label"] = normalized.map(label_mapping).astype("int64")

    print(f"\n{dataset_name} label verification")
    print("Original labels:")
    print(original_labels.value_counts())

    print("\nCleaned labels:")
    print(df["Label"].value_counts().rename(
        index={0: "Normal", 1: "Attack"}
    ))

    return df


def main():

    normal_df = load_swat_file(NORMAL_FILE)
    attack_df = load_swat_file(ATTACK_FILE)

    normal_df = clean_labels(normal_df, "Normal dataset")
    attack_df = clean_labels(attack_df, "Attack dataset")

    # Preserve the original dataset source.
    normal_df["Source"] = "Normal_v1"
    attack_df["Source"] = "Attack_v0"

    # Verify that the normal file contains only normal records.
    if not (normal_df["Label"] == 0).all():
        raise ValueError("Unexpected attack labels in Normal v1.")

    # Check feature consistency.
    normal_features = [
        c for c in normal_df.columns
        if c not in ["Timestamp", "Normal/Attack", "Label", "Source"]
    ]

    attack_features = [
        c for c in attack_df.columns
        if c not in ["Timestamp", "Normal/Attack", "Label", "Source"]
    ]

    if normal_features != attack_features:
        raise ValueError("Feature columns do not match.")

    print("\nFeature count:", len(normal_features))

    # Keep the two datasets separate at this stage.
    normal_df.to_parquet(
        OUTPUT_DIR / "normal_clean.parquet",
        index=False
    )

    attack_df.to_parquet(
        OUTPUT_DIR / "attack_clean.parquet",
        index=False
    )

    print("\nClean datasets saved successfully.")


if __name__ == "__main__":
    main()
