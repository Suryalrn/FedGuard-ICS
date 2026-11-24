
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler

try:
    from src.data.load_swat import (
        NORMAL_FILE,
        ATTACK_FILE,
        load_swat_file
    )
except ImportError:
    from load_swat import (
        NORMAL_FILE,
        ATTACK_FILE,
        load_swat_file
    )


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_TRAIN_RATIO = 0.80
INNER_TRAIN_RATIO = 0.80

# Keep the previously selected Attack v0 boundary method.


# ============================================================
# LABEL PREPARATION
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    labels = (
        df["Normal/Attack"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.lower()
    )

    df["Label"] = labels.map({
        "normal": 0,
        "attack": 1
    })

    if df["Label"].isna().any():
        raise ValueError("Unexpected label found.")

    df["Label"] = df["Label"].astype(np.int64)

    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


# ============================================================
# FEATURE IDENTIFICATION
# ============================================================

def get_feature_columns(df):

    excluded = {
        "Timestamp",
        "Normal/Attack",
        "Label",
        "Source"
    }

    features = [
        column for column in df.columns
        if column not in excluded
    ]

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="raise"
        )

    return features


# ============================================================
# CHRONOLOGICAL OUTER SPLIT
# ============================================================

def split_attack_at_transition(df):

    transition_indices = df.index[
        df["Label"].ne(df["Label"].shift())
    ].tolist()

    target = int(len(df) * OUTER_TRAIN_RATIO)

    candidates = [
        index for index in transition_indices
        if 0.60 * len(df) <= index <= 0.90 * len(df)
    ]

    if not candidates:
        raise ValueError(
            "No suitable attack label-transition boundary found."
        )

    split_index = min(
        candidates,
        key=lambda index: abs(index - target)
    )

    return (
        df.iloc[:split_index].copy().reset_index(drop=True),
        df.iloc[split_index:].copy().reset_index(drop=True)
    )


# ============================================================
# INNER CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

def split_train_validation(df):

    split_index = int(len(df) * INNER_TRAIN_RATIO)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid inner split.")

    train_core = df.iloc[:split_index].copy()
    validation = df.iloc[split_index:].copy()

    return (
        train_core.reset_index(drop=True),
        validation.reset_index(drop=True)
    )


# ============================================================
# SCALE AND SAVE ONE PARTITION
# ============================================================

def transform_partition(df, feature_columns, scaler):

    X = scaler.transform(
        df[feature_columns]
    ).astype(np.float32)

    y = df["Label"].to_numpy(dtype=np.int64)

    timestamps = pd.to_datetime(
        df["Timestamp"]
    ).to_numpy()

    return {
        "X": X,
        "y": y,
        "timestamps": timestamps
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading SWaT datasets...")

    normal_df = prepare_dataframe(
        load_swat_file(NORMAL_FILE)
    )

    attack_df = prepare_dataframe(
        load_swat_file(ATTACK_FILE)
    )

    feature_columns = get_feature_columns(normal_df)

    attack_features = get_feature_columns(attack_df)

    if feature_columns != attack_features:
        raise ValueError(
            "Normal and attack feature columns do not match."
        )

    print("Number of features:", len(feature_columns))

    # --------------------------------------------------------
    # Outer train/test split
    # --------------------------------------------------------

    normal_split = int(
        len(normal_df) * OUTER_TRAIN_RATIO
    )

    normal_train_outer = normal_df.iloc[:normal_split].copy()
    normal_test = normal_df.iloc[normal_split:].copy()

    (
        attack_train_outer,
        attack_test
    ) = split_attack_at_transition(attack_df)

    # --------------------------------------------------------
    # Inner train/validation split
    # --------------------------------------------------------

    (
        normal_train_core,
        normal_validation
    ) = split_train_validation(normal_train_outer)

    (
        attack_train_core,
        attack_validation
    ) = split_train_validation(attack_train_outer)

    print("\nROW-LEVEL SPLIT")
    print("Normal training core:", len(normal_train_core))
    print("Normal validation:", len(normal_validation))
    print("Normal test:", len(normal_test))

    print("Attack training core:", len(attack_train_core))
    print("Attack validation:", len(attack_validation))
    print("Attack test:", len(attack_test))

    # --------------------------------------------------------
    # Fit scaler ONLY on training core
    # --------------------------------------------------------

    scaler_training_data = pd.concat(
        [
            normal_train_core[feature_columns],
            attack_train_core[feature_columns]
        ],
        ignore_index=True
    )

    scaler = StandardScaler()

    scaler.fit(scaler_training_data)

    print("\nScaler fitted using training core only.")

    # --------------------------------------------------------
    # Transform each partition independently
    # --------------------------------------------------------

    partitions = {
        "train_normal": normal_train_core,
        "train_attack": attack_train_core,
        "val_normal": normal_validation,
        "val_attack": attack_validation,
        "test_normal": normal_test,
        "test_attack": attack_test
    }

    for name, df in partitions.items():

        transformed = transform_partition(
            df,
            feature_columns,
            scaler
        )

        joblib.dump(
            transformed,
            OUTPUT_DIR / f"{name}.joblib"
        )

        print(
            f"Saved {name}:",
            transformed["X"].shape
        )

    # --------------------------------------------------------
    # Save scaler and feature names
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        OUTPUT_DIR / "standard_scaler.joblib"
    )

    joblib.dump(
        feature_columns,
        OUTPUT_DIR / "feature_columns.joblib"
    )

    print("\nPREPROCESSING COMPLETE")
    print("Scaler fitted only on training core.")
    print("Saved partitions in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
