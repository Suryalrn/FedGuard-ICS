
import os
import numpy as np
import pandas as pd
import joblib


# ==========================================
# 1. Load data
# ==========================================

DATA_DIR = "data/processed"

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

scaler = joblib.load(
    os.path.join(DATA_DIR, "standard_scaler.joblib")
)

features = joblib.load(
    os.path.join(DATA_DIR, "feature_columns.joblib")
)


# ==========================================
# 2. Convert standardized values back
# ==========================================

X_train_raw = scaler.inverse_transform(X_train)
X_test_raw = scaler.inverse_transform(X_test)


# ==========================================
# 3. Features to investigate
# ==========================================

selected_features = [
    "P204",
    "P206",
    "P403",
    "P301",
    "P402",
    "AIT401"
]


# ==========================================
# 4. Inspect values and labels
# ==========================================

for feature in selected_features:

    if feature not in features:
        print(f"\nFeature not found: {feature}")
        continue

    idx = list(features).index(feature)

    train_values = X_train_raw[:, idx]
    test_values = X_test_raw[:, idx]

    print("\n" + "=" * 60)
    print("FEATURE:", feature)
    print("=" * 60)

    print("\nTraining unique values and counts:")

    train_unique, train_counts = np.unique(
        train_values,
        return_counts=True
    )

    for value, count in zip(train_unique, train_counts):

        print(
            f"Value: {value:.6f} | Count: {count}"
        )

    print("\nTest unique values and counts:")

    test_unique, test_counts = np.unique(
        test_values,
        return_counts=True
    )

    for value, count in zip(test_unique, test_counts):

        print(
            f"Value: {value:.6f} | Count: {count}"
        )

    print("\nTest values grouped by actual label:")

    for label in np.unique(y_test):

        mask = y_test == label

        print(
            f"Label {label}:",
            np.unique(test_values[mask], return_counts=True)
        )


print("\nFeature inspection completed.")
