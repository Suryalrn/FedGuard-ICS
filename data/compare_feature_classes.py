
import os
import numpy as np
import pandas as pd
import joblib


# ==========================================
# 1. Load datasets
# ==========================================

DATA_DIR = "data/processed"

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

features = joblib.load(
    os.path.join(DATA_DIR, "feature_columns.joblib")
)

scaler = joblib.load(
    os.path.join(DATA_DIR, "standard_scaler.joblib")
)


# ==========================================
# 2. Convert to original feature scale
# ==========================================

X_train = scaler.inverse_transform(X_train)
X_test = scaler.inverse_transform(X_test)


# ==========================================
# 3. Define feature groups
# ==========================================

groups = {
    "Train_Normal": (X_train, y_train, 0),
    "Train_Attack": (X_train, y_train, 1),
    "Test_Normal": (X_test, y_test, 0),
    "Test_Attack": (X_test, y_test, 1)
}


# ==========================================
# 4. Calculate statistics
# ==========================================

results = []

for i, feature in enumerate(features):

    row = {"Feature": feature}

    for group_name, (X, y, label) in groups.items():

        values = X[y == label, i]

        if len(values) == 0:
            continue

        row[group_name + "_Mean"] = np.mean(values)
        row[group_name + "_Std"] = np.std(values)
        row[group_name + "_Min"] = np.min(values)
        row[group_name + "_Max"] = np.max(values)

    results.append(row)


df = pd.DataFrame(results)


# ==========================================
# 5. Focus on previously identified features
# ==========================================

selected = [
    "P204",
    "P206",
    "P403",
    "P301",
    "P402",
    "AIT401",
    "AIT502",
    "AIT203",
    "AIT402",
    "AIT201",
    "AIT202"
]

print("\nSELECTED FEATURE COMPARISON")

print(
    df[df["Feature"].isin(selected)].to_string(index=False)
)


# ==========================================
# 6. Save report
# ==========================================

os.makedirs("reports", exist_ok=True)

output_path = "reports/feature_class_comparison.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
