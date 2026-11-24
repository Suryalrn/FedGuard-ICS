
import os
import numpy as np
import pandas as pd
import joblib


# ==========================================
# 1. Load row-level datasets
# ==========================================

DATA_DIR = "data/processed"

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

features = joblib.load(
    os.path.join(DATA_DIR, "feature_columns.joblib")
)


# ==========================================
# 2. Calculate statistics
# ==========================================

results = []

for i, feature in enumerate(features):

    train = X_train[:, i]
    test = X_test[:, i]

    results.append({
        "Feature": feature,

        "Train_Mean": np.mean(train),
        "Test_Mean": np.mean(test),

        "Train_Std": np.std(train),
        "Test_Std": np.std(test),

        "Train_Min": np.min(train),
        "Test_Min": np.min(test),

        "Train_Max": np.max(train),
        "Test_Max": np.max(test),

        "Train_Unique": np.unique(train).size,
        "Test_Unique": np.unique(test).size,

        "Test_Std_Ratio": (
            np.std(test) / np.std(train)
            if np.std(train) > 0 else np.nan
        )
    })


df = pd.DataFrame(results)


# ==========================================
# 3. Find almost constant test features
# ==========================================

df = df.sort_values(
    "Test_Std_Ratio",
    ascending=True
)

print("\nLOWEST TEST VARIABILITY FEATURES")

print(
    df[
        [
            "Feature",
            "Train_Std",
            "Test_Std",
            "Test_Std_Ratio",
            "Train_Unique",
            "Test_Unique"
        ]
    ].head(15).to_string(index=False)
)


# ==========================================
# 4. Save report
# ==========================================

os.makedirs("reports", exist_ok=True)

output = "reports/sensor_stability_analysis.csv"

df.to_csv(output, index=False)

print("\nSaved:", output)


# ==========================================
# 5. Summary
# ==========================================

print("\nSUMMARY")

print(
    "Features with test standard deviation ratio below 0.01:",
    (df["Test_Std_Ratio"] < 0.01).sum()
)

print(
    "Features with test standard deviation ratio below 0.10:",
    (df["Test_Std_Ratio"] < 0.10).sum()
)
