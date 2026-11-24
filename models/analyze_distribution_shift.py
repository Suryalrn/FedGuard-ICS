
import os
import numpy as np
import pandas as pd
import joblib

from scipy.stats import ks_2samp


# --------------------------------------------------
# 1. Load row-level data
# --------------------------------------------------

DATA_DIR = "data/processed"

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

feature_columns = joblib.load(
    os.path.join(DATA_DIR, "feature_columns.joblib")
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Number of features:", len(feature_columns))


# --------------------------------------------------
# 2. Validate dimensions
# --------------------------------------------------

if X_train.ndim != 2 or X_test.ndim != 2:
    raise ValueError("Expected two-dimensional row-level arrays.")

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Train and test feature counts do not match.")

if len(feature_columns) != X_train.shape[1]:
    raise ValueError("Feature names do not match array columns.")


# --------------------------------------------------
# 3. Calculate feature statistics
# --------------------------------------------------

results = []

for i, feature in enumerate(feature_columns):

    train_values = X_train[:, i]
    test_values = X_test[:, i]

    train_mean = np.mean(train_values)
    test_mean = np.mean(test_values)

    train_std = np.std(train_values)
    test_std = np.std(test_values)

    # Standardized mean difference
    pooled_std = np.sqrt(
        (train_std ** 2 + test_std ** 2) / 2
    )

    if pooled_std > 0:
        mean_difference = abs(
            train_mean - test_mean
        ) / pooled_std
    else:
        mean_difference = 0.0

    # Kolmogorov-Smirnov distribution comparison
    ks_statistic, ks_pvalue = ks_2samp(
        train_values,
        test_values
    )

    results.append({
        "Feature": feature,
        "Train_Mean": train_mean,
        "Test_Mean": test_mean,
        "Train_Std": train_std,
        "Test_Std": test_std,
        "Standardized_Mean_Difference": mean_difference,
        "KS_Statistic": ks_statistic,
        "KS_P_Value": ks_pvalue
    })


# --------------------------------------------------
# 4. Sort features by distribution difference
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Standardized_Mean_Difference",
    ascending=False
)

print("\nTOP 15 FEATURES WITH LARGEST DISTRIBUTION DIFFERENCE")

print(
    results_df.head(15).to_string(index=False)
)


# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

output_path = (
    "reports/feature_distribution_shift.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nAnalysis saved to:", output_path)


# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

print("\nDISTRIBUTION SHIFT SUMMARY")

print(
    "Features with standardized mean difference > 0.5:",
    (results_df["Standardized_Mean_Difference"] > 0.5).sum()
)

print(
    "Features with standardized mean difference > 1.0:",
    (results_df["Standardized_Mean_Difference"] > 1.0).sum()
)

print(
    "Features with KS statistic > 0.2:",
    (results_df["KS_Statistic"] > 0.2).sum()
)
