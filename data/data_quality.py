
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw/SWaT")

FILES = {
    "Normal": "SWaT_Dataset_Normal_v1.xlsx",
    "Attack": "SWaT_Dataset_Attack_v0.xlsx"
}


def inspect_dataset(name, filename):

    path = DATA_DIR / filename

    print("\n" + "=" * 75)
    print(f"DATA QUALITY REPORT: {name}")
    print("=" * 75)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    df = pd.read_excel(path, engine="openpyxl")

    # 1. Dataset dimensions
    print("\n1. DATASET SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # 2. Column names
    print("\n2. COLUMN NAMES")
    for col in df.columns:
        print(col)

    # 3. Data types
    print("\n3. DATA TYPES")
    print(df.dtypes)

    # 4. Missing values
    print("\n4. MISSING VALUES")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    if missing.sum() == 0:
        print("No missing values found.")

    # 5. Duplicate rows
    print("\n5. DUPLICATE ROWS")
    print("Duplicates:", df.duplicated().sum())

    # 6. Numerical statistics
    print("\n6. NUMERICAL STATISTICS")
    print(df.describe().T)

    # 7. Potential label columns
    print("\n7. POTENTIAL LABEL COLUMNS")

    for col in df.columns:
        if any(word in str(col).lower()
               for word in ["label", "attack", "normal", "state"]):
            print(f"\nColumn: {col}")
            print(df[col].value_counts(dropna=False).head(20))

    # 8. Save report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    safe_name = name.lower().replace(" ", "_")

    with open(report_dir / f"{safe_name}_quality.txt",
              "w", encoding="utf-8") as f:

        f.write(f"Dataset: {name}\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Column information:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\nMissing values:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\nDuplicate rows:\n")
        f.write(str(df.duplicated().sum()))
        f.write("\n\nStatistics:\n")
        f.write(df.describe().T.to_string())

    print("\nReport saved successfully.")

    return df


def main():

    for name, filename in FILES.items():
        inspect_dataset(name, filename)


if __name__ == "__main__":
    main()
