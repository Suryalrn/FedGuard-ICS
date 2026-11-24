
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/SWaT")

FILES = [
    "SWaT_Dataset_Normal_v1.xlsx",
    "SWaT_Dataset_Attack_v0.xlsx"
]

for filename in FILES:

    path = DATA_DIR / filename

    print("\n" + "=" * 90)
    print("FILE:", filename)
    print("=" * 90)

    # Read without assuming any row is the header
    df = pd.read_excel(
        path,
        header=None,
        nrows=8,
        engine="openpyxl"
    )

    # Display the first 8 original rows
    print("\nFIRST 8 ROWS:")
    print(df.to_string(index=False, header=False))

    # Display first column values
    print("\nFIRST COLUMN:")
    print(df.iloc[:, 0].tolist())

    # Display last column values
    print("\nLAST COLUMN:")
    print(df.iloc[:, -1].tolist())

    print("\nTOTAL COLUMNS:", df.shape[1])
