
from pathlib import Path
from openpyxl import load_workbook

# Dataset directory
DATA_DIR = Path("data/raw/SWaT")

FILES =[
    "SWaT_Dataset_Normal_v1.xlsx",
    "SWaT_Dataset_Attack_v0.xlsx"
]

def inspect_excel(file_name):
    file_path = DATA_DIR / file_name

    print("\n" + "=" * 70)
    print(f"INSPECTING: {file_name}")
    print("=" * 70)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    # Read workbook without loading the entire file into memory
    workbook = load_workbook(
        filename=file_path,
        read_only=True,
        data_only=True
    )

    print("Sheets:", workbook.sheetnames)

    for sheet in workbook.worksheets:
        print(f"\nSheet: {sheet.title}")
        print("Rows:", sheet.max_row)
        print("Columns:", sheet.max_column)

        # Display first 8 rows
        for row in sheet.iter_rows(
            min_row=1,
            max_row=min(8, sheet.max_row),
            values_only=True
        ):
            print(row)

    workbook.close()


def main():
    for file_name in FILES:
        inspect_excel(file_name)


if __name__ == "__main__":
    main()
