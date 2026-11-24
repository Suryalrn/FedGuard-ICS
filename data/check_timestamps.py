
import pandas as pd
from pathlib import Path


from load_swat import (
    NORMAL_FILE,
    ATTACK_FILE,
    load_swat_file
)

def inspect_timestamps(df, name):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    timestamps = df["Timestamp"].sort_values()

    print("Rows:", len(df))
    print("Start:", timestamps.iloc[0])
    print("End:", timestamps.iloc[-1])

    differences = timestamps.diff().dropna()

    print("\nMost common time intervals:")
    print(differences.value_counts().head(10))

    print("\nDuplicate timestamps:", timestamps.duplicated().sum())
    print("Missing timestamps:", timestamps.isna().sum())

    print("\nLargest time gaps:")
    print(differences.sort_values(ascending=False).head(10))


def main():
    normal_df = load_swat_file(NORMAL_FILE)
    attack_df = load_swat_file(ATTACK_FILE)

    inspect_timestamps(normal_df, "NORMAL V1")
    inspect_timestamps(attack_df, "ATTACK V0")

    normal_start = normal_df["Timestamp"].min()
    normal_end = normal_df["Timestamp"].max()
    attack_start = attack_df["Timestamp"].min()
    attack_end = attack_df["Timestamp"].max()

    print("\nOVERLAP CHECK")
    print("Normal ends:", normal_end)
    print("Attack starts:", attack_start)
    print("Time ranges overlap:", max(normal_start, attack_start)
          <= min(normal_end, attack_end))


if __name__ == "__main__":
    main()