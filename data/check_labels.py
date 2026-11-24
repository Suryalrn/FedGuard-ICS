import pandas as pd

from load_swat import (
    NORMAL_FILE,
    ATTACK_FILE,
    load_swat_file
)


def check_labels(df, name):

    print("\n" + "=" * 60)
    print(name, "LABEL DISTRIBUTION")
    print("=" * 60)

    counts = df["Normal/Attack"].value_counts(
        dropna=False
    )

    print(counts)

    print("\nPercentages:")

    print(
        df["Normal/Attack"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(3)
    )


normal_df = load_swat_file(NORMAL_FILE)
attack_df = load_swat_file(ATTACK_FILE)

check_labels(normal_df, "NORMAL")
check_labels(attack_df, "ATTACK")
