
import pandas as pd

from load_swat import ATTACK_FILE, load_swat_file


def main():
    df = load_swat_file(ATTACK_FILE)

    # Normalize labels, including the 37 "A ttack" entries.
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

    df = df.sort_values("Timestamp").reset_index(drop=True)

    print("\nOverall label distribution:")
    print(df["Label"].value_counts().rename(
        index={0: "Normal", 1: "Attack"}
    ))

    # Divide the recording into 10 chronological sections.
    print("\nLabel distribution across 10 chronological sections:")

    df["TimeSection"] = pd.qcut(
        df.index,
        q=10,
        labels=False
    )

    for section, group in df.groupby("TimeSection", observed=True):
        normal_count = (group["Label"] == 0).sum()
        attack_count = (group["Label"] == 1).sum()

        print(
            f"Section {section + 1}: "
            f"{group['Timestamp'].iloc[0]} to "
            f"{group['Timestamp'].iloc[-1]} | "
            f"Normal={normal_count:,}, "
            f"Attack={attack_count:,}"
        )

    # Find the first and last attack-labelled observations.
    attack_rows = df[df["Label"] == 1]

    print("\nAttack timestamp range:")
    print("First attack:", attack_rows["Timestamp"].min())
    print("Last attack: ", attack_rows["Timestamp"].max())

    # Count changes between consecutive labels.
    changes = df["Label"].ne(df["Label"].shift()).sum()

    print("\nNumber of label transitions:", changes)


if __name__ == "__main__":
    main()
