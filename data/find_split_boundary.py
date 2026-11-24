
import pandas as pd

from load_swat import ATTACK_FILE, load_swat_file


def main():
    df = load_swat_file(ATTACK_FILE)

    labels = (
        df["Normal/Attack"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.lower()
    )

    df["Label"] = labels.map({"normal": 0, "attack": 1})

    if df["Label"].isna().any():
        raise ValueError("Unexpected label found.")

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Identify boundaries where the label changes.
    transition_indices = df.index[
        df["Label"].ne(df["Label"].shift())
    ].tolist()

    # Approximate 80% chronological split.
    target = int(len(df) * 0.80)

    # Choose the closest label-transition boundary.
    candidates = [
        i for i in transition_indices
        if 0.60 * len(df) <= i <= 0.90 * len(df)
    ]

    split_index = min(candidates, key=lambda i: abs(i - target))

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    print("\nSPLIT INFORMATION")
    print("Total rows:", len(df))
    print("Target 80% index:", target)
    print("Selected split index:", split_index)
    print("Training percentage:", round(len(train) / len(df) * 100, 2))
    print("Testing percentage:", round(len(test) / len(df) * 100, 2))

    print("\nTRAINING PERIOD")
    print("Start:", train["Timestamp"].iloc[0])
    print("End:", train["Timestamp"].iloc[-1])
    print("Label counts:")
    print(train["Label"].value_counts().sort_index())

    print("\nTESTING PERIOD")
    print("Start:", test["Timestamp"].iloc[0])
    print("End:", test["Timestamp"].iloc[-1])
    print("Label counts:")
    print(test["Label"].value_counts().sort_index())

    print("\nBOUNDARY CHECK")
    print("Last training label:", train["Label"].iloc[-1])
    print("First testing label:", test["Label"].iloc[0])
    print("Last training timestamp:", train["Timestamp"].iloc[-1])
    print("First testing timestamp:", test["Timestamp"].iloc[0])


if __name__ == "__main__":
    main()
