
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

    df["Label"] = labels.map({
        "normal": 0,
        "attack": 1
    })

    if df["Label"].isna().any():
        raise ValueError("Unexpected label found.")

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Identify every point where the label changes.
    df["Change"] = df["Label"].ne(df["Label"].shift())

    change_indices = df.index[df["Change"]].tolist()

    events = []

    for i, start in enumerate(change_indices):
        end = (
            change_indices[i + 1] - 1
            if i + 1 < len(change_indices)
            else len(df) - 1
        )

        label = df.loc[start, "Label"]

        events.append({
            "Label": "Attack" if label == 1 else "Normal",
            "Start": df.loc[start, "Timestamp"],
            "End": df.loc[end, "Timestamp"],
            "Samples": end - start + 1
        })

    events_df = pd.DataFrame(events)

    attack_events = events_df[
        events_df["Label"] == "Attack"
    ].reset_index(drop=True)

    print("\nTotal attack periods:", len(attack_events))
    print("\nFirst 20 attack periods:")
    print(attack_events.head(20).to_string(index=False))

    print("\nLast 20 attack periods:")
    print(attack_events.tail(20).to_string(index=False))

    print("\nAttack period duration statistics:")
    print(attack_events["Samples"].describe())


if __name__ == "__main__":
    main()
