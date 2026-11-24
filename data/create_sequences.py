
import numpy as np
import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")

SEQUENCE_LENGTH = 60
STRIDE = 30


# ============================================================
# SPLIT RECORDINGS AT TIMESTAMP GAPS
# ============================================================

def split_at_time_gaps(timestamps):

    timestamps = pd.to_datetime(timestamps)

    differences = pd.Series(timestamps).diff()

    segment_ids = (
        differences.ne(pd.Timedelta(seconds=1))
    ).cumsum()

    return [
        np.asarray(indices, dtype=int)
        for _, indices in pd.Series(
            np.arange(len(timestamps))
        ).groupby(segment_ids)
    ]


# ============================================================
# CREATE SEQUENCES FOR ONE PARTITION
# ============================================================

def create_windows(data, source, partition):

    X = data["X"]
    y = data["y"]
    timestamps = data["timestamps"]

    all_X = []
    all_y = []
    metadata = []

    segments = split_at_time_gaps(timestamps)

    for segment_number, indices in enumerate(segments):

        if len(indices) < SEQUENCE_LENGTH:
            continue

        for start in range(
            0,
            len(indices) - SEQUENCE_LENGTH + 1,
            STRIDE
        ):

            window_indices = indices[
                start:start + SEQUENCE_LENGTH
            ]

            first_index = window_indices[0]
            last_index = window_indices[-1]

            all_X.append(
                X[window_indices]
            )

            # Label from final timestep
            all_y.append(
                y[last_index]
            )

            metadata.append({
                "Partition": partition,
                "Source": source,
                "Segment": segment_number,
                "Start": timestamps[first_index],
                "End": timestamps[last_index]
            })

    if not all_X:
        return None

    return (
        np.stack(all_X).astype(np.float32),
        np.asarray(all_y, dtype=np.int64),
        pd.DataFrame(metadata)
    )


# ============================================================
# CREATE COMBINED DATA FOR A PARTITION
# ============================================================

def create_partition(partition):

    sources = [
        "normal",
        "attack"
    ]

    results = []

    for source in sources:

        filename = (
            f"{partition}_{source}.joblib"
        )

        data = joblib.load(
            DATA_DIR / filename
        )

        result = create_windows(
            data,
            source,
            partition
        )

        if result is not None:
            results.append(result)

    if not results:
        raise ValueError(
            f"No sequences created for {partition}"
        )

    X = np.concatenate(
        [result[0] for result in results]
    )

    y = np.concatenate(
        [result[1] for result in results]
    )

    metadata = pd.concat(
        [result[2] for result in results],
        ignore_index=True
    )

    return X, y, metadata


# ============================================================
# SAVE PARTITION
# ============================================================

def save_partition(name, result):

    X, y, metadata = result

    np.save(
        DATA_DIR / f"X_{name}_seq.npy",
        X
    )

    np.save(
        DATA_DIR / f"y_{name}_seq.npy",
        y
    )

    metadata.to_csv(
        DATA_DIR / f"{name}_sequence_metadata.csv",
        index=False
    )

    print(f"\n{name.upper()} SEQUENCES")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Labels:")
    print(
        pd.Series(y).value_counts().sort_index()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Creating training sequences...")
    train = create_partition("train")

    print("\nCreating validation sequences...")
    validation = create_partition("val")

    print("\nCreating test sequences...")
    test = create_partition("test")

    save_partition("train", train)
    save_partition("val", validation)
    save_partition("test", test)

    print("\nSEQUENCE GENERATION COMPLETE")
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Stride:", STRIDE)
    print("Features:", train[0].shape[2])

    print("\nAll sequence files saved in:")
    print(DATA_DIR.resolve())


if __name__ == "__main__":
    main()
