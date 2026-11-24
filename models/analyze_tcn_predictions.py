
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tcn import FedGuardTCN


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. Load test data
# --------------------------------------------------

X_test = np.load("data/processed/X_test_seq.npy")
y_test = np.load("data/processed/y_test_seq.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

model = FedGuardTCN(
    input_features=51,
    hidden_channels=64,
    num_classes=2,
    dropout=0.2
).to(device)

model.load_state_dict(
    torch.load(
        "models/best_tcn.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# --------------------------------------------------
# 4. Get attack probabilities
# --------------------------------------------------

all_labels = []
all_probabilities = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        attack_probabilities = probabilities[:, 1]

        all_probabilities.extend(
            attack_probabilities.cpu().numpy()
        )

        all_labels.extend(
            y_batch.numpy()
        )


all_labels = np.array(all_labels)
all_probabilities = np.array(all_probabilities)


# --------------------------------------------------
# 5. Probability distribution
# --------------------------------------------------

normal_probs = all_probabilities[all_labels == 0]
attack_probs = all_probabilities[all_labels == 1]

print("\nNORMAL PROBABILITY DISTRIBUTION")
print(pd.Series(normal_probs).describe())

print("\nATTACK PROBABILITY DISTRIBUTION")
print(pd.Series(attack_probs).describe())


# --------------------------------------------------
# 6. Threshold experiments
# --------------------------------------------------

print("\nTHRESHOLD ANALYSIS")

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]

results = []

for threshold in thresholds:

    predictions = (
        all_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })


results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv(
    "models/tcn_threshold_analysis.csv",
    index=False
)

print("\nSaved: models/tcn_threshold_analysis.csv")
