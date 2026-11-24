
import numpy as np
import torch
import joblib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from tcn import FedGuardTCN


# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. Load untouched test sequences
X_test = np.load("data/processed/X_test_seq.npy")
y_test = np.load("data/processed/y_test_seq.npy")

print("Test X shape:", X_test.shape)
print("Test y shape:", y_test.shape)
print("Test labels:", np.unique(y_test, return_counts=True))


# 3. Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# 4. Load model architecture and saved weights
model = FedGuardTCN(
    input_features=51,
    hidden_channels=64,
    num_classes=2,
    dropout=0.2
).to(device)

checkpoint = torch.load(
    "models/best_tcn.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()


# 5. Make predictions
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)
        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(y_batch.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities[:, 1].cpu().numpy())


# 6. Calculate evaluation metrics
print("\n===== TEST RESULTS =====")

print("Accuracy:", accuracy_score(all_labels, all_predictions))
print("Precision:", precision_score(all_labels, all_predictions, zero_division=0))
print("Recall:", recall_score(all_labels, all_predictions, zero_division=0))
print("F1-score:", f1_score(all_labels, all_predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=["Normal", "Attack"],
    zero_division=0
))

if len(np.unique(all_labels)) == 2:
    print(
        "ROC-AUC:",
        roc_auc_score(all_labels, all_probabilities)
    )

    print(
        "PR-AUC:",
        average_precision_score(all_labels, all_probabilities)
    )
