
import os
import random
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tcn import FedGuardTCN


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5

INPUT_FEATURES = 51
HIDDEN_CHANNELS = 64
NUM_CLASSES = 2
DROPOUT = 0.2

DATA_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. LOAD SEQUENCE DATA
# ============================================================

X_train = np.load(
    os.path.join(DATA_DIR, "X_train_seq.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train_seq.npy")
)

metadata_path = os.path.join(
    DATA_DIR,
    "train_sequence_metadata.csv"
)

metadata = pd.read_csv(metadata_path)

print("\nLoaded training data")
print("X shape:", X_train.shape)
print("y shape:", y_train.shape)


# ============================================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

def chronological_validation_split(metadata):

    train_indices = []
    val_indices = []

    for _, group in metadata.groupby(
        ["Source", "Segment"]
    ):

        group = group.sort_values("Start")

        split_position = int(len(group) * 0.80)

        if split_position <= 0 or split_position >= len(group):
            continue

        cutoff = group.iloc[split_position]["Start"]

        train_group = group[
            group["End"] < cutoff
        ]

        val_group = group[
            group["Start"] >= cutoff
        ]

        train_indices.extend(train_group.index.tolist())
        val_indices.extend(val_group.index.tolist())

    return (
        np.array(sorted(train_indices), dtype=int),
        np.array(sorted(val_indices), dtype=int)
    )


train_idx, val_idx = chronological_validation_split(metadata)

if len(train_idx) == 0 or len(val_idx) == 0:
    raise ValueError(
        "Chronological split produced an empty training or validation set."
    )

if max(train_idx.max(), val_idx.max()) >= len(X_train):
    raise ValueError(
        "Metadata indices do not match the sequence arrays."
    )

X_train_final = X_train[train_idx]
y_train_final = y_train[train_idx]

X_val = X_train[val_idx]
y_val = y_train[val_idx]

print("\nTraining sequences:", len(X_train_final))
print("Validation sequences:", len(X_val))

print("\nTraining labels:")
print(pd.Series(y_train_final).value_counts().sort_index())

print("\nValidation labels:")
print(pd.Series(y_val).value_counts().sort_index())

if len(np.unique(y_train_final)) < 2:
    raise ValueError("Training data must contain both classes.")

if len(np.unique(y_val)) < 2:
    raise ValueError(
        "Validation data must contain both classes. "
        "Review the chronological split."
    )


# ============================================================
# 5. CONVERT DATA TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_final,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_final,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)


# ============================================================
# 6. CREATE DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 7. INITIALIZE TCN MODEL
# ============================================================

model = FedGuardTCN(
    input_features=INPUT_FEATURES,
    hidden_channels=HIDDEN_CHANNELS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)

print("\nTCN model initialized")


# ============================================================
# 8. CLASS-WEIGHTED LOSS
# ============================================================

class_counts = np.bincount(
    y_train_final,
    minlength=NUM_CLASSES
)

if np.any(class_counts == 0):
    raise ValueError(
        "Both classes must be present in training data."
    )

class_weights = (
    len(y_train_final) / (NUM_CLASSES * class_counts)
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass weights:", class_weights)

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


# ============================================================
# 11. VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_loss += loss.item() * len(y_batch)

            all_labels.extend(
                y_batch.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = total_loss / len(loader.dataset)

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return (
        avg_loss,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 12. TRAINING LOOP WITH BEST-F1 CHECKPOINT
# ============================================================

best_val_f1 = -1.0
epochs_without_improvement = 0

history = []

best_model_path = os.path.join(
    MODEL_DIR,
    "best_tcn.pth"
)

print("\nStarting TCN training...\n")


for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    (
        val_loss,
        precision,
        recall,
        val_f1,
        val_labels,
        val_predictions
    ) = validate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {val_f1:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "precision": precision,
        "recall": recall,
        "f1": val_f1
    })

    # Save model when validation F1 improves
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best F1 model saved!")

    else:

        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ============================================================
# 13. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = os.path.join(
    MODEL_DIR,
    "tcn_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# 14. TRAINING SUMMARY
# ============================================================

print("\n================================")
print("TCN TRAINING COMPLETED")
print("================================")

print("Best validation F1:", round(best_val_f1, 4))
print("Best model saved at:", best_model_path)
print("Training history saved at:", history_path)

print("\nTraining history:")
print(history_df.to_string(index=False))
