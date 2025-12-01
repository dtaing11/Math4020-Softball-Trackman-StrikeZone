
# train_strikezone_5feat.py
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from Strickzone_Predictve_Model.utils.model_weight_surgery import (
    load_and_upgrade_checkpoint_to_5_inputs,
    save_checkpoint,
    DEVICE,
)

# -------------------- Config --------------------
CSV_PATH = "../datasets/Trackman_Softball_Data.csv"   # <-- put your CSV path here
MODEL_PATH = "artifacts/strikezone_model.pt" # existing 3-input checkpoint
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

# Feature order for the new 5-dim input:
# [PlateLocHeight, PlateLocSide, Swing, BatterHand, PitcherHand]
FEATURE_NAMES = ["PlateLocHeight", "PlateLocSide", "Swing", "BatterHand", "PitcherHand"]
LABEL_COLUMN = "In Zone"  # target


# -------------------- Your model (no fallback) --------------------
try:
    from model import StrikeZonePredictionModel as SZP
except Exception as e:
    print("[ERROR] Could not import StrikeZonePredictionModel from model.py")
    print(f"       {e}")
    sys.exit(1)


def build_core_model_5in() -> nn.Module:
    """Strictly use your StrikeZonePredictionModel with 5 inputs."""
    try:
        return SZP(input_dim=5, output_dim=1)
    except TypeError:
        # If your class ignores input_dim/output_dim, just call it.
        return SZP()


# -------------------- Dataset --------------------
class StrikeZoneDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- Data loading & preprocessing --------------------
def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build:
      X_raw: [height, side, swing, batter_hand, pitcher_hand]
      y:     In Zone (0/1)
    """
    df = pd.read_csv(csv_path)

    # Core geometry features
    height = df["PlateLocHeight"].astype(float)
    side = df["PlateLocSide"].astype(float)

    # Swing and label
    swing = df["Swing"].astype(float)
    in_zone = df["In Zone"].astype(float)

    # Handedness
    batter_side_raw = df["BatterSide"].astype(str).str.upper().str.strip()
    pitcher_throws_raw = df["PitcherThrows"].astype(str).str.upper().str.strip()

    # Encode: 1 = Left, 0 = Right (or anything else)
    batter_hand = (batter_side_raw.str.startswith("L")).astype(np.float32)
    pitcher_hand = (pitcher_throws_raw.str.startswith("L")).astype(np.float32)

    # Drop rows with missing critical info
    mask = (
        height.notna()
        & side.notna()
        & swing.notna()
        & in_zone.notna()
    )
    height = height[mask]
    side = side[mask]
    swing = swing[mask]
    batter_hand = batter_hand[mask]
    pitcher_hand = pitcher_hand[mask]
    in_zone = in_zone[mask]

    X_raw = np.stack(
        [
            height.values,
            side.values,
            swing.values,
            batter_hand.values,
            pitcher_hand.values,
        ],
        axis=1,
    )
    y = in_zone.values
    return X_raw, y


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = std.copy()
    std[std == 0.0] = 1.0
    return (X - mean) / std


# -------------------- Train / Eval --------------------
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    return avg_loss, acc


# -------------------- Main --------------------
def main():
    # 1) Load dataset
    print(f"[INFO] Loading dataset from {CSV_PATH}")
    X_raw, y = load_dataset(CSV_PATH)
    print(f"[INFO] Loaded {X_raw.shape[0]} samples, feature dim = {X_raw.shape[1]}")

    # 2) Load + upgrade existing checkpoint (3 → 5) OR start fresh if none
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Found existing checkpoint at {MODEL_PATH}, upgrading to 5 inputs.")
        core_model, ckpt, x_mean, x_std = load_and_upgrade_checkpoint_to_5_inputs(
            MODEL_PATH,
            builder_5in=build_core_model_5in,
        )
    else:
        print("[INFO] No existing checkpoint found. Initializing new 5-input model.")
        core_model = build_core_model_5in().to(DEVICE)
        # Compute scaler from data
        x_mean = X_raw.mean(axis=0, keepdims=True).astype(np.float32)
        x_std = X_raw.std(axis=0, keepdims=True).astype(np.float32)
        x_std[x_std == 0.0] = 1.0
        ckpt = {
            "state_dict": core_model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
        }

    # 3) Standardize using scaler (from upgraded or fresh ckpt)
    X_std = standardize(X_raw, x_mean, x_std)

    # 4) Train/val split
    n = X_std.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X_std[train_idx], y[train_idx]
    X_val, y_val = X_std[val_idx], y[val_idx]

    train_ds = StrikeZoneDataset(X_train, y_train)
    val_ds = StrikeZoneDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5) Loss / optimizer
    model = core_model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 6) Training loop (fine-tuning on top of last checkpoint)
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

    # 7) Save updated checkpoint with new weights + scalers
    ckpt["state_dict"] = model.state_dict()
    ckpt["x_mean"] = x_mean
    ckpt["x_std"] = x_std
    save_checkpoint(ckpt, MODEL_PATH)
    print(f"[INFO] Saved updated checkpoint to {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
