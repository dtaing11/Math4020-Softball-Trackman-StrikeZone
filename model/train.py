# train_torch.py
"""
Train a binary classifier to predict if a pitch is called a strike,
using PlateLocHeight, PlateLocSide, and Swing as inputs.

The script:
- loads and cleans the dataset
- performs a stratified train/validation split (no scikit-learn)
- standardizes features (fit on train only)
- builds a model (uses StrikeZonePredictionModel if available, else a simple MLP)
- trains with BCEWithLogitsLoss and class-imbalance pos_weight
- early-stops on validation loss
- reports validation metrics and saves artifacts (model + loss curves)
- provides a small inference helper
"""

import os
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import StrikeZonePredictionModel as SZP  # expected local module

# -------------------- constants --------------------
STRIKE_CALLED = "StrikeCalled"
PITCH_CALLED = "PitchCall"
IS_STRIKE_CALLED = "is_Strike_Called"
STRIKEZONE_DATA_FILE_PATH = os.environ.get("STRIKEZONE_DATA", "../datasets/StrikeZoneData.csv")

PLAT_LOC_HEIGHT, PLAT_LOC_SIDE, SWING = "PlateLocHeight", "PlateLocSide", "Swing"
FEATURES = [PLAT_LOC_HEIGHT, PLAT_LOC_SIDE, SWING]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG_SEED = int(os.environ.get("SEED", 42))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
LR = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-5))
EPOCHS = int(os.environ.get("EPOCHS", 200))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.2))
PATIENCE = int(os.environ.get("PATIENCE", 15))
SAVE_DIR = os.environ.get("SAVE_DIR", "artifacts")

os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
if torch.cuda.is_available():
    # prefer higher precision matmul on supported GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# -------------------- load and clean dataset --------------------
df = pd.read_csv(STRIKEZONE_DATA_FILE_PATH)

# ensure required feature columns exist and are numeric
for col in FEATURES:
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")

if PITCH_CALLED not in df.columns:
    raise SystemExit(f"Missing required column: {PITCH_CALLED}")

# encode label: 1 for called strike, 0 otherwise
df[IS_STRIKE_CALLED] = (df[PITCH_CALLED] == STRIKE_CALLED).astype(int)

# drop rows with missing feature(s) or label
df = df.dropna(subset=FEATURES + [IS_STRIKE_CALLED]).reset_index(drop=True)

X_np = df[FEATURES].to_numpy(dtype=np.float32)
y_np = df[IS_STRIKE_CALLED].to_numpy(dtype=np.int64)  # (N,)

# -------------------- stratified split (no sklearn) --------------------
def stratified_split_indices(y: np.ndarray, val_ratio: float, seed: int = 42):
    """
    Return train/val indices for a stratified split.

    Args:
        y: array of class labels (shape: [N,])
        val_ratio: fraction for validation set (0 < val_ratio < 1)
        seed: RNG seed for reproducibility

    Returns:
        (train_idx, val_idx): two 1D numpy arrays of indices
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    train_idx, val_idx = [], []
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        split_c = int(round((1.0 - val_ratio) * len(idx_c)))
        train_idx.append(idx_c[:split_c])
        val_idx.append(idx_c[split_c:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


train_idx, val_idx = stratified_split_indices(y_np, VAL_SPLIT, RNG_SEED)

X_train, y_train = X_np[train_idx], y_np[train_idx]
X_val, y_val = X_np[val_idx], y_np[val_idx]

# -------------------- standardize features (fit on train only) --------------------
x_mean = X_train.mean(axis=0, keepdims=True)
x_std = X_train.std(axis=0, keepdims=True)
# avoid division by zero for constant columns
x_std[x_std == 0.0] = 1.0

X_train_std = (X_train - x_mean) / x_std
X_val_std = (X_val - x_mean) / x_std

# -------------------- dataset / dataloader --------------------
class SimpleArrayDataset(Dataset):
    """Thin tensor wrapper over numpy arrays for supervised learning."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # BCEWithLogitsLoss expects float targets with shape (N, 1)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


train_ds = SimpleArrayDataset(X_train_std, y_train)
val_ds = SimpleArrayDataset(X_val_std, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# -------------------- model resolution --------------------
def build_fallback_mlp(input_dim: int = 3, output_dim: int = 1) -> nn.Module:
    """
    Build a simple MLP used if StrikeZonePredictionModel is unavailable.
    The final layer outputs a single logit (no sigmoid).
    """
try:
    # handle model signature differences gracefully
    try:
        base_model = SZP(input_dim=len(FEATURES), output_dim=1)
    except TypeError:
        base_model = SZP()
except Exception as exc:
    print(f"[WARN] Could not instantiate StrikeZonePredictionModel ({exc}). Using fallback MLP.")
    base_model = build_fallback_mlp(input_dim=len(FEATURES), output_dim=1)


class EnsureColumnLogit(nn.Module):
    """Wraps a module to ensure output has shape (N, 1)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out


model = EnsureColumnLogit(base_model).to(DEVICE)

# -------------------- loss and optimizer --------------------
# compute pos_weight = (# negatives / # positives) to balance classes
pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
pos_weight = torch.tensor(
    [1.0 if pos == 0 else (neg / max(1.0, pos))],
    device=DEVICE,
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------- evaluation helper --------------------
def evaluate(loader: DataLoader):
    """
    Return average loss, true labels, and predicted probabilities for a dataloader.

    Args:
        loader: dataloader to evaluate

    Returns:
        (avg_loss, y_true, y_prob)
        - avg_loss (float)
        - y_true (np.ndarray shape [N, 1], dtype matches dataset targets)
        - y_prob (np.ndarray shape [N, 1], probabilities in [0, 1])
    """
    model.eval()
    total_loss = 0.0
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            ys.append(yb.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    n = len(loader.dataset)
    return total_loss / max(1, n), np.vstack(ys), np.vstack(ps)

# -------------------- training loop with early stopping --------------------
best_val_loss = float("inf")
best_state = None
patience_left = PATIENCE

train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)

    # compute a simple per-epoch gradient norm (reflects last mini-batch)
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().norm().item()
    print(f"Grad norm: {grad_norm:.4f}")

    train_loss = running / len(train_loader.dataset)
    val_loss, _, _ = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # check for validation improvement with small epsilon to avoid churn
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = {
            "model": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "feature_names": FEATURES,
            "rng_seed": RNG_SEED,
        }
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            print("Early stopping.")
            break

# load the best model weights before final evaluation
if best_state is not None:
    model.load_state_dict(best_state["model"])

# -------------------- validation metrics --------------------
val_loss, y_true, y_prob = evaluate(val_loader)
y_true = y_true.astype(np.int64)
y_pred = (y_prob >= 0.5).astype(np.int64)

tp = int(((y_pred == 1) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())

acc = (tp + tn) / max(1, (tp + tn + fp + fn))
prec = tp / max(1, (tp + fp))
rec = tp / max(1, (tp + fn))
f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

print("\n=== Validation Metrics ===")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1: {f1:.3f}")
print("Confusion matrix [[tn, fp], [fn, tp]]:")
print([[tn, fp], [fn, tp]])

# -------------------- save model and scaler stats --------------------
model_path = os.path.join(SAVE_DIR, "strikezone_model.pt")
torch.save(
    {
        "state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "feature_names": FEATURES,
        "rng_seed": RNG_SEED,
        "pos_weight": pos_weight.detach().cpu().numpy().tolist(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    },
    model_path,
)
print(f"\nSaved model to: {model_path}")

# -------------------- plot training vs validation loss --------------------
# use non-interactive backend to allow saving without a display
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot_path = os.path.join(SAVE_DIR, "loss_curve.png")
plt.savefig(loss_plot_path, dpi=150)
print(f"Saved loss curve to: {loss_plot_path}")

# -------------------- inference helper --------------------
def predict_is_called_strike(plate_loc_height: float, plate_loc_side: float, swing: float) -> float:
    """
    Predict the probability that an umpire calls a strike.

    Args:
        plate_loc_height: vertical pitch location (float).
        plate_loc_side: horizontal pitch location (float).
        swing: 1 if batter swung, 0 otherwise (float or int).

    Returns:
        Probability in [0, 1] (float).
    """
    model.eval()
    x = np.array([[plate_loc_height, plate_loc_side, swing]], dtype=np.float32)
    x = (x - x_mean) / x_std
    with torch.no_grad():
        logit = model(torch.from_numpy(x).to(DEVICE))
        prob = torch.sigmoid(logit).item()
    return prob


if __name__ == "__main__":
    # simple usage example for sanity checking
    print("Example prob:", predict_is_called_strike(1.95, 0.10, 0))
