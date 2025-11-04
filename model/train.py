# train_torch.py
import os
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import StrikeZonePredictionModel as SZP

# -------------------- Constants --------------------
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
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# -------------------- Load & clean --------------------
df = pd.read_csv(STRIKEZONE_DATA_FILE_PATH)

# Coerce numerics
for col in FEATURES:
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")

if PITCH_CALLED not in df.columns:
    raise SystemExit(f"Missing required column: {PITCH_CALLED}")

df[IS_STRIKE_CALLED] = (df[PITCH_CALLED] == STRIKE_CALLED).astype(int)

# Drop rows with NaN in features/label
df = df.dropna(subset=FEATURES + [IS_STRIKE_CALLED]).reset_index(drop=True)

X_np = df[FEATURES].to_numpy(dtype=np.float32)
y_np = df[IS_STRIKE_CALLED].to_numpy(dtype=np.int64)  # (N,)

# -------------------- Stratified split (no sklearn) --------------------
def stratified_split_indices(y, val_ratio, seed=42):
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
X_val,   y_val   = X_np[val_idx],   y_np[val_idx]

# -------------------- Standardize (fit on train only) --------------------
x_mean = X_train.mean(axis=0, keepdims=True)
x_std  = X_train.std(axis=0, keepdims=True)
x_std[x_std == 0.0] = 1.0

X_train_std = (X_train - x_mean) / x_std
X_val_std   = (X_val   - x_mean) / x_std

# -------------------- Dataset/Dataloader --------------------
class SimpleArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)  

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds = SimpleArrayDataset(X_train_std, y_train)
val_ds   = SimpleArrayDataset(X_val_std,   y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# -------------------- Model resolution --------------------
def build_fallback_mlp(input_dim=3, output_dim=1):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)  
    )

try:
    try:
        base_model = SZP(input_dim=len(FEATURES), output_dim=1)
    except TypeError:
        base_model = SZP()
except Exception as e:
    print(f"[WARN] Could not instantiate StrikeZonePredictionModel ({e}). Using fallback MLP.")
    base_model = build_fallback_mlp(input_dim=len(FEATURES), output_dim=1)

class _EnsureColumnLogit(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out

model = _EnsureColumnLogit(base_model).to(DEVICE)

# -------------------- Loss (with class imbalance handling) --------------------
pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
pos_weight = torch.tensor([1.0 if pos == 0 else (neg / max(1.0, pos))], device=DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------- Eval helper --------------------
def evaluate(loader):
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

# -------------------- Training loop with early stopping --------------------
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
        

    first_param = next(model.parameters())
    w_before = first_param.detach().clone()

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

# Load best state
if best_state is not None:
    model.load_state_dict(best_state["model"])

# -------------------- Validation metrics --------------------
val_loss, y_true, y_prob = evaluate(val_loader)
y_true = y_true.astype(np.int64)
y_pred = (y_prob >= 0.5).astype(np.int64)

tp = int(((y_pred == 1) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())
acc = (tp + tn) / max(1, (tp + tn + fp + fn))
prec = tp / max(1, (tp + fp))
rec  = tp / max(1, (tp + fn))
f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

print("\n=== Validation Metrics ===")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1: {f1:.3f}")
print("Confusion matrix [[tn, fp], [fn, tp]]:")
print([[tn, fp], [fn, tp]])

# -------------------- Save model + scaler stats --------------------
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

# -------------------- Plot training vs validation loss --------------------
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

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

# -------------------- Inference helper --------------------
def predict_is_called_strike(plate_loc_height, plate_loc_side, swing):
    model.eval()
    x = np.array([[plate_loc_height, plate_loc_side, swing]], dtype=np.float32)
    x = (x - x_mean) / x_std
    with torch.no_grad():
        logit = model(torch.from_numpy(x).to(DEVICE))
        prob = torch.sigmoid(logit).item()
    return prob

if __name__ == "__main__":
    print("Example prob:", predict_is_called_strike(1.95, 0.10, 0))
