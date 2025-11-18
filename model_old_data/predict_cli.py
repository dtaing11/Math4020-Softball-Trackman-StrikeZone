# predict_cli.py
import os
import sys
import numpy as np
import torch
from torch import nn
import matplotlib as plt

# Try to import your custom model (same as training)
try:
    from model import StrikeZonePredictionModel as SZP
except Exception:
    SZP = None

# -------------------- Config --------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/strikezone_model.pt")
THRESH = float(os.environ.get("THRESH", 0.5))  # decision threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = ["PlateLocHeight", "PlateLocSide", "Swing"]

# -------------------- Model builders --------------------
def build_fallback_mlp(input_dim=3, output_dim=1):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    )

class _EnsureColumnLogit(nn.Module):
    """Wrap to ensure output shape is (N,1) for BCEWithLogitsLoss compatibility"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out

def build_model_like_training():
    """
    Try to instantiate the same model as training:
      1) your custom StrikeZonePredictionModel(input_dim=3, output_dim=1)
      2) fallback MLP
    """
    if SZP is not None:
        try:
            return _EnsureColumnLogit(SZP(input_dim=3, output_dim=1))
        except TypeError:
            # If your class had no args
            return _EnsureColumnLogit(SZP())
    # Fallback
    return _EnsureColumnLogit(build_fallback_mlp(3, 1))

# -------------------- Load checkpoint --------------------
def load_checkpoint(model_path: str):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)
    ckpt = torch.load(model_path, map_location="cpu")
    return ckpt

def restore_model(ckpt):
    """
    Recreate the model architecture and load state_dict.
    If size mismatch occurs (e.g., different class than training), try fallback MLP.
    """
    # First, try with SZP (or fallback) in the same order used at train time
    for builder_name, builder in [
        ("custom", build_model_like_training),
        ("fallback", lambda: _EnsureColumnLogit(build_fallback_mlp(3, 1))),
    ]:
        model = builder().to(DEVICE)
        try:
            model.load_state_dict(ckpt["state_dict"])
            print(f"[INFO] Loaded model weights using {builder_name} architecture.")
            return model
        except Exception as e:
            print(f"[WARN] Failed to load with {builder_name}: {e}")
    print("[ERROR] Could not load model weights with available architectures.")
    sys.exit(1)

# -------------------- Preprocess --------------------
def standardize(x_row: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return (x_row - x_mean) / x_std

def prob_called_strike(model, x_row_std: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x_row_std.astype(np.float32)).to(DEVICE)
        logit = model(xb).item()  # single example
        prob = torch.sigmoid(torch.tensor(logit)).item()
    return prob

def parse_input(line: str):
    """
    Accepts inputs like:
      1.95 0.10 0
      height=1.95 side=0.10 swing=0
    Returns (height, side, swing)
    """
    line = line.strip()
    if "=" in line:
        parts = dict(kv.split("=", 1) for kv in line.split())
        h = float(parts.get("height") or parts.get("PlateLocHeight"))
        s = float(parts.get("side")   or parts.get("PlateLocSide"))
        sw = float(parts.get("swing") or parts.get("Swing"))
        return h, s, sw
    else:
        vals = line.split()
        if len(vals) != 3:
            raise ValueError("Please provide exactly three values: <height> <side> <swing>")
        h, s, sw = map(float, vals)
        return h, s, sw
# ----------------------------------------

# -------------------- Main loop --------------------
def main():
    ckpt = load_checkpoint(MODEL_PATH)

    # Load scaler stats
    x_mean = np.array(ckpt["x_mean"], dtype=np.float32)  # shape (1,3)
    x_std  = np.array(ckpt["x_std"],  dtype=np.float32)  # shape (1,3)
    # Safety
    x_std[x_std == 0.0] = 1.0

    # Rebuild & load model
    model = restore_model(ckpt)

    print("Strike Zone Called-Strike Predictor (type 'exit' to quit)")
    print("Enter inputs as: <PlaxteLocHeight> <PlateLocSide> <Swing(0 or 1)>")
    print("Example: 1.95 0.10 0")
    print(f"Decision threshold = {THRESH:.2f}\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if line.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break
        if not line:
            continue

        try:
            h, s, sw = parse_input(line)
        except Exception as e:
            print(f"[Input error] {e}")
            continue

        x_row = np.array([[h, s, sw]], dtype=np.float32)  # shape (1,3)
        x_row_std = standardize(x_row, x_mean, x_std)

        p = prob_called_strike(model, x_row_std)
        decision = "STRIKE (called)" if p >= THRESH else "BALL (not called)"
        print(f"Prob(StrikeCalled) = {p:.3f}  →  {decision}")

if __name__ == "__main__":
    main()
