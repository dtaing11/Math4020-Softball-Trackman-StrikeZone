# generate_heatmap_from_ckpt.py
import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES = ["PlateLocHeight", "PlateLocSide", "Swing"]  # (height, side, swing)

# ---------- Try to import your training model ----------
try:
    from model import StrikeZonePredictionModel as SZP
except Exception:
    SZP = None

# ---------- Builders ----------
def build_fallback_mlp(input_dim=3, output_dim=1):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    )

class _EnsureColumnLogit(nn.Module):
    """Ensure output shape is (N,1); applied AFTER we successfully load weights."""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out

# ---------- Key utilities ----------
def strip_prefix_from_keys(sd: dict, prefix: str) -> dict:
    """Return a copy with 'prefix' removed from keys that start with it."""
    new = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            new[k[len(prefix):]] = v
        else:
            new[k] = v
    return new

def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return torch.load(path, map_location="cpu")

# ---------- Restore model (robust) ----------
def restore_model(ckpt) -> nn.Module:
    """
    Load ckpt['state_dict'] with several strategies:
      A) custom class (direct):               keys like 'model.0.weight' stay
      B) custom class (.model submodule):     strip 'model.' -> load into obj.model
      C) fallback MLP (Sequential):           strip 'model.' -> load into plain nn.Sequential
    Always strip leading 'module.' first (DataParallel).
    Wrap with _EnsureColumnLogit only AFTER loading.
    """
    if "state_dict" not in ckpt or not isinstance(ckpt["state_dict"], dict):
        raise RuntimeError("Checkpoint missing 'state_dict' (dict).")

    sd = ckpt["state_dict"]

    # 1) strip DataParallel prefix first
    sd = strip_prefix_from_keys(sd, "module.")

    # ---- A) Try custom class, direct ----
    if SZP is not None:
        try:
            model_a = SZP(input_dim=3, output_dim=1) if _class_accepts_io(SZP) else SZP()
            model_a.load_state_dict(sd, strict=True)
            model_a.to(DEVICE).eval()
            print("[INFO] Loaded weights into custom class (direct).")
            return _EnsureColumnLogit(model_a)
        except Exception as e:
            print(f"[WARN] custom (direct) failed: {e}")

        # ---- B) Try loading into its .model submodule after stripping 'model.' ----
        try:
            sd_nomodel = strip_prefix_from_keys(sd, "model.")
            model_b = SZP(input_dim=3, output_dim=1) if _class_accepts_io(SZP) else SZP()
            # Must have a 'model' submodule
            if not hasattr(model_b, "model") or not isinstance(model_b.model, nn.Module):
                raise RuntimeError("Custom class has no .model submodule.")
            model_b.model.load_state_dict(sd_nomodel, strict=True)
            model_b.to(DEVICE).eval()
            print("[INFO] Loaded weights into custom class .model submodule.")
            return _EnsureColumnLogit(model_b)
        except Exception as e:
            print(f"[WARN] custom (.model) failed: {e}")

    # ---- C) Fallback: plain Sequential MLP (strip 'model.' first) ----
    try:
        sd_nomodel = strip_prefix_from_keys(sd, "model.")
        seq = build_fallback_mlp(3, 1)
        seq.load_state_dict(sd_nomodel, strict=True)
        seq.to(DEVICE).eval()
        print("[INFO] Loaded weights into fallback MLP (Sequential).")
        return _EnsureColumnLogit(seq)
    except Exception as e:
        print(f"[WARN] fallback MLP failed: {e}")

    raise RuntimeError("Could not load model weights with available strategies.")

def _class_accepts_io(cls) -> bool:
    """Best-effort: does ctor accept input_dim/output_dim?"""
    try:
        cls(input_dim=3, output_dim=1)
        return True
    except TypeError:
        return False
    except Exception:
        # Constructor may do heavy stuff; we only care about signature support
        return True

# ---------- Standardization & inference ----------
def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = std.copy()
    std[std == 0.0] = 1.0
    return (X - mean) / std

@torch.no_grad()
def model_probs(model: nn.Module, X_std: np.ndarray, outputs_logits: bool) -> np.ndarray:
    xb = torch.from_numpy(X_std.astype(np.float32)).to(DEVICE)
    out = model(xb)
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    elif out.ndim > 1 and out.size(-1) > 1:
        out = out[..., 1]
    if outputs_logits:
        probs = torch.sigmoid(out)
    else:
        probs = out.clamp(0, 1)
    return probs.detach().cpu().numpy()

# ---------- Heatmap ----------
def generate_heatmap(
    model: nn.Module,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    step: float,
    swing_fixed: float,
    outputs_logits: bool,
    save_path: str,
):
    xs = np.arange(xmin, xmax + 1e-9, step, dtype=np.float32)  # PlateLocSide
    ys = np.arange(ymin, ymax + 1e-9, step, dtype=np.float32)  # PlateLocHeight
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    # features in training order: [height, side, swing]
    swing = np.full_like(XX, swing_fixed, dtype=np.float32)
    X = np.stack([YY.ravel(), XX.ravel(), swing.ravel()], axis=1)  # (N,3)

    X_std = standardize(X, x_mean.astype(np.float32), x_std.astype(np.float32))
    P = model_probs(model, X_std, outputs_logits=outputs_logits).reshape(YY.shape)

    plt.figure(figsize=(6, 6), dpi=120)
    im = plt.imshow(
        P,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap="coolwarm",
        vmin=0.0, vmax=1.0,
        aspect="auto",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Probability")
    plt.title(f"Strike Zone Probability Heatmap (Swing={swing_fixed:.0f})")
    plt.xlabel("PlateLocSide (x)")
    plt.ylabel("PlateLocHeight (y)")

    cs = plt.contour(XX, YY, P, levels=[0.2, 0.4, 0.6, 0.8], linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    plt.show()

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate strike-zone probability heatmap from checkpoint.")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "artifacts/strikezone_model.pt"))
    ap.add_argument("--xmin", type=float, default=-4.0)
    ap.add_argument("--xmax", type=float, default= 4.0)
    ap.add_argument("--ymin", type=float, default=-2.0)
    ap.add_argument("--ymax", type=float, default= 6.0)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--swing", type=float, default=0.0, help="Fix Swing to 0 or 1 across the grid")
    ap.add_argument("--probs", action="store_true", help="Use if model already outputs probabilities (not logits)")
    ap.add_argument("--save", default="strikezone_heatmap.png")
    return ap.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    ckpt = load_checkpoint(args.model)

    if "x_mean" not in ckpt or "x_std" not in ckpt:
        raise RuntimeError("Checkpoint must contain 'x_mean' and 'x_std' for standardization.")
    x_mean = np.array(ckpt["x_mean"], dtype=np.float32).reshape(1, -1)  # (1,3)
    x_std  = np.array(ckpt["x_std"],  dtype=np.float32).reshape(1, -1)  # (1,3)

    model = restore_model(ckpt)

    generate_heatmap(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        xmin=args.xmin, xmax=args.xmax,
        ymin=args.ymin, ymax=args.ymax,
        step=args.step,
        swing_fixed=args.swing,
        outputs_logits=not args.probs,  # default assumes logits
        save_path=args.save,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
