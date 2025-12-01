# generate_heatmaps_5feat_from_ckpt.py
import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from Strickzone_Predictve_Model.utils.model_weight_surgery import (
    load_and_upgrade_checkpoint_to_5_inputs,
    DEVICE,
)

# ---------- Try to import your training model ----------
try:
    from model import StrikeZonePredictionModel as SZP
except Exception:
    SZP = None


class _EnsureColumnLogit(nn.Module):
    """Ensure output shape is (N,1) or (N,) consistently."""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out


def build_core_model_5in() -> nn.Module:
    """
    Strictly use your StrikeZonePredictionModel with 5 inputs.
    """
    if SZP is None:
        raise RuntimeError("StrikeZonePredictionModel (SZP) could not be imported from model.py")
    try:
        return SZP(input_dim=5, output_dim=1)
    except TypeError:
        # If your class ignores input_dim/output_dim, just call it
        return SZP()


# ---------- Standardization & inference ----------
def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = std.copy()
    std[std == 0.0] = 1.0
    return (X - mean) / std


@torch.no_grad()
def model_probs(model: nn.Module, X_std: np.ndarray, outputs_logits: bool) -> np.ndarray:
    xb = torch.from_numpy(X_std.astype(np.float32)).to(DEVICE)
    out = model(xb)  # (N,1) or (N,)
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    elif out.ndim > 1 and out.size(-1) > 1:
        # multiclass case: use class 1 as "strike" logit
        out = out[..., 1]
    if outputs_logits:
        probs = torch.sigmoid(out)
    else:
        probs = out.clamp(0, 1)
    return probs.detach().cpu().numpy()


# ---------- Heatmap for ONE combo ----------
def generate_heatmap_for_combo(
    model: nn.Module,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    step: float,
    swing: float,
    batter_hand: float,
    pitcher_hand: float,
    outputs_logits: bool,
    save_path: str,
):
    """
    Features in training order:
      [PlateLocHeight, PlateLocSide, Swing, BatterHand, PitcherHand]
    """
    xs = np.arange(xmin, xmax + 1e-9, step, dtype=np.float32)  # PlateLocSide
    ys = np.arange(ymin, ymax + 1e-9, step, dtype=np.float32)  # PlateLocHeight
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    swing_grid = np.full_like(XX, swing, dtype=np.float32)
    bh_grid = np.full_like(XX, batter_hand, dtype=np.float32)
    ph_grid = np.full_like(XX, pitcher_hand, dtype=np.float32)

    # (N,5): [height, side, swing, batter_hand, pitcher_hand]
    X = np.stack(
        [YY.ravel(), XX.ravel(), swing_grid.ravel(), bh_grid.ravel(), ph_grid.ravel()],
        axis=1,
    )

    X_std = standardize(X, x_mean.astype(np.float32), x_std.astype(np.float32))
    P = model_probs(model, X_std, outputs_logits=outputs_logits).reshape(YY.shape)

    plt.figure(figsize=(6, 6), dpi=120)
    im = plt.imshow(
        P,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Probability")

    title = (
        f"Strike Zone Prob Heatmap\n"
        f"Swing={int(swing)}, BatterHand={int(batter_hand)}, PitcherHand={int(pitcher_hand)}"
    )
    plt.title(title)
    plt.xlabel("PlateLocSide (x)")
    plt.ylabel("PlateLocHeight (y)")

    # ---- Contour lines ----
    cs = plt.contour(XX, YY, P, levels=[0.2, 0.4, 0.6, 0.8], linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # ---- Softball rulebook strike zone box ----
    # Horizontal: x in [-0.708, 0.708]
    # Vertical:   y in [1.5, 3.5]
    strike_x_min = -0.708
    strike_x_max = 0.708
    strike_y_min = 1.5
    strike_y_max = 3.5

    ax = plt.gca()
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (strike_x_min, strike_y_min),
        strike_x_max - strike_x_min,
        strike_y_max - strike_y_min,
        fill=False,
        linestyle="--",
        linewidth=1.5,
    )
    ax.add_patch(rect)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate strike-zone heatmaps for all combinations of "
                    "Swing, BatterHand, PitcherHand (each 0 or 1)."
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "artifacts/strikezone_model.pt"),
        help="Path to checkpoint (.pt) file",
    )
    ap.add_argument("--xmin", type=float, default=-4.0)
    ap.add_argument("--xmax", type=float, default=4.0)
    ap.add_argument("--ymin", type=float, default=-2.0)
    ap.add_argument("--ymax", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument(
        "--probs",
        action="store_true",
        help="Use this flag if the model already outputs probabilities (not logits)",
    )
    ap.add_argument(
        "--outdir",
        default="heatmaps_5feat",
        help="Directory to save all generated heatmaps",
    )
    return ap.parse_args()


# ---------- Main ----------
def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    # 1) Load and upgrade checkpoint to 5-input model + scalers
    core_model, ckpt, x_mean, x_std = load_and_upgrade_checkpoint_to_5_inputs(
        args.model,
        builder_5in=build_core_model_5in,
    )

    # Wrap to ensure (N,1) output shape
    model = _EnsureColumnLogit(core_model).to(DEVICE)
    model.eval()

    # 2) Generate all 2^3 = 8 combinations of (swing, batter_hand, pitcher_hand)
    combos = []
    for swing in (0.0, 1.0):
        for bh in (0.0, 1.0):
            for ph in (0.0, 1.0):
                combos.append((swing, bh, ph))

    os.makedirs(args.outdir, exist_ok=True)

    for swing, bh, ph in combos:
        fname = f"heatmap_s{swing:.0f}_bh{bh:.0f}_ph{ph:.0f}.png"
        save_path = os.path.join(args.outdir, fname)

        generate_heatmap_for_combo(
            model=model,
            x_mean=x_mean,
            x_std=x_std,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
            step=args.step,
            swing=swing,
            batter_hand=bh,
            pitcher_hand=ph,
            outputs_logits=not args.probs,  # default assumes logits
            save_path=save_path,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
