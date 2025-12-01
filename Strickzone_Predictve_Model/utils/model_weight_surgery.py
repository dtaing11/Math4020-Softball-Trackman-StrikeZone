# model_surgery.py
"""
Utilities to upgrade an old 3-input checkpoint to a 5-input model
using weight surgery (add two new input features with zero-initialized
weights in the first layer).

Intended feature order (example):
    [PlateLocHeight, PlateLocSide, Swing, BatterHand, PitcherHand]
"""

from __future__ import annotations
import os
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Checkpoint I/O --------------------


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def save_checkpoint(ckpt: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


# -------------------- State dict helpers --------------------


def _strip_prefix(sd: dict, prefix: str) -> dict:
    """Return a copy of sd with `prefix` removed from the beginning of keys that have it."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in sd.items()
    }


def _normalize_state_dict_keys(sd: dict) -> dict:
    """
    Normalize common wrappers:
      - remove 'module.' (DataParallel)
    Keep 'model.' because many models store the core net there.
    """
    sd = _strip_prefix(sd, "module.")
    return sd


def _expand_first_linear_column(
    sd: dict,
    weight_key: str,
    old_in: int,
    new_in: int,
) -> None:
    """
    In-place: expand sd[weight_key] from [out, old_in] to [out, new_in],
    copy old weights into first columns, zero-init new columns.
    """
    if weight_key not in sd:
        return

    W = sd[weight_key]
    if W.ndim != 2 or W.shape[1] != old_in:
        return

    out, _ = W.shape
    W_new = torch.zeros(out, new_in, dtype=W.dtype, device=W.device)
    W_new[:, :old_in] = W  # copy existing columns
    sd[weight_key] = W_new
    # bias stays unchanged (no key change needed)


# -------------------- Surgery core --------------------


def upgrade_state_dict_inputs(
    sd: dict,
    core_model: nn.Module,
    old_input_dim: int = 3,
    new_input_dim: int = 5,
) -> dict:
    """
    Given a checkpoint state_dict (possibly 3-input) and a freshly built
    `core_model` that expects 5 inputs, upgrade the first Linear layer's
    weight matrix so that:
        [out, old_input_dim] -> [out, new_input_dim]
    by zero-padding the new column(s).

    Returns a *new* state_dict (does not mutate original).
    """
    sd = _normalize_state_dict_keys(sd.copy())

    model_state = core_model.state_dict()
    first_w_key_model = None
    for k, v in model_state.items():
        if k.endswith(".weight") and v.ndim == 2:
            first_w_key_model = k
            break

    if first_w_key_model is None:
        return sd

    candidates = [first_w_key_model]
    if not first_w_key_model.startswith("model."):
        candidates.append("model." + first_w_key_model)

    first_w_key_ckpt = None
    for cand in candidates:
        if cand in sd:
            first_w_key_ckpt = cand
            break

    if first_w_key_ckpt is None:
        return sd

    W_ckpt = sd[first_w_key_ckpt]
    if W_ckpt.ndim != 2:
        return sd

    old_in = W_ckpt.shape[1]
    model_W = model_state[first_w_key_model]
    expected_in = model_W.shape[1]
    if old_in == expected_in:
        return sd

    if old_in != old_input_dim or expected_in != new_input_dim:
        return sd

    print(
        f"[model_surgery] Expanding first layer in_features "
        f"{old_in} -> {new_input_dim} (zero-init new columns)."
    )
    _expand_first_linear_column(sd, first_w_key_ckpt, old_in=old_in, new_in=new_input_dim)
    return sd


def upgrade_scaler_vectors(
    x_mean: np.ndarray,
    x_std: np.ndarray,
    old_input_dim: int = 3,
    new_input_dim: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upgrade feature scaler vectors from old_input_dim -> new_input_dim by:
      - appending mean=0 for the new features
      - appending std=1 for the new features
    For 3 -> 5, this adds **two** new dimensions (e.g., BatterHand, PitcherHand).
    """
    x_mean = np.array(x_mean, dtype=np.float32).reshape(1, -1)
    x_std = np.array(x_std, dtype=np.float32).reshape(1, -1)

    if x_mean.shape[1] == old_input_dim:
        extra_mean = np.zeros((1, new_input_dim - old_input_dim), dtype=np.float32)
        x_mean = np.concatenate([x_mean, extra_mean], axis=1)

    if x_std.shape[1] == old_input_dim:
        extra_std = np.ones((1, new_input_dim - old_input_dim), dtype=np.float32)
        x_std = np.concatenate([x_std, extra_std], axis=1)

    # Avoid divide-by-zero later
    x_std[x_std == 0.0] = 1.0
    return x_mean, x_std


# -------------------- High-level helper --------------------


def load_and_upgrade_checkpoint_to_5_inputs(
    model_path: str,
    builder_5in: Callable[[], nn.Module],
    old_input_dim: int = 3,
    new_input_dim: int = 5,
    device: torch.device = DEVICE,
) -> Tuple[nn.Module, dict, np.ndarray, np.ndarray]:
    """
    High-level convenience:
      1) Load checkpoint from `model_path`
      2) Build a 5-input core model with `builder_5in()`
      3) Do weight surgery on the first Linear layer if checkpoint is 3-input
      4) Upgrade x_mean/x_std to length 5
      5) Load weights into model and return:

         model, ckpt, x_mean, x_std

    - `builder_5in` should return your core network with 5 input features.
      Example:
        lambda: StrikeZonePredictionModel(input_dim=5, output_dim=1)
      or:
        lambda: nn.Sequential(
                  nn.Linear(5, 64), nn.ReLU(),
                  nn.Linear(64, 32), nn.Sigmoid(),
                  nn.Linear(32, 1)
                )
    """
    ckpt = load_checkpoint(model_path)
    if "state_dict" not in ckpt:
        raise RuntimeError("Checkpoint must contain 'state_dict' key.")

    raw_sd = ckpt["state_dict"]

    # 1) Build 5-input core model
    core_model = builder_5in().to(device)

    # 2) Adapt the state_dict if needed
    adapted_sd = upgrade_state_dict_inputs(
        raw_sd, core_model,
        old_input_dim=old_input_dim,
        new_input_dim=new_input_dim,
    )

    # 3) Load adapted weights (allow missing keys for safety)
    core_model.load_state_dict(adapted_sd, strict=False)
    core_model.eval()

    # 4) Upgrade scalers
    x_mean = ckpt.get("x_mean", np.zeros((1, old_input_dim), dtype=np.float32))
    x_std = ckpt.get("x_std", np.ones((1, old_input_dim), dtype=np.float32))

    x_mean_up, x_std_up = upgrade_scaler_vectors(
        x_mean, x_std,
        old_input_dim=old_input_dim,
        new_input_dim=new_input_dim,
    )

    return core_model, ckpt, x_mean_up, x_std_up
