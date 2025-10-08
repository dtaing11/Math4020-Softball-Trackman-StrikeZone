#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

DATASETS = {
    "2020": "../datasets/pitches_2020.csv",
    "2021": "../datasets/pitches_2021.csv",
    "2022": "../datasets/pitches_2022.csv",
    "2023": "../datasets/pitches_2023.csv",
    "2024": "../datasets/pitches_2024.csv",
}

def load_clean(file_path):
    """Load CSV and convert relevant columns to numeric without range filtering."""
    print(f"[INFO] Loading {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    cols = ["px", "pz", "is_strike", "sz_top", "sz_bot"]
    for c in cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["px", "pz", "is_strike"])
    df["is_strike"] = (df["is_strike"] > 0).astype(int)
    return df

def plot_heatmap(df, year, outdir="heatMap", bins_x=150 , bins_z=150):
    Path(outdir).mkdir(exist_ok=True)

    px_min, px_max = df["px"].quantile([0.01, 0.99])
    pz_min, pz_max = df["pz"].quantile([0.01, 0.99])

    H_all, xedges, zedges = np.histogram2d(
        df["px"], df["pz"],
        bins=(bins_x, bins_z),
        range=[[px_min, px_max], [pz_min, pz_max]],
    )
    H_strike, _, _ = np.histogram2d(
        df.loc[df["is_strike"] == 1, "px"],
        df.loc[df["is_strike"] == 1, "pz"],
        bins=(bins_x, bins_z),
        range=[[px_min, px_max], [pz_min, pz_max]],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(H_all > 0, H_strike / H_all, np.nan)

    sz_top = df["sz_top"].median()
    sz_bot = df["sz_bot"].median()
    if not np.isfinite(sz_top) or not np.isfinite(sz_bot) or sz_top <= sz_bot:
        sz_bot, sz_top = 1.5, 3.5

    plt.figure(figsize=(6,7))
    im = plt.imshow(
        rate.T, origin="lower",
        extent=[px_min, px_max, pz_min, pz_max],
        aspect="auto", cmap="coolwarm", vmin=0, vmax=1,
    )
    plt.colorbar(im, label="Strike Probability")
    plt.axhline(sz_bot, linestyle="--", color="black")
    plt.axhline(sz_top, linestyle="--", color="black")
    plate_half = 0.83
    plt.axvline(-plate_half, linestyle=":", color="black")
    plt.axvline( plate_half, linestyle=":", color="black")
    plt.axvline(0.0, color="black", linewidth=1, alpha=0.6)

    plt.xlabel("px (ft)")
    plt.ylabel("pz (ft)")
    plt.title(f"Strike Zone Heatmap ({year})")
    plt.tight_layout()

    out_path = Path(outdir) / f"strike_heatmap_{year}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[INFO] Saved {out_path}")

def main():
    for year, path in DATASETS.items():
        try:
            df = load_clean(path)
            plot_heatmap(df, year)
        except FileNotFoundError:
            print(f"[WARN] File not found for {year}: {path}")
        except Exception as e:
            print(f"[ERROR] Could not process {year}: {e}")

if __name__ == "__main__":
    main()
