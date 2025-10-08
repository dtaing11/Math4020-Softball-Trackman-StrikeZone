#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATASETS = {
    "2020": "../datasets/pitches_2020.csv",
    "2021": "../datasets/pitches_2021.csv",
    "2022": "../datasets/pitches_2022.csv",
    "2023": "../datasets/pitches_2023.csv",
    "2024": "../datasets/pitches_2024.csv",
}

def load_all_numeric(file_path: str) -> pd.DataFrame:
    """Load CSV, keep every row that has numeric px, pz, is_strike. No range filtering."""
    print(f"[INFO] Loading {file_path}")
    # read consistently to avoid DtypeWarning chunking issues
    df = pd.read_csv(file_path, low_memory=False)

    # required columns presence check
    required = ["px", "pz", "is_strike", "sz_top", "sz_bot"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # coerce to numeric but DO NOT clip/range-filter
    df["px"] = pd.to_numeric(df["px"], errors="coerce")
    df["pz"] = pd.to_numeric(df["pz"], errors="coerce")
    df["is_strike"] = pd.to_numeric(df["is_strike"], errors="coerce")

    # keep every numeric record; just drop rows that aren't numeric
    df = df.dropna(subset=["px", "pz", "is_strike"])

    # ensure is_strike is 0/1 if it's e.g. floats like 0.0/1.0
    df["is_strike"] = (df["is_strike"] > 0).astype(int)

    return df

def plot_scatter_all(df: pd.DataFrame, year: str, outdir="plots"):
    Path(outdir).mkdir(exist_ok=True)

    strikes = df[df["is_strike"] == 1]
    balls   = df[df["is_strike"] == 0]

    # compute median sz_top/sz_bot if present/valid; otherwise default
    sz_top = pd.to_numeric(df.get("sz_top"), errors="coerce").median()
    sz_bot = pd.to_numeric(df.get("sz_bot"), errors="coerce").median()
    if not (pd.notna(sz_top) and pd.notna(sz_bot) and sz_top > sz_bot):
        sz_bot, sz_top = 1.5, 3.5

    plt.figure(figsize=(6, 7))
    plt.scatter(balls["px"],   balls["pz"],   color="blue", s=8, alpha=0.5, label="Ball")
    plt.scatter(strikes["px"], strikes["pz"], color="red",  s=8, alpha=0.5, label="Strike")

    # zone guides (no x/y limits → show entire numeric range)
    plate_half = 0.83
    plt.axvline(-plate_half, linestyle=":", color="gray")
    plt.axvline( plate_half, linestyle=":", color="gray")
    plt.axhline(sz_bot, linestyle="--", color="gray")
    plt.axhline(sz_top, linestyle="--", color="gray")
    plt.axvline(0.0, linewidth=1, color="black", alpha=0.6)

    plt.xlabel("px (ft)")
    plt.ylabel("pz (ft)")
    plt.title(f"Pitch Locations ({year}) — Red = Strike, Blue = Ball")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path = Path(outdir) / f"strike_scatter_{year}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved {out_path}")

def main():
    for year, path in DATASETS.items():
        try:
            df = load_all_numeric(path)
            plot_scatter_all(df, year)
        except FileNotFoundError:
            print(f"[WARN] File not found for {year}: {path}")
        except Exception as e:
            print(f"[ERROR] Could not process {year}: {e}")

if __name__ == "__main__":
    main()
