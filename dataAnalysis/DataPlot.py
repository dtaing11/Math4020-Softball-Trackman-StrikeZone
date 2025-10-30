#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATASETS = {
    "StrikeZoneData": "../datasets/StrikeZoneData.csv",
}

Plat_LocSide = "PlateLocSide"
Plat_LocHeight = "PlateLocHeight"
Swing = "Swing"

def load_all_numeric(file_path: str) -> pd.DataFrame:
    """Load CSV, keep every row that has numeric Plat_LocSide, Plat_LocHeight, Swing. No range filtering."""
    print(f"[INFO] Loading {file_path}")
    # read consistently to avoid DtypeWarning chunking issues
    df = pd.read_csv(file_path, low_memory=False)

    # required columns presence check
    required = [Plat_LocSide, Plat_LocHeight, Swing]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # coerce to numeric but DO NOT clip/range-filter
    df[Plat_LocSide]   = pd.to_numeric(df[Plat_LocSide],   errors="coerce")
    df[Plat_LocHeight] = pd.to_numeric(df[Plat_LocHeight], errors="coerce")
    df[Swing]          = pd.to_numeric(df[Swing],          errors="coerce")


    # keep every numeric record; just drop rows that aren't numeric
    df = df.dropna(subset=[Plat_LocSide,Plat_LocHeight, Swing])

    # ensure Swing is 0/1 if it's e.g. floats like 0.0/1.0
    df[Swing] = (df[Swing] > 0).astype(int)

    return df

def plot_scatter_all(df: pd.DataFrame, year: str, outdir="plots"):
    Path(outdir).mkdir(exist_ok=True)

    strikes = df[df[Swing] == 1]
    balls   = df[df[Swing] == 0]


    plt.figure(figsize=(6, 7))
    plt.scatter(balls[Plat_LocSide],   balls[Plat_LocHeight],   color="blue", s=8, alpha=0.5, label="Not Swing")
    plt.scatter(strikes[Plat_LocSide], strikes[Plat_LocHeight], color="red",  s=8, alpha=0.5, label="Swing")

    # zone guides (no x/y limits → show entire numeric range)
    plate_half = 0.83
    plt.axvline(-plate_half, linestyle=":", color="gray")
    plt.axvline( plate_half, linestyle=":", color="gray")
    plt.axvline(0.0, linewidth=1, color="black", alpha=0.6)

    plt.xlabel("Plat_Locside (ft)")
    plt.ylabel("Plat_LocHeight (ft)")
    plt.title(f"Pitch Locations Red = Swing, Blue = Not Swing")
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
