"""
life_table_generator.py
=======================
Generate a synthetic Belgian female life table using Makeham-Gompertz
parameters calibrated to approximate HMD Belgian female 2010 mortality.

Run this script to produce data/belgian_female_lt.csv.

Parameters chosen:
  A = 0.0005  (accident/background hazard component)
  B = 0.00004 (Gompertz scale factor)
  c = 1.10    (Gompertz rate of ageing)
  l_0 = 100,000

Source approximation: Human Mortality Database, Belgium, Females, 2010.
Actual HMD data not redistributed; synthetic approximation used.
"""

import math
import os
import sys

import numpy as np
import pandas as pd

# Allow running from any directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mortality import generate_belgian_female_life_table


def main() -> None:
    """Generate and save the Belgian female life table."""
    df = generate_belgian_female_life_table(max_age=120)

    # Save to CSV
    out_path = os.path.join(HERE, "belgian_female_lt.csv")
    df.to_csv(out_path, index=False)
    print(f"Life table saved to: {out_path}")

    # Summary statistics
    print("\n=== Life Table Summary ===")
    print(f"Ages covered: {df['age'].min()} to {df['age'].max()}")
    print(f"Radix l_0: {df['lx'].iloc[0]:,.0f}")
    print(f"\nSurvival to selected ages:")
    for age in [40, 50, 60, 70, 80, 90, 100, 110]:
        row = df[df["age"] == age]
        if not row.empty:
            lx = row["lx"].values[0]
            surv = lx / df["lx"].iloc[0]
            print(f"  l_{age:3d} = {lx:10.2f}  (survival {surv:.4f})")

    print("\nOne-year survival probabilities at selected ages:")
    for age in [40, 60, 80, 100]:
        row = df[df["age"] == age]
        if not row.empty:
            px = row["px"].values[0]
            qx = row["qx"].values[0]
            print(f"  p_{age} = {px:.6f},  q_{age} = {qx:.6f}")

    # Validation: l_x is monotone decreasing
    lx_arr = df["lx"].values
    assert all(lx_arr[i] >= lx_arr[i + 1] for i in range(len(lx_arr) - 1)), \
        "Life table l_x is not monotone decreasing!"
    print("\nValidation: l_x is monotone decreasing ✓")

    # Validate p_x in (0, 1] for main ages
    px_arr = df["px"].values
    assert all(0 <= p <= 1 for p in px_arr), "p_x values out of [0,1]!"
    print("Validation: all p_x in [0, 1] ✓")

    return df


if __name__ == "__main__":
    main()
