"""
exp_3_relaxed.py
================
Experiment 3: Relaxed expected-matching mortality-control bounds.

Section 5.2 simplified setting:
  - Deterministic interest rate r=0.01
  - Zero market price of risk
  - Bang-bang optimal mortality controls
  - Constraints at maturity only vs. all intermediate dates

Products: GMAB (survival), GMIB (running), GMDB (death).
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mortality import load_life_table, get_annual_survival_probs
from relaxed_control import (
    relaxed_gmab_bounds,
    relaxed_gmib_bounds,
    relaxed_gmdb_bounds,
    balducci_gmab_value,
    balducci_gmib_value,
    balducci_gmdb_value,
)
import config

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp3() -> None:
    """Main runner for Experiment 3."""
    print("=" * 60)
    print("Experiment 3: Relaxed Mortality-Control Bounds")
    print("  Section 5.2: deterministic r=0.01, zero market price of risk")
    print("=" * 60)

    lt_path = os.path.join(ROOT, "data", "belgian_female_lt.csv")
    df_lt = load_life_table(lt_path)
    lx = df_lt["lx"].values

    A0 = config.A0
    r_g = config.R_G
    r_I = config.R_I
    beta = config.BETA
    pi_S = config.PI_S
    sigma_S = config.SIGMA_S
    r_det = config.R_DET
    delta_main = config.DELTA_MAIN
    n_T = config.N_T_PDE
    n_x = config.N_X_PDE

    # -----------------------------------------------------------------
    # Part A: GMAB, GMIB, GMDB bounds for all ages and maturities
    # -----------------------------------------------------------------
    print("\n--- Part A: Product bounds vs maturity ---")
    records = []

    for x in config.AGES_GMDB:
        for T in config.MATURITY_GRID:
            try:
                hat_p = get_annual_survival_probs(lx, x, T)
            except ValueError as e:
                print(f"  Skipping x={x}, T={T}: {e}")
                continue

            print(f"\n  Processing x={x}, T={T} (delta={delta_main}) ...")

            # Balducci baselines
            bl_gmab = balducci_gmab_value(hat_p, T, A0, r_g, pi_S, sigma_S, r_det, n_T, n_x)
            bl_gmib = balducci_gmib_value(hat_p, T, A0, r_g, r_I, beta, pi_S, sigma_S, r_det, n_T, n_x)
            bl_gmdb = balducci_gmdb_value(hat_p, T, A0, r_g, pi_S, sigma_S, r_det, n_T, n_x)

            # Relaxed bounds (maturity-only constraint)
            lo_gmab, up_gmab = relaxed_gmab_bounds(
                hat_p, T, delta_main, A0, r_g, r_I, beta, pi_S, sigma_S, r_det, n_T, n_x
            )
            lo_gmib, up_gmib = relaxed_gmib_bounds(
                hat_p, T, delta_main, A0, r_g, r_I, beta, pi_S, sigma_S, r_det, n_T, n_x
            )
            lo_gmdb, up_gmdb = relaxed_gmdb_bounds(
                hat_p, T, delta_main, A0, r_g, pi_S, sigma_S, r_det, n_T, n_x
            )

            print(
                f"    GMAB: baseline={bl_gmab:.4f}, [{lo_gmab:.4f}, {up_gmab:.4f}]"
            )
            print(
                f"    GMIB: baseline={bl_gmib:.4f}, [{lo_gmib:.4f}, {up_gmib:.4f}]"
            )
            print(
                f"    GMDB: baseline={bl_gmdb:.4f}, [{lo_gmdb:.4f}, {up_gmdb:.4f}]"
            )

            records.append({
                "age": x, "T": T, "delta": delta_main,
                "GMAB_baseline": bl_gmab,
                "GMAB_lower": lo_gmab, "GMAB_upper": up_gmab,
                "GMIB_baseline": bl_gmib,
                "GMIB_lower": lo_gmib, "GMIB_upper": up_gmib,
                "GMDB_baseline": bl_gmdb,
                "GMDB_lower": lo_gmdb, "GMDB_upper": up_gmdb,
            })

    df_res = pd.DataFrame(records)
    df_res.to_csv(os.path.join(RESULTS_DIR, "exp3_results.csv"), index=False)

    # -----------------------------------------------------------------
    # Part B: Delta-convergence study (x=60, T=5)
    # -----------------------------------------------------------------
    print("\n--- Part B: Delta-convergence study (x=60, T=5) ---")
    x_dc = 60
    T_dc = 5
    hat_p_dc = get_annual_survival_probs(lx, x_dc, T_dc)

    delta_records = []
    for delta in config.DELTA_GRID:
        lo_g, up_g = relaxed_gmab_bounds(
            hat_p_dc, T_dc, delta, A0, r_g, r_I, beta, pi_S, sigma_S, r_det, n_T, n_x
        )
        lo_i, up_i = relaxed_gmib_bounds(
            hat_p_dc, T_dc, delta, A0, r_g, r_I, beta, pi_S, sigma_S, r_det, n_T, n_x
        )
        lo_d, up_d = relaxed_gmdb_bounds(
            hat_p_dc, T_dc, delta, A0, r_g, pi_S, sigma_S, r_det, n_T, n_x
        )
        print(
            f"  delta={delta:.2f}: GMAB=[{lo_g:.4f},{up_g:.4f}], "
            f"GMIB=[{lo_i:.4f},{up_i:.4f}], GMDB=[{lo_d:.4f},{up_d:.4f}]"
        )
        delta_records.append({
            "delta": delta,
            "GMAB_lower": lo_g, "GMAB_upper": up_g,
            "GMIB_lower": lo_i, "GMIB_upper": up_i,
            "GMDB_lower": lo_d, "GMDB_upper": up_d,
        })

    df_delta = pd.DataFrame(delta_records)
    df_delta.to_csv(os.path.join(RESULTS_DIR, "exp3_delta_convergence.csv"), index=False)

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    _plot_product_bounds(df_res, "GMAB", RESULTS_DIR)
    _plot_product_bounds(df_res, "GMIB", RESULTS_DIR)
    _plot_product_bounds(df_res, "GMDB", RESULTS_DIR)
    _plot_delta_convergence(df_delta, RESULTS_DIR)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    print("\n--- Validation ---")
    fails = 0
    for _, row in df_res.iterrows():
        for prod in ["GMAB", "GMIB", "GMDB"]:
            lo = row[f"{prod}_lower"]
            up = row[f"{prod}_upper"]
            bl = row[f"{prod}_baseline"]
            if up < lo - 1e-4:
                print(f"  FAIL {prod} upper < lower: x={row['age']}, T={row['T']}")
                fails += 1

    # Monotonicity in delta
    for prod in ["GMAB", "GMIB", "GMDB"]:
        up_vals = df_delta[f"{prod}_upper"].values
        lo_vals = df_delta[f"{prod}_lower"].values
        if not np.all(np.diff(up_vals) >= -1e-4):
            print(f"  WARN: {prod} upper not monotone non-decreasing in delta")
        if not np.all(np.diff(lo_vals) <= 1e-4):
            print(f"  WARN: {prod} lower not monotone non-increasing in delta")

    if fails == 0:
        print("  All bound ordering checks passed ✓")

    print("\n=== Experiment 3 complete ===")


def _plot_product_bounds(df_res: pd.DataFrame, product: str, results_dir: str) -> None:
    """Plot relaxed bounds vs maturity for a given product."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, x in enumerate(config.AGES_GMDB):
        ax = axes[idx]
        sub = df_res[df_res["age"] == x]
        if sub.empty:
            continue
        ax.fill_between(
            sub["T"], sub[f"{product}_lower"], sub[f"{product}_upper"],
            alpha=0.25, color="gray", label="Relaxed interval"
        )
        ax.plot(sub["T"], sub[f"{product}_upper"], "r--", linewidth=1.5, label="Relaxed upper")
        ax.plot(sub["T"], sub[f"{product}_lower"], "b--", linewidth=1.5, label="Relaxed lower")
        ax.plot(sub["T"], sub[f"{product}_baseline"], "k-.", linewidth=1.5, label="Balducci baseline")
        ax.set_title(f"{product} bounds: age {x}")
        ax.set_xlabel("Maturity T")
        ax.set_ylabel(f"{product} value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Relaxed {product} Bounds vs Maturity (delta={config.DELTA_MAIN})")
    plt.tight_layout()
    fname = os.path.join(results_dir, f"exp3_{product.lower()}_bounds.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def _plot_delta_convergence(df_delta: pd.DataFrame, results_dir: str) -> None:
    """Plot delta-convergence for all products."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, product in enumerate(["GMAB", "GMIB", "GMDB"]):
        ax = axes[idx]
        ax.fill_between(
            df_delta["delta"],
            df_delta[f"{product}_lower"],
            df_delta[f"{product}_upper"],
            alpha=0.3, color="gray",
        )
        ax.plot(df_delta["delta"], df_delta[f"{product}_upper"], "r--", linewidth=1.5,
                label="Worst-case (upper)")
        ax.plot(df_delta["delta"], df_delta[f"{product}_lower"], "b--", linewidth=1.5,
                label="Best-case (lower)")
        ax.set_xlabel(r"$\delta$ (control band radius)")
        ax.set_ylabel(f"{product} value")
        ax.set_title(f"{product} delta-convergence (x=60, T=5)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(r"Relaxed Bound Width vs $\delta$ (x=60, T=5)")
    plt.tight_layout()
    fname = os.path.join(results_dir, "exp3_delta_convergence.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    run_exp3()
