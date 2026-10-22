"""
exp_1_strict_survival.py
========================
Experiment 1: Strict mortality-matching replication.
- Survival-curve construction under UDD, CFM, Balducci assumptions.
- GMAB strict value and invariance verification.
- GMIB strict upper and lower bounds.
- Comparison of strict vs. benchmark valuations.

Maturity grid: T in {1, 2, 3, 5, 7, 10, 15} (documented replication choice).
"""

import os
import sys

# Path setup
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mortality import (
    load_life_table,
    get_annual_survival_probs,
    get_cumulative_survival,
    survival_curve_udd,
    survival_curve_cfm,
    survival_curve_balducci,
    survival_curve_strict_upper,
    survival_curve_strict_lower,
)
from finance import (
    zero_coupon_bond,
    sigma_Y_squared,
    gmab_price,
    gmib_integrand,
    precompute_gmib_integrand,
)
from strict_gmab_gmib import (
    gmab_strict,
    verify_gmab_invariance,
    gmib_strict_upper,
    gmib_strict_lower,
    gmib_benchmark,
    verify_annual_product_matching,
)
import config

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp1() -> None:
    """Main runner for Experiment 1."""
    print("=" * 60)
    print("Experiment 1: Strict Mortality-Matching Replication")
    print("=" * 60)

    # Load life table
    lt_path = os.path.join(ROOT, "data", "belgian_female_lt.csv")
    df_lt = load_life_table(lt_path)
    lx = df_lt["lx"].values
    print(f"Life table loaded: {len(df_lt)} ages, l_0 = {lx[0]:.0f}")

    fwd_params = config.FWD_PARAMS
    A0 = config.A0
    r_g = config.R_G
    r_I = config.R_I
    beta = config.BETA
    pi_S = config.PI_S
    pi_P = config.PI_P
    sigma_S = config.SIGMA_S
    sigma_r = config.SIGMA_R
    kappa = config.KAPPA
    rho = config.RHO
    n_grid = config.N_GRID

    # -----------------------------------------------------------------
    # Part A: Survival curve comparison (Figure 5.1 style)
    # -----------------------------------------------------------------
    print("\n--- Part A: Survival curves ---")

    for x, T_surv in [(40, 5), (100, 5)]:
        hat_p = get_annual_survival_probs(lx, x, T_surv)
        times = np.linspace(0.0, T_surv, 5001)

        surv_udd = survival_curve_udd(hat_p, times)
        surv_cfm = survival_curve_cfm(hat_p, times)
        surv_bal = survival_curve_balducci(hat_p, times)
        surv_up = survival_curve_strict_upper(hat_p, times)
        surv_lo = survival_curve_strict_lower(hat_p, times)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, surv_udd, label="UDD", color="steelblue", linewidth=1.5)
        ax.plot(times, surv_cfm, label="CFM", color="darkorange", linewidth=1.5, linestyle="--")
        ax.plot(times, surv_bal, label="Balducci", color="forestgreen", linewidth=1.5, linestyle="-.")
        ax.plot(times, surv_up, label="Strict upper", color="firebrick", linewidth=1.2, linestyle=":")
        ax.plot(times, surv_lo, label="Strict lower", color="purple", linewidth=1.2, linestyle=":")
        ax.set_xlabel("Years from start")
        ax.set_ylabel(r"${}_{s}p_x$")
        ax.set_title(f"Survival curves: age x={x}, T={T_surv} years")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"exp1_survival_x{x}_T{T_surv}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")

        # Validation: monotonicity
        assert np.all(np.diff(surv_udd) <= 1e-12), "UDD not monotone!"
        assert np.all(np.diff(surv_cfm) <= 1e-12), "CFM not monotone!"
        assert np.all(np.diff(surv_bal) <= 1e-12), "Balducci not monotone!"
        print(f"  x={x}: survival curves monotone ✓")

        # Validation: strict ordering at non-integer times
        interior_mask = (times % 1 > 0.05) & (times % 1 < 0.95)
        if interior_mask.any():
            assert np.all(surv_up[interior_mask] >= surv_udd[interior_mask] - 1e-10), \
                "Strict upper not >= UDD!"
            assert np.all(surv_lo[interior_mask] <= surv_udd[interior_mask] + 1e-10), \
                "Strict lower not <= UDD!"
            print(f"  x={x}: strict ordering verified ✓")

    # -----------------------------------------------------------------
    # Part B: GMAB strict invariance
    # -----------------------------------------------------------------
    print("\n--- Part B: GMAB strict invariance ---")

    invariance_records = []
    for x in config.AGES:
        for T in config.MATURITY_GRID:
            try:
                hat_p = get_annual_survival_probs(lx, x, T)
            except ValueError:
                continue
            inv = verify_gmab_invariance(
                hat_p, T, A0, r_g, pi_S, pi_P, sigma_S, sigma_r, kappa, rho,
                fwd_params, tol=config.PRICING_TOL,
            )
            invariance_records.append({
                "age": x, "T": T,
                "V_a": inv["value_a"],
                "V_b": inv["value_b"],
                "V_c": inv["value_c"],
                "max_diff": inv["max_diff"],
                "passes": inv["passes"],
                "surv_T": inv["cumulative_survival_T"],
                "C0": inv["C0_T"],
            })
            status = "✓" if inv["passes"] else "✗"
            print(
                f"  x={x:3d}, T={T:2d}: V={inv['value_a']:.4f}, "
                f"max_diff={inv['max_diff']:.2e} {status}"
            )

    df_inv = pd.DataFrame(invariance_records)
    df_inv.to_csv(os.path.join(RESULTS_DIR, "exp1_gmab_invariance.csv"), index=False)

    # Print invariance table
    print("\nGMAB Invariance Table:")
    print(df_inv[["age", "T", "V_a", "max_diff", "passes"]].to_string(index=False))

    # -----------------------------------------------------------------
    # Part C: GMIB strict bounds and benchmarks
    # -----------------------------------------------------------------
    print("\n--- Part C: GMIB strict bounds ---")

    gmib_records = []
    for x in [40, 60, 80]:
        for T in config.MATURITY_GRID:
            try:
                hat_p = get_annual_survival_probs(lx, x, T)
            except ValueError:
                continue

            upper = gmib_strict_upper(
                hat_p, T, A0, r_g, r_I, beta, pi_S, pi_P, sigma_S, sigma_r, kappa, rho,
                fwd_params, n_grid=n_grid,
            )
            lower = gmib_strict_lower(
                hat_p, T, A0, r_g, r_I, beta, pi_S, pi_P, sigma_S, sigma_r, kappa, rho,
                fwd_params, n_grid=n_grid,
            )
            v_udd = gmib_benchmark(
                hat_p, T, "UDD", A0, r_g, r_I, beta, pi_S, pi_P, sigma_S, sigma_r,
                kappa, rho, fwd_params, n_grid=n_grid,
            )
            v_cfm = gmib_benchmark(
                hat_p, T, "CFM", A0, r_g, r_I, beta, pi_S, pi_P, sigma_S, sigma_r,
                kappa, rho, fwd_params, n_grid=n_grid,
            )
            v_bal = gmib_benchmark(
                hat_p, T, "Balducci", A0, r_g, r_I, beta, pi_S, pi_P, sigma_S, sigma_r,
                kappa, rho, fwd_params, n_grid=n_grid,
            )

            rel_width = 100.0 * (upper - lower) / lower if lower > 0 else float("nan")
            gmib_records.append({
                "age": x, "T": T,
                "upper": upper, "lower": lower,
                "UDD": v_udd, "CFM": v_cfm, "Balducci": v_bal,
                "rel_width_pct": rel_width,
            })
            print(
                f"  x={x:2d}, T={T:2d}: "
                f"lower={lower:.4f}, upper={upper:.4f}, "
                f"UDD={v_udd:.4f}, Balducci={v_bal:.4f}, "
                f"rel_width={rel_width:.2f}%"
            )

    df_gmib = pd.DataFrame(gmib_records)
    df_gmib.to_csv(os.path.join(RESULTS_DIR, "exp1_gmib_bounds.csv"), index=False)

    # -----------------------------------------------------------------
    # Plots: GMIB values vs maturity
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for idx, x in enumerate([40, 60, 80]):
        ax = axes[idx]
        sub = df_gmib[df_gmib["age"] == x].copy()
        if sub.empty:
            continue
        ax.fill_between(sub["T"], sub["lower"], sub["upper"],
                        alpha=0.25, color="gray", label="Strict interval")
        ax.plot(sub["T"], sub["upper"], "r--", linewidth=1.5, label="Strict upper")
        ax.plot(sub["T"], sub["lower"], "b--", linewidth=1.5, label="Strict lower")
        ax.plot(sub["T"], sub["UDD"], "b-", linewidth=1.5, label="UDD")
        ax.plot(sub["T"], sub["CFM"], "g-", linewidth=1.5, label="CFM")
        ax.plot(sub["T"], sub["Balducci"], "k-.", linewidth=1.5, label="Balducci")
        ax.set_title(f"GMIB bounds: age {x}")
        ax.set_xlabel("Maturity T")
        ax.set_ylabel("GMIB value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Strict GMIB Upper and Lower Bounds vs Maturity")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_gmib_bounds.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: exp1_gmib_bounds.png")

    # Relative width plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"40": "steelblue", "60": "darkorange", "80": "forestgreen"}
    for x in [40, 60, 80]:
        sub = df_gmib[df_gmib["age"] == x]
        ax.plot(sub["T"], sub["rel_width_pct"], marker="o",
                label=f"x={x}", color=colors[str(x)])
    ax.set_xlabel("Maturity T")
    ax.set_ylabel(r"Relative width $100 \cdot (\bar{V}_0 - \underline{V}_0)/\underline{V}_0$ (%)")
    ax.set_title("GMIB strict bound relative width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_gmib_relative_width.png"), dpi=150)
    plt.close(fig)
    print("Saved: exp1_gmib_relative_width.png")

    # -----------------------------------------------------------------
    # Save combined CSV
    # -----------------------------------------------------------------
    df_all = df_gmib.copy()
    df_all.to_csv(os.path.join(RESULTS_DIR, "exp1_results.csv"), index=False)

    # -----------------------------------------------------------------
    # Validation checks
    # -----------------------------------------------------------------
    print("\n--- Validation Checks ---")
    fails = 0
    for _, row in df_gmib.iterrows():
        if row["upper"] < row["lower"] - 1e-6:
            print(f"  FAIL: upper < lower at x={row['age']}, T={row['T']}")
            fails += 1
        if row["upper"] < row["UDD"] - 1e-4:
            print(f"  FAIL: upper < UDD at x={row['age']}, T={row['T']}")
            fails += 1
        if row["lower"] > row["UDD"] + 1e-4:
            print(f"  FAIL: lower > UDD at x={row['age']}, T={row['T']}")
            fails += 1
    if fails == 0:
        print("  All GMIB ordering checks passed ✓")
    else:
        print(f"  {fails} ordering check(s) FAILED ✗")

    print("\n=== Experiment 1 complete ===")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    run_exp1()
