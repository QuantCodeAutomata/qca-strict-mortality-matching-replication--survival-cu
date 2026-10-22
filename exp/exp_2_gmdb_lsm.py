"""
exp_2_gmdb_lsm.py
=================
Experiment 2: Strict GMDB bounds via Least-Squares Monte Carlo.

Note on path counts: Paper uses N_TRAIN=200,000 and N_EVAL=400,000.
For computational feasibility, we use configurable counts defaulting to
N_TRAIN=20,000 and N_EVAL=40,000 in this replication.
Set USE_PAPER_PATHS=True to use the paper's exact counts (slow).

Maturity grid: T in {1, 2, 3, 5, 7, 10, 15} (documented replication choice).
"""

import os
import sys
import time
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

from mortality import load_life_table, get_annual_survival_probs, get_cumulative_survival
from finance import forward_rate, zero_coupon_bond
from lsm_gmdb import (
    simulate_paths_annual,
    lsm_basis,
    _lsm_train,
    annual_gmdb_bounds,
    strict_gmdb_bounds,
    stopping_diagnostics,
)
import config

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path count configuration
USE_PAPER_PATHS = False
N_TRAIN = config.N_TRAIN if USE_PAPER_PATHS else config.N_TRAIN_FAST
N_EVAL = config.N_EVAL if USE_PAPER_PATHS else config.N_EVAL_FAST
N_STEPS = config.N_STEPS_PER_YEAR


def compute_annual_weights(hat_p: np.ndarray, T: int) -> np.ndarray:
    """Compute death probability weights w_j for each year j."""
    weights = np.empty(T)
    for j in range(T):
        cum = float(np.prod(hat_p[:j])) if j > 0 else 1.0
        weights[j] = (1.0 - hat_p[j]) * cum
    return weights


def gmdb_benchmark_mc(
    T: int,
    n_paths: int,
    n_steps: int,
    hat_p: np.ndarray,
    assumption: str,
    r0: float,
    kappa: float,
    sigma_r: float,
    pi_S: float,
    sigma_S: float,
    pi_P: float,
    rho: float,
    A0: float,
    r_g: float,
    fwd_params: dict,
    seed: int = 42,
) -> float:
    """
    Monte Carlo benchmark GMDB value under UDD/CFM/Balducci.

    Uses within-year death timing implied by the given assumption
    to weight discounted payoffs.

    Parameters
    ----------
    T : int
        Maturity in years.
    assumption : str
        'UDD', 'CFM', or 'Balducci'.
    ... (standard financial parameters)

    Returns
    -------
    float
        Benchmark GMDB value.
    """
    total_value = 0.0
    for j in range(T):
        cum_surv = float(np.prod(hat_p[:j])) if j > 0 else 1.0
        q_j = 1.0 - hat_p[j]
        if q_j < 1e-12:
            continue

        # Death time density within year j: depends on assumption
        # UDD: uniform on [j, j+1)
        # CFM: exponential within year
        # Balducci: Balducci density within year
        # For GMDB benchmark, integrate E[disc * payoff | death in (j,j+1)] over death time

        # Simulate paths for this year
        data = simulate_paths_annual(
            j=j, n_paths=n_paths, n_steps=n_steps,
            r0=r0, kappa=kappa, sigma_r=sigma_r,
            pi_S=pi_S, sigma_S=sigma_S, pi_P=pi_P, rho=rho,
            A0=A0, r_g=r_g, fwd_params=fwd_params,
            seed=seed + j * 1000,
        )
        disc = data["discount"]  # (n_paths, n_steps+1)
        payoff = data["payoff"]  # (n_paths, n_steps+1)

        # For each path, average the death-weighted expected payoff
        # UDD: uniform death within year => average over all steps
        if assumption == "UDD":
            weights_within = np.ones(n_steps + 1) / (n_steps + 1)
        elif assumption == "CFM":
            # CFM: exponential density
            mu = -np.log(hat_p[j])  # constant force
            dt = 1.0 / n_steps
            t_steps = np.arange(n_steps + 1) * dt
            weights_within = np.exp(-mu * t_steps) * mu
            weights_within /= (weights_within.sum() + 1e-16)
        elif assumption == "Balducci":
            # Balducci density: f(u) = q_j * hat_p[j] / (hat_p[j] + u*q_j)^2
            dt = 1.0 / n_steps
            u_steps = np.arange(n_steps + 1) * dt
            q_j_val = 1.0 - hat_p[j]
            p_j = hat_p[j]
            denom = (p_j + u_steps * q_j_val) ** 2
            density = q_j_val * p_j / (denom + 1e-16)
            density[denom <= 0] = 0.0
            total_d = density.sum()
            weights_within = density / (total_d + 1e-16) if total_d > 0 else np.ones(n_steps + 1) / (n_steps + 1)
        else:
            raise ValueError(f"Unknown assumption: {assumption}")

        # Weighted average discounted payoff for this year
        # E[disc_j(tau) * payoff_j(tau)] weighted by death timing
        val_j = np.mean(np.sum(disc * payoff * weights_within[np.newaxis, :], axis=1))
        total_value += cum_surv * q_j * val_j

    return total_value


def run_exp2() -> None:
    """Main runner for Experiment 2."""
    print("=" * 60)
    print("Experiment 2: GMDB Bounds via LSM")
    print(f"  N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_STEPS={N_STEPS}")
    if not USE_PAPER_PATHS:
        print("  NOTE: Using reduced path counts for speed. Set USE_PAPER_PATHS=True for full replication.")
    print("=" * 60)

    lt_path = os.path.join(ROOT, "data", "belgian_female_lt.csv")
    df_lt = load_life_table(lt_path)
    lx = df_lt["lx"].values

    fwd_params = config.FWD_PARAMS
    A0 = config.A0
    r_g = config.R_G
    r0 = config.R0
    pi_S = config.PI_S
    pi_P = config.PI_P
    sigma_S = config.SIGMA_S
    sigma_r = config.SIGMA_R
    kappa = config.KAPPA
    rho = config.RHO
    seed = config.RNG_SEED

    records = []

    for x in config.AGES_GMDB:
        for T in config.MATURITY_GRID:
            t_start = time.time()
            try:
                hat_p = get_annual_survival_probs(lx, x, T)
            except ValueError as e:
                print(f"  Skipping x={x}, T={T}: {e}")
                continue

            # Annual death weights
            weights = compute_annual_weights(hat_p, T)
            surv_T = float(np.prod(hat_p[:T]))
            weight_sum = float(np.sum(weights))
            print(f"\n  x={x}, T={T}: w_sum={weight_sum:.4f}, 1-surv_T={1-surv_T:.4f}")

            # Strict GMDB bounds via LSM
            result = strict_gmdb_bounds(
                hat_p=hat_p, T=T,
                n_train=N_TRAIN, n_eval=N_EVAL, n_steps=N_STEPS,
                r0=r0, kappa=kappa, sigma_r=sigma_r,
                pi_S=pi_S, sigma_S=sigma_S, pi_P=pi_P, rho=rho,
                A0=A0, r_g=r_g, fwd_params=fwd_params, seed=seed,
            )

            # Benchmarks via MC
            v_udd = gmdb_benchmark_mc(
                T, N_EVAL // T, N_STEPS, hat_p, "UDD",
                r0, kappa, sigma_r, pi_S, sigma_S, pi_P, rho, A0, r_g, fwd_params, seed,
            )
            v_cfm = gmdb_benchmark_mc(
                T, N_EVAL // T, N_STEPS, hat_p, "CFM",
                r0, kappa, sigma_r, pi_S, sigma_S, pi_P, rho, A0, r_g, fwd_params, seed,
            )
            v_bal = gmdb_benchmark_mc(
                T, N_EVAL // T, N_STEPS, hat_p, "Balducci",
                r0, kappa, sigma_r, pi_S, sigma_S, pi_P, rho, A0, r_g, fwd_params, seed,
            )

            elapsed = time.time() - t_start
            print(
                f"    upper={result['upper']:.4f}, lower={result['lower']:.4f}, "
                f"UDD={v_udd:.4f}, Bal={v_bal:.4f} [{elapsed:.1f}s]"
            )

            records.append({
                "age": x, "T": T,
                "upper": result["upper"],
                "lower": result["lower"],
                "se_upper": result["se_upper"],
                "se_lower": result["se_lower"],
                "UDD": v_udd,
                "CFM": v_cfm,
                "Balducci": v_bal,
                "weight_sum": weight_sum,
                "surv_T": surv_T,
            })

    df_res = pd.DataFrame(records)
    df_res.to_csv(os.path.join(RESULTS_DIR, "exp2_results.csv"), index=False)
    print(f"\nResults saved to exp2_results.csv")

    # -----------------------------------------------------------------
    # Stopping diagnostics for x=80, T=10
    # -----------------------------------------------------------------
    print("\n--- Stopping diagnostics: x=80, T=10 ---")
    try:
        hat_p_diag = get_annual_survival_probs(lx, 80, 10)
        diag_records = []
        for j in range(10):
            data = simulate_paths_annual(
                j=j, n_paths=min(N_EVAL, 20000), n_steps=N_STEPS,
                r0=r0, kappa=kappa, sigma_r=sigma_r,
                pi_S=pi_S, sigma_S=sigma_S, pi_P=pi_P, rho=rho,
                A0=A0, r_g=r_g, fwd_params=fwd_params, seed=seed + j,
            )
            stop_up = _lsm_train(
                data["discount"], data["payoff"], data["log_m"], data["r"],
                min(N_EVAL, 20000), N_STEPS, maximize=True,
            )
            stop_lo = _lsm_train(
                data["discount"], data["payoff"], data["log_m"], data["r"],
                min(N_EVAL, 20000), N_STEPS, maximize=False,
            )
            diag = stopping_diagnostics(stop_up, stop_lo, N_STEPS)
            diag_records.append({
                "year": j,
                "frac_left_upper": diag["upper"]["frac_left"],
                "frac_interior_upper": diag["upper"]["frac_interior"],
                "frac_right_upper": diag["upper"]["frac_right"],
                "frac_left_lower": diag["lower"]["frac_left"],
                "frac_interior_lower": diag["lower"]["frac_interior"],
                "frac_right_lower": diag["lower"]["frac_right"],
            })
            print(
                f"  Year {j}: upper=[L:{diag['upper']['frac_left']:.2f} "
                f"I:{diag['upper']['frac_interior']:.2f} "
                f"R:{diag['upper']['frac_right']:.2f}], "
                f"lower=[L:{diag['lower']['frac_left']:.2f} "
                f"I:{diag['lower']['frac_interior']:.2f} "
                f"R:{diag['lower']['frac_right']:.2f}]"
            )

        df_diag = pd.DataFrame(diag_records)
        df_diag.to_csv(os.path.join(RESULTS_DIR, "exp2_stopping_diagnostics.csv"), index=False)

        # Interior stopping time histogram (year 0, upper)
        data0 = simulate_paths_annual(
            j=0, n_paths=min(N_EVAL, 20000), n_steps=N_STEPS,
            r0=r0, kappa=kappa, sigma_r=sigma_r,
            pi_S=pi_S, sigma_S=sigma_S, pi_P=pi_P, rho=rho,
            A0=A0, r_g=r_g, fwd_params=fwd_params, seed=seed,
        )
        stop_up0 = _lsm_train(
            data0["discount"], data0["payoff"], data0["log_m"], data0["r"],
            min(N_EVAL, 20000), N_STEPS, maximize=True,
        )
        diag0 = stopping_diagnostics(stop_up0, stop_up0, N_STEPS)
        interior_times = diag0["upper"]["interior_stopping_times"]

        if len(interior_times) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(interior_times, bins=20, color="steelblue", edgecolor="white",
                    alpha=0.8, density=True)
            ax.set_xlabel("Interior stopping time (fraction of year)")
            ax.set_ylabel("Density")
            ax.set_title("Distribution of interior stopping times (year 0, upper policy, x=80)")
            plt.tight_layout()
            fig.savefig(os.path.join(RESULTS_DIR, "exp2_stopping_hist.png"), dpi=150)
            plt.close(fig)
            print("  Saved: exp2_stopping_hist.png")
    except Exception as e:
        print(f"  Diagnostics failed: {e}")

    # -----------------------------------------------------------------
    # Plots: GMDB bounds vs maturity
    # -----------------------------------------------------------------
    if not df_res.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors_bnd = {"upper": "firebrick", "lower": "navy"}
        for idx, x in enumerate(config.AGES_GMDB):
            ax = axes[idx]
            sub = df_res[df_res["age"] == x]
            if sub.empty:
                continue
            ax.fill_between(sub["T"], sub["lower"], sub["upper"],
                            alpha=0.2, color="gray", label="Strict interval")
            ax.plot(sub["T"], sub["upper"], "r--", linewidth=1.5, label="Strict upper")
            ax.plot(sub["T"], sub["lower"], "b--", linewidth=1.5, label="Strict lower")
            ax.plot(sub["T"], sub["UDD"], "b-", linewidth=1.5, label="UDD")
            ax.plot(sub["T"], sub["Balducci"], "k-.", linewidth=1.5, label="Balducci")
            ax.set_title(f"GMDB bounds: age {x}")
            ax.set_xlabel("Maturity T")
            ax.set_ylabel("GMDB value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.suptitle("Strict GMDB Upper and Lower Bounds vs Maturity (LSM)")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "exp2_gmdb_bounds.png"), dpi=150)
        plt.close(fig)
        print("Saved: exp2_gmdb_bounds.png")

        # Annual weights plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for idx, x in enumerate(config.AGES_GMDB):
            ax = axes[idx]
            T_demo = 10
            try:
                hp = get_annual_survival_probs(lx, x, T_demo)
                w = compute_annual_weights(hp, T_demo)
                ax.bar(np.arange(T_demo), w, color="steelblue", alpha=0.7, edgecolor="navy")
            except Exception:
                pass
            ax.set_title(f"Annual weights w_j: age {x}, T={T_demo}")
            ax.set_xlabel("Year j")
            ax.set_ylabel("Weight w_j")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "exp2_annual_weights.png"), dpi=150)
        plt.close(fig)
        print("Saved: exp2_annual_weights.png")

    print("\n=== Experiment 2 complete ===")


if __name__ == "__main__":
    run_exp2()
