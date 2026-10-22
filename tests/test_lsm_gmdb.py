"""
tests/test_lsm_gmdb.py
======================
Tests for the LSM GMDB module: simulation, regression basis,
stopping policies, and annual bound computation.

Uses small path counts to keep tests fast.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mortality import generate_belgian_female_life_table, get_annual_survival_probs
from lsm_gmdb import (
    simulate_paths_annual,
    lsm_basis,
    _lsm_train,
    annual_gmdb_bounds,
    strict_gmdb_bounds,
    stopping_diagnostics,
)
import config


PARAMS = config.FWD_PARAMS
FIN = {
    "r0": config.R0,
    "kappa": config.KAPPA,
    "sigma_r": config.SIGMA_R,
    "pi_S": config.PI_S,
    "sigma_S": config.SIGMA_S,
    "pi_P": config.PI_P,
    "rho": config.RHO,
    "A0": config.A0,
    "r_g": config.R_G,
}

# Reduced sizes for tests
N_TRAIN_TEST = 2000
N_EVAL_TEST = 4000
N_STEPS_TEST = 26  # ~1/4 of paper's 104
SEED = 123


@pytest.fixture
def life_table():
    return generate_belgian_female_life_table(max_age=120)


@pytest.fixture
def hat_p_x40_T3(life_table):
    return get_annual_survival_probs(life_table["lx"].values, 40, 3)


@pytest.fixture
def hat_p_x80_T5(life_table):
    return get_annual_survival_probs(life_table["lx"].values, 80, 5)


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulatePaths:
    def test_output_keys(self):
        """simulate_paths_annual returns expected keys."""
        data = simulate_paths_annual(
            j=0, n_paths=100, n_steps=10,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        for key in ["A", "r", "discount", "payoff", "log_m"]:
            assert key in data

    def test_output_shapes(self):
        """All output arrays have shape (n_paths, n_steps+1)."""
        n_paths, n_steps = 100, 10
        data = simulate_paths_annual(
            j=0, n_paths=n_paths, n_steps=n_steps,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        for key in ["A", "r", "discount", "payoff", "log_m"]:
            assert data[key].shape == (n_paths, n_steps + 1), f"Wrong shape for {key}"

    def test_A_positive(self):
        """Fund values A_t are always positive."""
        data = simulate_paths_annual(
            j=0, n_paths=200, n_steps=20,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        assert np.all(data["A"] > 0.0)

    def test_discount_starts_at_one(self):
        """Discount factor at t=0 is 1.0."""
        data = simulate_paths_annual(
            j=0, n_paths=100, n_steps=10,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        assert np.allclose(data["discount"][:, 0], 1.0)

    def test_discount_decreasing(self):
        """Discount factor decreases over time (positive rates)."""
        data = simulate_paths_annual(
            j=0, n_paths=500, n_steps=20,
            r0=0.05, kappa=FIN["kappa"], sigma_r=0.01,
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        # On average, discount should decrease
        mean_disc = np.mean(data["discount"], axis=0)
        assert mean_disc[-1] < mean_disc[0]

    def test_payoff_is_max_A_G(self):
        """Payoff = max(A, G) >= A and >= G."""
        data = simulate_paths_annual(
            j=0, n_paths=200, n_steps=10,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        A = data["A"]
        payoff = data["payoff"]
        n_steps = A.shape[1] - 1
        G0 = 100.0
        for k in range(n_steps + 1):
            t_k = k / n_steps
            G_t = G0 * math.exp(config.R_G * t_k)
            assert np.all(payoff[:, k] >= A[:, k] - 1e-8)
            assert np.all(payoff[:, k] >= G_t - 1e-6)

    def test_reproducibility(self):
        """Same seed produces identical paths."""
        d1 = simulate_paths_annual(
            j=0, n_paths=100, n_steps=10, r0=FIN["r0"],
            kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=42,
        )
        d2 = simulate_paths_annual(
            j=0, n_paths=100, n_steps=10, r0=FIN["r0"],
            kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=42,
        )
        assert np.allclose(d1["A"], d2["A"])


# ---------------------------------------------------------------------------
# LSM basis tests
# ---------------------------------------------------------------------------

class TestLSMBasis:
    def test_shape(self):
        """Basis matrix has shape (n_paths, 10)."""
        n = 100
        lm = np.random.randn(n)
        r = np.random.randn(n)
        X = lsm_basis(lm, r)
        assert X.shape == (n, 10)

    def test_intercept_column(self):
        """First column is all ones."""
        n = 50
        X = lsm_basis(np.zeros(n), np.zeros(n))
        assert np.all(X[:, 0] == 1.0)

    def test_pure_X1(self):
        """X1 column: at r=0, columns 1 is X1."""
        X1 = np.array([1.0, 2.0, 3.0])
        X = lsm_basis(X1, np.zeros(3))
        assert np.allclose(X[:, 1], X1)

    def test_pure_X2(self):
        """X2 column: at lm=0, column 2 is X2."""
        X2 = np.array([0.1, 0.2, 0.3])
        X = lsm_basis(np.zeros(3), X2)
        assert np.allclose(X[:, 2], X2)

    def test_degree_3_monomials(self):
        """X1^3 and X2^3 are in the basis."""
        X1 = np.array([2.0])
        X2 = np.array([0.5])
        X = lsm_basis(X1, X2)
        assert abs(X[0, 6] - X1[0] ** 3) < 1e-14  # X1^3
        assert abs(X[0, 9] - X2[0] ** 3) < 1e-14  # X2^3


# ---------------------------------------------------------------------------
# Annual GMDB bounds tests
# ---------------------------------------------------------------------------

class TestAnnualGMDBBounds:
    def test_upper_ge_lower(self):
        """For each policy year, upper bound >= lower bound."""
        for j in [0, 1, 2]:
            upper, lower, se_u, se_l, _, _ = annual_gmdb_bounds(
                j=j, n_train=N_TRAIN_TEST, n_eval=N_EVAL_TEST, n_steps=N_STEPS_TEST,
                r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
                pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
                rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
                fwd_params=PARAMS, seed=SEED,
            )
            assert upper >= lower - 1.0, f"Upper {upper:.3f} < lower {lower:.3f} at j={j}"

    def test_upper_bounded_by_payoff(self):
        """Upper bound is bounded by a reasonable payoff range."""
        upper, lower, _, _, _, _ = annual_gmdb_bounds(
            j=0, n_train=N_TRAIN_TEST, n_eval=N_EVAL_TEST, n_steps=N_STEPS_TEST,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        assert 50 < upper < 250, f"Upper bound out of range: {upper}"
        assert 50 < lower < 250, f"Lower bound out of range: {lower}"

    def test_standard_errors_finite(self):
        """Standard errors are finite."""
        _, _, se_u, se_l, _, _ = annual_gmdb_bounds(
            j=0, n_train=N_TRAIN_TEST, n_eval=N_EVAL_TEST, n_steps=N_STEPS_TEST,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        assert math.isfinite(se_u) and se_u >= 0.0
        assert math.isfinite(se_l) and se_l >= 0.0


# ---------------------------------------------------------------------------
# Strict GMDB bounds (aggregated) tests
# ---------------------------------------------------------------------------

class TestStrictGMDBBounds:
    def test_weight_sum(self, hat_p_x40_T3):
        """Annual weights sum to 1 - surv_T."""
        result = strict_gmdb_bounds(
            hat_p=hat_p_x40_T3, T=3,
            n_train=N_TRAIN_TEST, n_eval=N_EVAL_TEST, n_steps=N_STEPS_TEST,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        surv_T = float(np.prod(hat_p_x40_T3[:3]))
        assert abs(result["weight_sum"] - (1.0 - surv_T)) < 1e-8

    def test_total_upper_ge_lower(self, hat_p_x40_T3):
        """Total strict upper >= total strict lower."""
        result = strict_gmdb_bounds(
            hat_p=hat_p_x40_T3, T=3,
            n_train=N_TRAIN_TEST, n_eval=N_EVAL_TEST, n_steps=N_STEPS_TEST,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        assert result["upper"] >= result["lower"] - 1.0

    def test_output_keys(self, hat_p_x40_T3):
        """Result dict has all required keys."""
        result = strict_gmdb_bounds(
            hat_p=hat_p_x40_T3, T=3,
            n_train=N_TRAIN_TEST, n_eval=N_EVAL_TEST, n_steps=N_STEPS_TEST,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        for key in ["upper", "lower", "weights", "annual_upper", "annual_lower"]:
            assert key in result

    def test_high_mortality_age_has_nontrivial_weights(self, hat_p_x80_T5):
        """For high mortality age, weights are positive and nontrivial."""
        result = strict_gmdb_bounds(
            hat_p=hat_p_x80_T5, T=5,
            n_train=N_TRAIN_TEST, n_eval=N_EVAL_TEST, n_steps=N_STEPS_TEST,
            r0=FIN["r0"], kappa=FIN["kappa"], sigma_r=FIN["sigma_r"],
            pi_S=FIN["pi_S"], sigma_S=FIN["sigma_S"], pi_P=FIN["pi_P"],
            rho=FIN["rho"], A0=FIN["A0"], r_g=FIN["r_g"],
            fwd_params=PARAMS, seed=SEED,
        )
        # At x=80, mortality is high; weights should be significant
        assert np.all(result["weights"] >= 0.0)
        assert float(np.sum(result["weights"])) > 0.1


# ---------------------------------------------------------------------------
# Stopping diagnostics tests
# ---------------------------------------------------------------------------

class TestStoppingDiagnostics:
    def test_fractions_sum_to_one(self):
        """Left + interior + right fractions sum to 1."""
        n_paths = 1000
        n_steps = 26
        rng = np.random.default_rng(42)
        stop_up = rng.integers(0, n_steps + 1, size=n_paths)
        stop_lo = rng.integers(0, n_steps + 1, size=n_paths)
        diag = stopping_diagnostics(stop_up, stop_lo, n_steps)
        for key in ["upper", "lower"]:
            total = diag[key]["frac_left"] + diag[key]["frac_interior"] + diag[key]["frac_right"]
            assert abs(total - 1.0) < 1e-10, f"{key} fractions sum to {total}"

    def test_all_left_endpoint(self):
        """All stopping at 0 gives frac_left=1."""
        n = 500
        stop = np.zeros(n, dtype=int)
        diag = stopping_diagnostics(stop, stop, 26)
        assert abs(diag["upper"]["frac_left"] - 1.0) < 1e-14

    def test_all_right_endpoint(self):
        """All stopping at n_steps gives frac_right=1."""
        n = 500
        n_steps = 26
        stop = np.full(n, n_steps, dtype=int)
        diag = stopping_diagnostics(stop, stop, n_steps)
        assert abs(diag["upper"]["frac_right"] - 1.0) < 1e-14

    def test_interior_times_in_unit_interval(self):
        """Interior stopping times are in (0, 1)."""
        n = 1000
        n_steps = 26
        stop = np.full(n, n_steps // 2, dtype=int)
        diag = stopping_diagnostics(stop, stop, n_steps)
        interior_times = diag["upper"]["interior_stopping_times"]
        if len(interior_times) > 0:
            assert np.all(interior_times > 0.0)
            assert np.all(interior_times < 1.0)
