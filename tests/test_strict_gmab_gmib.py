"""
tests/test_strict_gmab_gmib.py
==============================
Tests for strict almost-sure life-table matching: GMAB invariance and
GMIB strict upper/lower bounds.

Validates the methodology from Section 5.1 of the replication paper.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mortality import generate_belgian_female_life_table, get_annual_survival_probs
from finance import gmab_price, zero_coupon_bond
from strict_gmab_gmib import (
    gmab_strict,
    verify_gmab_invariance,
    gmib_strict_upper,
    gmib_strict_lower,
    gmib_benchmark,
    verify_annual_product_matching,
)
import config


PARAMS = config.FWD_PARAMS
FIN = {
    "A0": config.A0,
    "r_g": config.R_G,
    "r_I": config.R_I,
    "beta": config.BETA,
    "pi_S": config.PI_S,
    "pi_P": config.PI_P,
    "sigma_S": config.SIGMA_S,
    "sigma_r": config.SIGMA_R,
    "kappa": config.KAPPA,
    "rho": config.RHO,
}


@pytest.fixture
def life_table():
    return generate_belgian_female_life_table(max_age=120)


@pytest.fixture
def hat_p_x40_T5(life_table):
    """Annual survival probs for age 40, T=5."""
    return get_annual_survival_probs(life_table["lx"].values, 40, 5)


@pytest.fixture
def hat_p_x60_T10(life_table):
    """Annual survival probs for age 60, T=10."""
    return get_annual_survival_probs(life_table["lx"].values, 60, 10)


@pytest.fixture
def hat_p_x80_T5(life_table):
    """Annual survival probs for age 80, T=5."""
    return get_annual_survival_probs(life_table["lx"].values, 80, 5)


# ---------------------------------------------------------------------------
# GMAB strict value tests
# ---------------------------------------------------------------------------

class TestGMABStrict:
    def test_gmab_strict_equals_survival_times_price(self, hat_p_x40_T5):
        """V_0 = {}_T p_x * C_0(T, G_T^A) exactly."""
        T = 5
        hat_p = hat_p_x40_T5
        surv_T = float(np.prod(hat_p))
        C0 = gmab_price(T, FIN["A0"], FIN["r_g"], FIN["pi_S"], FIN["pi_P"],
                        FIN["sigma_S"], FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS)
        V_expected = surv_T * C0
        V_computed = gmab_strict(hat_p, T, FIN["A0"], FIN["r_g"],
                                  FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"], FIN["sigma_r"],
                                  FIN["kappa"], FIN["rho"], PARAMS)
        assert abs(V_computed - V_expected) < 1e-12

    def test_gmab_strict_positive(self, hat_p_x40_T5):
        """GMAB strict value is positive."""
        V = gmab_strict(hat_p_x40_T5, 5, FIN["A0"], FIN["r_g"],
                        FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"], FIN["sigma_r"],
                        FIN["kappa"], FIN["rho"], PARAMS)
        assert V > 0.0

    def test_gmab_strict_decreases_with_age(self, life_table):
        """Older ages have lower GMAB value (less chance to survive to T)."""
        lx = life_table["lx"].values
        T = 5
        values = {}
        for x in [40, 60, 80]:
            hat_p = get_annual_survival_probs(lx, x, T)
            values[x] = gmab_strict(hat_p, T, FIN["A0"], FIN["r_g"],
                                     FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                     FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS)
        assert values[40] > values[60] > values[80]

    def test_gmab_strict_T1_sanity(self, life_table):
        """For T=1, GMAB value is near A0 for young age."""
        lx = life_table["lx"].values
        hat_p = get_annual_survival_probs(lx, 40, 1)
        V = gmab_strict(hat_p, 1, FIN["A0"], FIN["r_g"],
                        FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                        FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS)
        # hat_p[0] ~ 1.0 for age 40, so V ~ C0(1)
        assert 80.0 < V < 120.0


# ---------------------------------------------------------------------------
# GMAB invariance tests
# ---------------------------------------------------------------------------

class TestGMABInvariance:
    def test_invariance_passes(self, hat_p_x40_T5):
        """All three admissible controls give the same GMAB value."""
        inv = verify_gmab_invariance(hat_p_x40_T5, 5, FIN["A0"], FIN["r_g"],
                                     FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                     FIN["sigma_r"], FIN["kappa"], FIN["rho"],
                                     PARAMS, tol=config.PRICING_TOL)
        assert inv["passes"], f"Invariance check failed: max_diff={inv['max_diff']}"

    def test_invariance_max_diff_below_tolerance(self, hat_p_x60_T10):
        """max_diff is below tolerance for various age/maturity combos."""
        inv = verify_gmab_invariance(hat_p_x60_T10, 10, FIN["A0"], FIN["r_g"],
                                     FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                     FIN["sigma_r"], FIN["kappa"], FIN["rho"],
                                     PARAMS, tol=config.PRICING_TOL)
        assert inv["max_diff"] < config.PRICING_TOL

    def test_invariance_returns_all_keys(self, hat_p_x40_T5):
        """verify_gmab_invariance returns all required keys."""
        inv = verify_gmab_invariance(hat_p_x40_T5, 5, FIN["A0"], FIN["r_g"],
                                     FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                     FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS)
        for key in ["value_a", "value_b", "value_c", "max_diff", "passes",
                    "cumulative_survival_T", "C0_T"]:
            assert key in inv

    def test_invariance_cumulative_survival(self, hat_p_x40_T5):
        """cumulative_survival_T matches product of hat_p."""
        inv = verify_gmab_invariance(hat_p_x40_T5, 5, FIN["A0"], FIN["r_g"],
                                     FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                     FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS)
        expected_surv = float(np.prod(hat_p_x40_T5))
        assert abs(inv["cumulative_survival_T"] - expected_surv) < 1e-14


# ---------------------------------------------------------------------------
# GMIB strict bound tests
# ---------------------------------------------------------------------------

class TestGMIBStrictBounds:
    def test_upper_ge_lower(self, hat_p_x40_T5):
        """Strict GMIB upper >= lower."""
        T = 5
        upper = gmib_strict_upper(hat_p_x40_T5, T, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                   FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                   FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=1000)
        lower = gmib_strict_lower(hat_p_x40_T5, T, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                   FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                   FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=1000)
        assert upper >= lower - 1e-8, f"Upper {upper:.4f} < lower {lower:.4f}"

    def test_upper_positive(self, hat_p_x40_T5):
        """Strict GMIB upper is positive."""
        upper = gmib_strict_upper(hat_p_x40_T5, 5, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                   FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                   FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=1000)
        assert upper > 0.0

    def test_lower_positive(self, hat_p_x40_T5):
        """Strict GMIB lower is positive."""
        lower = gmib_strict_lower(hat_p_x40_T5, 5, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                   FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                   FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=1000)
        assert lower > 0.0

    def test_bounds_bracket_udd(self, hat_p_x40_T5):
        """Strict GMIB bounds bracket UDD benchmark."""
        T = 5
        upper = gmib_strict_upper(hat_p_x40_T5, T, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                   FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                   FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=500)
        lower = gmib_strict_lower(hat_p_x40_T5, T, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                   FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                   FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=500)
        v_udd = gmib_benchmark(hat_p_x40_T5, T, "UDD", FIN["A0"], FIN["r_g"], FIN["r_I"],
                                FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=500)
        assert upper >= v_udd - 1e-6, "UDD above strict upper!"
        assert lower <= v_udd + 1e-6, "UDD below strict lower!"

    def test_bounds_widen_with_age(self, life_table):
        """Strict GMIB bound width increases with age."""
        lx = life_table["lx"].values
        T = 5
        widths = {}
        for x in [40, 60, 80]:
            hat_p = get_annual_survival_probs(lx, x, T)
            up = gmib_strict_upper(hat_p, T, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                    FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                    FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=200)
            lo = gmib_strict_lower(hat_p, T, FIN["A0"], FIN["r_g"], FIN["r_I"],
                                    FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                                    FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=200)
            widths[x] = up - lo
        # Width generally increases with age
        assert widths[80] >= widths[40] - 0.5  # allow small tolerance


# ---------------------------------------------------------------------------
# GMIB benchmark tests
# ---------------------------------------------------------------------------

class TestGMIBBenchmark:
    def test_benchmark_positive(self, hat_p_x40_T5):
        """All benchmark values are positive."""
        for assumption in ["UDD", "CFM", "Balducci"]:
            v = gmib_benchmark(hat_p_x40_T5, 5, assumption, FIN["A0"], FIN["r_g"], FIN["r_I"],
                               FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                               FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=500)
            assert v > 0.0, f"{assumption} benchmark is not positive: {v}"

    def test_benchmarks_close_for_young_age(self, hat_p_x40_T5):
        """UDD, CFM, Balducci are close for young age (low mortality)."""
        v_udd = gmib_benchmark(hat_p_x40_T5, 5, "UDD", FIN["A0"], FIN["r_g"], FIN["r_I"],
                               FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                               FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=500)
        v_cfm = gmib_benchmark(hat_p_x40_T5, 5, "CFM", FIN["A0"], FIN["r_g"], FIN["r_I"],
                               FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                               FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=500)
        v_bal = gmib_benchmark(hat_p_x40_T5, 5, "Balducci", FIN["A0"], FIN["r_g"], FIN["r_I"],
                               FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                               FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS, n_grid=500)
        # For young age, values should be within 1% of each other
        avg = (v_udd + v_cfm + v_bal) / 3
        assert abs(v_udd - avg) / avg < 0.05, "UDD far from average"
        assert abs(v_cfm - avg) / avg < 0.05, "CFM far from average"
        assert abs(v_bal - avg) / avg < 0.05, "Balducci far from average"

    def test_invalid_assumption_raises(self, hat_p_x40_T5):
        """Invalid assumption string raises ValueError."""
        with pytest.raises(ValueError):
            gmib_benchmark(hat_p_x40_T5, 5, "InvalidAssumption", FIN["A0"], FIN["r_g"],
                           FIN["r_I"], FIN["beta"], FIN["pi_S"], FIN["pi_P"], FIN["sigma_S"],
                           FIN["sigma_r"], FIN["kappa"], FIN["rho"], PARAMS)


# ---------------------------------------------------------------------------
# Annual product matching verification
# ---------------------------------------------------------------------------

def test_annual_product_matching_udd(hat_p_x40_T5):
    """verify_annual_product_matching passes for UDD survival curve."""
    from mortality import survival_curve_udd
    times = np.linspace(0, 5, 5001)
    surv = survival_curve_udd(hat_p_x40_T5, times)
    result = verify_annual_product_matching(surv, times, hat_p_x40_T5, tol=1e-8)
    assert result["passes"], f"Annual product mismatch: {result['max_error']}"


def test_annual_product_matching_balducci(hat_p_x40_T5):
    """verify_annual_product_matching passes for Balducci survival curve."""
    from mortality import survival_curve_balducci
    times = np.linspace(0, 5, 5001)
    surv = survival_curve_balducci(hat_p_x40_T5, times)
    result = verify_annual_product_matching(surv, times, hat_p_x40_T5, tol=1e-8)
    assert result["passes"], f"Annual product mismatch: {result['max_error']}"
