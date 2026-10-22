"""
tests/test_relaxed_control.py
==============================
Tests for relaxed mortality control: Balducci baseline hazard,
bang-bang HJB PDE solver, and product-specific bounds.

Validates Section 5.2 methodology (deterministic r, zero market price of risk).
"""

import math
import sys
import os

import numpy as np
import pytest
import scipy.integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mortality import generate_belgian_female_life_table, get_annual_survival_probs
from relaxed_control import (
    balducci_hazard_rate,
    balducci_survival,
    fd_solver_product,
    relaxed_gmab_bounds,
    relaxed_gmib_bounds,
    relaxed_gmdb_bounds,
    balducci_gmab_value,
    balducci_gmib_value,
    balducci_gmdb_value,
)
import config


FIN = {
    "A0": config.A0,
    "r_g": config.R_G,
    "r_I": config.R_I,
    "beta": config.BETA,
    "pi_S": config.PI_S,
    "sigma_S": config.SIGMA_S,
    "r": config.R_DET,
}

# Small grid sizes for fast testing
N_T_TEST = 100
N_X_TEST = 60
DELTA_TEST = 1.0


@pytest.fixture
def life_table():
    return generate_belgian_female_life_table(max_age=120)


@pytest.fixture
def hat_p_x60_T3(life_table):
    return get_annual_survival_probs(life_table["lx"].values, 60, 3)


@pytest.fixture
def hat_p_x40_T5(life_table):
    return get_annual_survival_probs(life_table["lx"].values, 40, 5)


@pytest.fixture
def hat_p_x80_T3(life_table):
    return get_annual_survival_probs(life_table["lx"].values, 80, 3)


# ---------------------------------------------------------------------------
# Balducci hazard rate tests
# ---------------------------------------------------------------------------

class TestBalducciHazardRate:
    def test_nonnegative(self, hat_p_x60_T3):
        """Balducci hazard is non-negative."""
        for t in np.linspace(0.0, 3.0, 31):
            j = min(int(math.floor(t)), len(hat_p_x60_T3) - 1)
            u = t - j
            mu = balducci_hazard_rate(hat_p_x60_T3, j, u)
            assert mu >= 0.0

    def test_hazard_at_year_start(self, hat_p_x60_T3):
        """Balducci hazard at u=0: (1 - hat_p[j]) / hat_p[j]."""
        j = 0
        p = hat_p_x60_T3[j]
        expected = (1.0 - p) / p
        val = balducci_hazard_rate(hat_p_x60_T3, j, 0.0)
        assert abs(val - expected) < 1e-12

    def test_hazard_integrates_to_minus_log_p(self, hat_p_x60_T3):
        """Integral of Balducci hazard over year = -log(hat_p[j])."""
        for j in range(len(hat_p_x60_T3)):
            integral, _ = scipy.integrate.quad(
                lambda u: balducci_hazard_rate(hat_p_x60_T3, j, u),
                0.0, 1.0,
            )
            expected = -math.log(hat_p_x60_T3[j])
            assert abs(integral - expected) < 1e-6, f"Year {j}: {integral} vs {expected}"


class TestBalducciSurvival:
    def test_at_zero(self, hat_p_x60_T3):
        """Balducci survival at t=0 is 1."""
        assert abs(balducci_survival(hat_p_x60_T3, 0.0) - 1.0) < 1e-14

    def test_at_integer(self, hat_p_x60_T3):
        """Balducci survival at integer j equals cumulative product."""
        for j in range(1, len(hat_p_x60_T3) + 1):
            val = balducci_survival(hat_p_x60_T3, float(j))
            expected = float(np.prod(hat_p_x60_T3[:j]))
            assert abs(val - expected) < 1e-10, f"Mismatch at j={j}"

    def test_monotone(self, hat_p_x60_T3):
        """Balducci survival is non-increasing."""
        times = np.linspace(0, 3, 100)
        vals = [balducci_survival(hat_p_x60_T3, t) for t in times]
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-10)

    def test_positive(self, hat_p_x60_T3):
        """Balducci survival is positive for finite ages."""
        for t in [0.0, 0.5, 1.0, 2.0, 2.9]:
            val = balducci_survival(hat_p_x60_T3, t)
            assert val > 0.0


# ---------------------------------------------------------------------------
# PDE solver tests
# ---------------------------------------------------------------------------

class TestFDSolverProduct:
    def test_gmab_value_positive(self, hat_p_x60_T3):
        """GMAB PDE value at y=0 is positive."""
        val = fd_solver_product(
            "GMAB", "worst", hat_p_x60_T3, 3.0, DELTA_TEST,
            N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
        )
        assert val > 0.0

    def test_gmib_value_positive(self, hat_p_x60_T3):
        """GMIB PDE value at y=0 is positive."""
        val = fd_solver_product(
            "GMIB", "worst", hat_p_x60_T3, 3.0, DELTA_TEST,
            N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
        )
        assert val > 0.0

    def test_gmdb_value_positive(self, hat_p_x60_T3):
        """GMDB PDE value at y=0 is positive."""
        val = fd_solver_product(
            "GMDB", "worst", hat_p_x60_T3, 3.0, DELTA_TEST,
            N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
        )
        assert val > 0.0

    def test_invalid_product_raises(self, hat_p_x60_T3):
        """Invalid product string raises ValueError."""
        with pytest.raises(ValueError):
            fd_solver_product(
                "INVALID", "worst", hat_p_x60_T3, 3.0, DELTA_TEST,
                N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
                FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
            )

    def test_gmab_worst_le_best(self, hat_p_x60_T3):
        """GMAB worst-case value <= best-case (more mortality = lower survival value)."""
        worst = fd_solver_product(
            "GMAB", "worst", hat_p_x60_T3, 3.0, DELTA_TEST,
            N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
        )
        best = fd_solver_product(
            "GMAB", "best", hat_p_x60_T3, 3.0, DELTA_TEST,
            N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
        )
        assert worst <= best + 0.5  # allow small tolerance for PDE discretization

    def test_gmdb_worst_ge_best(self, hat_p_x60_T3):
        """GMDB worst-case value >= best-case (more mortality = higher death value)."""
        worst = fd_solver_product(
            "GMDB", "worst", hat_p_x60_T3, 3.0, DELTA_TEST,
            N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
        )
        best = fd_solver_product(
            "GMDB", "best", hat_p_x60_T3, 3.0, DELTA_TEST,
            N_T_TEST, N_X_TEST, FIN["r"], FIN["pi_S"], FIN["sigma_S"],
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
        )
        assert worst >= best - 0.5


# ---------------------------------------------------------------------------
# Relaxed bound wrapper tests
# ---------------------------------------------------------------------------

class TestRelaxedGMABBounds:
    def test_upper_ge_lower(self, hat_p_x60_T3):
        """Relaxed GMAB: upper >= lower."""
        lower, upper = relaxed_gmab_bounds(
            hat_p_x60_T3, 3.0, DELTA_TEST,
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        assert upper >= lower - 0.5, f"GMAB upper {upper:.3f} < lower {lower:.3f}"

    def test_larger_delta_widens_interval(self, hat_p_x60_T3):
        """Larger delta -> wider relaxed GMAB bound interval."""
        lo1, up1 = relaxed_gmab_bounds(
            hat_p_x60_T3, 3.0, 0.5,
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        lo2, up2 = relaxed_gmab_bounds(
            hat_p_x60_T3, 3.0, 2.0,
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        interval1 = up1 - lo1
        interval2 = up2 - lo2
        assert interval2 >= interval1 - 0.3, \
            f"Larger delta {interval2:.4f} < smaller delta {interval1:.4f}"


class TestRelaxedGMIBBounds:
    def test_upper_ge_lower(self, hat_p_x60_T3):
        """Relaxed GMIB: upper >= lower."""
        lower, upper = relaxed_gmib_bounds(
            hat_p_x60_T3, 3.0, DELTA_TEST,
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        assert upper >= lower - 0.5

    def test_both_positive(self, hat_p_x60_T3):
        """Both GMIB bounds are positive."""
        lower, upper = relaxed_gmib_bounds(
            hat_p_x60_T3, 3.0, DELTA_TEST,
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        assert lower >= 0.0
        assert upper >= 0.0


class TestRelaxedGMDBBounds:
    def test_upper_ge_lower(self, hat_p_x60_T3):
        """Relaxed GMDB: upper >= lower."""
        lower, upper = relaxed_gmdb_bounds(
            hat_p_x60_T3, 3.0, DELTA_TEST,
            FIN["A0"], FIN["r_g"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        assert upper >= lower - 0.5

    def test_both_positive(self, hat_p_x60_T3):
        """Both GMDB bounds are positive."""
        lower, upper = relaxed_gmdb_bounds(
            hat_p_x60_T3, 3.0, DELTA_TEST,
            FIN["A0"], FIN["r_g"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        assert lower >= 0.0
        assert upper >= 0.0


# ---------------------------------------------------------------------------
# Balducci baseline value tests
# ---------------------------------------------------------------------------

class TestBalducciBaseline:
    def test_gmab_baseline_positive(self, hat_p_x40_T5):
        """Balducci GMAB baseline is positive."""
        val = balducci_gmab_value(hat_p_x40_T5, 5.0, FIN["A0"], FIN["r_g"],
                                  FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST)
        assert val > 0.0

    def test_gmib_baseline_positive(self, hat_p_x40_T5):
        """Balducci GMIB baseline is positive."""
        val = balducci_gmib_value(hat_p_x40_T5, 5.0, FIN["A0"], FIN["r_g"],
                                  FIN["r_I"], FIN["beta"],
                                  FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST)
        assert val > 0.0

    def test_gmdb_baseline_positive(self, hat_p_x40_T5):
        """Balducci GMDB baseline is positive."""
        val = balducci_gmdb_value(hat_p_x40_T5, 5.0, FIN["A0"], FIN["r_g"],
                                  FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST)
        assert val > 0.0

    def test_baseline_within_relaxed_bounds(self, hat_p_x60_T3):
        """Balducci baseline should be within or near the relaxed bounds."""
        baseline = balducci_gmab_value(hat_p_x60_T3, 3.0, FIN["A0"], FIN["r_g"],
                                       FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST)
        lower, upper = relaxed_gmab_bounds(
            hat_p_x60_T3, 3.0, DELTA_TEST,
            FIN["A0"], FIN["r_g"], FIN["r_I"], FIN["beta"],
            FIN["pi_S"], FIN["sigma_S"], FIN["r"], N_T_TEST, N_X_TEST,
        )
        # Baseline (delta=0 limit) should be close to or within [lower, upper]
        assert lower - 1.0 <= baseline <= upper + 1.0, \
            f"Baseline {baseline:.3f} outside [{lower:.3f}, {upper:.3f}]"
