"""
tests/test_finance.py
=====================
Unit tests for the finance module: forward curve, Hull-White bond pricing,
GMAB pricing, and GMIB integrand.

Tests verify mathematical properties described in Proposition 2.2 and
Section 4 of the replication paper.
"""

import math
import sys
import os

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finance import (
    forward_rate,
    forward_rate_array,
    forward_rate_deriv,
    theta,
    B_hw,
    zero_coupon_bond,
    bond_price,
    sigma_Y_squared,
    gmab_price,
    gmib_integrand,
    precompute_gmib_integrand,
)
import config


PARAMS = config.FWD_PARAMS
KAPPA = config.KAPPA
SIGMA_R = config.SIGMA_R


# ---------------------------------------------------------------------------
# Forward curve tests
# ---------------------------------------------------------------------------

class TestForwardRate:
    def test_forward_at_zero(self):
        """f(0,0) = b + c + 0 = b + c."""
        a, b, c, d = PARAMS["a"], PARAMS["b"], PARAMS["c"], PARAMS["d"]
        expected = b + c
        val = forward_rate(0.0, a, b, c, d)
        assert abs(val - expected) < 1e-14

    def test_forward_positive_for_long_maturities(self):
        """f(0,t) -> b as t -> infinity."""
        a, b, c, d = PARAMS["a"], PARAMS["b"], PARAMS["c"], PARAMS["d"]
        val_long = forward_rate(100.0, a, b, c, d)
        assert abs(val_long - b) < 1e-4

    def test_forward_array_vs_scalar(self):
        """Vectorized forward rate matches scalar."""
        a, b, c, d = PARAMS["a"], PARAMS["b"], PARAMS["c"], PARAMS["d"]
        times = np.array([0.0, 1.0, 5.0, 10.0])
        scalar_vals = np.array([forward_rate(t, a, b, c, d) for t in times])
        array_vals = forward_rate_array(times, a, b, c, d)
        assert np.allclose(scalar_vals, array_vals, atol=1e-14)

    def test_forward_derivative_numerical(self):
        """Analytical derivative matches numerical finite difference."""
        a, b, c, d = PARAMS["a"], PARAMS["b"], PARAMS["c"], PARAMS["d"]
        t = 1.5
        h = 1e-5
        analytical = forward_rate_deriv(t, a, b, c, d)
        numerical = (forward_rate(t + h, a, b, c, d) - forward_rate(t - h, a, b, c, d)) / (2 * h)
        assert abs(analytical - numerical) < 1e-8


# ---------------------------------------------------------------------------
# Hull-White B function tests
# ---------------------------------------------------------------------------

class TestBHullWhite:
    def test_B_zero_when_t_equals_T(self):
        """B(T,T) = 0."""
        B = B_hw(5.0, 5.0, KAPPA)
        assert abs(B) < 1e-14

    def test_B_positive_when_t_less_T(self):
        """B(t,T) > 0 for t < T."""
        B = B_hw(1.0, 5.0, KAPPA)
        assert B > 0.0

    def test_B_formula(self):
        """B(t,T) = (1/kappa)*(1 - exp(-kappa*(T-t)))."""
        t, T = 2.0, 7.0
        kappa = 0.1
        expected = (1.0 / kappa) * (1.0 - math.exp(-kappa * (T - t)))
        val = B_hw(t, T, kappa)
        assert abs(val - expected) < 1e-14

    def test_B_upper_bounded_by_1_over_kappa(self):
        """B(t,T) <= 1/kappa."""
        kappa = 0.1
        for T in [1, 5, 10, 50]:
            B = B_hw(0.0, float(T), kappa)
            assert B <= 1.0 / kappa + 1e-14


# ---------------------------------------------------------------------------
# Zero-coupon bond price tests
# ---------------------------------------------------------------------------

class TestZeroCouponBond:
    def test_P00_equals_one(self):
        """P(0,0) = 1."""
        P = zero_coupon_bond(0.0, PARAMS)
        assert abs(P - 1.0) < 1e-12

    def test_P_positive(self):
        """P(0,T) > 0 for all T > 0."""
        for T in [0.5, 1, 5, 10, 15]:
            P = zero_coupon_bond(float(T), PARAMS)
            assert P > 0.0

    def test_P_less_than_one(self):
        """P(0,T) < 1 for positive forward rates."""
        for T in [1, 5, 10]:
            P = zero_coupon_bond(float(T), PARAMS)
            assert P < 1.0, f"P(0,{T}) = {P:.4f} >= 1"

    def test_P_decreasing_in_T(self):
        """P(0,T) is decreasing in T (positive forward rates)."""
        maturities = [1.0, 2.0, 5.0, 10.0, 15.0]
        prices = [zero_coupon_bond(T, PARAMS) for T in maturities]
        diffs = np.diff(prices)
        assert np.all(diffs < 0.0), "P(0,T) not decreasing in T"

    def test_P_theta_calibration(self):
        """theta(t) is computed consistently from forward curve parameters."""
        t = 2.0
        theta_val = theta(t, KAPPA, SIGMA_R, PARAMS["a"], PARAMS["b"], PARAMS["c"], PARAMS["d"])
        assert np.isfinite(theta_val)


# ---------------------------------------------------------------------------
# sigma_Y_squared tests
# ---------------------------------------------------------------------------

class TestSigmaYSquared:
    def test_positive(self):
        """sigma_Y^2 is non-negative."""
        sig2 = sigma_Y_squared(5.0, config.PI_S, config.SIGMA_S, config.PI_P, SIGMA_R, KAPPA, config.RHO, PARAMS)
        assert sig2 >= 0.0

    def test_increases_with_T(self):
        """sigma_Y^2 increases with T (quadratic variation accumulates)."""
        vals = [sigma_Y_squared(T, config.PI_S, config.SIGMA_S, config.PI_P, SIGMA_R, KAPPA, config.RHO, PARAMS)
                for T in [1.0, 5.0, 10.0]]
        assert vals[0] < vals[1] < vals[2]

    def test_zero_volatility_gives_zero(self):
        """Zero pi_S and pi_P means no variance."""
        sig2 = sigma_Y_squared(5.0, 0.0, config.SIGMA_S, 0.0, SIGMA_R, KAPPA, config.RHO, PARAMS)
        assert abs(sig2) < 1e-12

    def test_stock_only_allocation(self):
        """With pi_P=0, sigma_Y^2 = pi_S^2 * sigma_S^2 * T."""
        T = 5.0
        pi_S = 1.0
        sigma_S = 0.2
        sig2 = sigma_Y_squared(T, pi_S, sigma_S, 0.0, SIGMA_R, KAPPA, config.RHO, PARAMS)
        expected = pi_S ** 2 * sigma_S ** 2 * T
        assert abs(sig2 - expected) < 1e-6


# ---------------------------------------------------------------------------
# GMAB pricing tests
# ---------------------------------------------------------------------------

class TestGMABPrice:
    def test_gmab_positive(self):
        """GMAB price is positive."""
        price = gmab_price(5.0, config.A0, config.R_G, config.PI_S, config.PI_P,
                           config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS)
        assert price > 0.0

    def test_gmab_above_atm_price(self):
        """GMAB price >= A0 (portfolio is at least worth A0 in present value)."""
        price = gmab_price(5.0, config.A0, config.R_G, config.PI_S, config.PI_P,
                           config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS)
        # max(A_T, G_T) >= A_T and E[exp(-int r)*A_T] = A0 (by martingale)
        assert price >= config.A0 - 0.1  # small tolerance for numerical errors

    def test_gmab_zero_guarantee_rate(self):
        """With r_g=0, GMAB = max of fund and par."""
        price = gmab_price(1.0, 100.0, 0.0, config.PI_S, config.PI_P,
                           config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS)
        assert price >= 100.0 * zero_coupon_bond(1.0, PARAMS) - 1.0

    def test_gmab_increases_with_volatility(self):
        """GMAB price is increasing in sigma_S (higher optionality)."""
        p_low = gmab_price(5.0, config.A0, config.R_G, config.PI_S, config.PI_P,
                           0.1, SIGMA_R, KAPPA, config.RHO, PARAMS)
        p_high = gmab_price(5.0, config.A0, config.R_G, config.PI_S, config.PI_P,
                            0.3, SIGMA_R, KAPPA, config.RHO, PARAMS)
        assert p_high >= p_low

    def test_gmab_short_maturity(self):
        """GMAB price at T=1 is reasonable."""
        price = gmab_price(1.0, config.A0, config.R_G, config.PI_S, config.PI_P,
                           config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS)
        assert 50.0 < price < 200.0  # sanity range

    def test_gmab_long_maturity(self):
        """GMAB price at T=15 is finite and positive."""
        price = gmab_price(15.0, config.A0, config.R_G, config.PI_S, config.PI_P,
                           config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS)
        assert 0 < price < 1000.0


# ---------------------------------------------------------------------------
# GMIB integrand tests
# ---------------------------------------------------------------------------

class TestGMIBIntegrand:
    def test_at_s0_equals_A0_rI(self):
        """At s=0, C_0(0, G_0^I) = A0 * r_I."""
        val = gmib_integrand(0.0, config.A0, config.R_G, config.R_I, config.BETA,
                             config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R,
                             KAPPA, config.RHO, PARAMS)
        assert abs(val - config.A0 * config.R_I) < 1e-10

    def test_positive_for_all_s(self):
        """C_0(s, G_s^I) >= 0 for all s > 0."""
        for s in [0.1, 1.0, 5.0, 10.0]:
            val = gmib_integrand(s, config.A0, config.R_G, config.R_I, config.BETA,
                                 config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R,
                                 KAPPA, config.RHO, PARAMS)
            assert val >= 0.0, f"Negative GMIB integrand at s={s}: {val}"

    def test_fixed_income_floor(self):
        """C_0(s, G_s^I) >= A0 * r_I * P(0,s)."""
        for s in [1.0, 5.0, 10.0]:
            val = gmib_integrand(s, config.A0, config.R_G, config.R_I, config.BETA,
                                 config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R,
                                 KAPPA, config.RHO, PARAMS)
            floor = config.A0 * config.R_I * zero_coupon_bond(s, PARAMS)
            assert val >= floor - 1e-10, f"GMIB integrand below floor at s={s}"

    def test_beta_zero_reduces_to_fixed_income(self):
        """With beta=0, C_0 = A0 * r_I * P(0,s)."""
        s = 5.0
        val = gmib_integrand(s, config.A0, config.R_G, config.R_I, 0.0,
                             config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R,
                             KAPPA, config.RHO, PARAMS)
        expected = config.A0 * config.R_I * zero_coupon_bond(s, PARAMS)
        assert abs(val - expected) < 1e-8

    def test_increases_with_beta(self):
        """C_0 increases with beta (more participation)."""
        s = 5.0
        v1 = gmib_integrand(s, config.A0, config.R_G, config.R_I, 0.01,
                             config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R,
                             KAPPA, config.RHO, PARAMS)
        v2 = gmib_integrand(s, config.A0, config.R_G, config.R_I, 0.1,
                             config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R,
                             KAPPA, config.RHO, PARAMS)
        assert v2 >= v1


# ---------------------------------------------------------------------------
# Precompute integrand grid tests
# ---------------------------------------------------------------------------

def test_precompute_gmib_integrand_shape():
    """Precomputed grid has correct shape."""
    T, n = 5.0, 100
    times, values = precompute_gmib_integrand(
        T, n, config.A0, config.R_G, config.R_I, config.BETA,
        config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS
    )
    assert len(times) == n + 1
    assert len(values) == n + 1


def test_precompute_gmib_integrand_positive():
    """All precomputed values are non-negative."""
    T, n = 5.0, 50
    _, values = precompute_gmib_integrand(
        T, n, config.A0, config.R_G, config.R_I, config.BETA,
        config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS
    )
    assert np.all(values >= 0.0)


def test_precompute_gmib_integrand_matches_direct():
    """Precomputed values match direct gmib_integrand calls."""
    T, n = 3.0, 30
    times, values = precompute_gmib_integrand(
        T, n, config.A0, config.R_G, config.R_I, config.BETA,
        config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R, KAPPA, config.RHO, PARAMS
    )
    direct = np.array([
        gmib_integrand(s, config.A0, config.R_G, config.R_I, config.BETA,
                       config.PI_S, config.PI_P, config.SIGMA_S, SIGMA_R,
                       KAPPA, config.RHO, PARAMS)
        for s in times
    ])
    assert np.allclose(values, direct, atol=1e-12)
