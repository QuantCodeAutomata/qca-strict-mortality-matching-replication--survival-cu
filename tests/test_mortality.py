"""
tests/test_mortality.py
=======================
Unit tests for the mortality module: life-table construction, within-year
survival curves (UDD, CFM, Balducci), strict extremal survival curves.

Validates mathematical properties described in Section 3 of the paper.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mortality import (
    generate_belgian_female_life_table,
    load_life_table,
    get_annual_survival_probs,
    get_cumulative_survival,
    survival_udd_scalar,
    survival_cfm_scalar,
    survival_balducci_scalar,
    survival_curve_udd,
    survival_curve_cfm,
    survival_curve_balducci,
    survival_curve_strict_upper,
    survival_curve_strict_lower,
    balducci_hazard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_hat_p():
    """Simple test survival probabilities (5 years)."""
    return np.array([0.99, 0.98, 0.97, 0.96, 0.95])


@pytest.fixture
def old_hat_p():
    """High-mortality survival probabilities for old age."""
    return np.array([0.80, 0.75, 0.70, 0.65, 0.60])


@pytest.fixture
def life_table():
    """Generated Belgian female life table."""
    return generate_belgian_female_life_table(max_age=120)


# ---------------------------------------------------------------------------
# Life table generation tests
# ---------------------------------------------------------------------------

def test_life_table_has_required_columns(life_table):
    """Life table must have age, lx, px, qx columns."""
    for col in ["age", "lx", "px", "qx"]:
        assert col in life_table.columns, f"Missing column: {col}"


def test_life_table_radix(life_table):
    """Radix l_0 = 100,000."""
    assert abs(life_table["lx"].iloc[0] - 100_000.0) < 1e-6


def test_life_table_monotone_decreasing(life_table):
    """l_x must be monotone non-increasing."""
    lx = life_table["lx"].values
    diffs = np.diff(lx)
    assert np.all(diffs <= 0.0 + 1e-10), "l_x not monotone non-increasing"


def test_life_table_px_in_unit_interval(life_table):
    """All p_x must lie in [0, 1]."""
    px = life_table["px"].values
    assert np.all(px >= 0.0), "p_x contains negatives"
    assert np.all(px <= 1.0 + 1e-10), "p_x exceeds 1"


def test_life_table_px_qx_sum_to_one(life_table):
    """p_x + q_x = 1 for all ages."""
    sums = life_table["px"] + life_table["qx"]
    # Exclude terminal age where both may be non-standard
    assert np.allclose(sums.values[:-1], 1.0, atol=1e-10)


def test_life_table_covers_required_ages(life_table):
    """Life table must cover ages 40 through 115."""
    ages = life_table["age"].values
    for a in [40, 60, 80, 100, 115]:
        assert a in ages, f"Age {a} not in life table"


def test_life_table_max_age(life_table):
    """Max age default is 120."""
    assert life_table["age"].max() == 120


def test_life_table_custom_max_age():
    """Custom max_age is respected."""
    df = generate_belgian_female_life_table(max_age=100)
    assert df["age"].max() == 100


# ---------------------------------------------------------------------------
# Annual survival probability extraction tests
# ---------------------------------------------------------------------------

def test_get_annual_survival_probs_length(life_table):
    """hat_p length equals ceil(T)."""
    lx = life_table["lx"].values
    hat_p = get_annual_survival_probs(lx, 60, 5.0)
    assert len(hat_p) == 5


def test_get_annual_survival_probs_values_in_unit(life_table):
    """hat_p values in (0, 1]."""
    lx = life_table["lx"].values
    hat_p = get_annual_survival_probs(lx, 60, 10)
    assert np.all(hat_p > 0.0)
    assert np.all(hat_p <= 1.0 + 1e-10)


def test_get_annual_survival_probs_noninteger_T(life_table):
    """ceil(T) years are extracted for non-integer T."""
    lx = life_table["lx"].values
    hat_p = get_annual_survival_probs(lx, 60, 5.5)
    assert len(hat_p) == 6


def test_get_annual_survival_probs_age_100(life_table):
    """Survival probs at old age are low but positive."""
    lx = life_table["lx"].values
    hat_p = get_annual_survival_probs(lx, 100, 5)
    assert np.all(hat_p >= 0.0)
    assert np.all(hat_p < 1.0)


def test_get_annual_survival_out_of_range(life_table):
    """Out-of-range age raises ValueError."""
    lx = life_table["lx"].values
    with pytest.raises(ValueError):
        get_annual_survival_probs(lx, 119, 5)  # would need age 123


# ---------------------------------------------------------------------------
# Cumulative survival tests
# ---------------------------------------------------------------------------

def test_cumulative_survival_j0(simple_hat_p):
    """j=0 cumulative survival is 1."""
    assert get_cumulative_survival(simple_hat_p, 0) == 1.0


def test_cumulative_survival_j1(simple_hat_p):
    """j=1 cumulative survival equals hat_p[0]."""
    assert abs(get_cumulative_survival(simple_hat_p, 1) - simple_hat_p[0]) < 1e-14


def test_cumulative_survival_j5(simple_hat_p):
    """j=5 cumulative survival is product of all hat_p."""
    expected = float(np.prod(simple_hat_p))
    assert abs(get_cumulative_survival(simple_hat_p, 5) - expected) < 1e-14


def test_cumulative_survival_monotone(simple_hat_p):
    """Cumulative survival is non-increasing in j."""
    survs = [get_cumulative_survival(simple_hat_p, j) for j in range(6)]
    diffs = np.diff(survs)
    assert np.all(diffs <= 0.0 + 1e-14)


# ---------------------------------------------------------------------------
# Within-year survival scalar tests
# ---------------------------------------------------------------------------

class TestUDDScalar:
    def test_at_integer_times(self, simple_hat_p):
        """UDD scalar: at integer u, equals cumulative survival."""
        for j in range(5):
            val = survival_udd_scalar(simple_hat_p, float(j))
            expected = get_cumulative_survival(simple_hat_p, j)
            assert abs(val - expected) < 1e-14, f"UDD failed at j={j}"

    def test_half_year(self, simple_hat_p):
        """UDD at u=0.5: linear interpolation."""
        val = survival_udd_scalar(simple_hat_p, 0.5)
        expected = 1.0 - 0.5 * (1.0 - simple_hat_p[0])
        assert abs(val - expected) < 1e-14

    def test_monotone(self, simple_hat_p):
        """UDD survival is non-increasing."""
        times = np.linspace(0, 5, 201)
        vals = [survival_udd_scalar(simple_hat_p, t) for t in times]
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-12)


class TestCFMScalar:
    def test_at_integer_times(self, simple_hat_p):
        """CFM scalar: at integer u, equals cumulative survival."""
        for j in range(5):
            val = survival_cfm_scalar(simple_hat_p, float(j))
            expected = get_cumulative_survival(simple_hat_p, j)
            assert abs(val - expected) < 1e-14

    def test_half_year(self, simple_hat_p):
        """CFM at u=0.5: geometric interpolation."""
        val = survival_cfm_scalar(simple_hat_p, 0.5)
        expected = simple_hat_p[0] ** 0.5
        assert abs(val - expected) < 1e-14

    def test_monotone(self, simple_hat_p):
        """CFM survival is non-increasing."""
        times = np.linspace(0, 5, 201)
        vals = [survival_cfm_scalar(simple_hat_p, t) for t in times]
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-12)


class TestBalducciScalar:
    def test_at_integer_times(self, simple_hat_p):
        """Balducci scalar: at integer u, equals cumulative survival."""
        for j in range(5):
            val = survival_balducci_scalar(simple_hat_p, float(j))
            expected = get_cumulative_survival(simple_hat_p, j)
            assert abs(val - expected) < 1e-10

    def test_formula(self, simple_hat_p):
        """Balducci s_j(y) = p / (y + (1-y)*p)."""
        y = 0.3
        p = simple_hat_p[0]
        expected = p / (y + (1.0 - y) * p)
        val = survival_balducci_scalar(simple_hat_p, y)
        assert abs(val - expected) < 1e-14

    def test_monotone(self, simple_hat_p):
        """Balducci survival is non-increasing."""
        times = np.linspace(0, 5, 201)
        vals = [survival_balducci_scalar(simple_hat_p, t) for t in times]
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-12)


# ---------------------------------------------------------------------------
# Annual product matching tests
# ---------------------------------------------------------------------------

def _check_annual_products(surv_func, hat_p: np.ndarray, tol: float = 1e-10) -> None:
    """Helper: check annual product matching."""
    n = len(hat_p)
    for j in range(1, n + 1):
        sv_before = surv_func(float(j - 1))
        sv_after = surv_func(float(j))
        ratio = sv_after / sv_before
        assert abs(ratio - hat_p[j - 1]) < tol, \
            f"Annual product mismatch at j={j}: {ratio:.8f} vs {hat_p[j-1]:.8f}"


def test_udd_annual_product_matching(simple_hat_p):
    """UDD survival matches annual life-table products exactly."""
    _check_annual_products(
        lambda u: survival_udd_scalar(simple_hat_p, u),
        simple_hat_p, tol=1e-12,
    )


def test_cfm_annual_product_matching(simple_hat_p):
    """CFM survival matches annual life-table products exactly."""
    _check_annual_products(
        lambda u: survival_cfm_scalar(simple_hat_p, u),
        simple_hat_p, tol=1e-12,
    )


def test_balducci_annual_product_matching(simple_hat_p):
    """Balducci survival matches annual life-table products exactly."""
    _check_annual_products(
        lambda u: survival_balducci_scalar(simple_hat_p, u),
        simple_hat_p, tol=1e-10,
    )


# ---------------------------------------------------------------------------
# Vectorized survival curve tests
# ---------------------------------------------------------------------------

def test_udd_curve_values_match_scalar(simple_hat_p):
    """Vectorized UDD matches scalar version pointwise."""
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.7, 3.0, 4.5, 5.0])
    vec_vals = survival_curve_udd(simple_hat_p, times)
    scalar_vals = np.array([survival_udd_scalar(simple_hat_p, t) for t in times])
    assert np.allclose(vec_vals, scalar_vals, atol=1e-14)


def test_cfm_curve_values_match_scalar(simple_hat_p):
    """Vectorized CFM matches scalar version pointwise."""
    times = np.array([0.0, 0.3, 1.0, 2.5, 3.0, 4.0, 5.0])
    vec_vals = survival_curve_cfm(simple_hat_p, times)
    scalar_vals = np.array([survival_cfm_scalar(simple_hat_p, t) for t in times])
    assert np.allclose(vec_vals, scalar_vals, atol=1e-14)


def test_balducci_curve_values_match_scalar(simple_hat_p):
    """Vectorized Balducci matches scalar version pointwise."""
    times = np.array([0.0, 0.25, 1.0, 1.75, 2.0, 3.5, 5.0])
    vec_vals = survival_curve_balducci(simple_hat_p, times)
    scalar_vals = np.array([survival_balducci_scalar(simple_hat_p, t) for t in times])
    assert np.allclose(vec_vals, scalar_vals, atol=1e-14)


# ---------------------------------------------------------------------------
# Strict extremal survival curve tests
# ---------------------------------------------------------------------------

class TestStrictUpperSurvival:
    def test_at_zero(self, simple_hat_p):
        """Strict upper survival at t=0 is 1."""
        val = survival_curve_strict_upper(simple_hat_p, np.array([0.0]))[0]
        assert abs(val - 1.0) < 1e-14

    def test_at_integer_j(self, simple_hat_p):
        """Strict upper at integer j equals {}_j p_x."""
        for j in range(1, 5):
            val = survival_curve_strict_upper(simple_hat_p, np.array([float(j)]))[0]
            expected = get_cumulative_survival(simple_hat_p, j)
            assert abs(val - expected) < 1e-14

    def test_upper_ge_udd(self, simple_hat_p):
        """Strict upper >= UDD at all non-integer times."""
        times = np.linspace(0, 5, 501)
        surv_up = survival_curve_strict_upper(simple_hat_p, times)
        surv_udd = survival_curve_udd(simple_hat_p, times)
        interior = (times % 1 > 0.01) & (times % 1 < 0.99)
        assert np.all(surv_up[interior] >= surv_udd[interior] - 1e-10), \
            "Strict upper < UDD at some interior time!"

    def test_step_function(self, simple_hat_p):
        """Strict upper is constant within each year."""
        # At t=0.1, 0.5, 0.9: all should equal {}_0 p_x = 1.0
        times_yr0 = np.array([0.1, 0.5, 0.9])
        vals = survival_curve_strict_upper(simple_hat_p, times_yr0)
        assert np.all(np.isclose(vals, 1.0, atol=1e-14))


class TestStrictLowerSurvival:
    def test_at_zero(self, simple_hat_p):
        """Strict lower survival at t=0 is 1."""
        val = survival_curve_strict_lower(simple_hat_p, np.array([0.0]))[0]
        assert abs(val - 1.0) < 1e-14

    def test_at_integer_j(self, simple_hat_p):
        """Strict lower at integer j equals {}_j p_x (same as upper)."""
        for j in range(0, 5):
            val = survival_curve_strict_lower(simple_hat_p, np.array([float(j)]))[0]
            expected = get_cumulative_survival(simple_hat_p, j)
            assert abs(val - expected) < 1e-14

    def test_lower_le_udd(self, simple_hat_p):
        """Strict lower <= UDD at all non-integer times."""
        times = np.linspace(0, 5, 501)
        surv_lo = survival_curve_strict_lower(simple_hat_p, times)
        surv_udd = survival_curve_udd(simple_hat_p, times)
        interior = (times % 1 > 0.01) & (times % 1 < 0.99)
        assert np.all(surv_lo[interior] <= surv_udd[interior] + 1e-10), \
            "Strict lower > UDD at some interior time!"

    def test_ordering_strict_upper_ge_lower(self, simple_hat_p):
        """Strict upper >= strict lower pointwise."""
        times = np.linspace(0, 5, 501)
        surv_up = survival_curve_strict_upper(simple_hat_p, times)
        surv_lo = survival_curve_strict_lower(simple_hat_p, times)
        assert np.all(surv_up >= surv_lo - 1e-10), "Strict upper < lower at some time!"


# ---------------------------------------------------------------------------
# Balducci hazard rate tests
# ---------------------------------------------------------------------------

def test_balducci_hazard_positive(simple_hat_p):
    """Balducci hazard is non-negative."""
    for t in [0.0, 0.5, 1.0, 2.7]:
        mu = balducci_hazard(simple_hat_p, t)
        assert mu >= 0.0, f"Negative hazard at t={t}"


def test_balducci_hazard_integrates_to_annual_mortality(simple_hat_p):
    """Integrate Balducci hazard over each year => recovers -log(hat_p[j])."""
    import scipy.integrate as sci
    for j in range(len(simple_hat_p)):
        integral, _ = sci.quad(
            lambda t: balducci_hazard(simple_hat_p, j + t), 0.0, 1.0
        )
        expected = -math.log(simple_hat_p[j])
        assert abs(integral - expected) < 1e-6, \
            f"Balducci hazard integral mismatch in year {j}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_year_hat_p():
    """Single-year survival probs work correctly."""
    hat_p = np.array([0.95])
    val = survival_udd_scalar(hat_p, 0.5)
    assert 0 < val < 1.0

    val_cfm = survival_cfm_scalar(hat_p, 0.5)
    assert 0 < val_cfm < 1.0

    val_bal = survival_balducci_scalar(hat_p, 0.5)
    assert 0 < val_bal < 1.0


def test_perfect_survival():
    """hat_p = 1 means no deaths; all assumptions agree."""
    hat_p = np.array([1.0, 1.0, 1.0])
    for t in [0.0, 0.5, 1.0, 2.0, 3.0]:
        assert abs(survival_udd_scalar(hat_p, t) - 1.0) < 1e-14
        assert abs(survival_cfm_scalar(hat_p, t) - 1.0) < 1e-14
        assert abs(survival_balducci_scalar(hat_p, t) - 1.0) < 1e-14


def test_high_mortality_hat_p(old_hat_p):
    """High-mortality survival probs: all assumptions give values in (0,1)."""
    for t in np.linspace(0, 5, 21):
        for fn in [survival_udd_scalar, survival_cfm_scalar, survival_balducci_scalar]:
            val = fn(old_hat_p, t)
            assert 0 <= val <= 1.0 + 1e-10, f"{fn.__name__} out of range at t={t}"
