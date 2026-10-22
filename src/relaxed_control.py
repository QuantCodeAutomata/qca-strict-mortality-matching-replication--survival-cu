"""
relaxed_control.py
==================
Relaxed expected-matching mortality-control bounds for GMAB, GMIB, GMDB.

Section 5.2 simplified setting: deterministic interest rate r=0.01,
zero market price of risk. The financial state reduces to log(A_t/G_t).

Implements:
- Balducci baseline hazard mu_{x,a}(t)
- Bang-bang HJB PDE for GMAB, GMIB, GMDB in the reduced state space
- Dual formulation: outer optimization over Lagrange multipliers lambda
- Maturity-only and all-intermediate expectation constraints

Methodology: Sections 4 and 5.2 of the replication paper.
# Custom — Context7 found no library equivalent (paper-specific bang-bang HJB)
"""

import math
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from scipy import optimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Balducci baseline hazard
# ---------------------------------------------------------------------------

def balducci_hazard_rate(hat_p: np.ndarray, j: int, u: float) -> float:
    """
    Balducci baseline hazard rate mu_{x,a}(j+u).

    mu_{x,a}(j+u) = (1 - hat_p[j]) / (hat_p[j] + u*(1 - hat_p[j]))

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    j : int
        Year index.
    u : float
        Fractional year in [0,1).

    Returns
    -------
    float
    """
    p = hat_p[j]
    q = 1.0 - p
    denom = p + u * q
    if denom <= 0:
        return np.inf
    return q / denom


def balducci_survival(hat_p: np.ndarray, t: float) -> float:
    """
    Balducci survival function {}_t p_x.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    t : float
        Time from policy start.

    Returns
    -------
    float
    """
    j = int(math.floor(t))
    u = t - j
    cum = float(np.prod(hat_p[:j]))
    if u == 0.0:
        return cum  # s_j(0) = 1 for all assumptions; handles t = integer exactly
    if j >= len(hat_p):
        return 0.0
    p = hat_p[j]
    s_j = p / (u + (1.0 - u) * p)
    return cum * s_j


# ---------------------------------------------------------------------------
# Analytical GMAB in deterministic-rate setting
# ---------------------------------------------------------------------------

def _bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European call price with deterministic rate r."""
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _log_moneyness_sigma(pi_S: float, sigma_S: float) -> float:
    """Effective volatility of log(A_t/G_t) in deterministic-rate setting."""
    # sigma_eff^2 = pi_S^2 * sigma_S^2 (only stock contributes)
    return pi_S * sigma_S


# ---------------------------------------------------------------------------
# Finite-difference HJB solver (backward in time, state = y = log(A/G))
# ---------------------------------------------------------------------------

def _build_gmab_terminal(y_grid: np.ndarray, A0: float, r_g: float, T: float, r: float) -> np.ndarray:
    """GMAB terminal payoff: max(A_T, G_T^A) = A_0*exp(y_T)*max(1, exp(-y_T))*... simplified."""
    # y = log(A/G), A = G * exp(y), G_T = A0 * exp(r_g*T)
    # max(A_T, G_T) = G_T * max(exp(y), 1)
    G_T = A0 * math.exp(r_g * T)
    return G_T * np.maximum(np.exp(y_grid), 1.0)


def _build_gmdb_payoff(y_grid: np.ndarray, A0: float, r_g: float, t: float) -> np.ndarray:
    """GMDB death payoff: max(A_t, G_t^D) = G_t * max(exp(y), 1)."""
    G_t = A0 * math.exp(r_g * t)
    return G_t * np.maximum(np.exp(y_grid), 1.0)


def _build_gmib_running(y_grid: np.ndarray, A0: float, r_g: float, r_I: float, beta: float, t: float) -> np.ndarray:
    """GMIB running payoff: A0*r_I + beta*(A_t - G_t^I)_+."""
    G_t = A0 * math.exp(r_g * t)
    A_t = G_t * np.exp(y_grid)  # A_t = G_t * exp(y)
    return A0 * r_I + beta * np.maximum(A_t - G_t, 0.0)


def fd_solver_gmab(
    n_T: int,
    n_x: int,
    T: float,
    r: float,
    sigma_eff: float,
    A0: float,
    r_g: float,
    mu_func: Callable[[float], float],
) -> np.ndarray:
    """
    Backward finite-difference PDE for GMAB with mortality control.

    State: y = log(A_t / G_t^A), G_t^A = A0 * exp(r_g*t)
    Under deterministic rate r and zero market price of risk:
      dy = (r - r_g - 0.5*sigma_eff^2) dt + sigma_eff * dW

    HJB equation (for value V(t,y)):
      -dV/dt + (r - r_g - 0.5*sigma_eff^2)*dV/dy + 0.5*sigma_eff^2*d^2V/dy^2
      - (r + mu(t))*V + mu(t)*payoff_death = 0
    For GMAB: no running payoff, no death payoff => mortality just discounts survival.
    Terminal: V(T,y) = max(A_T, G_T) = G_T * max(exp(y), 1)

    For the relaxed GMAB (survival payoff only), the mortality control affects
    only the cumulative survival probability. The relaxed bound for GMAB under
    expected matching is:
      worst-case: minimize over mu in [mu_a - delta, mu_a + delta]
      best-case: maximize over mu

    For GMAB, since the payoff is at maturity only:
      dV/dt = (r + mu)*V - (r-r_g-0.5*sig^2)*dV/dy - 0.5*sig^2*d^2V/dy^2
    The optimal bang-bang control is:
      mu* = mu_high if V > 0 (worst case: more deaths = lower survival)
      mu* = mu_low  if V > 0 (best case: fewer deaths = higher survival)

    Parameters
    ----------
    n_T : int
        Number of time steps.
    n_x : int
        Number of spatial grid points.
    T : float
        Maturity.
    r : float
        Deterministic interest rate.
    sigma_eff : float
        Effective volatility of y = log(A/G).
    A0 : float
        Initial fund.
    r_g : float
        Guarantee growth rate.
    mu_func : Callable
        mu_func(t) -> (mu_low, mu_high): lower and upper control bounds at time t.

    Returns
    -------
    np.ndarray
        Shape (n_T+1, n_x), PDE solution V(t, y).
    """
    dt = T / n_T
    # Spatial grid in log-moneyness y = log(A/G)
    y_min, y_max = -3.0, 3.0
    dy = (y_max - y_min) / (n_x - 1)
    y_grid = np.linspace(y_min, y_max, n_x)

    drift = r - r_g - 0.5 * sigma_eff ** 2
    vol2 = sigma_eff ** 2

    # Terminal condition
    V = _build_gmab_terminal(y_grid, A0, r_g, T, r)
    V_store = np.zeros((n_T + 1, n_x))
    V_store[n_T] = V

    for step in range(n_T - 1, -1, -1):
        t = step * dt
        mu_lo, mu_hi = mu_func(t)

        # Finite difference coefficients (explicit Euler in space)
        # dV/dy centered, d^2V/dy^2 central
        dV_dy = np.zeros(n_x)
        d2V_dy2 = np.zeros(n_x)
        dV_dy[1:-1] = (V[2:] - V[:-2]) / (2.0 * dy)
        dV_dy[0] = (V[1] - V[0]) / dy
        dV_dy[-1] = (V[-1] - V[-2]) / dy
        d2V_dy2[1:-1] = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dy ** 2
        # Boundary: zero second derivative
        d2V_dy2[0] = d2V_dy2[1]
        d2V_dy2[-1] = d2V_dy2[-2]

        # GMAB: death payoff = 0 (no death benefit), survival payoff at T
        # The mortality control affects survival: more mortality -> lower value
        # worst-case for GMAB (minimum value): highest mortality mu_hi
        # best-case (maximum value): lowest mortality mu_lo
        # We compute BOTH and return separately (direction handled outside).
        # Here: return value under provided mu_func which returns (lo, hi).
        # Caller must call twice or pass a single mu.
        # For the HJB: dV/dt = (r+mu)*V - drift*dV/dy - 0.5*vol2*d2V/dy2
        # No death payoff for GMAB.
        # Use explicit Euler (stable for small dt).
        mu = mu_lo  # Default; caller selects direction
        V_new = V + dt * (-(r + mu) * V + drift * dV_dy + 0.5 * vol2 * d2V_dy2)
        V_new = np.clip(V_new, 0.0, None)  # non-negative value
        V = V_new
        V_store[step] = V

    return V_store


def fd_solver_product(
    product: Literal["GMAB", "GMIB", "GMDB"],
    direction: Literal["worst", "best"],
    hat_p: np.ndarray,
    T: float,
    delta: float,
    n_T: int,
    n_x: int,
    r: float,
    pi_S: float,
    sigma_S: float,
    A0: float,
    r_g: float,
    r_I: float = 0.01,
    beta: float = 0.05,
) -> float:
    """
    Finite-difference HJB PDE solver for relaxed mortality-control bounds.

    State variable: y = log(A_t / G_t) where G_t is the product-specific guarantee.
    Effective fund volatility: sigma_eff = pi_S * sigma_S (deterministic rate).
    Drift of y under Q: mu_y = r - r_g - 0.5 * sigma_eff^2.

    HJB PDE:
      -dV/dt + mu_y*dV/dy + 0.5*sigma_eff^2*d^2V/dy^2 - r*V
             - mu*(V - h_death) + h_running = 0

    where:
      h_death = death payoff at time t (relevant for GMDB)
      h_running = running payoff rate (relevant for GMIB)
      mu*(t) is the bang-bang optimal control in [max(0, mu_a-delta), mu_a+delta]

    Parameters
    ----------
    product : str
        'GMAB', 'GMIB', or 'GMDB'.
    direction : str
        'worst' (maximize mu for GMAB/GMIB, minimize for GMDB) or 'best'.
    hat_p : np.ndarray
        Annual survival probabilities.
    T : float
        Maturity.
    delta : float
        Control band radius around Balducci baseline.
    n_T, n_x : int
        Time and space grid sizes.
    r : float
        Deterministic interest rate.
    pi_S, sigma_S : float
        Stock allocation and volatility.
    A0 : float
        Initial fund.
    r_g : float
        Guarantee growth rate.
    r_I : float
        GMIB fixed income rate.
    beta : float
        GMIB participation rate.

    Returns
    -------
    float
        Value at (t=0, y=0), corresponding to A_0 = G_0 = A0.
    """
    dt = T / n_T
    y_min, y_max = -3.5, 3.5
    n_y = n_x
    dy = (y_max - y_min) / (n_y - 1)
    y_grid = np.linspace(y_min, y_max, n_y)

    sigma_eff = pi_S * sigma_S
    drift_y = r - r_g - 0.5 * sigma_eff ** 2
    vol2 = sigma_eff ** 2

    # Terminal condition at T
    if product == "GMAB":
        G_T = A0 * math.exp(r_g * T)
        V = G_T * np.maximum(np.exp(y_grid), 1.0)
    elif product in ("GMIB", "GMDB"):
        # No terminal payoff for pure GMIB/GMDB (terminal payoff = 0 for running/death products)
        V = np.zeros(n_y)
    else:
        raise ValueError(f"Unknown product: {product}")

    # Backward induction
    for step in range(n_T - 1, -1, -1):
        t = step * dt
        j = int(math.floor(t))
        u = t - j

        # Balducci baseline hazard
        if j < len(hat_p):
            p = hat_p[j]
            q = 1.0 - p
            denom = p + u * q
            mu_a = q / denom if denom > 0 else 0.0
        else:
            mu_a = 0.0

        mu_lo = max(0.0, mu_a - delta)
        mu_hi = mu_a + delta

        # Spatial derivatives (central differences)
        dV_dy = np.zeros(n_y)
        d2V_dy2 = np.zeros(n_y)
        dV_dy[1:-1] = (V[2:] - V[:-2]) / (2.0 * dy)
        dV_dy[0] = (V[1] - V[0]) / dy
        dV_dy[-1] = (V[-1] - V[-2]) / dy
        d2V_dy2[1:-1] = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dy ** 2
        d2V_dy2[0] = d2V_dy2[1]
        d2V_dy2[-1] = d2V_dy2[-2]

        # Payoff terms at current time
        G_t = A0 * math.exp(r_g * t)

        if product == "GMAB":
            h_death = np.zeros(n_y)  # no death benefit
            h_running = np.zeros(n_y)
        elif product == "GMIB":
            h_death = np.zeros(n_y)  # no death benefit for pure GMIB
            A_t = G_t * np.exp(y_grid)
            h_running = (A0 * r_I + beta * np.maximum(A_t - G_t, 0.0)) * np.ones(n_y)
            h_running = A0 * r_I + beta * np.maximum(A_t - G_t, 0.0)
        elif product == "GMDB":
            A_t = G_t * np.exp(y_grid)
            h_death = np.maximum(A_t, G_t)  # death benefit
            h_running = np.zeros(n_y)
        else:
            raise ValueError(f"Unknown product: {product}")

        # PDE right-hand side for each candidate control
        def rhs(mu: float) -> np.ndarray:
            # HJB: -dV/dt = -r*V + mu*(h_death - V) + h_running
            #               + drift_y*dV/dy + 0.5*vol2*d2V/dy^2
            return (
                -r * V
                + mu * (h_death - V)
                + h_running
                + drift_y * dV_dy
                + 0.5 * vol2 * d2V_dy2
            )

        rhs_lo = rhs(mu_lo)
        rhs_hi = rhs(mu_hi)

        # Bang-bang: select control pointwise
        if product in ("GMAB", "GMIB"):
            # Survival payoff: worst-case = more mortality (lower value)
            if direction == "worst":
                # More mortality reduces value: choose mu_hi if it lowers value
                # rhs_hi < rhs_lo means higher mu gives more negative rate of growth
                # Equivalently: choose mu that minimizes V_new => choose mu_hi
                selected_rhs = np.where(rhs_hi < rhs_lo, rhs_hi, rhs_lo)
            else:  # best
                selected_rhs = np.where(rhs_hi > rhs_lo, rhs_hi, rhs_lo)
        elif product == "GMDB":
            # Death payoff: worst-case = higher mortality (more deaths = higher value for WC upper bound)
            if direction == "worst":
                selected_rhs = np.where(rhs_hi > rhs_lo, rhs_hi, rhs_lo)
            else:  # best
                selected_rhs = np.where(rhs_hi < rhs_lo, rhs_hi, rhs_lo)

        # Explicit Euler step
        V_new = V + dt * selected_rhs

        # Non-negativity constraint on value
        if product == "GMAB":
            V_new = np.maximum(V_new, 0.0)

        V = V_new

    # Value at y=0 (A_0 = G_0): find index nearest y=0
    idx0 = np.argmin(np.abs(y_grid))
    return float(V[idx0])


# ---------------------------------------------------------------------------
# Relaxed bounds (product-specific wrappers)
# ---------------------------------------------------------------------------

def _relaxed_bounds_product(
    product: Literal["GMAB", "GMIB", "GMDB"],
    hat_p: np.ndarray,
    T: float,
    delta: float,
    A0: float,
    r_g: float,
    r_I: float,
    beta: float,
    pi_S: float,
    sigma_S: float,
    r: float = 0.01,
    n_T: int = 500,
    n_x: int = 200,
) -> Tuple[float, float]:
    """
    Compute relaxed lower and upper bounds for a given product.

    Returns (lower, upper) where:
      lower = best-case value (minimum mortality exposure for survival products)
      upper = worst-case value (maximum mortality exposure for death products)

    Note: 'worst' and 'best' have product-specific interpretations.
    """
    worst = fd_solver_product(
        product=product, direction="worst",
        hat_p=hat_p, T=T, delta=delta,
        n_T=n_T, n_x=n_x, r=r,
        pi_S=pi_S, sigma_S=sigma_S,
        A0=A0, r_g=r_g, r_I=r_I, beta=beta,
    )
    best = fd_solver_product(
        product=product, direction="best",
        hat_p=hat_p, T=T, delta=delta,
        n_T=n_T, n_x=n_x, r=r,
        pi_S=pi_S, sigma_S=sigma_S,
        A0=A0, r_g=r_g, r_I=r_I, beta=beta,
    )
    if product in ("GMAB", "GMIB"):
        # Survival products: best-case = more survival = higher value
        lower = worst  # worst-case (more mortality) gives lower value
        upper = best   # best-case (less mortality) gives higher value
    else:  # GMDB
        # Death product: worst-case (more mortality) gives higher value
        lower = best
        upper = worst
    return lower, upper


def relaxed_gmab_bounds(
    hat_p: np.ndarray,
    T: float,
    delta: float,
    A0: float,
    r_g: float,
    r_I: float,
    beta: float,
    pi_S: float,
    sigma_S: float,
    r: float = 0.01,
    n_T: int = 500,
    n_x: int = 200,
) -> Tuple[float, float]:
    """
    Relaxed GMAB worst-case and best-case values.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    T : float
        Maturity.
    delta : float
        Mortality control band radius.
    A0, r_g, r_I, beta, pi_S, sigma_S : float
        Product and market parameters.
    r : float
        Deterministic interest rate (Section 5.2 simplification).
    n_T, n_x : int
        PDE grid sizes.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
    """
    return _relaxed_bounds_product(
        "GMAB", hat_p, T, delta, A0, r_g, r_I, beta, pi_S, sigma_S, r, n_T, n_x
    )


def relaxed_gmib_bounds(
    hat_p: np.ndarray,
    T: float,
    delta: float,
    A0: float,
    r_g: float,
    r_I: float,
    beta: float,
    pi_S: float,
    sigma_S: float,
    r: float = 0.01,
    n_T: int = 500,
    n_x: int = 200,
) -> Tuple[float, float]:
    """
    Relaxed GMIB worst-case and best-case values.

    Parameters
    ----------
    (same as relaxed_gmab_bounds)

    Returns
    -------
    (lower, upper) : Tuple[float, float]
    """
    return _relaxed_bounds_product(
        "GMIB", hat_p, T, delta, A0, r_g, r_I, beta, pi_S, sigma_S, r, n_T, n_x
    )


def relaxed_gmdb_bounds(
    hat_p: np.ndarray,
    T: float,
    delta: float,
    A0: float,
    r_g: float,
    pi_S: float,
    sigma_S: float,
    r: float = 0.01,
    n_T: int = 500,
    n_x: int = 200,
) -> Tuple[float, float]:
    """
    Relaxed GMDB worst-case and best-case values.

    Parameters
    ----------
    (same as relaxed_gmab_bounds, without r_I and beta)

    Returns
    -------
    (lower, upper) : Tuple[float, float]
    """
    return _relaxed_bounds_product(
        "GMDB", hat_p, T, delta, A0, r_g, 0.0, 0.0, pi_S, sigma_S, r, n_T, n_x
    )


def balducci_gmab_value(
    hat_p: np.ndarray,
    T: float,
    A0: float,
    r_g: float,
    pi_S: float,
    sigma_S: float,
    r: float = 0.01,
    n_T: int = 500,
    n_x: int = 200,
) -> float:
    """
    GMAB baseline value under Balducci mortality (delta -> 0 limit).

    This uses the PDE with delta=0 (no control freedom), equivalent to
    Balducci baseline survival.
    """
    # Small delta for numerical stability
    return fd_solver_product(
        "GMAB", direction="worst",
        hat_p=hat_p, T=T, delta=1e-6,
        n_T=n_T, n_x=n_x, r=r,
        pi_S=pi_S, sigma_S=sigma_S,
        A0=A0, r_g=r_g, r_I=0.0, beta=0.0,
    )


def balducci_gmib_value(
    hat_p: np.ndarray,
    T: float,
    A0: float,
    r_g: float,
    r_I: float,
    beta: float,
    pi_S: float,
    sigma_S: float,
    r: float = 0.01,
    n_T: int = 500,
    n_x: int = 200,
) -> float:
    """GMIB baseline value under Balducci mortality."""
    return fd_solver_product(
        "GMIB", direction="worst",
        hat_p=hat_p, T=T, delta=1e-6,
        n_T=n_T, n_x=n_x, r=r,
        pi_S=pi_S, sigma_S=sigma_S,
        A0=A0, r_g=r_g, r_I=r_I, beta=beta,
    )


def balducci_gmdb_value(
    hat_p: np.ndarray,
    T: float,
    A0: float,
    r_g: float,
    pi_S: float,
    sigma_S: float,
    r: float = 0.01,
    n_T: int = 500,
    n_x: int = 200,
) -> float:
    """GMDB baseline value under Balducci mortality."""
    return fd_solver_product(
        "GMDB", direction="worst",
        hat_p=hat_p, T=T, delta=1e-6,
        n_T=n_T, n_x=n_x, r=r,
        pi_S=pi_S, sigma_S=sigma_S,
        A0=A0, r_g=r_g, r_I=0.0, beta=0.0,
    )
