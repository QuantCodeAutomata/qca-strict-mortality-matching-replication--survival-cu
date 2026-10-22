"""
finance.py
==========
Risk-neutral financial model under Hull-White short rate and GBM stock.

Implements:
- Nelson-Siegel-Svensson-style initial forward curve f(0,t)
- Hull-White theta(t) calibration from the forward curve
- Analytical zero-coupon bond prices P(0,T) and P(t,T)
- Variance sigma_Y^2 of the log-fund relative to P(0,T)
- GMAB analytical pricing (call-on-forward style)
- GMIB running-payoff integrand C_0(s, G_s^I)

Methodology follows Proposition 2.2 and Section 4 of the replication paper.
Context7 confirmed: custom implementation — no library equivalent for the
paper-specific bond+fund option formulas with constant-mix allocation.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Forward curve
# ---------------------------------------------------------------------------

def forward_rate(
    t: float,
    a: float = 0.76,
    b: float = 0.015,
    c: float = -0.005,
    d: float = 0.02,
) -> float:
    """
    Initial instantaneous forward rate.

    f(0,t) = b + c*exp(-a*t) + d*a*t*exp(-a*t)

    Parameters
    ----------
    t : float
        Maturity (>=0).
    a, b, c, d : float
        Curve parameters (Nelson-Siegel style).

    Returns
    -------
    float
    """
    eat = math.exp(-a * t)
    return b + c * eat + d * a * t * eat


def forward_rate_array(
    t: np.ndarray,
    a: float = 0.76,
    b: float = 0.015,
    c: float = -0.005,
    d: float = 0.02,
) -> np.ndarray:
    """Vectorized version of forward_rate."""
    t = np.asarray(t, dtype=float)
    eat = np.exp(-a * t)
    return b + c * eat + d * a * t * eat


def forward_rate_deriv(
    t: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float:
    """
    Derivative of the forward rate: df(0,t)/dt.

    df/dt = -a*c*exp(-a*t) + d*a*exp(-a*t)*(1 - a*t)

    Parameters
    ----------
    t : float
        Maturity.
    a, b, c, d : float
        Forward curve parameters.

    Returns
    -------
    float
    """
    eat = math.exp(-a * t)
    return -a * c * eat + d * a * eat * (1.0 - a * t)


def theta(
    t: float,
    kappa: float,
    sigma_r: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float:
    """
    Hull-White mean reversion level theta(t).

    theta(t) = (1/kappa)*df(0,t)/dt + f(0,t) + sigma_r^2/(2*kappa^2)*(1-exp(-2*kappa*t))

    Parameters
    ----------
    t : float
        Time.
    kappa : float
        Mean reversion speed.
    sigma_r : float
        Short-rate volatility.
    a, b, c, d : float
        Forward curve parameters.

    Returns
    -------
    float
    """
    dfdt = forward_rate_deriv(t, a, b, c, d)
    f0t = forward_rate(t, a, b, c, d)
    hump = sigma_r ** 2 / (2.0 * kappa ** 2) * (1.0 - math.exp(-2.0 * kappa * t))
    return dfdt / kappa + f0t + hump


# ---------------------------------------------------------------------------
# Bond pricing (Hull-White)
# ---------------------------------------------------------------------------

def B_hw(t: float, T: float, kappa: float) -> float:
    """
    Hull-White B function: B(t,T) = (1/kappa)*(1 - exp(-kappa*(T-t))).

    Parameters
    ----------
    t : float
        Current time.
    T : float
        Maturity.
    kappa : float
        Mean reversion speed.

    Returns
    -------
    float
        Non-negative for t < T.
    """
    return (1.0 / kappa) * (1.0 - math.exp(-kappa * (T - t)))


def zero_coupon_bond(T: float, fwd_params: Dict[str, float]) -> float:
    """
    Initial zero-coupon bond price P(0,T) = exp(-integral_0^T f(0,s) ds).

    Uses numerical integration via scipy.integrate.quad.
    # Using scipy.integrate.quad — Context7 confirmed

    Parameters
    ----------
    T : float
        Maturity.
    fwd_params : dict
        Keys: a, b, c, d — forward curve parameters.

    Returns
    -------
    float
    """
    if T <= 0:
        return 1.0
    a, b, c, d = fwd_params["a"], fwd_params["b"], fwd_params["c"], fwd_params["d"]
    integral, _ = integrate.quad(
        lambda s: forward_rate(s, a, b, c, d), 0.0, T, limit=200
    )
    return math.exp(-integral)


def bond_price(
    t: float,
    T: float,
    r_t: float,
    kappa: float,
    sigma_r: float,
    fwd_params: Dict[str, float],
) -> float:
    """
    Hull-White bond price P(t,T) using Proposition 2.2.

    P(t,T) = P(0,T)/P(0,t) * exp(-B(t,T)*(r_t - f(0,t))
             - sigma_r^2/(4*kappa)*B(t,T)^2*(1-exp(-2*kappa*t)))

    Parameters
    ----------
    t : float
        Current time.
    T : float
        Maturity.
    r_t : float
        Current short rate.
    kappa, sigma_r : float
        Hull-White parameters.
    fwd_params : dict
        Forward curve parameters.

    Returns
    -------
    float
    """
    if t >= T:
        return 1.0
    a = fwd_params["a"]
    b = fwd_params["b"]
    c = fwd_params["c"]
    d = fwd_params["d"]

    p0T = zero_coupon_bond(T, fwd_params)
    p0t = zero_coupon_bond(t, fwd_params)
    Btt = B_hw(t, T, kappa)
    f0t = forward_rate(t, a, b, c, d)

    exponent = (
        -Btt * (r_t - f0t)
        - sigma_r ** 2 / (4.0 * kappa) * Btt ** 2 * (1.0 - math.exp(-2.0 * kappa * t))
    )
    return (p0T / p0t) * math.exp(exponent)


# ---------------------------------------------------------------------------
# Variance of log-fund
# ---------------------------------------------------------------------------

def sigma_Y_squared(
    T: float,
    pi_S: float,
    sigma_S: float,
    pi_P: float,
    sigma_r: float,
    kappa: float,
    rho: float,
    fwd_params: Dict[str, float],
) -> float:
    """
    Variance sigma_Y^2 of log(A_T) relative to the money-market account.

    Under the constant-mix fund:
      dA_t/A_t = r_t dt + (pi_S*sigma_S*e1 - pi_P*sigma_r*B(t,T)*e2) Sigma dW^Q

    The integrated quadratic variation gives:
      sigma_Y^2 = integral_0^T [pi_S^2*sigma_S^2
                                + pi_P^2*sigma_r^2*B(s,T)^2
                                - 2*pi_S*pi_P*sigma_S*sigma_r*rho*B(s,T)] ds

    Uses scipy.integrate.quad for numerical precision.

    Parameters
    ----------
    T : float
        Maturity.
    pi_S, sigma_S : float
        Stock allocation and volatility.
    pi_P, sigma_r : float
        Bond allocation and short-rate volatility.
    kappa : float
        Hull-White mean reversion.
    rho : float
        Correlation between stock and short-rate Brownian motions.
    fwd_params : dict
        Forward curve parameters (used for B_hw via kappa).

    Returns
    -------
    float
        Non-negative variance.
    """
    def integrand(s: float) -> float:
        Bst = B_hw(s, T, kappa)
        term1 = pi_S ** 2 * sigma_S ** 2
        term2 = pi_P ** 2 * sigma_r ** 2 * Bst ** 2
        term3 = 2.0 * pi_S * pi_P * sigma_S * sigma_r * rho * Bst
        return term1 + term2 - term3

    result, _ = integrate.quad(integrand, 0.0, T, limit=200)
    return max(result, 0.0)  # numerical guard


# ---------------------------------------------------------------------------
# GMAB pricing (survival-to-maturity payoff)
# ---------------------------------------------------------------------------

def gmab_price(
    T: float,
    A0: float,
    r_g: float,
    pi_S: float,
    pi_P: float,
    sigma_S: float,
    sigma_r: float,
    kappa: float,
    rho: float,
    fwd_params: Dict[str, float],
) -> float:
    """
    Price of GMAB option: C_0(T, G_T^A) = E^Q[e^{-int_0^T r} max(A_T, G_T^A)].

    max(A_T, G_T^A) = A_T + (G_T^A - A_T)^+ = G_T^A + (A_T - G_T^A)^+

    Using Black-style formula with log-normal A_T under Q:
      E^Q[e^{-int r} A_T] = A_0
      E^Q[e^{-int r} G_T^A] = G_T^A * P(0,T)

    So: C_0 = A_0 + G_T^A * P(0,T) * Phi(-d2) - A_0 * Phi(-d1)
    where d1 = (log(A_0 / (G_T^A * P(0,T))) + sigma_Y^2/2) / sigma_Y
          d2 = d1 - sigma_Y

    Parameters
    ----------
    T : float
        Maturity.
    A0 : float
        Initial fund value.
    r_g : float
        Guarantee growth rate.
    pi_S, pi_P : float
        Stock and bond allocations.
    sigma_S, sigma_r : float
        Volatilities.
    kappa : float
        Hull-White mean reversion.
    rho : float
        Correlation.
    fwd_params : dict
        Forward curve parameters.

    Returns
    -------
    float
        GMAB price C_0(T, G_T^A).
    """
    G_T = A0 * math.exp(r_g * T)
    P0T = zero_coupon_bond(T, fwd_params)
    sig2 = sigma_Y_squared(T, pi_S, sigma_S, pi_P, sigma_r, kappa, rho, fwd_params)
    sig = math.sqrt(sig2)

    if sig < 1e-14:
        # Deterministic case: max(A_T, G_T^A) with known A_T
        # A_T = A0 / P(0,T) on average => compare with G_T
        return max(A0, G_T * P0T)

    d1 = (math.log(A0 / (G_T * P0T)) + 0.5 * sig2) / sig
    d2 = d1 - sig

    # C_0 = E^Q[e^{-∫r}(A_T + (G_T - A_T)^+)] = A_0 + G_T*P(0,T)*Phi(-d2) - A_0*Phi(-d1)
    price = A0 + G_T * P0T * norm.cdf(-d2) - A0 * norm.cdf(-d1)
    return price


# ---------------------------------------------------------------------------
# GMIB integrand C_0(s, G_s^I)
# ---------------------------------------------------------------------------

def gmib_integrand(
    s: float,
    A0: float,
    r_g: float,
    r_I: float,
    beta: float,
    pi_S: float,
    pi_P: float,
    sigma_S: float,
    sigma_r: float,
    kappa: float,
    rho: float,
    fwd_params: Dict[str, float],
) -> float:
    """
    GMIB benefit density integrand C_0(s, G_s^I).

    Running payoff: H_tilde(A_s, G_s^I) = A_0*r_I + beta*(A_s - G_s^I)_+
    with G_s^I = A_0 * exp(r_g * s).

    Expected discounted value:
      C_0(s, G_s^I) = A_0*r_I*P(0,s)
                    + beta*(A_0*Phi(-d1(0,s)) - G_s^I*P(0,s)*Phi(-d2(0,s)))

    where:
      d1(0,s) = (log(A_0/(G_s^I * P(0,s))) + sigma_Y^2(s)/2) / sigma_Y(s)
      d2(0,s) = d1(0,s) - sigma_Y(s)

    Note: (A-G)_+ = call option on fund vs guarantee.
    The Phi(-d1), Phi(-d2) terms come from the put version via put-call parity:
      E[e^{-∫r}(A_s - G_s^I)_+] = A_0*Phi(d1_call) - G_s^I*P(0,s)*Phi(d2_call)
    where d1_call = (log(A0/(G_s^I*P(0,s))) + sig2/2)/sig
    So the formula uses Phi(d1), Phi(d2) NOT negated for the call.
    Double check: max(A,G) = G + (A-G)+ and E[e^{-∫r}(A_s-G_s^I)_+] call formula.

    Parameters
    ----------
    s : float
        Running payoff time.
    A0 : float
        Initial fund value.
    r_g : float
        Guarantee growth rate.
    r_I : float
        GMIB fixed income rate.
    beta : float
        GMIB participation rate.
    pi_S, pi_P : float
        Allocations.
    sigma_S, sigma_r : float
        Volatilities.
    kappa : float
        Hull-White mean reversion.
    rho : float
        Correlation.
    fwd_params : dict
        Forward curve parameters.

    Returns
    -------
    float
        Non-negative integrand value C_0(s, G_s^I).
    """
    if s <= 1e-12:
        return A0 * r_I  # at s=0, P=1, call=0

    G_s = A0 * math.exp(r_g * s)
    P0s = zero_coupon_bond(s, fwd_params)
    sig2 = sigma_Y_squared(s, pi_S, sigma_S, pi_P, sigma_r, kappa, rho, fwd_params)
    sig = math.sqrt(sig2)

    fixed_income = A0 * r_I * P0s

    if sig < 1e-14:
        # Deterministic: call value = max(A0/P0s - G_s, 0) * P0s
        A_det = A0 / P0s if P0s > 0 else A0
        call = max(A_det - G_s, 0.0) * P0s
        return fixed_income + beta * call

    # Black-Scholes style call: E^Q[e^{-∫r}(A_s - G_s)_+]
    # = A0*Phi(d1) - G_s*P0s*Phi(d2)
    d1 = (math.log(A0 / (G_s * P0s)) + 0.5 * sig2) / sig
    d2 = d1 - sig

    call = A0 * norm.cdf(d1) - G_s * P0s * norm.cdf(d2)
    return fixed_income + beta * call


def precompute_gmib_integrand(
    T: float,
    n_grid: int,
    A0: float,
    r_g: float,
    r_I: float,
    beta: float,
    pi_S: float,
    pi_P: float,
    sigma_S: float,
    sigma_r: float,
    kappa: float,
    rho: float,
    fwd_params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute C_0(s, G_s^I) on a dense time grid [0, T].

    Parameters
    ----------
    T : float
        Upper integration limit.
    n_grid : int
        Number of grid points (excluding 0).
    A0, r_g, r_I, beta, pi_S, pi_P, sigma_S, sigma_r, kappa, rho : float
        Model and product parameters.
    fwd_params : dict
        Forward curve parameters.

    Returns
    -------
    (times, values) : Tuple[np.ndarray, np.ndarray]
        times: shape (n_grid+1,), values: shape (n_grid+1,).
    """
    times = np.linspace(0.0, T, n_grid + 1)
    values = np.array(
        [
            gmib_integrand(
                s, A0, r_g, r_I, beta, pi_S, pi_P, sigma_S, sigma_r, kappa, rho, fwd_params
            )
            for s in times
        ]
    )
    return times, values
