"""
mortality.py
============
Life-table loading and within-year survival curve construction under
UDD, CFM (Constant Force of Mortality), and Balducci assumptions, plus
extremal survival curves for strict almost-sure matching.

Methodology follows Section 3 and Section 5.1 of the replication paper.
"""

import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Life-table generation (Makeham-Gompertz calibrated to Belgian female HMD 2010)
# ---------------------------------------------------------------------------

def generate_belgian_female_life_table(max_age: int = 120) -> pd.DataFrame:
    """
    Generate a synthetic Belgian female life table using Makeham-Gompertz.

    Parameters calibrated to approximate HMD Belgian female 2010 mortality:
      A = 0.0005 (accident / background hazard)
      B = 0.00004 (Gompertz scale)
      c = 1.10    (Gompertz rate-of-ageing)
    Radix l_0 = 100,000.

    Parameters
    ----------
    max_age : int
        Maximum age to compute (inclusive). Default 120.

    Returns
    -------
    pd.DataFrame
        Columns: age (int), lx (float), px (float), qx (float).
    """
    A_mk = 0.0005
    B_mk = 0.00004
    c_mk = 1.10
    l0 = 100_000.0

    ages = np.arange(0, max_age + 1, dtype=float)
    # Integrated Makeham-Gompertz hazard: -log(l_x/l_0) = A*x + B*(c^x - 1)/log(c)
    log_c = math.log(c_mk)
    cum_hazard = A_mk * ages + B_mk * (c_mk ** ages - 1.0) / log_c
    lx = l0 * np.exp(-cum_hazard)

    df = pd.DataFrame({"age": ages.astype(int), "lx": lx})
    # one-year survival and mortality probabilities
    px = np.empty(len(df))
    qx = np.empty(len(df))
    for i in range(len(df) - 1):
        px[i] = df["lx"].iloc[i + 1] / df["lx"].iloc[i]
        qx[i] = 1.0 - px[i]
    # Terminal age: certain death
    px[-1] = 0.0
    qx[-1] = 1.0
    df["px"] = px
    df["qx"] = qx
    return df


def load_life_table(path: str | None = None) -> pd.DataFrame:
    """
    Load the Belgian female life table from a CSV file.

    If *path* is None, looks for
    ``data/belgian_female_lt.csv`` relative to this file's package root.
    Falls back to generating the table in memory if the file does not exist.

    Parameters
    ----------
    path : str or None
        Explicit path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: age, lx, px, qx.
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "belgian_female_lt.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        # Ensure px and qx columns exist
        if "px" not in df.columns:
            px = np.empty(len(df))
            qx = np.empty(len(df))
            for i in range(len(df) - 1):
                px[i] = df["lx"].iloc[i + 1] / df["lx"].iloc[i]
                qx[i] = 1.0 - px[i]
            px[-1] = 0.0
            qx[-1] = 1.0
            df["px"] = px
            df["qx"] = qx
        return df
    else:
        # Generate on the fly
        return generate_belgian_female_life_table()


def get_annual_survival_probs(
    lx_series: np.ndarray | pd.Series,
    x: int,
    T: float,
) -> np.ndarray:
    """
    Extract one-year survival probabilities hat_p_j = l_{x+j+1}/l_{x+j}.

    Parameters
    ----------
    lx_series : array-like
        l_x values indexed from age 0 (index i corresponds to age i).
    x : int
        Initial age of the policyholder.
    T : float
        Policy maturity (possibly non-integer; uses ceil(T) years).

    Returns
    -------
    np.ndarray
        Array hat_p of length ceil(T) with hat_p[j] = p_{x+j}.
    """
    n_years = math.ceil(T)
    lx_arr = np.asarray(lx_series, dtype=float)
    hat_p = np.empty(n_years)
    for j in range(n_years):
        age_j = x + j
        if age_j + 1 >= len(lx_arr):
            raise ValueError(
                f"Life table does not cover age {age_j + 1}. "
                "Extend the table or reduce maturity."
            )
        l_j = lx_arr[age_j]
        l_j1 = lx_arr[age_j + 1]
        if l_j <= 0:
            raise ValueError(f"l_{{{age_j}}} = 0; no survivors at age {age_j}.")
        hat_p[j] = l_j1 / l_j
    return hat_p


def get_cumulative_survival(hat_p: np.ndarray, j: int) -> float:
    """
    Compute cumulative j-year survival: {}_j p_x = prod_{i=0}^{j-1} hat_p[i].

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    j : int
        Number of full years (j=0 returns 1.0).

    Returns
    -------
    float
    """
    if j == 0:
        return 1.0
    return float(np.prod(hat_p[:j]))


# ---------------------------------------------------------------------------
# Within-year survival scalars (fractional year y in [0,1))
# ---------------------------------------------------------------------------

def _split_u(u: float) -> Tuple[int, float]:
    """Split u = j + y into integer part j and fractional part y in [0,1)."""
    j = int(math.floor(u))
    y = u - j
    return j, y


def survival_udd_scalar(hat_p: np.ndarray, u: float) -> float:
    """
    UDD within-year survival at duration u from initial age x.

    s_j(y) = 1 - y*(1 - hat_p[j])  for u = j + y, y in [0,1).
    At integer u = j with y = 0: s_j(0) = 1, so result = {}_j p_x.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    u : float
        Duration from start (non-negative).

    Returns
    -------
    float
        {}_u p_x under UDD assumption.
    """
    j, y = _split_u(u)
    cum = get_cumulative_survival(hat_p, j)
    if y == 0.0 or j >= len(hat_p):
        return cum  # s_j(0) = 1 for all assumptions
    s_j = 1.0 - y * (1.0 - hat_p[j])
    return cum * s_j


def survival_cfm_scalar(hat_p: np.ndarray, u: float) -> float:
    """
    CFM (Constant Force of Mortality) within-year survival at duration u.

    s_j(y) = hat_p[j]^y.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    u : float
        Duration from start.

    Returns
    -------
    float
    """
    j, y = _split_u(u)
    cum = get_cumulative_survival(hat_p, j)
    if y == 0.0 or j >= len(hat_p):
        return cum
    s_j = hat_p[j] ** y
    return cum * s_j


def survival_balducci_scalar(hat_p: np.ndarray, u: float) -> float:
    """
    Balducci within-year survival at duration u.

    s_j(y) = hat_p[j] / (y + (1-y)*hat_p[j]).

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    u : float
        Duration from start.

    Returns
    -------
    float
    """
    j, y = _split_u(u)
    cum = get_cumulative_survival(hat_p, j)
    if y == 0.0 or j >= len(hat_p):
        return cum
    p = hat_p[j]
    s_j = p / (y + (1.0 - y) * p)
    return cum * s_j


# ---------------------------------------------------------------------------
# Within-year survival vectorized (over time arrays)
# ---------------------------------------------------------------------------

def survival_curve_udd(hat_p: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Vectorized UDD survival curve: {}_s p_x for s in times.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities of length >= ceil(max(times)).
    times : np.ndarray
        Array of evaluation times in [0, len(hat_p)].

    Returns
    -------
    np.ndarray
        Survival values, same shape as times.
    """
    times = np.asarray(times, dtype=float)
    result = np.empty_like(times)
    j_arr = np.floor(times).astype(int)
    y_arr = times - j_arr
    for i, (j, y) in enumerate(zip(j_arr, y_arr)):
        cum = get_cumulative_survival(hat_p, j)
        if y == 0.0 or j >= len(hat_p):
            result[i] = cum
        else:
            s_j = 1.0 - y * (1.0 - hat_p[j])
            result[i] = cum * s_j
    return result


def survival_curve_cfm(hat_p: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Vectorized CFM survival curve: {}_s p_x for s in times.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    times : np.ndarray
        Evaluation times.

    Returns
    -------
    np.ndarray
    """
    times = np.asarray(times, dtype=float)
    result = np.empty_like(times)
    j_arr = np.floor(times).astype(int)
    y_arr = times - j_arr
    for i, (j, y) in enumerate(zip(j_arr, y_arr)):
        cum = get_cumulative_survival(hat_p, j)
        if y == 0.0 or j >= len(hat_p):
            result[i] = cum
        else:
            s_j = hat_p[j] ** y
            result[i] = cum * s_j
    return result


def survival_curve_balducci(hat_p: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Vectorized Balducci survival curve: {}_s p_x for s in times.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    times : np.ndarray
        Evaluation times.

    Returns
    -------
    np.ndarray
    """
    times = np.asarray(times, dtype=float)
    result = np.empty_like(times)
    j_arr = np.floor(times).astype(int)
    y_arr = times - j_arr
    for i, (j, y) in enumerate(zip(j_arr, y_arr)):
        cum = get_cumulative_survival(hat_p, j)
        if y == 0.0 or j >= len(hat_p):
            result[i] = cum
        else:
            p = hat_p[j]
            s_j = p / (y + (1.0 - y) * p)
            result[i] = cum * s_j
    return result


# ---------------------------------------------------------------------------
# Strict extremal survival curves (Sections 3 and 5.1)
# ---------------------------------------------------------------------------

def survival_curve_strict_upper(hat_p: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Strict upper GMIB extremal survival curve.

    Concentrates deaths at year-end.  For s in [j, j+1):
        bar_p*(s) = prod_{k: k+1 <= s} hat_p[k]

    At integer s = j: bar_p*(j) = prod_{k=0}^{j-1} hat_p[k] = {}_j p_x.
    For non-integer s in (j, j+1): bar_p*(s) = {}_j p_x  (no deaths yet in year j).

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    times : np.ndarray
        Evaluation times in [0, len(hat_p)].

    Returns
    -------
    np.ndarray
    """
    times = np.asarray(times, dtype=float)
    result = np.empty_like(times)
    for i, s in enumerate(times):
        # k+1 <= s  iff  k <= s-1  iff  k <= floor(s) - 1  when s is non-integer,
        # or k <= s-1 when s is integer.
        # Equivalently: k+1 <= floor(s) is the correct condition.
        # At s in [j, j+1): bar_p*(s) = prod_{k=0}^{j-1} hat_p[k] = {}_j p_x
        j = int(math.floor(s))
        # If s is exactly integer, k+1 <= s means k+1 <= j, i.e., k <= j-1
        # so we still get prod_{k=0}^{j-1} = {}_j p_x
        result[i] = get_cumulative_survival(hat_p, j)
    return result


def survival_curve_strict_lower(hat_p: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Strict lower GMIB extremal survival curve (limiting).

    Concentrates deaths at year-start (left limit).  For s in (j, j+1]:
        underline_p*(s) = prod_{k: k < s} hat_p[k]

    For s in (j, j+1): underline_p*(s) = prod_{k=0}^{j} hat_p[k] = {}_{j+1} p_x.
    At s = 0: underline_p*(0) = 1.

    This survival is the *limit* of controls pushing deaths to time j+1/n -> j+.

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    times : np.ndarray
        Evaluation times in [0, len(hat_p)].

    Returns
    -------
    np.ndarray
    """
    times = np.asarray(times, dtype=float)
    result = np.empty_like(times)
    for i, s in enumerate(times):
        if s <= 0.0:
            result[i] = 1.0
        else:
            # For s > 0: k < s means k <= floor(s) - eps, i.e., k <= floor(s-eps)
            # For non-integer s in (j, j+1): k < s means k <= j.
            # For integer s = j: k < j means k <= j-1, so product is {}_j p_x.
            j = int(math.floor(s))
            y = s - j
            if y > 0:
                # s in (j, j+1): include k=j
                result[i] = get_cumulative_survival(hat_p, j + 1)
            else:
                # s = j (integer): k < j means k = 0,...,j-1
                result[i] = get_cumulative_survival(hat_p, j)
    return result


# ---------------------------------------------------------------------------
# Balducci hazard rate (for relaxed control baseline)
# ---------------------------------------------------------------------------

def balducci_hazard(hat_p: np.ndarray, t: float) -> float:
    """
    Balducci baseline hazard rate at time t = j + u.

    mu_{x,a}(j+u) = (1 - hat_p[j]) / (hat_p[j] + u*(1 - hat_p[j]))

    Parameters
    ----------
    hat_p : np.ndarray
        Annual survival probabilities.
    t : float
        Time from start of policy.

    Returns
    -------
    float
        Non-negative hazard rate.
    """
    j, u = _split_u(t)
    p = hat_p[j]
    qj = 1.0 - p
    denom = p + u * qj
    if denom <= 0:
        return np.inf
    return qj / denom
