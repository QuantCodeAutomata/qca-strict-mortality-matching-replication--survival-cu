"""
Configuration for strict mortality-matching replication experiment.
Belgian female life table source: Human Mortality Database (HMD), 2010
(synthetic Makeham-Gompertz approximation).
Maturity grid choice: T in {1,2,3,5,7,10,15} — documented replication choice.
"""

LIFE_TABLE_SOURCE = "HMD Belgian female 2010 (synthetic Makeham-Gompertz approximation)"
MATURITY_GRID = [1, 2, 3, 5, 7, 10, 15]
AGES = [40, 60, 80, 100]
AGES_GMDB = [40, 60, 80]

# Financial parameters
A0 = 100.0
R_G = 0.03
R0 = 0.01
PI_S = 0.6
PI_P = 0.2
SIGMA_S = 0.2
SIGMA_R = 0.02
RHO = 0.3
KAPPA = 0.1

# GMIB parameters
R_I = 0.01
BETA = 0.05

# Forward curve parameters f(0,t) = b + c*exp(-a*t) + d*a*t*exp(-a*t)
FWD_A = 0.76
FWD_B = 0.015
FWD_C = -0.005
FWD_D = 0.02
FWD_PARAMS = {"a": FWD_A, "b": FWD_B, "c": FWD_C, "d": FWD_D}

# Numerical parameters
N_GRID = 10000   # dense time grid for integrals
ANNUAL_PRODUCT_TOL = 1e-10
PRICING_TOL = 1e-6

# LSM parameters (exp_2)
N_TRAIN = 200000
N_EVAL = 400000
N_TRAIN_FAST = 20000    # reduced for speed in tests/quick runs
N_EVAL_FAST = 40000
N_STEPS_PER_YEAR = 104
RNG_SEED = 42

# Relaxed control parameters (exp_3)
DELTA_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DELTA_MAIN = 1.0

# Relaxed PDE grid
N_T_PDE = 500
N_X_PDE = 200

# Deterministic rate for relaxed experiment (Section 5.2 simplification)
R_DET = 0.01
