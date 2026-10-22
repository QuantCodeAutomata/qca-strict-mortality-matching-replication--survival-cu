# Strict Mortality-Matching Replication

Reproducible Python implementation of strict almost-sure life-table matching
for insurance variable annuity products (GMAB, GMIB, GMDB) under within-year
mortality ambiguity.

## Repository Structure

```
repo/
├── data/                # Life table generation script + generated CSV
│   └── life_table_generator.py
├── src/                 # Core implementation modules
│   ├── config.py        # (symlinked from root)
│   ├── mortality.py     # Life table, UDD/CFM/Balducci survival curves
│   ├── finance.py       # Hull-White bond pricing, GMAB/GMIB formulas
│   ├── strict_gmab_gmib.py  # Strict GMAB invariance, GMIB bounds
│   ├── lsm_gmdb.py      # LSM Monte Carlo for GMDB stopping problems
│   └── relaxed_control.py   # Bang-bang HJB PDE for relaxed bounds
├── exp/                 # Experiment scripts
│   ├── exp_1_strict_survival.py   # Exp 1: survival curves, GMAB/GMIB
│   ├── exp_2_gmdb_lsm.py          # Exp 2: GMDB bounds via LSM
│   └── exp_3_relaxed.py           # Exp 3: relaxed mortality controls
├── tests/               # pytest test suite
│   ├── test_mortality.py
│   ├── test_finance.py
│   ├── test_strict_gmab_gmib.py
│   ├── test_lsm_gmdb.py
│   └── test_relaxed_control.py
├── results/             # Output: plots, tables, RESULTS.md
│   └── RESULTS.md
├── config.py            # All parameters (life table source, financial params)
└── README.md
```

## Experiments

### Experiment 1 — Strict Survival Curves and GMAB/GMIB Bounds

Reproduces:
- Within-year survival curves under UDD, CFM, and Balducci assumptions
- Strict extremal survival curves (year-end vs year-start death timing)
- GMAB strict value and invariance over all admissible controls
- GMIB strict upper and lower bounds and benchmark comparisons

```bash
python exp/exp_1_strict_survival.py
```

### Experiment 2 — GMDB Bounds via LSM

Reproduces:
- Strict GMDB upper/lower bounds via one-year optimal stopping
- LSM backward induction with 104 steps/year, degree-3 polynomial basis
- Stopping policy diagnostics (left/interior/right-endpoint frequencies)
- Annual death weights and benchmark comparisons

```bash
python exp/exp_2_gmdb_lsm.py
```

*Note*: Full replication uses 200,000 training + 400,000 evaluation paths. Default uses 20,000 / 40,000 for speed. Set `USE_PAPER_PATHS=True` in `exp_2_gmdb_lsm.py` for full run.

### Experiment 3 — Relaxed Mortality-Control Bounds

Reproduces:
- Bang-bang HJB PDE bounds for GMAB, GMIB, GMDB under Section 5.2 simplification
- Relaxed admissible set: expectation matching at maturity and at intermediate dates
- Delta-convergence study (x=60, T=5)
- Comparison with Balducci baseline

```bash
python exp/exp_3_relaxed.py
```

## Setup

```bash
# Generate life table
python data/life_table_generator.py

# Run tests
pytest tests/ -v

# Run all experiments
python exp/exp_1_strict_survival.py
python exp/exp_2_gmdb_lsm.py
python exp/exp_3_relaxed.py
```

## Parameters

All parameters are in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| A₀ | 100 | Initial fund value |
| r_g | 0.03 | Guarantee growth rate |
| r₀ | 0.01 | Initial short rate |
| π_S | 0.6 | Stock allocation |
| π_P | 0.2 | Bond allocation |
| σ_S | 0.20 | Stock volatility |
| σ_r | 0.02 | Short-rate volatility |
| ρ | 0.3 | Correlation |
| κ | 0.1 | HW mean reversion |
| r_I | 0.01 | GMIB fixed income rate |
| β | 0.05 | GMIB participation rate |

Forward curve: f(0,t) = b + c·e^{-at} + d·a·t·e^{-at}
with a=0.76, b=0.015, c=-0.005, d=0.02.

## Life Table

Synthetic Belgian female life table generated via Makeham-Gompertz approximation calibrated to HMD Belgian female 2010 mortality:
- A=0.0005, B=0.00004, c=1.10, radix l₀=100,000.

## Maturity Grid

T ∈ {1, 2, 3, 5, 7, 10, 15} — documented replication choice (exact grid not specified in paper).
