# Iterative Solvers for Leontief Systems

Comparing iterative methods for solving large-scale Leontief input-output systems of the form (I - A)x = b, where A is a sparse non-negative matrix with spectral radius ρ(A) < 1.

## Overview

This project implements and compares three iterative solvers:

- **Neumann Series**: Fixed-point iteration x^{k+1} = Ax^k + b
- **GMRES**: Generalized Minimal Residual method (Saad & Schultz, 1986)
- **BiCGSTAB**: Bi-Conjugate Gradient Stabilized (van der Vorst, 1992)

We analyze convergence behavior across different matrix sizes (n = 1000 to 10000) and spectral radii (ρ = 0.5 to 0.999).

## Usage

```bash
pip install -r requirements.txt
cd src
python main.py
```

Results and plots are saved to `src/results/` and `src/plots/`.

## Project Structure

```
src/
├── main.py                    # Run experiments
├── solvers/
│   ├── neumann_solver.py      # Neumann series iteration
│   ├── gmres_solver.py        # GMRES implementation
│   ├── bicgstab_solver.py     # BiCGSTAB implementation
│   └── direct_solver.py       # Direct solve (baseline)
├── experiments/
│   ├── experiment_runner.py   # Batch experiment framework
│   └── plotting.py            # Visualization
└── utils/
    └── generate_matrix.py     # Leontief matrix generation
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
