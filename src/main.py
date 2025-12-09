#!/usr/bin/env python3
"""Main entry point for solver comparison experiments."""

import sys
from pathlib import Path

src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# flake8: noqa: E402
from solvers.neumann_solver import neumann_solver
from solvers.gmres_solver import solve as gmres_solver
from solvers.bicgstab_solver import solve as bicgstab_solver
from solvers.direct_solver import solve as direct_solver
from experiments.experiment_runner import run_batch_experiments
from experiments.plotting import (
    plot_runtime,
    plot_matvecs,
    plot_convergence,
    plot_convergence_comparison,
    create_summary_table,
)


SOLVERS_ITERATIVE = {
    "neumann": neumann_solver,
    "gmres": gmres_solver,
    "bicgstab": bicgstab_solver,
}

SOLVERS_ALL = {
    **SOLVERS_ITERATIVE,
    "direct": direct_solver,
}

DIRECT_MAX_SIZE = 1000  # direct solver too slow beyond this

SIZES = [int(1e3), int(5e3), int(1e4)]
SIZES_SMALL = [n for n in SIZES if n <= DIRECT_MAX_SIZE]
SIZES_LARGE = [n for n in SIZES if n > DIRECT_MAX_SIZE]

RHOS = [0.5, 0.9, 0.95, 0.99, 0.999]
DENSITY = 0.01
TOL = 1e-8
MAX_ITER = int(1e4)


def main():
    """Run complete experiment pipeline."""
    print("Iterative Solver Comparison for Leontief Systems")
    print(f"Configuration: sizes={SIZES}, rhos={RHOS}, density={DENSITY}")
    print(f"Tol={TOL}, max_iter={MAX_ITER}")
    print(f"Solvers: {list(SOLVERS_ITERATIVE.keys())}, direct enabled for n <= {DIRECT_MAX_SIZE}\n")

    results = {}

    # small systems: can afford to run direct solver for ground truth
    if SIZES_SMALL:
        print(f"Running all solvers for n in {SIZES_SMALL}")
        results_small = run_batch_experiments(
            sizes=SIZES_SMALL,
            rhos=RHOS,
            density=DENSITY,
            solvers=SOLVERS_ALL,
            tol=TOL,
            max_iter=MAX_ITER,
            seed=42,
            verbose=True,
        )
        results.update(results_small)

    # large systems: skip direct (would take forever)
    if SIZES_LARGE:
        print(f"\nRunning iterative solvers only for n in {SIZES_LARGE}")
        results_large = run_batch_experiments(
            sizes=SIZES_LARGE,
            rhos=RHOS,
            density=DENSITY,
            solvers=SOLVERS_ITERATIVE,
            tol=TOL,
            max_iter=MAX_ITER,
            seed=42,
            verbose=True,
        )
        results.update(results_large)

    print("\nGenerating plots...")
    runtime_path = plot_runtime(results)
    print(f"Runtime plot: {runtime_path}")
    matvec_path = plot_matvecs(results)
    print(f"Matvec plot: {matvec_path}")

    for n in SIZES:
        for rho in RHOS:
            if (n, rho) in results:
                path = plot_convergence(results[(n, rho)], n, rho)

    for n in SIZES:
        try:
            path = plot_convergence_comparison(results, n)
        except ValueError:
            pass  # not enough data for this size

    print("\nResults Summary:")
    print(create_summary_table(results))

    print("\nKey Observations:")

    for n in SIZES:
        print(f"\nSystem Size n = {n}:")
        for rho in RHOS:
            if (n, rho) not in results:
                continue

            solver_results = results[(n, rho)]
            print(f"\nρ = {rho}:")

            for solver_name, result in solver_results.items():
                converged = result.get('converged', False)
                iters = result.get('iterations', 0)
                matvecs = result.get('matvecs', 0)
                runtime = result.get('runtime', 0.0)
                residual = result.get('final_residual', float('inf'))

                status = "✓" if converged else "✗"
                print(f"  {status} {solver_name:10s}: {iters:5d} iters, "
                      f"{matvecs:5d} matvecs, {runtime:7.3f}s, "
                      f"residual={residual:.2e}")

    print("\nWinner Analsyis (by matvec count):")
    for n in SIZES:
        print(f"\nSystem Size n = {n}:")
        for rho in RHOS:
            if (n, rho) not in results:
                continue

            solver_results = results[(n, rho)]
            # find winner among converged solvers (not that useful tbf)
            best_solver = None
            best_matvecs = float('inf')
            for solver_name, result in solver_results.items():
                if result.get('converged', False):
                    matvecs = result.get('matvecs', float('inf'))
                    if matvecs < best_matvecs:
                        best_matvecs = matvecs
                        best_solver = solver_name

            if best_solver:
                print(f"  rho = {rho}: {best_solver.upper()} ({best_matvecs} matvecs)")
            else:
                print(f"  rho = {rho}: No solver converged!")

    print("\nExperiment complete.")
    print("Results saved to: results.pkl")
    print("Plots saved to: plots/")


if __name__ == "__main__":
    main()

