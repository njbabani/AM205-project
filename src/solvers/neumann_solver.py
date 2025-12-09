"""Neumann series solver using fixed-point iteration.

Based on PageRank power iteration (Page & Brin, 1998). Uses relative residual
for comparison with Krylov methods.
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, List
import time


def neumann_solver(
    A: sparse.spmatrix | np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
    verbose: bool = True
) -> Tuple[np.ndarray, dict]:
    """Solve (I - A)x = 1 using fixed-point iteration x^{k+1} = A @ x^k + 1.
    
    Converges when ρ(A) < 1. Uses relative residual for comparison with Krylov methods.
    """
    n = A.shape[0]
    x = np.ones(n)  # initial guess
    one = np.ones(n)
    b_norm = np.linalg.norm(one)
    
    delta_history = []  # tracks L1 norm
    residual_history = []
    
    start_time = time.perf_counter()
    
    for k in range(max_iter):
        x_new = A @ x + one  # power iteration
        
        delta = np.linalg.norm(x_new - x, ord=1)
        delta_history.append(delta)
        
        # compute true residual for fair comparison w/ krylov methods
        residual = np.linalg.norm(one - (x_new - A @ x_new)) / b_norm
        residual_history.append(residual)
        
        x = x_new
        
        if verbose and (k < 10 or k % 10 == 0):
            print(f"Iter {k:4d}: δ = {delta:.2e}, rel_residual = {residual:.2e}")
        
        if residual < tol:
            elapsed = time.perf_counter() - start_time
            if verbose:
                print(f"\nConverged in {k+1} iterations ({elapsed:.4f}s)")
                print(f"Final relative residual: {residual:.2e}")
            return x, {
                'converged': True,
                'iterations': k + 1,
                'delta_history': delta_history,
                'residual_history': residual_history,
                'time': elapsed
            }
    
    elapsed = time.perf_counter() - start_time
    if verbose:
        print(f"\nDid NOT converge after {max_iter} iterations ({elapsed:.4f}s)")
        print(f"Final relative residual = {residual_history[-1]:.2e}")
    
    return x, {
        'converged': False,
        'iterations': max_iter,
        'delta_history': delta_history,
        'residual_history': residual_history,
        'time': elapsed
    }


def verify_solution(A: sparse.spmatrix | np.ndarray, x: np.ndarray) -> float:
    """Compute relative residual to verifiy solution."""
    one = np.ones(A.shape[0])
    residual = one - (x - A @ x)
    return np.linalg.norm(residual) / np.linalg.norm(one)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.generate_matrix import generate_leontief_matrix
    
    for rho_target in [0.5, 0.9, 0.95, 0.99]:
        A, actual_rho = generate_leontief_matrix(n=1000, density=0.01, rho_target=rho_target)
        print(f"Testing ρ(A) = {rho_target}, actual = {actual_rho:.6f}")
        x, info = neumann_solver(A, tol=1e-10, max_iter=500, verbose=True)
        err = verify_solution(A, x)
        print(f"Residual: {err:.2e}\n")

