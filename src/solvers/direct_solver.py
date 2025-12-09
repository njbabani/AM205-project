"""Direct solver using Gaussian elimination / sparse LU factorization.

Used as baseline for accuracy verification. Direct methods scale poorly
(O(n³) dense, O(n^{1.5}-O(n²) sparse) so iterative methods are preferred
for large systems.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Union

from solvers.core import SolverResult, Timer


def solve(
    M: Union[sparse.spmatrix, np.ndarray],
    b: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 0,
    verbose: bool = False
) -> SolverResult:
    """Solve M x = b using direct Gaussian elimination / sparse LU."""
    timer = Timer()
    timer.start()

    # scipy's spsolve uses SuperLU under the hood
    if sparse.issparse(M):
        x = spsolve(M.tocsr(), b)
    else:
        x = np.linalg.solve(M, b)

    runtime = timer.stop()

    r = b - M @ x
    res_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(b)

    if b_norm > 0:
        rel_res = res_norm / b_norm
    else:
        rel_res = res_norm

    if b_norm > 0:
        converged = res_norm <= tol * b_norm
    else:
        converged = res_norm <= tol

    if verbose:
        print(f"Direct solver complted:")
        print(f"  Runtime:  {runtime:.6f} s")
        print(f"  Residual: {res_norm:.2e}")
        print(f"  ||b||:    {b_norm:.2e}")
        print(f"  Relative: {rel_res:.2e}")
        print(f"  Converged: {converged}")

    return SolverResult(
        x=x,
        residuals=[rel_res],
        iterations=1,  # "1 iteration" for plotting purposes
        matvecs=0,  # n/a for direct methods
        runtime=runtime,
        converged=converged
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    from utils.generate_matrix import generate_leontief_matrix

    n = 200
    A, actual_rho = generate_leontief_matrix(n=n, density=0.02, rho_target=0.9, seed=42)
    M = sparse.eye(n, format='csr') - A
    b = np.ones(n)
    result = solve(M, b, tol=1e-12, verbose=True)
    print(f"x range: [{result.x.min():.6f}, {result.x.max():.6f}]")
