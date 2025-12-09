"""GMRES implementation following Saad & Schultz (1986)."""

import numpy as np
from scipy import sparse
from typing import Optional, Callable, Tuple, List

from .core import SolverResult, MatVecCounter, Timer


def _apply_givens_rotation(
    h: np.ndarray,
    cs: np.ndarray,
    sn: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Givens rotations for progressive QR factoriation (Saad & Schultz 1986)."""
    # apply previous rotations to new column
    for i in range(k):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]
        h[i] = temp

    # new rotation coefficients
    r = np.sqrt(h[k] ** 2 + h[k + 1] ** 2)
    if r > 1e-16:
        cs[k] = h[k] / r
        sn[k] = h[k + 1] / r
    else:
        cs[k] = 1.0
        sn[k] = 0.0

    h[k] = r
    h[k + 1] = 0.0

    return h, cs, sn


def _arnoldi_step(
    A_op: Callable[[np.ndarray], np.ndarray],
    V: np.ndarray,
    k: int,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """One step of Arnoldi process using Modified Gram-Schmidt (Saad & Schultz 1986)."""
    w = A_op(V[:, k])
    if precond is not None:
        w = precond(w)

    h = np.zeros(k + 2)

    # modified Gram-Schmidt orthogonalization
    for i in range(k + 1):
        h[i] = np.dot(w, V[:, i])
        w = w - h[i] * V[:, i]

    h_subdiag = np.linalg.norm(w)
    h[k + 1] = h_subdiag

    if h_subdiag > 1e-14:
        v_new = w / h_subdiag
    else:
        v_new = np.zeros_like(w)

    return h, v_new, h_subdiag


def _solve_upper_triangular(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve R @ y = b via back substitution."""
    n = len(b)
    if n == 0:
        return np.array([])

    y = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) > 1e-14:
            y[i] = (b[i] - np.dot(R[i, i + 1:], y[i + 1:])) / R[i, i]
        else:
            y[i] = 0.0

    return y


def gmres_iteration(
    A_op: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    m: int,
    tol: float,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Tuple[np.ndarray, List[float], int, bool]:
    """One cycle of GMRES(m) iteration (Saad & Schultz 1986, Algorithm 4)."""
    n = len(b)

    r0 = b - A_op(x0)
    if precond is not None:
        r0 = precond(r0)

    beta = np.linalg.norm(r0)
    residuals = [beta]

    if beta < tol:
        return x0.copy(), residuals, 0, True

    V = np.zeros((n, m + 1))
    V[:, 0] = r0 / beta

    H = np.zeros((m + 1, m))
    cs = np.zeros(m)
    sn = np.zeros(m)

    g = np.zeros(m + 1)
    g[0] = beta

    converged = False
    num_iters = 0

    for k in range(m):
        h_col, v_new, h_subdiag = _arnoldi_step(A_op, V, k, precond)
        V[:, k + 1] = v_new

        h_col, cs, sn = _apply_givens_rotation(h_col, cs, sn, k)
        H[:k + 2, k] = h_col

        g_old = g[k]
        g[k] = cs[k] * g_old
        g[k + 1] = -sn[k] * g_old

        num_iters = k + 1

        res_norm = abs(g[k + 1])
        residuals.append(res_norm)

        if res_norm < tol:
            converged = True
            y = _solve_upper_triangular(H[:k + 1, :k + 1], g[:k + 1])
            x = x0 + V[:, :k + 1] @ y
            return x, residuals, num_iters, converged

        if h_subdiag < 1e-14:
            # Krylov subspace exhausted - we've found the exact solution
            y = _solve_upper_triangular(H[:k + 1, :k + 1], g[:k + 1])
            x = x0 + V[:, :k + 1] @ y
            return x, residuals, num_iters, True

    y = _solve_upper_triangular(H[:m, :m], g[:m])
    x = x0 + V[:, :m] @ y

    return x, residuals, num_iters, converged


def solve(
    M: sparse.spmatrix | np.ndarray,
    b: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 1000,
    restart: int = 30,
    x0: Optional[np.ndarray] = None,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    verbose: bool = False
) -> SolverResult:
    """Solve M @ x = b using restarted GMRES(m) (Saad & Schultz 1986)."""
    n = M.shape[0]
    counter = MatVecCounter(M)

    def A_op(x):
        return counter.matvec(x)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        return SolverResult(
            x=np.zeros(n),
            residuals=[0.0],
            iterations=0,
            matvecs=0,
            runtime=0.0,
            converged=True
        )

    abs_tol = tol * b_norm
    all_residuals = []

    timer = Timer()
    timer.start()

    total_iters = 0
    outer_iter = 0
    converged = False

    # outer loop handles restarts
    while total_iters < max_iter:
        remaining = max_iter - total_iters
        cycle_m = min(restart, remaining)

        if cycle_m <= 0:
            break

        x, residuals, inner_iters, converged = gmres_iteration(
            A_op, b, x, cycle_m, abs_tol, precond
        )

        relative_residuals = [r / b_norm for r in residuals]

        # skip duplicate residual at restart boundaries
        if outer_iter == 0:
            all_residuals.extend(relative_residuals)
        else:
            all_residuals.extend(relative_residuals[1:])

        total_iters += inner_iters
        outer_iter += 1

        if verbose:
            print(f"GMRES restart {outer_iter}: {inner_iters} iters, "
                  f"residual = {relative_residuals[-1]:.2e}")

        if converged:
            break

        if inner_iters == 0:
            break

    runtime = timer.stop()

    final_res = np.linalg.norm(b - M @ x) / b_norm
    converged = final_res < tol

    if verbose:
        status = "converged" if converged else "did NOT converge"
        print(f"\nGMRES {status} in {total_iters} iterations "
              f"({counter.count} matvecs, {runtime:.4f}s)")
        print(f"Final relative residual: {final_res:.2e}")

    return SolverResult(
        x=x,
        residuals=all_residuals,
        iterations=total_iters,
        matvecs=counter.count,
        runtime=runtime,
        converged=converged
    )


def _run_verification_tests():
    """Run verification tests."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    all_passed = True
    n = 100
    I = sparse.eye(n, format='csr')
    b = np.random.randn(n)
    result = solve(I, b, tol=1e-12, max_iter=100, restart=30, verbose=False)
    error = np.linalg.norm(result.x - b)

    if result.converged and result.iterations <= 2 and error < 1e-10:
        pass
    else:
        all_passed = False
    np.random.seed(42)
    n = 100
    I = sparse.eye(n, format='csr')
    E = sparse.random(n, n, density=0.1, format='csr')
    M = I + 0.1 * E
    b = np.ones(n)
    result = solve(M, b, tol=1e-10, max_iter=200, restart=50, verbose=False)
    residual = np.linalg.norm(b - M @ result.x)

    if result.converged and residual < 1e-8:
        pass
    else:
        all_passed = False
    n = 50
    D = sparse.diags(np.arange(1, n + 1, dtype=float), format='csr')
    b = np.ones(n)
    result = solve(D, b, tol=1e-12, max_iter=100, restart=50, verbose=False)
    x_exact = 1.0 / np.arange(1, n + 1)
    error = np.linalg.norm(result.x - x_exact)

    if result.converged and error < 1e-10:
        pass
    else:
        all_passed = False
    try:
        from utils.generate_matrix import generate_leontief_matrix
        from .bicgstab_solver import solve as bicgstab_solve

        A, actual_rho = generate_leontief_matrix(n=500, density=0.01,
                                                  rho_target=0.9, seed=123)
        n = A.shape[0]
        M = sparse.eye(n) - A
        b = np.ones(n)

        result_gmres = solve(M, b, tol=1e-10, max_iter=500, restart=50,
                             verbose=False)
        result_bicg = bicgstab_solve(M, b, tol=1e-10, max_iter=500,
                                     verbose=False)

        gmres_res = np.linalg.norm(b - M @ result_gmres.x)

        if result_gmres.converged and gmres_res < 1e-8:
            pass
        else:
            all_passed = False

    except ImportError:
        pass
    np.random.seed(99)
    n = 100
    A = sparse.random(n, n, density=0.1, format='csr')
    M = sparse.eye(n) + 0.5 * A
    b = np.ones(n)
    result = solve(M, b, tol=1e-10, max_iter=200, restart=30, verbose=False)

    return all_passed


if __name__ == "__main__":
    _run_verification_tests()
