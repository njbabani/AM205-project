"""BiCGSTAB implementation following van der Vorst (1992)."""

import numpy as np
from scipy import sparse
from typing import Optional, Callable, Tuple

from .core import SolverResult, MatVecCounter, Timer


def solve(
    M: sparse.spmatrix | np.ndarray,
    b: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 1000,
    x0: Optional[np.ndarray] = None,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    verbose: bool = False
) -> SolverResult:
    """Solve M @ x = b using BiCGSTAB (van der Vorst 1992)."""
    n = M.shape[0]
    counter = MatVecCounter(M)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - counter.matvec(x)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        return SolverResult(
            x=np.zeros(n),
            residuals=[0.0],
            iterations=0,
            matvecs=counter.count,
            runtime=0.0,
            converged=True
        )

    abs_tol = tol * b_norm
    r_tilde = r.copy()  # shadow residual - kept fixed throughout

    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0

    v = np.zeros(n)
    p = np.zeros(n)

    residuals = [np.linalg.norm(r) / b_norm]

    timer = Timer()
    timer.start()

    converged = False

    for i in range(1, max_iter + 1):
        # BiCG part
        rho = np.dot(r_tilde, r)

        if abs(rho) < 1e-14:
            if verbose:
                print(f"BiCGSTAB breakdown: ρ = {rho:.2e} at iterration {i}")
            break

        beta = (rho / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * v)

        if precond is not None:
            p_hat = precond(p)
            v = counter.matvec(p_hat)
        else:
            v = counter.matvec(p)

        r_tilde_dot_v = np.dot(r_tilde, v)

        if abs(r_tilde_dot_v) < 1e-14:
            if verbose:
                print(f"BiCGSTAB breakdown: (r̃_0, v) = {r_tilde_dot_v:.2e} at iteration {i}")
            break

        alpha = rho / r_tilde_dot_v
        s = r - alpha * v  # intermediate residual

        s_norm = np.linalg.norm(s)
        if s_norm < abs_tol:
            if precond is not None:
                x = x + alpha * p_hat
            else:
                x = x + alpha * p
            residuals.append(s_norm / b_norm)
            converged = True
            break

        if precond is not None:
            s_hat = precond(s)
            t = counter.matvec(s_hat)
        else:
            t = counter.matvec(s)

        # stabilization step
        t_dot_t = np.dot(t, t)
        if abs(t_dot_t) < 1e-14:
            omega = 0.0
        else:
            omega = np.dot(t, s) / t_dot_t

        if precond is not None:
            x = x + alpha * p_hat + omega * s_hat
        else:
            x = x + alpha * p + omega * s

        r = s - omega * t

        r_norm = np.linalg.norm(r)
        residuals.append(r_norm / b_norm)

        if verbose and (i <= 10 or i % 10 == 0):
            print(f"Iter {i:4d}: ||r||/||b|| = {r_norm / b_norm:.2e}")

        if r_norm < abs_tol:
            converged = True
            break

        if abs(omega) < 1e-14:
            if verbose:
                print(f"BiCGSTAB stagnation: ω = {omega:.2e} at iteration {i}")
            break

        rho_prev = rho

    runtime = timer.stop()

    final_res = np.linalg.norm(b - M @ x) / b_norm
    converged = final_res < tol

    if verbose:
        status = "converged" if converged else "did NOT converge"
        print(f"\nBiCGSTAB {status} in {i} iterations "
              f"({counter.count} matvecs, {runtime:.4f}s)")
        print(f"Final relative residual: {final_res:.2e}")

    return SolverResult(
        x=x,
        residuals=residuals,
        iterations=i,
        matvecs=counter.count,
        runtime=runtime,
        converged=converged
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.generate_matrix import generate_leontief_matrix

    for rho_target in [0.5, 0.9, 0.95]:
        A, actual_rho = generate_leontief_matrix(n=500, density=0.01, rho_target=rho_target)
        n = A.shape[0]
        M = sparse.eye(n) - A
        b = np.ones(n)
        result = solve(M, b, tol=1e-10, max_iter=2000, verbose=True)
        residual = np.linalg.norm(b - M @ result.x)
        print(f"Residual: {residual:.2e}\n")
