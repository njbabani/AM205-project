"""Generate sparse nonnegative matrices with controllable spectral radius."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from typing import Tuple, Optional


def generate_leontief_matrix(
    n: int,
    density: float = 0.01,
    rho_target: float = 0.9,
    seed: Optional[int] = None
) -> Tuple[sparse.csr_matrix, float]:
    """Generate sparse nonnegative matrix with target spectral radius."""
    if not 0 < rho_target < 1:
        raise ValueError("rho_target must be in (0, 1)")
    
    if seed is not None:
        np.random.seed(seed)
    
    # random sparse matrix with uniform positive entries
    nnz_expected = int(n * n * density)
    rows = np.random.randint(0, n, size=nnz_expected)
    cols = np.random.randint(0, n, size=nnz_expected)
    data = np.random.uniform(0.1, 1.0, size=nnz_expected)
    
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A.tocoo().tocsr()  # dedupe any overlapping entries
    
    # scale to hit target spectral radius
    actual_rho = spectral_radius(A)
    if actual_rho > 0:
        scale = rho_target / actual_rho
        A = A * scale
    
    final_rho = spectral_radius(A)
    return A, final_rho


def spectral_radius(A: sparse.spmatrix, tol: float = 1e-6) -> float:
    """Compute spectral radius ρ(A) = max|λ_i|."""
    n = A.shape[0]
    
    # small matrices: just compute all eigenvalues
    if n < 3:
        eigvals = np.linalg.eigvals(A.toarray())
        return np.max(np.abs(eigvals))
    
    try:
        # use ARPACK for large matrices
        k = min(6, n - 2)
        eigvals, _ = eigs(A.astype(float), k=k, which='LM', tol=tol)
        return np.max(np.abs(eigvals))
    except Exception:
        # fallback if ARPACK fails (can happen for weird matrices)
        eigvals = np.linalg.eigvals(A.toarray())
        return np.max(np.abs(eigvals))


def matrix_info(A: sparse.spmatrix) -> dict:
    """Return summary statisics about the matrix."""
    n = A.shape[0]
    nnz = A.nnz
    rho = spectral_radius(A)
    row_sums = np.array(A.sum(axis=1)).flatten()
    
    return {
        'n': n,
        'nnz': nnz,
        'density': nnz / (n * n),
        'spectral_radius': rho,
        'row_sum_min': row_sums.min(),
        'row_sum_max': row_sums.max(),
        'row_sum_mean': row_sums.mean(),
        'condition_approx': 1 / (1 - rho) if rho < 1 else np.inf
    }


if __name__ == "__main__":
    for rho_target in [0.5, 0.7, 0.9, 0.95, 0.99]:
        A, actual_rho = generate_leontief_matrix(n=500, density=0.02, rho_target=rho_target, seed=42)
        info = matrix_info(A)
        print(f"ρ={rho_target:.2f}: actual={info['spectral_radius']:.6f}, "
              f"condition≈{info['condition_approx']:.1f}")
