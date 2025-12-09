"""Functions for running solver comparison experiments."""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Any, Callable
import pickle
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.generate_matrix import generate_leontief_matrix, matrix_info
from solvers.core import SolverResult


def build_system_from_target(
    n: int,
    rho: float,
    density: float,
    seed: int = None
) -> Tuple[sparse.spmatrix, sparse.spmatrix, np.ndarray, dict]:
    """Build Leontief system M = (I - A), b = ones from target parameters."""
    A, actual_rho = generate_leontief_matrix(n=n, density=density, rho_target=rho, seed=seed)
    I = sparse.eye(n, format='csr')
    M = I - A
    b = np.ones(n)
    info = matrix_info(A)
    info['target_rho'] = rho
    info['actual_rho'] = actual_rho
    return A, M, b, info


def _normalize_solver_result(result: Any, solver_name: str) -> Dict[str, Any]:
    """Normalize solver output to consistent dictionary format."""
    if isinstance(result, SolverResult):
        return {
            'x': result.x,
            'converged': result.converged,
            'iterations': result.iterations,
            'residuals': result.residuals,
            'matvecs': result.matvecs,
            'runtime': result.runtime,
        }
    elif isinstance(result, tuple) and len(result) == 2:
        # neumann solver returns (x, info_dict)
        x, info = result
        return {
            'x': x,
            'converged': info.get('converged', False),
            'iterations': info.get('iterations', 0),
            'residuals': info.get('residual_history', []),
            'matvecs': info.get('iterations', 0),
            'runtime': info.get('time', 0.0),
        }
    else:
        raise ValueError(f"Unknown result type from solver '{solver_name}': {type(result)}")


def run_single_experiment(
    n: int,
    rho: float,
    density: float,
    solvers: Dict[str, Callable],
    tol: float = 1e-8,
    max_iter: int = 1000,
    seed: int = None,
    verbose: bool = True
) -> Tuple[Dict[str, Dict], Dict]:
    """Run experiment comparing solvers on one system."""
    if verbose:
        print(f"\nRunning experiment: n={n}, ρ={rho}, density={density}")

    A, M, b, sys_info = build_system_from_target(n, rho, density, seed)

    if verbose:
        print(f"Generated system: actual ρ(A) = {sys_info['actual_rho']:.6f}")
        print(f"Matrix: {sys_info['n']}x{sys_info['n']}, {sys_info['nnz']} nonzeros")

    results = {}

    for name, solver_fn in solvers.items():
        if verbose:
            print(f"\nRunning {name}...")

        try:
            start_time = time.perf_counter()

            # neumann takes A directly, others take M = I - A
            if name.lower() == 'neumann':
                result = solver_fn(A, tol=tol, max_iter=max_iter, verbose=False)
            else:
                result = solver_fn(M, b, tol=tol, max_iter=max_iter, verbose=False)

            elapsed = time.perf_counter() - start_time
            result_dict = _normalize_solver_result(result, name)

            if result_dict['x'] is not None:
                b_norm = np.linalg.norm(b)
                if b_norm > 1e-14:
                    final_residual = np.linalg.norm(b - M @ result_dict['x']) / b_norm
                else:
                    final_residual = np.linalg.norm(b - M @ result_dict['x'])
                result_dict['final_residual'] = final_residual
            else:
                result_dict['final_residual'] = float('inf')

            results[name] = result_dict

            if verbose:
                status = "✓ converged" if result_dict['converged'] else "✗ did not converge"
                print(f"  {status} in {result_dict['iterations']} iterations")
                print(f"  Runtime: {result_dict['runtime']:.4f}s, Matvecs: {result_dict['matvecs']}")
                print(f"  Final residual: {result_dict['final_residual']:.2e}")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results[name] = {
                'x': None,
                'converged': False,
                'iterations': 0,
                'residuals': [],
                'matvecs': 0,
                'runtime': 0.0,
                'final_residual': float('inf'),
                'error': str(e),
            }

    metadata = {
        'n': n,
        'rho': rho,
        'density': density,
        'tol': tol,
        'max_iter': max_iter,
        'seed': seed,
        'system_info': sys_info,
    }

    return results, metadata


def run_batch_experiments(
    sizes: List[int],
    rhos: List[float],
    density: float,
    solvers: Dict[str, Callable],
    tol: float = 1e-8,
    max_iter: int = 1000,
    seed: int = 42,
    output_dir: str = None,
    verbose: bool = True
) -> Dict[Tuple[int, float], Dict[str, Dict]]:
    """Run batch experiments over grid of paramters. Saves results to results.pkl."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        total = len(sizes) * len(rhos)
        print(f"\nBatch experiment: {total} configurations")
        print(f"Sizes: {sizes}, Spectral radii: {rhos}, Solvers: {list(solvers.keys())}")

    all_results = {}
    all_metadata = {}
    experiment_num = 0

    for n in sizes:
        for rho in rhos:
            experiment_num += 1
            exp_seed = seed + experiment_num if seed is not None else None

            if verbose:
                print(f"\n[{experiment_num}/{len(sizes)*len(rhos)}] ", end="")

            results, metadata = run_single_experiment(
                n=n,
                rho=rho,
                density=density,
                solvers=solvers,
                tol=tol,
                max_iter=max_iter,
                seed=exp_seed,
                verbose=verbose
            )

            all_results[(n, rho)] = results
            all_metadata[(n, rho)] = metadata

    output_file = output_dir / 'results.pkl'
    save_data = {
        'results': all_results,
        'metadata': all_metadata,
        'config': {
            'sizes': sizes,
            'rhos': rhos,
            'density': density,
            'tol': tol,
            'max_iter': max_iter,
            'solver_names': list(solvers.keys()),
        }
    }

    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_make_serializable(item) for item in obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    serializable_data = _make_serializable(save_data)

    with open(output_file, 'wb') as f:
        pickle.dump(serializable_data, f)

    if verbose:
        print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    A, M, b, info = build_system_from_target(n=500, rho=0.9, density=0.01)
    print(f"Built system: {info['n']}x{info['n']}, ρ={info['actual_rho']:.4f}")

    try:
        from solvers.bicgstab_solver import solve as bicgstab_solve
        test_solvers = {'bicgstab': bicgstab_solve}
        results, meta = run_single_experiment(
            n=500, rho=0.9, density=0.01,
            solvers=test_solvers,
            tol=1e-8, max_iter=500,
            verbose=True
        )
    except ImportError:
        pass

