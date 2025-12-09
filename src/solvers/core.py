"""Shared utilities for iterative solvers."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from scipy import sparse
import time


@dataclass
class SolverResult:
    """Result container for iterative solvers."""
    x: np.ndarray
    residuals: List[float] = field(default_factory=list)
    iterations: int = 0
    matvecs: int = 0
    runtime: float = 0.0
    converged: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'converged': self.converged,
            'iterations': self.iterations,
            'residual_history': self.residuals,
            'matvecs': self.matvecs,
            'time': self.runtime,
        }


def residual_norm(
    M: sparse.spmatrix | np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    ord: Optional[int] = 2
) -> float:
    """Compute ||b - M @ x||."""
    r = b - M @ x
    return np.linalg.norm(r, ord=ord)


class MatVecCounter:
    """Counts matrix-vector multiplications."""

    def __init__(self, M: sparse.spmatrix | np.ndarray):
        self.M = M
        self.count = 0
        self.shape = M.shape

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute M @ x and increment counter."""
        self.count += 1
        return self.M @ x

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Support @ operator."""
        return self.matvec(x)

    def reset(self) -> None:
        """Reset counter to zero."""
        self.count = 0


class Timer:
    """Simple timer for code blocks."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    def start(self) -> None:
        """Start timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timer and return elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

