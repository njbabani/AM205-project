"""Plotting utilities for solver comparison. Plots saved to plots/ directory."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from pathlib import Path


SOLVER_COLORS = {
    'neumann': '#1f77b4',    # Blue
    'gmres': '#ff7f0e',       # Orange
    'bicgstab': '#2ca02c',    # Green
    'direct': '#d62728',      # Red
}

SOLVER_MARKERS = {
    'neumann': 'o',
    'gmres': 's',
    'bicgstab': '^',
    'direct': 'D',            # Diamond
}

SOLVER_LINESTYLES = {
    'neumann': '-',
    'gmres': '--',
    'bicgstab': '-.',
    'direct': ':',
}


def _ensure_plot_dir(base_dir: Path = None) -> Path:
    """Ensure plots directory exists."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / 'plots'
    else:
        base_dir = Path(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _get_solver_style(solver_name: str) -> dict:
    """Get consistant styling for a solver."""
    name_lower = solver_name.lower()
    return {
        'color': SOLVER_COLORS.get(name_lower, '#333333'),
        'marker': SOLVER_MARKERS.get(name_lower, 'o'),
        'linestyle': SOLVER_LINESTYLES.get(name_lower, '-'),
    }


def plot_convergence(
    result_dict: Dict[str, Dict],
    n: int,
    rho: float,
    output_dir: Path = None,
    show: bool = False
) -> str:
    """Plot residual convergence history for all solvers."""
    plot_dir = _ensure_plot_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    max_iters = 0
    min_iters = float('inf')
    all_residuals = []

    for solver_name, result in result_dict.items():
        residuals = result.get('residuals', [])
        if len(residuals) == 0:
            continue
            
        iters = len(residuals)
        max_iters = max(max_iters, iters)
        min_iters = min(min_iters, iters)
        all_residuals.append((solver_name, residuals))

        style = _get_solver_style(solver_name)
        iterations = np.arange(len(residuals))

        ax.semilogy(
            iterations,
            residuals,
            label=solver_name.upper(),
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            markevery=max(1, len(residuals) // 20),  # don't clutter with markers
            markersize=6,
            linewidth=1.5,
        )

    # inset zoom for when one solver is way faster than others
    if min_iters > 5 and max_iters > 50 * min_iters:
        # Create inset (top right)
        ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3]) 
        
        # Limit inset
        inset_limit = int(min_iters * 2.5)
        
        for solver_name, residuals in all_residuals:
            style = _get_solver_style(solver_name)
            if len(residuals) > 0:
                limit = min(len(residuals), inset_limit + 1)
                ax_ins.semilogy(
                    np.arange(limit),
                    residuals[:limit],
                    color=style['color'],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markevery=max(1, limit // 10),
                    markersize=4,
                    linewidth=1.5
                )
        
        ax_ins.set_xlim(0, inset_limit)
        ax_ins.set_title(f'Zoom: First {inset_limit} Iterations', fontsize=10)
        ax_ins.grid(True, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'Relative Residual $\|r\|/\|b\|$', fontsize=12)
    ax.set_title(f'Convergence History (n={n}, ρ={rho})', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)

    # Add horizontal line at typical tolerance
    ax.axhline(y=1e-8, color='gray', linestyle=':', alpha=0.5, label='tol=1e-8')

    plt.tight_layout()

    # Save figure
    # Replace decimal point with 'p' for filename compatibility
    rho_str = str(rho).replace('.', 'p')
    filename = f'convergence_n{n}_rho{rho_str}.png'
    filepath = plot_dir / filename

    fig.savefig(filepath, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(filepath)


def plot_runtime(
    results: Dict[Tuple[int, float], Dict[str, Dict]],
    output_dir: Path = None,
    show: bool = False
) -> str:
    """Plot runtime vs specral radius for each solver."""
    plot_dir = _ensure_plot_dir(output_dir)

    # Extract unique sizes and rhos
    all_keys = list(results.keys())
    sizes = sorted(set(k[0] for k in all_keys))
    rhos = sorted(set(k[1] for k in all_keys))

    # Get solver names from first result
    first_result = results[all_keys[0]]
    solver_names = list(first_result.keys())

    # Create subplot for each size
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5), squeeze=False)

    for idx, n in enumerate(sizes):
        ax = axes[0, idx]

        for solver_name in solver_names:
            style = _get_solver_style(solver_name)
            runtimes = []
            valid_rhos = []

            for rho in rhos:
                if (n, rho) in results:
                    result = results[(n, rho)].get(solver_name, {})
                    runtime = result.get('runtime', None)
                    if runtime is not None and runtime > 0:
                        runtimes.append(runtime)
                        valid_rhos.append(rho)

            if len(runtimes) > 0:
                ax.plot(
                    valid_rhos,
                    runtimes,
                    label=solver_name.upper(),
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel('Spectral Radius ρ(A)', fontsize=12)
        ax.set_ylabel('Runtime (seconds)', fontsize=12)
        ax.set_title(f'n = {n}', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(rhos) - 0.05, max(rhos) + 0.05)

    fig.suptitle('Runtime vs Spectral Radius', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save figure
    filepath = plot_dir / 'runtime.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(filepath)


def plot_matvecs(
    results: Dict[Tuple[int, float], Dict[str, Dict]],
    output_dir: Path = None,
    show: bool = False
) -> str:
    """Plot matrix-vector product count vs spectral radius."""
    plot_dir = _ensure_plot_dir(output_dir)

    # Extract unique sizes and rhos
    all_keys = list(results.keys())
    sizes = sorted(set(k[0] for k in all_keys))
    rhos = sorted(set(k[1] for k in all_keys))

    # Get solver names from first result
    first_result = results[all_keys[0]]
    solver_names = list(first_result.keys())

    # Create subplot for each size
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5), squeeze=False)

    for idx, n in enumerate(sizes):
        ax = axes[0, idx]

        for solver_name in solver_names:
            style = _get_solver_style(solver_name)
            matvecs_list = []
            valid_rhos = []

            for rho in rhos:
                if (n, rho) in results:
                    result = results[(n, rho)].get(solver_name, {})
                    matvecs = result.get('matvecs', None)
                    if matvecs is not None and matvecs > 0:
                        matvecs_list.append(matvecs)
                        valid_rhos.append(rho)

            if len(matvecs_list) > 0:
                ax.semilogy(
                    valid_rhos,
                    matvecs_list,
                    label=solver_name.upper(),
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel('Spectral Radius ρ(A)', fontsize=12)
        ax.set_ylabel('Matrix-Vector Products', fontsize=12)
        ax.set_title(f'n = {n}', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(min(rhos) - 0.05, max(rhos) + 0.05)

    fig.suptitle('Matrix-Vector Operations vs Spectral Radius', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save figure
    filepath = plot_dir / 'matvecs.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(filepath)


def plot_convergence_comparison(
    results: Dict[Tuple[int, float], Dict[str, Dict]],
    n: int,
    output_dir: Path = None,
    show: bool = False
) -> str:
    """Create multi-panel convergence plot for fixed sytem size."""
    plot_dir = _ensure_plot_dir(output_dir)

    # Get all rhos for this size
    rhos = sorted([k[1] for k in results.keys() if k[0] == n])

    if len(rhos) == 0:
        raise ValueError(f"No results found for n={n}")

    # Create subplots in a grid
    n_plots = len(rhos)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, rho in enumerate(rhos):
        ax = axes_flat[idx]
        result_dict = results.get((n, rho), {})
        
        # Track max iterations for inset decision
        max_iters = 0
        min_iters = float('inf')
        all_residuals = []

        for solver_name, result in result_dict.items():
            residuals = result.get('residuals', [])
            if len(residuals) == 0:
                continue
            
            iters = len(residuals)
            max_iters = max(max_iters, iters)
            min_iters = min(min_iters, iters)
            all_residuals.append((solver_name, residuals))

            style = _get_solver_style(solver_name)
            iterations = np.arange(len(residuals))

            ax.semilogy(
                iterations,
                residuals,
                label=solver_name.upper(),
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=1.5,
            )
        
        # Add inset if there's a huge disparity in iterations (e.g. > 50x difference)
        # and we have enough data points to make it worth showing
        if min_iters > 5 and max_iters > 50 * min_iters:
            # Create inset in top right (x=0.65, y=0.65, w=0.3, h=0.3)
            ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
            
            # Limit inset to just past the slow solver's convergence or a multiple of fast solver
            inset_limit = int(min_iters * 2.5) 
            
            for solver_name, residuals in all_residuals:
                style = _get_solver_style(solver_name)
                # Only plot first N iterations
                if len(residuals) > 0:
                    limit = min(len(residuals), inset_limit + 1)
                    ax_ins.semilogy(
                        np.arange(limit),
                        residuals[:limit],
                        color=style['color'],
                        linestyle=style['linestyle'],
                        linewidth=1.5
                    )
            
            ax_ins.set_xlim(0, inset_limit)
            ax_ins.set_title(f'First {inset_limit} iters', fontsize=8)
            ax_ins.grid(True, alpha=0.2)
            ax_ins.tick_params(labelsize=8)

        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(r'Relative Residual $\|r\|/\|b\|$', fontsize=11)
        ax.set_title(f'ρ = {rho}', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.axhline(y=1e-8, color='gray', linestyle=':', alpha=0.5)

    # Hide empty subplots
    for idx in range(n_plots, rows * cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f'Convergence Comparison (n = {n})', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    filepath = plot_dir / f'convergence_comparison_n{n}.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(filepath)


def create_summary_table(
    results: Dict[Tuple[int, float], Dict[str, Dict]]
) -> str:
    """Create formatted text summary table of all results."""
    lines = []
    lines.append("=" * 100)
    lines.append("SOLVER COMPARISON SUMMARY")
    lines.append("=" * 100)
    lines.append(f"{'n':>6} | {'rho':>6} | {'Solver':>10} | {'Conv':>5} | "
                 f"{'Iters':>8} | {'Matvecs':>8} | {'Time (s)':>10} | {'Residual':>12}")
    lines.append("-" * 100)

    for (n, rho), solver_results in sorted(results.items()):
        for solver_name, result in solver_results.items():
            conv = "Yes" if result.get('converged', False) else "No"
            iters = result.get('iterations', 0)
            matvecs = result.get('matvecs', 0)
            runtime = result.get('runtime', 0.0)
            residual = result.get('final_residual', float('inf'))

            lines.append(
                f"{n:>6} | {rho:>6.2f} | {solver_name:>10} | {conv:>5} | "
                f"{iters:>8} | {matvecs:>8} | {runtime:>10.4f} | {residual:>12.2e}"
            )
        lines.append("-" * 100)

    lines.append("=" * 100)
    return "\n".join(lines)


def plot_scaling_runtime(
    results: Dict[int, Dict[str, Dict]],
    rho: float,
    timeout: float,
    output_dir: Path = None,
    show: bool = False
) -> str:
    """Create log-log plot of runtime vs system size for scalling comparison."""
    plot_dir = _ensure_plot_dir(output_dir)
    
    # Extract sizes and sort
    sizes = sorted(results.keys())
    
    # Solver display configuration
    solver_config = {
        'direct': {
            'label': 'Direct (LU)',
            'color': '#d62728',  # Red
            'marker': 'D',
            'linestyle': '-',
        },
        'neumann': {
            'label': 'Neumann Series',
            'color': '#1f77b4',  # Blue
            'marker': 'o',
            'linestyle': '-',
        },
        'gmres': {
            'label': 'GMRES',
            'color': '#ff7f0e',  # Orange
            'marker': 's',
            'linestyle': '-',
        },
        'bicgstab': {
            'label': 'BiCGSTAB',
            'color': '#2ca02c',  # Green
            'marker': '^',
            'linestyle': '-',
        },
    }
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot each solver
    for solver_name, config in solver_config.items():
        n_vals = []
        runtimes = []
        
        for n in sizes:
            if n not in results:
                continue
            solver_result = results[n].get(solver_name, {})
            
            # Skip if timed out or no runtime
            if solver_result.get('timed_out', False):
                continue
            
            runtime = solver_result.get('runtime', None)
            if runtime is not None and runtime > 0:
                n_vals.append(n)
                runtimes.append(runtime)
        
        if len(n_vals) > 0:
            ax.loglog(
                n_vals,
                runtimes,
                label=config['label'],
                color=config['color'],
                marker=config['marker'],
                linestyle=config['linestyle'],
                linewidth=2,
                markersize=8,
            )
    
    # Add reference slopes for scaling
    # O(n) reference line
    n_ref = np.array([sizes[0], sizes[-1]])
    # Scale to pass through a reasonable point
    if len(sizes) >= 2:
        # Get a reference runtime from an iterative method at smallest n
        ref_runtime = None
        for solver in ['gmres', 'bicgstab', 'neumann']:
            if sizes[0] in results and solver in results[sizes[0]]:
                r = results[sizes[0]][solver].get('runtime', 0)
                if r > 0:
                    ref_runtime = r
                    break
        
        if ref_runtime is not None:
            # O(n) line
            y_linear = ref_runtime * (n_ref / n_ref[0])
            ax.loglog(n_ref, y_linear, '--', color='gray', alpha=0.5, 
                     linewidth=1.5, label='O(n) reference')
            
            # O(n²) line (scaled to start higher)
            y_quad = ref_runtime * 10 * (n_ref / n_ref[0])**2
            ax.loglog(n_ref, y_quad, ':', color='gray', alpha=0.5,
                     linewidth=1.5, label='O(n²) reference')
    
    # Formatting
    ax.set_xlabel('System Size n', fontsize=13)
    ax.set_ylabel('Runtime (seconds)', fontsize=13)
    ax.set_title(f'Runtime Scaling: Direct vs Iterative Methods (ρ = {rho})', fontsize=14)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, which='both')
    
    # Set axis limits with some padding
    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.2)
    
    # Add annotation about timeout if direct didn't complete for all sizes
    direct_completed = sum(
        1 for n in sizes 
        if n in results and 'direct' in results[n] 
        and not results[n]['direct'].get('timed_out', False)
    )
    
    if direct_completed < len(sizes):
        # Find the last n where direct completed
        last_direct_n = 0
        for n in sorted(sizes):
            if n in results and 'direct' in results[n]:
                if not results[n]['direct'].get('timed_out', False):
                    last_direct_n = n
        
        if last_direct_n > 0:
            ax.annotate(
                f'Direct: timeout at n > {last_direct_n:,}',
                xy=(0.98, 0.02),
                xycoords='axes fraction',
                ha='right',
                va='bottom',
                fontsize=10,
                style='italic',
                color='#d62728',
            )
    
    plt.tight_layout()
    
    # Save figure
    filepath = plot_dir / 'scaling_runtime.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return str(filepath)


if __name__ == "__main__":
    dummy_results = {
        (1000, 0.5): {
            'gmres': {
                'residuals': [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001],
                'runtime': 0.1,
                'matvecs': 6,
                'converged': True,
                'iterations': 6,
                'final_residual': 1e-4,
            },
            'bicgstab': {
                'residuals': [1.0, 0.3, 0.05, 0.005, 0.0005],
                'runtime': 0.08,
                'matvecs': 10,
                'converged': True,
                'iterations': 5,
                'final_residual': 5e-4,
            },
        },
        (1000, 0.9): {
            'gmres': {
                'residuals': [1.0] + [0.9**i for i in range(1, 50)],
                'runtime': 0.5,
                'matvecs': 50,
                'converged': True,
                'iterations': 50,
                'final_residual': 1e-8,
            },
            'bicgstab': {
                'residuals': [1.0] + [0.85**i for i in range(1, 40)],
                'runtime': 0.4,
                'matvecs': 80,
                'converged': True,
                'iterations': 40,
                'final_residual': 1e-8,
            },
        },
    }

    path = plot_convergence(dummy_results[(1000, 0.5)], n=1000, rho=0.5)
    print(f"Convergence plot: {path}")
    path = plot_runtime(dummy_results)
    print(f"Runtime plot: {path}")
    path = plot_matvecs(dummy_results)
    print(f"Matvecs plot: {path}")
    print(create_summary_table(dummy_results))

