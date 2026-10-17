"""Parallel execution utilities for knockoff voting.

This module provides optimized parallel execution for knockoff_filter_voting
with the key optimization of precomputing diag_s (SDP solution) once before
the parallel loop.

Example
-------
>>> from slide_py.knockoff import knockoff_filter_voting
>>> result = knockoff_filter_voting(X, y, niter=500, n_jobs=-1)  # Auto-parallel
"""

from typing import Optional, Callable, Tuple, List
import warnings
import numpy as np
import multiprocessing

from .create import create_gaussian, create_second_order
from .solve import create_solve_sdp, create_solve_asdp, create_solve_equi
from .filter import (
    _estimate_covariance_corpcor,
    knockoff_threshold,
    VotingResult,
)


def _precompute_knockoff_params(
    X: np.ndarray,
    method: str = 'auto',
    shrink: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute knockoff parameters that are constant across iterations.

    This is the key optimization: compute mu, Sigma, and diag_s ONCE
    instead of recomputing the expensive SDP optimization 500 times.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Data matrix.
    method : str, default='auto'
        Method for diag_s optimization: 'sdp', 'asdp', 'equi', or 'auto'.
        'auto' uses ASDP for p > 500, else SDP.
    shrink : bool, default=False
        Whether to force R-compatible Schäfer-Strimmer covariance shrinkage.

    Returns
    -------
    mu : np.ndarray of shape (p,)
        Mean vector.
    Sigma : np.ndarray of shape (p, p)
        Covariance matrix (possibly shrunk).
    diag_s : np.ndarray of shape (p,)
        SDP solution for knockoff construction.
    """
    n, p = X.shape

    # Estimate mean
    mu = np.mean(X, axis=0)

    Sigma = _estimate_covariance_corpcor(X, shrink=shrink)

    # Determine method
    if method == 'auto':
        method = 'asdp' if p > 500 else 'sdp'

    # Compute diag_s (the expensive SDP optimization)
    if method == 'equi':
        diag_s = create_solve_equi(Sigma)
    elif method == 'sdp':
        diag_s = create_solve_sdp(Sigma)
    else:  # asdp
        diag_s = create_solve_asdp(Sigma)

    # Match R: if SDP returns all zeros, warn but do not fall back.
    # R returns X as knockoffs, giving zero power (no variables selected).
    if np.all(diag_s == 0):
        warnings.warn(
            "SDP solver returned all-zero solution. "
            "Knockoffs will have no power."
        )

    return mu, Sigma, diag_s


def _single_knockoff_iteration(
    X: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    diag_s: np.ndarray,
    fdr: float,
    offset: int,
    seed: int,
    statistic_name: str = 'coefdiff',
) -> List[int]:
    """
    Run a single knockoff iteration with precomputed parameters.

    This is the worker function for parallel execution.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    y : np.ndarray
        Response vector.
    mu : np.ndarray
        Precomputed mean vector.
    Sigma : np.ndarray
        Precomputed covariance matrix.
    diag_s : np.ndarray
        Precomputed SDP solution.
    fdr : float
        Target FDR.
    offset : int
        Knockoff offset (0 or 1).
    seed : int
        Random seed for this iteration.
    statistic_name : str
        Name of statistic to use ('coefdiff', 'lambdasmax', 'lambdadiff').

    Returns
    -------
    List[int]
        Indices of selected variables.
    """
    np.random.seed(seed)

    # Create knockoffs using precomputed diag_s (skips SDP!)
    Xk = create_gaussian(X, mu, Sigma, diag_s=diag_s)

    # Compute statistics
    if statistic_name == 'coefdiff':
        from .stats import stat_glmnet_coefdiff
        W = stat_glmnet_coefdiff(X, Xk, y)
    elif statistic_name == 'lambdasmax':
        from .stats import stat_glmnet_lambdasmax
        W = stat_glmnet_lambdasmax(X, Xk, y)
    elif statistic_name == 'lambdadiff':
        from .stats import stat_glmnet_lambdadiff
        W = stat_glmnet_lambdadiff(X, Xk, y)
    else:
        raise ValueError(f"Unknown statistic: {statistic_name}")

    # Apply threshold
    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    return np.where(W >= t)[0].tolist()


def _worker_wrapper(args):
    """Wrapper for multiprocessing (must be at module level for pickling)."""
    return _single_knockoff_iteration(*args)


def knockoff_voting_parallel_joblib(
    X: np.ndarray,
    y: np.ndarray,
    fdr: float = 0.10,
    offset: int = 0,
    niter: int = 1000,
    spec: float = 0.3,
    n_jobs: int = -1,
    base_seed: int = 42,
    verbose: bool = False,
    batch_size: int = 100,
    statistic_name: str = 'coefdiff',
    **kwargs
) -> VotingResult:
    """
    Parallel knockoff voting using joblib with precomputed diag_s.

    This is the optimized implementation that:
    1. Precomputes mu, Sigma, diag_s ONCE before the parallel loop
    2. Uses joblib for efficient parallel execution
    3. Processes iterations in batches to control memory

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Matrix of predictors.
    y : np.ndarray of shape (n,)
        Response vector.
    fdr : float, default=0.10
        Target false discovery rate.
    offset : int, default=0
        Knockoff offset (0 = standard, 1 = knockoffs+).
    niter : int, default=1000
        Number of knockoff iterations. Matches R SLIDE default.
    spec : float, default=0.3
        Specificity threshold for voting. Matches R SLIDE default.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores).
    base_seed : int, default=42
        Base random seed for reproducibility.
    verbose : bool, default=False
        Print progress information.
    batch_size : int, default=100
        Number of iterations per batch.
    statistic_name : str, default='coefdiff'
        Statistic to use ('coefdiff', 'lambdasmax', 'lambdadiff').

    Returns
    -------
    VotingResult
        Voting results with selection counts and frequencies.

    Notes
    -----
    Expected speedup: 4-8x on multi-core systems due to:
    - Single SDP computation instead of niter
    - Parallel LASSO path computation
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "joblib is required for parallel execution. "
            "Install with: pip install joblib"
        )

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n, p = X.shape

    # Determine n_jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, niter))

    # ========== PRECOMPUTE SHARED STATE ==========
    if verbose:
        print("Precomputing knockoff parameters (SDP optimization)...")

    mu, Sigma, diag_s = _precompute_knockoff_params(X)

    if verbose:
        print(f"  diag_s max={np.max(diag_s):.4f}, min={np.min(diag_s):.4f}")
        print(f"Running {niter} iterations with {n_jobs} workers...")

    # ========== PARALLEL EXECUTION ==========
    selection_counts = np.zeros(p, dtype=np.int32)

    # Progress tracking
    try:
        from tqdm import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False

    # Process in batches
    batches = list(range(0, niter, batch_size))
    if use_tqdm:
        batches = tqdm(batches, desc="Knockoff voting", unit="batch")

    for batch_start in batches:
        batch_end = min(batch_start + batch_size, niter)

        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_single_knockoff_iteration)(
                X, y, mu, Sigma, diag_s, fdr, offset, base_seed + i, statistic_name
            )
            for i in range(batch_start, batch_end)
        )

        # Aggregate results
        for selected in results:
            for idx in selected:
                selection_counts[idx] += 1

        if verbose and not use_tqdm and batch_end % 100 == 0:
            print(f"  Completed {batch_end}/{niter} iterations")

    # ========== COMPUTE RESULTS ==========
    selection_frequency = selection_counts / niter
    min_selections = int(np.ceil(niter * spec))
    selected = np.sort(np.where(selection_counts >= min_selections)[0])

    if verbose:
        print(f"Selected {len(selected)} variables (>= {min_selections} selections)")

    return VotingResult(
        selection_counts=selection_counts,
        selection_frequency=selection_frequency,
        selected=selected,
        threshold=spec,
        niter=niter,
        spec=spec,
        min_selections=min_selections
    )


def knockoff_voting_parallel_futures(
    X: np.ndarray,
    y: np.ndarray,
    fdr: float = 0.10,
    offset: int = 0,
    niter: int = 1000,
    spec: float = 0.3,
    n_jobs: int = -1,
    base_seed: int = 42,
    verbose: bool = False,
    statistic_name: str = 'coefdiff',
    **kwargs
) -> VotingResult:
    """
    Parallel knockoff voting using concurrent.futures (standard library).

    This is an alternative to the joblib version that uses only standard
    library components. May be slightly slower due to less optimization.

    Parameters
    ----------
    (Same as knockoff_voting_parallel_joblib)

    Returns
    -------
    VotingResult
        Voting results with selection counts and frequencies.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n, p = X.shape

    # Determine n_jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, niter))

    # Precompute shared state
    if verbose:
        print("Precomputing knockoff parameters...")

    mu, Sigma, diag_s = _precompute_knockoff_params(X)

    if verbose:
        print(f"Running {niter} iterations with {n_jobs} workers...")

    # Prepare arguments for each iteration
    args_list = [
        (X, y, mu, Sigma, diag_s, fdr, offset, base_seed + i, statistic_name)
        for i in range(niter)
    ]

    # Parallel execution
    selection_counts = np.zeros(p, dtype=np.int32)
    completed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_worker_wrapper, args) for args in args_list]

        for future in as_completed(futures):
            try:
                selected = future.result()
                for idx in selected:
                    selection_counts[idx] += 1
            except Exception as e:
                warnings.warn(f"Knockoff iteration failed: {e}")

            completed += 1
            if verbose and completed % 100 == 0:
                print(f"  Completed {completed}/{niter} iterations")

    # Compute results
    selection_frequency = selection_counts / niter
    min_selections = int(np.ceil(niter * spec))
    selected = np.sort(np.where(selection_counts >= min_selections)[0])

    return VotingResult(
        selection_counts=selection_counts,
        selection_frequency=selection_frequency,
        selected=selected,
        threshold=spec,
        niter=niter,
        spec=spec,
        min_selections=min_selections
    )


# Convenience function to select best parallel backend
def knockoff_voting_parallel(
    X: np.ndarray,
    y: np.ndarray,
    backend: str = 'auto',
    **kwargs
) -> VotingResult:
    """
    Parallel knockoff voting with automatic backend selection.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    y : np.ndarray
        Response vector.
    backend : str, default='auto'
        Parallel backend: 'joblib', 'futures', or 'auto'.
        'auto' uses joblib if available, else futures.
    **kwargs
        Additional arguments passed to the parallel function.

    Returns
    -------
    VotingResult
        Voting results.
    """
    if backend == 'auto':
        try:
            import joblib
            backend = 'joblib'
        except ImportError:
            backend = 'futures'

    if backend == 'joblib':
        return knockoff_voting_parallel_joblib(X, y, **kwargs)
    elif backend == 'futures':
        return knockoff_voting_parallel_futures(X, y, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'joblib' or 'futures'.")
