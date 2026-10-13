"""Optimization solvers for knockoff construction."""

from typing import List, Tuple, Optional
import logging
import warnings
import numpy as np
from scipy import linalg
from scipy.cluster.hierarchy import linkage, fcluster

from .utils import is_posdef, cov2cor

logger = logging.getLogger(__name__)

SDP_SOLVERS = ("dsdp", "cvxpy")

_DSDP_MISSING_MESSAGE = (
    "The DSDP C extension (loveslide.dsdp_solver.pydsdp5) is not available, "
    "and knockoff SDPs require it by default to match R's Rdsdp solver.\n"
    "Build it from the repository root with a C compiler and BLAS/LAPACK:\n"
    "  pip install -e .            (or: python setup.py build_ext --inplace)\n"
    "  HPC: module load gcc openblas | Debian/Ubuntu: apt install libopenblas-dev | "
    "macOS: brew install openblas | conda: conda install openblas\n"
    "To use the slower, non-R-matching cvxpy solver instead, pass "
    "sdp_solver='cvxpy' (SLIDE input_params key 'sdp_solver')."
)


def _get_sdp_solver(solver: str = "dsdp") -> str:
    """Return the requested SDP solver name after checking that it can be imported.

    DSDP is the default and is required unless the caller explicitly selects
    'cvxpy'; there is no silent fallback between the two.
    """
    if solver not in SDP_SOLVERS:
        raise ValueError(f"sdp_solver must be one of {SDP_SOLVERS}, got {solver!r}")

    if solver == "cvxpy":
        import cvxpy  # noqa: F401
        return solver

    try:
        from .pydsdp import dsdp  # noqa: F401
    except ImportError as e:
        raise ImportError(_DSDP_MISSING_MESSAGE) from e
    return solver


def _solve_sdp_cvxpy(
    G: np.ndarray,
    gaptol: float = 1e-6,
    maxit: int = 10000,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve knockoff SDP using cvxpy with SCS solver.

    This is a fallback when DSDP is not available.

    Parameters
    ----------
    G : np.ndarray of shape (p, p)
        Correlation matrix.
    gaptol : float
        Tolerance for solver.
    maxit : int
        Maximum iterations.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    np.ndarray of shape (p,)
        Solution s (in correlation scale, values in [0, 1]).
    """
    import cvxpy as cp

    p = G.shape[0]

    # Define optimization variable
    s = cp.Variable(p)

    # Constraints: 0 <= s <= 1, 2*G - diag(s) >= 0 (PSD)
    constraints = [
        s >= 0,
        s <= 1,
        2 * G - cp.diag(s) >> 0  # PSD constraint
    ]

    # Objective: maximize sum(s)
    objective = cp.Maximize(cp.sum(s))

    # Solve
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, eps=gaptol, max_iters=maxit, verbose=verbose)
    except Exception as e:
        warnings.warn(f"cvxpy SCS solver failed: {e}. Trying CVXOPT...")
        try:
            prob.solve(solver=cp.CVXOPT, verbose=verbose)
        except Exception as e2:
            warnings.warn(f"cvxpy CVXOPT solver also failed: {e2}")
            return np.zeros(p)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        warnings.warn(f"cvxpy solver returned status: {prob.status}")
        if s.value is None:
            return np.zeros(p)

    s_val = np.clip(s.value, 0, 1)
    return s_val


def create_solve_equi(Sigma: np.ndarray, **kwargs) -> np.ndarray:
    """
    Optimization for equi-correlated knockoffs.

    Computes the closed-form solution to the semidefinite programming problem:
        maximize    s
        subject to  0 <= s <= 1
                    2*Sigma - s*I >= 0

    The closed-form solution is s = min(2 * lambda_min(Sigma), 1).

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Positive-definite covariance matrix.

    Returns
    -------
    np.ndarray of shape (p,)
        The solution s to the optimization problem.
    """
    Sigma = np.asarray(Sigma)

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Covariance matrix must be symmetric")

    p = Sigma.shape[0]
    tol = 1e-10

    # Convert to correlation matrix
    G = cov2cor(Sigma)

    # Check positive-definiteness
    if not is_posdef(G):
        raise ValueError("The covariance matrix is not positive-definite: cannot solve SDP")

    # Compute smallest eigenvalue
    if p > 2:
        try:
            from scipy.sparse.linalg import eigsh
            # Use sparse eigenvalue solver for efficiency
            maxiter = 100000
            lambda_min = eigsh(
                G, k=1, which='SA',
                return_eigenvectors=False,
                maxiter=maxiter,
                tol=1e-8
            )[0]
        except Exception:
            # Fallback to dense solver
            warnings.warn(
                "Sparse eigenvalue solver did not converge. "
                "Using dense computation instead."
            )
            eigenvalues = linalg.eigvalsh(G)
            lambda_min = eigenvalues[0]  # Already sorted ascending
    else:
        eigenvalues = linalg.eigvalsh(G)
        lambda_min = eigenvalues[0]

    if lambda_min < 0:
        raise ValueError(
            "The covariance matrix is not positive-definite. "
            "Cannot create equi-correlated knockoffs."
        )

    # Closed-form solution
    s = np.ones(p) * min(2 * lambda_min, 1)

    # Compensate for numerical errors (feasibility)
    # Match R's feasibility shrinkage: allow up to 10% (s_eps <= 0.1)
    s_eps = 1e-8
    psd = False
    while s_eps <= 0.1:
        if is_posdef(2 * G - np.diag(s * (1 - s_eps))):
            psd = True
            break
        s_eps *= 10

    s = s * (1 - s_eps)

    # Scale back for original covariance matrix
    return s * np.diag(Sigma)


def create_solve_sdp(
    Sigma: np.ndarray,
    gaptol: float = 1e-6,
    maxit: int = 1000,
    verbose: bool = False,
    solver: str = "dsdp",
    **kwargs
) -> np.ndarray:
    """
    Full SDP optimization for knockoffs.

    Solves the semidefinite programming problem:
        maximize    sum(s)
        subject to  0 <= s <= 1
                    2*Sigma - diag(s) >= 0 (positive semidefinite)

    DSDP (exact R compatibility) is used unless ``solver='cvxpy'`` is passed.

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Positive-definite covariance matrix.
    gaptol : float, default=1e-6
        Tolerance for duality gap. Matches R knockoff package default.
    maxit : int, default=1000
        Maximum number of iterations for the solver.
    verbose : bool, default=False
        Whether to display progress.
    solver : {'dsdp', 'cvxpy'}, default='dsdp'
        SDP solver. 'dsdp' requires the bundled C extension and raises
        ImportError when it is missing; 'cvxpy' must be selected explicitly.

    Returns
    -------
    np.ndarray of shape (p,)
        The solution s to the optimization problem.

    Raises
    ------
    ImportError
        If the selected solver is not installed.
    RuntimeError
        If DSDP raises while solving.
    """
    from scipy import sparse

    solver = _get_sdp_solver(solver)

    Sigma = np.asarray(Sigma)

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Covariance matrix must be symmetric")

    # Convert to correlation matrix
    G = cov2cor(Sigma)
    p = G.shape[0]

    # Check positive-definiteness
    if not is_posdef(G):
        warnings.warn(
            "The covariance matrix is not positive-definite: "
            "knockoffs may not have power."
        )

    if verbose:
        logger.info(f"Solving SDP using {solver}")

    if solver == 'cvxpy':
        s_val = _solve_sdp_cvxpy(G, gaptol=gaptol, maxit=maxit, verbose=verbose)
    else:
        # Use DSDP (exact R compatibility)
        from .pydsdp import dsdp

        # Build SDP problem in SeDuMi format (matching R's Rdsdp interface)
        # Problem: maximize sum(s) s.t. 0 <= s <= 1, 2*G - diag(s) >= 0

        # Linear constraints for 0 <= s <= 1:
        # -s <= 0  (i.e., s >= 0)
        # s <= 1
        Cl1 = np.zeros(p)  # RHS for s >= 0
        Al1 = -sparse.eye(p)  # -I for s >= 0
        Cl2 = np.ones(p)  # RHS for s <= 1
        Al2 = sparse.eye(p)  # I for s <= 1

        # SDP constraint: 2*G - diag(s) >= 0
        # The constraint matrix As maps s to the diagonal of the SDP cone
        # Each column j of As puts s_j on the (j,j) diagonal entry
        # In vectorized form (column-major), diagonal entry (j,j) is at index j*p + j
        d_As = np.diag(np.eye(p)).flatten()  # Indicator for diagonal entries
        As_rows = []
        As_cols = []
        As_data = []
        for j in range(p):
            # Entry (j,j) in column-major vectorization is at index j*p + j
            As_rows.append(j)
            As_cols.append(j * p + j)
            As_data.append(1.0)
        As = sparse.csr_matrix((As_data, (As_rows, As_cols)), shape=(p, p * p))

        # Cone constraint data: 2*G vectorized (column-major)
        Cs = (2 * G).flatten(order='F')

        # Assemble A matrix: [Al1 | Al2 | As]
        # A has shape (p, 2*p + p*p)
        A = sparse.hstack([Al1.T, Al2.T, As])

        # Assemble c vector: [Cl1; Cl2; Cs]
        c = np.concatenate([Cl1, Cl2, Cs])

        # Cone specification
        K = {'l': 2 * p, 's': [p]}

        # Objective: maximize sum(s), i.e., b = ones(p)
        b = np.ones(p)

        # Solver options (matching R's Rdsdp defaults)
        OPTIONS = {
            'gaptol': gaptol,
            'maxit': maxit,
            'logsummary': 0,
            'outputstats': 0,
            'print': 1 if verbose else 0,
        }

        # Convert to numpy matrix format expected by pydsdp
        from numpy import matrix as np_matrix
        A_mat = np_matrix(A.toarray())
        b_mat = np_matrix(b).T
        c_mat = np_matrix(c).T

        # Solve SDP with DSDP
        try:
            sol = dsdp(A_mat, b_mat, c_mat, K, OPTIONS)
        except Exception as e:
            raise RuntimeError(f"DSDP solver failed for a {p}x{p} SDP: {e}") from e

        # Check solution status
        if sol['STATS']['stype'] != 'PDFeasible':
            warnings.warn(
                f"The SDP solver returned a non-feasible solution "
                f"(status: {sol['STATS']['stype']}). Knockoffs may lose power."
            )

        # Extract and clip solution
        s_val = np.asarray(sol['y']).flatten()
        s_val = np.clip(s_val, 0, 1)

    # Compensate for numerical errors (feasibility)
    if verbose:
        logger.info("Verifying that the solution is correct")

    # Match R's feasibility shrinkage: allow up to 10% (s_eps <= 0.1)
    s_eps = 1e-8
    psd = False
    while (not psd) and (s_eps <= 0.1):
        if is_posdef(2 * G - np.diag(s_val * (1 - s_eps)), tol=1e-9):
            psd = True
        else:
            s_eps *= 10

    s_val = s_val * (1 - s_eps)
    s_val = np.clip(s_val, 0, 1)

    # Verify solution
    if np.all(s_val == 0):
        warnings.warn(
            "In creation of SDP knockoffs, procedure failed. "
            "Knockoffs will have no power."
        )

    # Scale back for original covariance matrix
    diag_s = s_val * np.diag(Sigma)

    # Ensure feasibility on original Sigma (important when G was regularized)
    # The constraint is: 2*Sigma - diag(diag_s) >= 0
    # Use eigenvalue-based scaling to ensure feasibility
    test_sigma = 2 * Sigma - np.diag(diag_s)
    min_eig = np.linalg.eigvalsh(test_sigma)[0]

    if min_eig < 1e-9:
        # Scale diag_s down to ensure positive definiteness
        # Need: 2*Sigma - alpha*diag(diag_s) >= 0 for some alpha < 1
        # Binary search for the largest feasible alpha
        alpha_low, alpha_high = 0.0, 1.0
        for _ in range(50):  # Binary search iterations
            alpha_mid = (alpha_low + alpha_high) / 2
            test_mat = 2 * Sigma - np.diag(diag_s * alpha_mid)
            if is_posdef(test_mat, tol=1e-9):
                alpha_low = alpha_mid
            else:
                alpha_high = alpha_mid
            if alpha_high - alpha_low < 1e-6:
                break

        # Use a slightly conservative alpha
        alpha = alpha_low * 0.99
        diag_s = diag_s * alpha
        if verbose:
            logger.info(f"Sigma feasibility: scaled by {alpha:.3f}")

    return diag_s


def _merge_clusters(clusters: np.ndarray, max_size: int) -> np.ndarray:
    """
    Merge consecutive clusters while ensuring no cluster exceeds max_size.

    Parameters
    ----------
    clusters : array-like of shape (p,)
        Cluster assignments (1-indexed).
    max_size : int
        Maximum cluster size.

    Returns
    -------
    np.ndarray of shape (p,)
        New cluster assignments (1-indexed).
    """
    unique_clusters = np.unique(clusters)
    cluster_sizes = {k: np.sum(clusters == k) for k in unique_clusters}

    clusters_new = np.zeros_like(clusters)
    g = 1
    g_size = 0

    for k in sorted(unique_clusters):
        if g_size + cluster_sizes[k] > max_size:
            g += 1
            g_size = 0
        clusters_new[clusters == k] = g
        g_size += cluster_sizes[k]

    return clusters_new


def _divide_sdp(
    Sigma: np.ndarray,
    max_size: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Approximate a covariance matrix by a block diagonal matrix.

    Uses hierarchical clustering to divide variables into groups.

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Covariance matrix.
    max_size : int
        Maximum block size.

    Returns
    -------
    clusters : np.ndarray of shape (p,)
        Cluster assignments (1-indexed).
    sub_sigmas : list of np.ndarray
        List of covariance sub-matrices for each cluster.
    """
    p = Sigma.shape[0]

    # Convert to dissimilarity matrix with small perturbation
    eps = np.random.randn(p, p) * 1e-6
    G = cov2cor(Sigma)
    dissimilarity = 1 - np.abs(G + eps)

    # Make symmetric and extract condensed form for linkage
    dissimilarity = (dissimilarity + dissimilarity.T) / 2
    np.fill_diagonal(dissimilarity, 0)

    # Convert to condensed distance matrix
    from scipy.spatial.distance import squareform
    distance = squareform(dissimilarity)

    # Hierarchical clustering with single linkage
    Z = linkage(distance, method='single')

    # Binary search for optimal number of clusters
    n_blocks_min = 1
    n_blocks_max = p

    for _ in range(100):
        n_blocks = (n_blocks_min + n_blocks_max + 1) // 2
        clusters = fcluster(Z, n_blocks, criterion='maxclust')
        max_cluster_size = max(np.bincount(clusters)[1:]) if np.max(clusters) > 0 else p

        if max_cluster_size <= max_size:
            n_blocks_max = n_blocks
        if max_cluster_size >= max_size:
            n_blocks_min = n_blocks
        if n_blocks_min == n_blocks_max:
            break

    # Merge small clusters
    clusters_new = _merge_clusters(clusters, max_size)
    while not np.array_equal(clusters_new, clusters):
        clusters = clusters_new
        clusters_new = _merge_clusters(clusters, max_size)
    clusters = clusters_new

    # Create covariance sub-matrices
    n_clusters = int(np.max(clusters))
    sub_sigmas = []
    for k in range(1, n_clusters + 1):
        indices = np.where(clusters == k)[0]
        sub_sigmas.append(Sigma[np.ix_(indices, indices)])

    return clusters, sub_sigmas


def create_solve_asdp(
    Sigma: np.ndarray,
    max_size: int = 500,
    gaptol: float = 1e-6,
    maxit: int = 1000,
    verbose: bool = False,
    solver: str = "dsdp",
    **kwargs
) -> np.ndarray:
    """
    Approximate SDP optimization for knockoffs.

    For high-dimensional problems (p > max_size), this function approximates
    the covariance matrix as block-diagonal and solves smaller SDPs.

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Positive-definite covariance matrix.
    max_size : int, default=500
        Maximum size of each block.
    gaptol : float, default=1e-6
        Tolerance for duality gap. Matches R knockoff package default.
    maxit : int, default=1000
        Maximum number of iterations.
    verbose : bool, default=False
        Whether to display progress.
    solver : {'dsdp', 'cvxpy'}, default='dsdp'
        SDP solver passed to :func:`create_solve_sdp`.

    Returns
    -------
    np.ndarray of shape (p,)
        The solution s to the optimization problem.
    """
    Sigma = np.asarray(Sigma)

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Covariance matrix must be symmetric")

    p = Sigma.shape[0]

    # If small enough, use full SDP
    if p <= max_size:
        return create_solve_sdp(
            Sigma, gaptol=gaptol, maxit=maxit, verbose=verbose, solver=solver
        )

    if verbose:
        logger.info(f"Dividing the problem into subproblems of size <= {max_size}")

    # Divide into blocks
    clusters, sub_sigmas = _divide_sdp(Sigma, max_size)
    n_blocks = int(np.max(clusters))

    if verbose:
        logger.info(f"Solving {n_blocks} smaller SDPs")

    # Solve SDPs for each block
    s_list = []
    for k in range(n_blocks):
        s_k = create_solve_sdp(
            sub_sigmas[k], gaptol=gaptol, maxit=maxit, verbose=False, solver=solver
        )
        s_list.append(s_k)
        if verbose:
            logger.info(f"Block {k + 1}/{n_blocks} done")

    # Assemble solutions
    idx_count = np.zeros(n_blocks, dtype=int)
    s_asdp = np.zeros(p)

    for j in range(p):
        cluster_j = clusters[j] - 1  # Convert to 0-indexed
        s_asdp[j] = s_list[cluster_j][idx_count[cluster_j]]
        idx_count[cluster_j] += 1

    # Maximize shrinkage factor via binary search
    if verbose:
        logger.info(f"Combining the solutions of the {n_blocks} smaller SDPs")

    gamma_range = np.linspace(0, 1, 1000)

    def check_psd(gamma_idx: int) -> float:
        """Check if 2*Sigma - gamma*diag(s) is positive definite."""
        gamma = gamma_range[gamma_idx]
        G = 2 * Sigma - gamma * np.diag(s_asdp)
        try:
            from scipy.sparse.linalg import eigsh
            lambda_min = eigsh(
                G, k=1, which='SA',
                return_eigenvectors=False,
                maxiter=100000,
                tol=1e-12
            )[0]
        except Exception:
            lambda_min = np.min(linalg.eigvalsh(G))
        return lambda_min

    # Binary search for optimal gamma
    low, high = 0, len(gamma_range) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if check_psd(mid) > 0:
            low = mid
        else:
            high = mid - 1

    gamma_opt = gamma_range[low]
    s_asdp_scaled = gamma_opt * s_asdp

    if verbose:
        logger.info("Verifying that the solution is correct")

    # Verify solution
    if not is_posdef(2 * Sigma - np.diag(s_asdp_scaled)):
        warnings.warn(
            "In creation of approximate SDP knockoffs, procedure failed. "
            "Knockoffs will have no power."
        )
        s_asdp_scaled = np.zeros(p)

    return s_asdp_scaled
