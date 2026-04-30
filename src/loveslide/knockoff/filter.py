"""Main knockoff filter pipeline."""

from dataclasses import dataclass
from typing import Optional, Callable, Union, List, Dict, Any
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scipy import linalg

from .create import create_second_order, create_fixed, KnockoffVariables
from .solve import create_solve_equi, create_solve_sdp, create_solve_asdp
from .utils import is_posdef


@dataclass
class KnockoffResult:
    """
    Result of the knockoff filter procedure.

    Attributes
    ----------
    X : np.ndarray
        Matrix of original variables.
    Xk : np.ndarray
        Matrix of knockoff variables.
    y : np.ndarray
        Response vector.
    statistic : np.ndarray
        Computed test statistics W.
    threshold : float
        Selection threshold.
    selected : np.ndarray
        Indices of selected variables.
    feature_names : list, optional
        Names of selected features (if provided).
    """
    X: np.ndarray
    Xk: np.ndarray
    y: np.ndarray
    statistic: np.ndarray
    threshold: float
    selected: np.ndarray
    feature_names: Optional[List[str]] = None

    def __repr__(self) -> str:
        n_selected = len(self.selected)
        p = self.X.shape[1]
        return (
            f"KnockoffResult(\n"
            f"  n_features={p},\n"
            f"  n_selected={n_selected},\n"
            f"  selected={self.selected.tolist()},\n"
            f"  threshold={self.threshold:.4f}\n"
            f")"
        )


def knockoff_threshold(
    W: np.ndarray,
    fdr: float = 0.10,
    offset: int = 1,
    **kwargs
) -> float:
    """
    Compute the threshold for the knockoff filter.

    Parameters
    ----------
    W : array-like of shape (p,)
        Test statistics.
    fdr : float, default=0.10
        Target false discovery rate.
    offset : {0, 1}, default=1
        The value 1 yields a slightly more conservative procedure ("knockoffs+")
        that controls the FDR according to the usual definition, while an
        offset of 0 controls a modified FDR.

    Returns
    -------
    float
        The threshold for variable selection.
    """
    W = np.asarray(W)

    if offset not in [0, 1]:
        raise ValueError("offset must be either 0 or 1")

    # Candidate thresholds: 0 and absolute values of W
    ts = np.sort(np.concatenate([[0], np.abs(W)]))

    # For each threshold, compute FDP estimate
    for t in ts:
        numerator = offset + np.sum(W <= -t)
        denominator = max(1, np.sum(W >= t))
        ratio = numerator / denominator
        if ratio <= fdr:
            return t

    return np.inf


def knockoff_filter(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 1,
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> KnockoffResult:
    """
    Run the Knockoff Filter for controlled variable selection.

    This function creates knockoffs, computes importance statistics,
    and selects variables while controlling the false discovery rate.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix of predictors.
    y : array-like of shape (n,)
        Response vector.
    knockoffs : callable, optional
        Function to construct knockoffs. Takes a (n, p) matrix as input
        and returns knockoff variables. Default: create_second_order.
    statistic : callable, optional
        Function to compute importance statistics. Takes (X, Xk, y) as input
        and returns a (p,) array of statistics W.
        Default: stat_glmnet_coefdiff.
    fdr : float, default=0.10
        Target false discovery rate.
    offset : {0, 1}, default=1
        Offset for computing the rejection threshold.
        1 = more conservative ("knockoffs+"), 0 = modified FDR.
    feature_names : list of str, optional
        Names of features for labeling selected variables.

    Returns
    -------
    KnockoffResult
        Object containing X, Xk, y, statistic, threshold, and selected.

    References
    ----------
    Candes et al., Panning for Gold: Model-free Knockoffs for
    High-dimensional Controlled Variable Selection, arXiv:1610.02351 (2016).

    Barber and Candes, Controlling the false discovery rate via knockoffs.
    Ann. Statist. 43 (2015), no. 5, 2055--2085.

    Examples
    --------
    >>> import numpy as np
    >>> from knockoff import knockoff_filter
    >>> n, p = 100, 50
    >>> X = np.random.randn(n, p)
    >>> beta = np.zeros(p)
    >>> beta[:5] = 3.0
    >>> y = X @ beta + np.random.randn(n)
    >>> result = knockoff_filter(X, y, fdr=0.1)
    >>> print(result.selected)
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # Validate input types
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must be numeric")

    if not (np.issubdtype(y.dtype, np.number) or np.issubdtype(y.dtype, np.object_)):
        raise ValueError("y must be numeric or categorical")

    if y.ndim > 1:
        y = y.ravel()

    if offset not in [0, 1]:
        raise ValueError("offset must be either 0 or 1")

    # Validate dimensions
    n, p = X.shape
    if len(y) != n:
        raise ValueError(f"Length of y ({len(y)}) must match number of rows in X ({n})")

    # Extract feature names if not provided
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = list(X.columns)

    # Set default knockoff constructor
    if knockoffs is None:
        knockoffs = create_second_order

    # Set default statistic
    if statistic is None:
        from .stats import stat_glmnet_coefdiff
        statistic = stat_glmnet_coefdiff

    # Handle create_fixed specially (needs y for augmentation)
    if knockoffs is create_fixed:
        knockoffs = lambda x: create_fixed(x, y=y)

    # Create knockoff variables
    knock_variables = knockoffs(X)

    # Handle different return types from knockoff constructors
    if isinstance(knock_variables, KnockoffVariables):
        X = knock_variables.X
        Xk = knock_variables.Xk
        if knock_variables.y is not None:
            y = knock_variables.y
    elif isinstance(knock_variables, np.ndarray):
        Xk = knock_variables
    else:
        raise ValueError(
            f"Knockoff constructor returned unexpected type: {type(knock_variables)}"
        )

    # Compute statistics
    W = statistic(X, Xk, y)

    # Run the knockoff filter
    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    selected = np.sort(np.where(W >= t)[0])

    # Get feature names for selected variables
    selected_names = None
    if feature_names is not None:
        selected_names = [feature_names[i] for i in selected]

    return KnockoffResult(
        X=X,
        Xk=Xk,
        y=y,
        statistic=W,
        threshold=t,
        selected=selected,
        feature_names=selected_names
    )


@dataclass
class VotingResult:
    """
    Result of the knockoff voting procedure (SLIDE-style).

    Attributes
    ----------
    selection_counts : np.ndarray
        Number of times each variable was selected across iterations.
    selection_frequency : np.ndarray
        Proportion of iterations where each variable was selected (= W).
    selected : np.ndarray
        Indices of variables selected in >= spec proportion of runs.
    threshold : float
        The spec threshold used for selection.
    niter : int
        Number of knockoff iterations.
    spec : float
        Specificity threshold.
    min_selections : int
        Minimum selections required (= ceiling(niter * spec)).
    selected_list : List[np.ndarray], optional
        List of selected indices per iteration (for findOptIter).
    optimal_iter : int, optional
        Index of the optimal iteration chosen by findOptIter.
    """
    selection_counts: np.ndarray
    selection_frequency: np.ndarray  # This is the "W" for voting
    selected: np.ndarray
    threshold: float
    niter: int
    spec: float
    min_selections: int
    selected_list: Optional[List[np.ndarray]] = None
    optimal_iter: Optional[int] = None

    def __repr__(self) -> str:
        n_selected = len(self.selected)
        p = len(self.selection_counts)
        opt_iter_str = f", optimal_iter={self.optimal_iter}" if self.optimal_iter is not None else ""
        return (
            f"VotingResult(\n"
            f"  n_features={p},\n"
            f"  n_selected={n_selected},\n"
            f"  selected={self.selected.tolist()},\n"
            f"  niter={self.niter},\n"
            f"  spec={self.spec},\n"
            f"  min_selections={self.min_selections}{opt_iter_str}\n"
            f")"
        )


def _run_single_knockoff(args):
    """
    Worker function for parallel knockoff execution.

    Returns the indices of selected variables for one iteration.
    """
    X, y, knockoffs, statistic, fdr, offset, seed = args

    # Set random seed for this iteration
    np.random.seed(seed)

    try:
        result = knockoff_filter(
            X, y,
            knockoffs=knockoffs,
            statistic=statistic,
            fdr=fdr,
            offset=offset
        )
        return result.selected.tolist()
    except Exception as e:
        warnings.warn(f"Knockoff iteration failed (seed={seed}): {e}")
        return []


def _cov_shrink_corpcor(X: np.ndarray) -> np.ndarray:
    """Shrink covariance matrix using the Schäfer-Strimmer (2005) method.

    Reimplements R's ``corpcor::cov.shrink`` in pure Python/NumPy.
    The algorithm separately shrinks variances (toward the median variance)
    and correlations (toward the identity matrix), then recombines them.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Data matrix (observations x variables).

    Returns
    -------
    np.ndarray of shape (p, p)
        Shrinkage-regularised covariance matrix.
    """
    n, p = X.shape
    # Uniform weights (unweighted case)
    w = np.full(n, 1.0 / n)
    w2 = np.sum(w ** 2)          # 1/n
    h1 = 1.0 / (1.0 - w2)       # n/(n-1)
    h1w2 = w2 / (1.0 - w2)      # 1/(n-1)

    # ── Weighted moments (mean, variance) ──────────────────────────────
    wm = np.sum(w[:, None] * X, axis=0)                    # weighted mean
    wv = h1 * (np.sum(w[:, None] * X ** 2, axis=0) - wm ** 2)  # weighted var
    wv = np.maximum(wv, 0.0)

    # ── Estimate lambda_var (variance shrinkage intensity) ─────────────
    xc = X - wm[None, :]   # centred
    zz = xc ** 2
    q1 = np.sum(w[:, None] * zz, axis=0)
    q2 = np.sum(w[:, None] * zz ** 2, axis=0) - q1 ** 2
    target_var = np.median(wv)
    num_var = np.sum(q2)
    den_var = np.sum((q1 - target_var / h1) ** 2)
    lambda_var = max(0.0, min(1.0, num_var / den_var * h1w2)) if den_var != 0 else 1.0

    # Shrunk variances
    vs = lambda_var * target_var + (1.0 - lambda_var) * wv
    sc_shrunk = np.sqrt(vs)

    # ── Estimate lambda (correlation shrinkage intensity) ──────────────
    # IMPORTANT: R's estimate.lambda standardises with the ORIGINAL sample sd,
    # not the shrunk sd.  The variance and correlation shrinkage are independent.
    sc_orig = np.sqrt(wv)
    sc_safe = np.where(sc_orig > 0, sc_orig, 1.0)
    xs = xc / sc_safe[None, :]                      # standardised
    xs[:, sc_orig == 0] = 0.0

    sw = np.sqrt(w)
    xsw = xs * sw[:, None]                          # xs * sqrt(w)

    # SVD-based computation of sE2R and sER2
    U, d, Vt = np.linalg.svd(xsw, full_matrices=False)
    sE2R = (np.sum(xsw * ((U * d[None, :] ** 3) @ Vt))
            - np.sum(np.sum(xsw ** 2, axis=0) ** 2))

    xs2w = xs ** 2 * sw[:, None]
    # sER2 = 2 * sum over i<j of (xs2w[:,i] * xs2w[:,j])
    # Equivalent to sum(rowsums^2) - sum(colsums_of_squares^2)
    # But R computes it with cumsum trick; replicate exactly:
    cumrev = np.cumsum(xs2w[:, ::-1], axis=1)[:, ::-1]  # cumsum from right
    sER2 = 2.0 * np.sum(xs2w[:, :(p - 1)] * cumrev[:, 1:])

    den_cor = sE2R
    num_cor = sER2 - sE2R
    lambda_cor = max(0.0, min(1.0, num_cor / den_cor * h1w2)) if den_cor != 0 else 1.0

    # Shrunk correlation matrix  R_shrunk = (1 - lambda) * R_sample + lambda * I
    # R_sample = h1 * (xsw^T @ xsw)  but we only need (1-lambda)*R_sample + lambda*I
    R_sample = h1 * (xsw.T @ xsw)
    R_shrunk = (1.0 - lambda_cor) * R_sample
    np.fill_diagonal(R_shrunk, 1.0)

    # ── Combine: Sigma_shrunk = D^{1/2} R_shrunk D^{1/2} ──────────────
    Sigma = R_shrunk * sc_shrunk[:, None] * sc_shrunk[None, :]

    return Sigma


def _prepare_knockoff_cache(
    X: np.ndarray,
    method: str = 'asdp',
    shrink: bool = False,
    match_r: bool = False,
    sdp_solver: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Pre-compute invariant values for knockoff generation.

    These computations are CONSTANT across all voting iterations since X doesn't change.
    By caching them, we avoid redundant SDP solving, covariance estimation, and Cholesky
    decomposition on every iteration.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Matrix of predictors.
    method : str, default='asdp'
        Method for minimizing correlation between original and knockoffs.
        One of 'asdp', 'sdp', 'equi'.
    shrink : bool, default=False
        Whether to shrink the estimated covariance matrix.
    match_r : bool, default=False
        If True, skip the condition number check for auto-shrinkage.

    Returns
    -------
    dict
        Dictionary containing cached values:
        - 'mu': Mean vector (p,)
        - 'Sigma': Covariance matrix (p, p)
        - 'diag_s': Diagonal of knockoff covariance (p,)
        - 'SigmaInv_s': Sigma^{-1} @ diag(s) (p, p)
        - 'L': Cholesky decomposition of Sigma_k (p, p)
        - 'method': Method used (may differ from input if fallback occurred)
        - 'degenerate': True if knockoff construction failed
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    if method not in ['asdp', 'sdp', 'equi']:
        raise ValueError(f"method must be 'asdp', 'sdp', or 'equi', got '{method}'")

    # Do not use ASDP unless p > 500
    if p <= 500 and method == 'asdp':
        method = 'sdp'

    # Auto-enable shrinkage for n <= 1.25*p (R-style regularization)
    # Skip this check if match_r=True (R doesn't auto-shrink based on n/p ratio)
    if not shrink and not match_r and n <= 1.25 * p:
        warnings.warn(
            f"n={n}, p={p} (n/p={n/p:.2f}): Insufficient samples for stable covariance. "
            f"Auto-enabling Ledoit-Wolf shrinkage for better knockoff power."
        )
        shrink = True

    # Estimate the mean vector
    mu = np.mean(X, axis=0)

    # Estimate the covariance matrix
    if not shrink:
        Sigma = np.cov(X, rowvar=False, ddof=1)
        if Sigma.ndim == 0:
            Sigma = np.array([[Sigma]])

        if not is_posdef(Sigma):
            shrink = True

        # Auto-enable shrinkage for ill-conditioned matrices (unless match_r)
        if not shrink and not match_r:
            cond_num = np.linalg.cond(Sigma)
            if cond_num > 1e5:
                warnings.warn(
                    f"Covariance matrix is ill-conditioned (cond={cond_num:.1e}). "
                    f"Auto-enabling Ledoit-Wolf shrinkage for better knockoff power."
                )
                shrink = True

    if shrink:
        if match_r:
            # Use Schäfer-Strimmer (2005) shrinkage to match R's corpcor::cov.shrink
            Sigma = _cov_shrink_corpcor(X)
        else:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(X)
                Sigma = lw.covariance_
            except ImportError:
                warnings.warn(
                    "sklearn is not installed. Using manual shrinkage."
                )
                S = np.cov(X, rowvar=False, ddof=1)
                if S.ndim == 0:
                    S = np.array([[S]])
                trace_S = np.trace(S)
                shrinkage_param = min(1.0, max(0.0, (p / n)))
                Sigma = (1 - shrinkage_param) * S + shrinkage_param * (trace_S / p) * np.eye(p)

    # Compute diag_s using the solver (or external solver, e.g. R's SDP)
    if sdp_solver is not None:
        diag_s = sdp_solver(Sigma, method)
    elif method == 'equi':
        diag_s = create_solve_equi(Sigma)
    elif method == 'sdp':
        diag_s = create_solve_sdp(Sigma)
    else:  # asdp
        diag_s = create_solve_asdp(Sigma)

    diag_s = np.asarray(diag_s)
    if diag_s.ndim == 2:
        diag_s = np.diag(diag_s)

    # Match R's create.gaussian: if all(diag_s == 0), warn and mark degenerate.
    # R does NOT fall back to equicorrelated — it returns X as knockoffs,
    # giving W ~0 and no variables selected (zero power).
    if np.all(diag_s == 0):
        warnings.warn(
            "The conditional knockoff covariance matrix is not positive definite. "
            "Knockoffs will have no power."
        )
        return {
            'mu': mu,
            'Sigma': Sigma,
            'diag_s': diag_s,
            'SigmaInv_s': None,
            'L': None,
            'method': method,
            'degenerate': True
        }

    # Compute knockoff distribution parameters
    diag_s_matrix = np.diag(diag_s)
    SigmaInv_s = linalg.solve(Sigma, diag_s_matrix)

    # Sigma_k = 2*diag(s) - diag(s) @ SigmaInv_s
    Sigma_k = 2 * diag_s_matrix - diag_s_matrix @ SigmaInv_s

    # Cholesky decomposition with R-style scaled regularization
    try:
        L = linalg.cholesky(Sigma_k, lower=True)
    except linalg.LinAlgError:
        max_diag = np.max(np.diag(Sigma_k))
        eps = 1e-10 * max(1.0, max_diag)
        try:
            L = linalg.cholesky(Sigma_k + eps * np.eye(p), lower=True)
        except linalg.LinAlgError:
            while eps < 1:
                eps *= 10
                try:
                    L = linalg.cholesky(Sigma_k + eps * np.eye(p), lower=True)
                    break
                except linalg.LinAlgError:
                    continue
            else:
                raise ValueError("Cholesky decomposition failed even with large regularization")

    return {
        'mu': mu,
        'Sigma': Sigma,
        'diag_s': diag_s,
        'SigmaInv_s': SigmaInv_s,
        'L': L,
        'method': method,
        'degenerate': False
    }


def _sample_knockoffs_from_cache(
    X: np.ndarray,
    cache: Dict[str, Any]
) -> np.ndarray:
    """
    Generate knockoff variables using pre-computed cache.

    This is the ONLY computation that varies across voting iterations -
    the random sampling step.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Matrix of original variables.
    cache : dict
        Pre-computed cache from _prepare_knockoff_cache.

    Returns
    -------
    np.ndarray of shape (n, p)
        Matrix of knockoff variables.
    """
    if cache.get('degenerate', False):
        # Return copy of X if knockoff construction failed
        return X.copy()

    n, p = X.shape
    mu = cache['mu']
    SigmaInv_s = cache['SigmaInv_s']
    L = cache['L']

    # mu_k = X - (X - mu) @ SigmaInv_s
    mu_k = X - (X - mu) @ SigmaInv_s

    # Sample knockoffs: X_k = mu_k + randn(n, p) @ L.T
    X_k = mu_k + np.random.randn(n, p) @ L.T

    return X_k


def _cached_iteration(X, y, cache, statistic, fdr, offset, seed):
    """Single cached knockoff iteration for parallel execution."""
    np.random.seed(seed)
    Xk = _sample_knockoffs_from_cache(X, cache)
    W = statistic(X, Xk, y)
    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    selected = np.where(W >= t)[0]
    return selected


def find_opt_iter(
    freq_vars: np.ndarray,
    selected_list: List[np.ndarray]
) -> tuple:
    """
    Find the iteration with maximum overlap with frequent variables (R SLIDE behavior).

    This implements R SLIDE's findOptIter() function which refines the selection
    by choosing variables from ONE optimal iteration rather than all variables
    above the threshold.

    The algorithm:
    1. Find iterations with maximum overlap with freq_vars (the threshold-passing variables)
    2. Tie-breaker: choose the iteration with the smallest total selection
    3. Return the selected variables from that ONE iteration

    Parameters
    ----------
    freq_vars : np.ndarray
        Indices of variables that passed the threshold (selected in >= spec * niter runs).
    selected_list : List[np.ndarray]
        List of selected variable indices for each iteration.

    Returns
    -------
    tuple
        (selected_vars, optimal_iter) where:
        - selected_vars: np.ndarray of indices from the optimal iteration
        - optimal_iter: int index of the chosen iteration

    Notes
    -----
    This matches R SLIDE's findOptIter() exactly:
    ```r
    mm <- max(unlist(lapply(selected_list, function(x) { sum(x %in% freq_vars) })))
    max_overlap_ind <- which(... == mm)
    overlap_list_len <- sapply(max_overlap_ind, function(x) { length(selected_list[[x]]) })
    selected_run <- max_overlap_ind[which.min(overlap_list_len)]
    selected_vars <- selected_list[[selected_run]]
    ```

    Examples
    --------
    >>> freq_vars = np.array([0, 2, 5])  # Variables passing threshold
    >>> selected_list = [
    ...     np.array([0, 2, 5, 7]),    # iter 0: 3 overlap, size 4
    ...     np.array([0, 2, 5]),       # iter 1: 3 overlap, size 3 (winner - smallest)
    ...     np.array([0, 2]),          # iter 2: 2 overlap, size 2
    ... ]
    >>> selected, opt_iter = find_opt_iter(freq_vars, selected_list)
    >>> opt_iter
    1
    >>> selected
    array([0, 2, 5])
    """
    if len(freq_vars) == 0:
        # No frequent variables - return empty
        return np.array([], dtype=int), None

    if len(selected_list) == 0:
        return np.array([], dtype=int), None

    freq_vars_set = set(freq_vars)

    # Compute overlap for each iteration
    overlaps = []
    for i, sel in enumerate(selected_list):
        if sel is None or len(sel) == 0:
            overlaps.append(0)
        else:
            overlap = len(set(sel) & freq_vars_set)
            overlaps.append(overlap)

    overlaps = np.array(overlaps)

    # Find iterations with maximum overlap
    max_overlap = np.max(overlaps)
    if max_overlap == 0:
        # No overlap at all - return freq_vars as-is (fallback)
        return freq_vars, None

    max_overlap_indices = np.where(overlaps == max_overlap)[0]

    # Tie-breaker: choose iteration with smallest selection set
    selection_sizes = np.array([
        len(selected_list[i]) if selected_list[i] is not None else 0
        for i in max_overlap_indices
    ])

    # Find the index within max_overlap_indices that has smallest size
    min_size_idx = np.argmin(selection_sizes)
    optimal_iter = max_overlap_indices[min_size_idx]

    # Return variables from that ONE iteration
    selected_vars = selected_list[optimal_iter]
    if selected_vars is None:
        selected_vars = np.array([], dtype=int)
    else:
        selected_vars = np.sort(np.asarray(selected_vars))

    return selected_vars, int(optimal_iter)


def knockoff_filter_voting(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 0,
    niter: int = 1000,
    spec: float = 0.1,
    n_jobs: int = 1,
    base_seed: int = 42,
    verbose: bool = False,
    match_r: bool = True,
    use_cache: bool = True,
    slide_selection: bool = False,
    return_selected_list: bool = False,
    **kwargs
) -> VotingResult:
    """
    Run the Knockoff Filter with SLIDE-style voting for stable variable selection.

    This function runs the knockoff filter multiple times with different random
    seeds and selects variables that appear in at least `spec` proportion of runs.
    This matches R's SLIDE voting methodology.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix of predictors.
    y : array-like of shape (n,)
        Response vector.
    knockoffs : callable, optional
        Function to construct knockoffs. Default: create_second_order.
    statistic : callable, optional
        Function to compute importance statistics. Default: stat_glmnet_coefdiff.
    fdr : float, default=0.10
        Target false discovery rate for each knockoff run.
    offset : {0, 1}, default=0
        Offset for knockoff threshold. 0 = standard knockoff (matches R's knockoff.filter),
        1 = knockoffs+ (more conservative).
    niter : int, default=1000
        Number of knockoff iterations to run. Matches R SLIDE default.
    spec : float, default=0.1
        Specificity threshold - keep variables selected in >= spec * niter runs.
        Matches R SLIDE default.
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all available cores.
    base_seed : int, default=42
        Base seed for reproducibility. Each iteration uses base_seed + i.
    verbose : bool, default=False
        Print progress information.
    match_r : bool, default=True
        If True (default), skip the automatic Ledoit-Wolf shrinkage condition
        number check to match R's knockoff.filter behavior exactly. Set to False
        to enable Python's more conservative auto-shrinkage for ill-conditioned cases.
    use_cache : bool, default=True
        If True (default), pre-compute invariant quantities (covariance, SDP solution,
        Cholesky decomposition) once before the voting loop. This provides 3-4x speedup
        by avoiding redundant computations. Only applies when using default knockoffs.
        Set to False to use the original uncached behavior.
    slide_selection : bool, default=False
        If True, use R SLIDE's findOptIter() selection refinement:
        - Find iterations with maximum overlap with threshold-passing variables
        - Tie-breaker: choose iteration with smallest selection set
        - Return variables from that ONE iteration
        If False (default), return ALL variables passing the threshold.
    return_selected_list : bool, default=False
        If True, store the list of selected indices per iteration in the result.
        Required for findOptIter but uses more memory.

    Returns
    -------
    VotingResult
        Object containing selection_counts, selection_frequency (W), selected,
        and voting parameters.

    Notes
    -----
    This implements the SLIDE voting approach:
    1. Run knockoff filter `niter` times with different random seeds
    2. Count how often each variable is selected
    3. Keep variables selected in >= spec proportion of runs

    The `selection_frequency` field serves as the "W" statistic for voting,
    representing the proportion of runs where each variable was selected.

    Performance Optimization (use_cache=True):
    When using default second-order knockoffs, the covariance matrix estimation,
    SDP solver, and Cholesky decomposition are CONSTANT across all voting iterations
    since X doesn't change. By pre-computing these once, we avoid redundant work
    and achieve 3-4x speedup (e.g., from ~23 min to ~6-8 min for 500 iterations).

    References
    ----------
    Barber and Candes, Controlling the false discovery rate via knockoffs.
    Ann. Statist. 43 (2015), no. 5, 2055--2085.

    Examples
    --------
    >>> import numpy as np
    >>> from loveslide.knockoff import knockoff_filter_voting
    >>> n, p = 100, 50
    >>> X = np.random.randn(n, p)
    >>> beta = np.zeros(p)
    >>> beta[:5] = 3.0
    >>> y = X @ beta + np.random.randn(n)
    >>> result = knockoff_filter_voting(X, y, niter=100, spec=0.1)
    >>> print(result.selected)
    """
    # Convert inputs
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if y.ndim > 1:
        y = y.ravel()

    n, p = X.shape

    # Validate parameters
    if niter < 1:
        raise ValueError(f"niter must be >= 1, got {niter}")
    if not 0 < spec <= 1:
        raise ValueError(f"spec must be in (0, 1], got {spec}")

    # Set defaults
    # If match_r is True and no custom knockoffs function provided, wrap create_second_order
    if knockoffs is None:
        if match_r:
            knockoffs = lambda X: create_second_order(X, match_r=True)
        else:
            knockoffs = create_second_order
    if statistic is None:
        from .stats import stat_glmnet_coefdiff
        statistic = stat_glmnet_coefdiff

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, niter))

    # Initialize selection counts and optional list
    selection_counts = np.zeros(p, dtype=np.int32)
    selected_list = [] if (slide_selection or return_selected_list) else None

    if verbose:
        print(f"Running {niter} knockoff iterations with {n_jobs} jobs...")

    # Prepare arguments for each iteration
    # Note: We can't pickle lambdas, so we pass knockoffs/statistic as None
    # and rely on defaults, or we run sequentially

    # Extract sdp_solver from kwargs (used by r_knockoffs backend)
    sdp_solver = kwargs.pop('sdp_solver', None)

    # Check if we can use caching optimization (default knockoffs and use_cache enabled)
    can_use_cache = use_cache and (knockoffs is create_second_order or
                 (knockoffs is not None and callable(knockoffs) and
                  hasattr(knockoffs, '__name__') and knockoffs.__name__ == '<lambda>' and match_r))

    # Force caching when an external SDP solver is provided (r_knockoffs hybrid path)
    if sdp_solver is not None and use_cache:
        can_use_cache = True

    if can_use_cache:
        # OPTIMIZED PATH: Pre-compute invariant quantities once
        # This avoids redundant SDP solving, covariance estimation, and Cholesky
        # decomposition on every iteration (3-4x speedup)
        if verbose:
            solver_name = "R SDP" if sdp_solver is not None else "Python"
            print(f"  Using cached knockoff computation ({solver_name} solver)...")

        try:
            cache = _prepare_knockoff_cache(X, method='asdp', shrink=False, match_r=match_r,
                                            sdp_solver=sdp_solver)
        except Exception as e:
            warnings.warn(f"Cache preparation failed: {e}. Falling back to uncached path.")
            can_use_cache = False

    if can_use_cache:
        # Fast path with caching - supports both sequential and parallel
        if n_jobs > 1:
            from joblib import Parallel, delayed
            results_list = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_cached_iteration)(
                    X, y, cache, statistic, fdr, offset, base_seed + i
                )
                for i in range(niter)
            )
            for i, selected_iter in enumerate(results_list):
                for idx in selected_iter:
                    selection_counts[idx] += 1
                if selected_list is not None:
                    selected_list.append(selected_iter.copy())
            if verbose:
                print(f"  Completed {niter}/{niter} iterations ({n_jobs} parallel jobs)")
        else:
            for i in range(niter):
                seed = base_seed + i
                np.random.seed(seed)

                try:
                    # Only random sampling varies per iteration
                    Xk = _sample_knockoffs_from_cache(X, cache)

                    # Compute statistics and threshold
                    W = statistic(X, Xk, y)
                    t = knockoff_threshold(W, fdr=fdr, offset=offset)
                    selected_iter = np.where(W >= t)[0]

                    for idx in selected_iter:
                        selection_counts[idx] += 1

                    # Store selected list if needed for findOptIter
                    if selected_list is not None:
                        selected_list.append(selected_iter.copy())

                    if verbose and (i + 1) % 50 == 0:
                        print(f"  Completed {i + 1}/{niter} iterations")

                except Exception as e:
                    warnings.warn(f"Knockoff iteration {i} failed: {e}")
                    if selected_list is not None:
                        selected_list.append(np.array([], dtype=int))
    elif n_jobs == 1:
        # Sequential uncached path for custom knockoffs
        for i in range(niter):
            seed = base_seed + i
            np.random.seed(seed)

            try:
                result = knockoff_filter(
                    X, y,
                    knockoffs=knockoffs,
                    statistic=statistic,
                    fdr=fdr,
                    offset=offset
                )
                for idx in result.selected:
                    selection_counts[idx] += 1

                # Store selected list if needed for findOptIter
                if selected_list is not None:
                    selected_list.append(result.selected.copy())

                if verbose and (i + 1) % 50 == 0:
                    print(f"  Completed {i + 1}/{niter} iterations")

            except Exception as e:
                warnings.warn(f"Knockoff iteration {i} failed: {e}")
                if selected_list is not None:
                    selected_list.append(np.array([], dtype=int))
    else:
        # Parallel uncached execution (only works with default knockoffs/statistic)
        is_default_knockoffs = (knockoffs is create_second_order or
                                (match_r and callable(knockoffs) and knockoffs.__name__ == '<lambda>'))
        if not is_default_knockoffs or statistic is not None:
            warnings.warn(
                "Parallel execution with custom knockoffs/statistic may not work. "
                "Falling back to sequential execution."
            )
            return knockoff_filter_voting(
                X, y, knockoffs=knockoffs, statistic=statistic,
                fdr=fdr, offset=offset, niter=niter, spec=spec,
                n_jobs=1, base_seed=base_seed, verbose=verbose,
                match_r=match_r, slide_selection=slide_selection,
                return_selected_list=return_selected_list, **kwargs
            )

        # Prepare arguments
        args_list = [
            (X, y, None, None, fdr, offset, base_seed + i)
            for i in range(niter)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_run_single_knockoff, args) for args in args_list]

            for i, future in enumerate(as_completed(futures)):
                try:
                    selected_iter = future.result()
                    for idx in selected_iter:
                        selection_counts[idx] += 1
                    # Store selected list if needed for findOptIter
                    if selected_list is not None:
                        selected_list.append(np.array(selected_iter, dtype=int))
                except Exception as e:
                    warnings.warn(f"Knockoff iteration failed: {e}")
                    if selected_list is not None:
                        selected_list.append(np.array([], dtype=int))

                if verbose and (i + 1) % 50 == 0:
                    print(f"  Completed {i + 1}/{niter} iterations")

    # Compute selection frequency (this is the "W" for voting)
    selection_frequency = selection_counts / niter

    # Select variables appearing in >= spec proportion of runs
    min_selections = int(np.ceil(niter * spec))
    freq_vars = np.where(selection_counts >= min_selections)[0]
    freq_vars = np.sort(freq_vars)

    # Apply findOptIter refinement if requested (R SLIDE behavior)
    optimal_iter = None
    if slide_selection and selected_list is not None and len(freq_vars) > 0:
        selected, optimal_iter = find_opt_iter(freq_vars, selected_list)
        if verbose:
            print(f"  findOptIter: chose iteration {optimal_iter} with {len(selected)} variables")
    else:
        selected = freq_vars

    if verbose:
        print(f"  Selected {len(selected)} variables (>= {min_selections} selections)")

    return VotingResult(
        selection_counts=selection_counts,
        selection_frequency=selection_frequency,
        selected=selected,
        threshold=spec,
        niter=niter,
        spec=spec,
        min_selections=min_selections,
        selected_list=selected_list if return_selected_list else None,
        optimal_iter=optimal_iter
    )


def knockoff_filter_voting_slide(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 0,
    niter: int = 1000,
    spec: float = 0.1,
    f_size: int = 100,
    n_jobs: int = 1,
    base_seed: int = 42,
    verbose: bool = False,
    match_r: bool = True,
    use_cache: bool = True,
    slide_selection: bool = True,
    **kwargs
) -> VotingResult:
    """
    Run the Knockoff Filter with full R SLIDE methodology.

    This function implements R SLIDE's complete voting procedure including:
    1. Feature chunking (f_size parameter)
    2. Knockoff voting on each chunk
    3. findOptIter refinement
    4. Two-stage screening when multiple chunks

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix of predictors.
    y : array-like of shape (n,)
        Response vector.
    knockoffs : callable, optional
        Function to construct knockoffs. Default: create_second_order.
    statistic : callable, optional
        Function to compute importance statistics. Default: stat_glmnet_coefdiff.
    fdr : float, default=0.10
        Target false discovery rate for each knockoff run.
    offset : {0, 1}, default=0
        Offset for knockoff threshold. 0 = standard knockoff (matches R's knockoff.filter),
        1 = knockoffs+ (more conservative).
    niter : int, default=1000
        Number of knockoff iterations to run. Matches R SLIDE default.
    spec : float, default=0.1
        Specificity threshold - keep variables selected in >= spec * niter runs.
        Matches R SLIDE default.
    f_size : int, default=100
        Maximum number of features per chunk. Features are split into
        ceil(p / f_size) chunks, and knockoff voting is run on each chunk separately.
        This matches R SLIDE's selectShortFreq() default behavior.
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all available cores.
    base_seed : int, default=42
        Base seed for reproducibility.
    verbose : bool, default=False
        Print progress information.
    match_r : bool, default=True
        If True (default), skip the automatic Ledoit-Wolf shrinkage condition number
        check to match R's knockoff.filter behavior.
    use_cache : bool, default=True
        If True (default), pre-compute invariant quantities once per chunk.

    Returns
    -------
    VotingResult
        Object containing selection_counts, selection_frequency (W), selected,
        and voting parameters.

    Notes
    -----
    This implements R SLIDE's selectShortFreq() procedure:

    1. **Feature chunking**: Split p features into ceil(p/f_size) chunks
    2. **Per-chunk voting**: Run knockoff_filter_voting on each chunk with slide_selection=True
    3. **findOptIter refinement**: For each chunk, apply findOptIter to select variables
       from ONE optimal iteration rather than all threshold-passing variables
    4. **Two-stage screening**: If n_splits > 1, combine screened variables from all chunks
       and re-run knockoff voting on the combined set

    When p <= f_size, no chunking is needed and this behaves like knockoff_filter_voting
    with slide_selection=True.

    References
    ----------
    SLIDE: Significant Latent factor Interaction Discovery and Exploration across
    biological domains.

    Examples
    --------
    >>> import numpy as np
    >>> from loveslide.knockoff import knockoff_filter_voting_slide
    >>> n, p = 100, 200
    >>> X = np.random.randn(n, p)
    >>> beta = np.zeros(p)
    >>> beta[:10] = 3.0
    >>> y = X @ beta + np.random.randn(n)
    >>> # With chunking (p=200 > f_size=100 means 2 chunks)
    >>> result = knockoff_filter_voting_slide(X, y, niter=100, spec=0.1, f_size=100)
    >>> print(result.selected)
    """
    import math

    # Convert inputs
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if y.ndim > 1:
        y = y.ravel()

    n, p = X.shape

    # Validate parameters
    if f_size < 1:
        raise ValueError(f"f_size must be >= 1, got {f_size}")

    # Determine number of chunks
    n_splits = int(math.ceil(p / f_size))

    if n_splits == 1:
        # No chunking needed - just run voting with SLIDE selection
        if verbose:
            print(f"Single chunk (p={p} <= f_size={f_size}), using direct voting...")
        return knockoff_filter_voting(
            X, y,
            knockoffs=knockoffs,
            statistic=statistic,
            fdr=fdr,
            offset=offset,
            niter=niter,
            spec=spec,
            n_jobs=n_jobs,
            base_seed=base_seed,
            verbose=verbose,
            match_r=match_r,
            use_cache=use_cache,
            slide_selection=slide_selection,
            return_selected_list=True,
            **kwargs
        )

    # Multiple chunks - implement R SLIDE's two-stage screening
    if verbose:
        print(f"Chunking: p={p} into {n_splits} chunks of ~{f_size} features...")

    # Calculate chunk boundaries
    feature_split = int(math.ceil(p / n_splits))
    feature_starts = list(range(0, p, feature_split))
    feature_stops = [min(start + feature_split, p) for start in feature_starts]

    # Stage 1: Run knockoff voting on each chunk
    screen_var = []

    for chunk_idx, (start, stop) in enumerate(zip(feature_starts, feature_stops)):
        if verbose:
            print(f"  Chunk {chunk_idx + 1}/{n_splits}: features [{start}, {stop})")

        X_chunk = X[:, start:stop]

        # Run voting on this chunk
        result_chunk = knockoff_filter_voting(
            X_chunk, y,
            knockoffs=knockoffs,
            statistic=statistic,
            fdr=fdr,
            offset=offset,
            niter=niter,
            spec=spec,
            n_jobs=n_jobs,
            base_seed=base_seed,
            verbose=verbose,
            match_r=match_r,
            use_cache=use_cache,
            slide_selection=slide_selection,
            return_selected_list=True,
            **kwargs
        )

        # Map chunk indices back to global indices
        chunk_selected = result_chunk.selected + start
        if verbose:
            print(f"    Selected {len(chunk_selected)} variables from chunk")

        screen_var.extend(chunk_selected.tolist())

    screen_var = np.array(screen_var, dtype=int)

    if verbose:
        print(f"  Stage 1 complete: {len(screen_var)} screened variables")

    # Stage 2: Re-run knockoff voting on combined screened variables
    if len(screen_var) <= 1:
        # Not enough variables to re-screen
        if verbose:
            print(f"  Skipping Stage 2 (only {len(screen_var)} variables)")

        # Return a VotingResult with global indices
        selection_counts = np.zeros(p, dtype=np.int32)
        for idx in screen_var:
            selection_counts[idx] = int(np.ceil(niter * spec))  # Mark as selected

        return VotingResult(
            selection_counts=selection_counts,
            selection_frequency=selection_counts / niter,
            selected=np.sort(screen_var),
            threshold=spec,
            niter=niter,
            spec=spec,
            min_selections=int(np.ceil(niter * spec)),
            selected_list=None,
            optimal_iter=None
        )

    if verbose:
        print(f"  Stage 2: Re-screening {len(screen_var)} combined variables...")

    X_screen = X[:, screen_var]

    # Run final voting on screened variables
    final_result = knockoff_filter_voting(
        X_screen, y,
        knockoffs=knockoffs,
        statistic=statistic,
        fdr=fdr,
        offset=offset,
        niter=niter,
        spec=spec,
        n_jobs=n_jobs,
        base_seed=base_seed,
        verbose=verbose,
        match_r=match_r,
        use_cache=use_cache,
        slide_selection=slide_selection,
        return_selected_list=True,
        **kwargs
    )

    # Map screened indices back to global indices
    final_selected = screen_var[final_result.selected]

    if verbose:
        print(f"  Final selection: {len(final_selected)} variables")

    # Build global selection counts and frequency
    selection_counts = np.zeros(p, dtype=np.int32)
    for i, local_idx in enumerate(range(len(screen_var))):
        global_idx = screen_var[local_idx]
        selection_counts[global_idx] = final_result.selection_counts[local_idx]

    selection_frequency = selection_counts / niter

    return VotingResult(
        selection_counts=selection_counts,
        selection_frequency=selection_frequency,
        selected=np.sort(final_selected),
        threshold=spec,
        niter=niter,
        spec=spec,
        min_selections=final_result.min_selections,
        selected_list=None,  # Not meaningful after two-stage
        optimal_iter=final_result.optimal_iter
    )
