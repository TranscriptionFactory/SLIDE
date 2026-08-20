import numpy as np
import pandas as pd
import os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import math
from pqdm.processes import pqdm
from functools import partial
from tqdm import tqdm
import copy
import logging

from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, lasso_path

logger = logging.getLogger(__name__)


def _import_rpy2():
    """Lazy rpy2 import with graceful CFFI error handling.

    Returns rpy2.robjects module. Raises RuntimeError with actionable advice
    if rpy2 fails to initialize (common in subprocess/loky contexts).
    """
    try:
        import rpy2.robjects as robjects
        return robjects
    except Exception as e:
        raise RuntimeError(
            f"rpy2 initialization failed: {e}\n"
            "This often occurs when rpy2 is first imported inside a subprocess "
            "(joblib/loky). Solutions:\n"
            "  1. Add `import rpy2.robjects` at the top of your main script\n"
            "  2. Use backend='r_knockoffs' (Python voting + R SDP) instead of "
            "backend='r' (full R pipeline)"
        ) from e


# Import from bundled knockoff package
from .knockoff.filter import (
    knockoff_filter,
    knockoff_threshold,
    knockoff_filter_voting_slide,
    VotingResult,
)
from .knockoff.create import create_gaussian, KnockoffVariables
from .knockoff.stats import (
    stat_glmnet_lambdasmax,
    stat_glmnet_lambdadiff,
    stat_glmnet_coefdiff,
    stat_sqrt_lasso,
    stat_stability_selection,
    stat_random_forest,
    stat_lasso_lambdasmax,
    stat_lasso_lambdadiff,
    stat_lasso_coefdiff,
)
from .knockoff.solve import create_solve_equi, create_solve_sdp, create_solve_asdp
from .knockoff.utils import is_posdef


def _rlist_get(robj, name):
    """Access named element of an R list, compatible with rpy2 3.5.x and 3.6.x.

    rpy2 3.5 supports result['name']; rpy2 3.6 deprecated OrdDict and broke
    string-key access.  Fall back to names-based integer indexing.
    In rpy2 3.6.x, .names may be a method (callable) rather than a property.
    """
    try:
        return robj[name]
    except TypeError:
        r_names = robj.names
        if callable(r_names):
            r_names = r_names()
        names = list(r_names)
        return robj[names.index(name)]


def _create_second_order_r(X: np.ndarray) -> np.ndarray:
    """Wrapper around R's knockoff::create.second_order().

    This function generates knockoff variables using R's implementation,
    which may produce slightly different results than the Python implementation
    due to numerical differences in the SDP solver and random sampling.

    Requires rpy2 and the R knockoff package to be installed.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Knockoff feature matrix of shape (n_samples, n_features).

    Raises
    ------
    ImportError
        If rpy2 or the R knockoff package is not installed.
    """
    robjects = _import_rpy2()
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr

    converter = robjects.default_converter + numpy2ri.converter + pandas2ri.converter
    with converter.context():
        knockoff_r = importr("knockoff")
        X_r = robjects.r["as.matrix"](pd.DataFrame(X))
        Xk_r = knockoff_r.create_second_order(X_r)
        Xk = np.array(Xk_r)

    return Xk


def _solve_sdp_r(Sigma: np.ndarray, method: str = "sdp") -> np.ndarray:
    """Compute knockoff diagonal s-vector using R's SDP solver via rpy2.

    Calls R's create.solve_sdp (or create.solve_asdp/equi) once to get the
    diagonal of the knockoff covariance matrix.  The result can be fed into
    the Python cached iteration loop so that R's SDP solution is used for
    knockoff generation without re-invoking R on every iteration.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix of shape (p, p).
    method : str
        SDP method: 'sdp' (default), 'asdp', or 'equi'.

    Returns
    -------
    np.ndarray
        Diagonal s-vector of shape (p,).
    """
    robjects = _import_rpy2()
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    converter = robjects.default_converter + numpy2ri.converter
    with converter.context():
        knockoff_r = importr("knockoff")

        if method == "equi":
            diag_s_r = knockoff_r.create_solve_equi(Sigma)
        elif method == "asdp":
            diag_s_r = knockoff_r.create_solve_asdp(Sigma)
        else:  # sdp
            diag_s_r = knockoff_r.create_solve_sdp(Sigma)

        diag_s = np.asarray(diag_s_r).flatten()

    return diag_s


def _sample_gaussian_r(X, cache, seed=None):
    """Sample one Gaussian knockoff matrix in R using cached parameters."""
    robjects = _import_rpy2()
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    converter = robjects.default_converter + numpy2ri.converter
    with converter.context():
        if seed is not None:
            robjects.r["set.seed"](int(seed))
        knockoff_r = importr("knockoff")
        diag_s = np.diag(cache["diag_s"])
        sampled = knockoff_r.create_gaussian(
            X,
            cache["mu"],
            cache["Sigma"],
            diag_s=diag_s,
        )
        return np.asarray(sampled)


def _single_knockoff_iteration_python(
    z, y, fdr, method, shrink, offset, statistic, mu, Sigma, diag_s
):
    """Execute a single knockoff filter iteration with cached covariance.

    This function is designed to be called in parallel via joblib.
    """
    # Import inside function for parallel execution (pickle serialization)
    from loveslide.knockoff.filter import knockoff_threshold
    from loveslide.knockoff.create import create_gaussian

    # Generate knockoffs using cached SDP solution
    Xk = create_gaussian(z, mu, Sigma, method=method, diag_s=diag_s)

    # Compute statistics
    W = statistic(z, Xk, y.flatten())

    # Run the knockoff filter
    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    selected = np.where(W >= t)[0]

    return selected.tolist() if len(selected) > 0 else []


class Knockoffs:
    def __init__(self, y, z2, model="LR"):

        # self.z1 = self.scale_features(z1)
        self.z2 = self.scale_features(z2)
        self.y = y
        self.n = self.y.shape[0]
        # self.n, self.k = self.z1.shape
        self.l = self.z2.shape[1]
        # self.interaction_terms = self.get_interaction_terms(self.z1, self.z2)

        if model == "lasso":
            self.model = Lasso(alpha=0.1)
        elif model == "LR":
            self.model = LinearRegression()
        else:
            raise ValueError("Model not supported")

    def add_z1(self, z1=None, marginal_idxs=None):
        if marginal_idxs is not None and z1 is None:
            z1 = self.z2[:, marginal_idxs]
            self.z2 = np.delete(self.z2, marginal_idxs, axis=1)

        n, self.k = z1.shape

        assert n == self.n

        self.z1 = self.scale_features(z1)
        self.interaction_terms = self.get_interaction_terms(self.z1, self.z2)

    @staticmethod
    def scale_features(X, minmax=False, feature_range=(-1, 1)):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if minmax:
            scaler = MinMaxScaler(feature_range=feature_range)
        else:
            scaler = StandardScaler()

        scaler.fit(X)
        return scaler.transform(X)

    @staticmethod
    def correct_y(z_marginal, y):
        """Correct y for marginal effect (residuals from lm(y ~ z_marginal)).

        This matches R's correctIt function in interactionSLIDE.R which removes
        the marginal's contribution from y before testing interactions.

        Parameters
        ----------
        z_marginal : np.ndarray
            Marginal latent factor values, shape (n_samples,).
        y : np.ndarray
            Response vector, shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        np.ndarray
            Residuals from linear regression of y on z_marginal.
        """
        y_flat = y.flatten() if hasattr(y, "flatten") else np.array(y).flatten()
        z_2d = z_marginal.reshape(-1, 1)
        reg = LinearRegression().fit(z_2d, y_flat)
        return y_flat - reg.predict(z_2d)

    @staticmethod
    def get_interaction_terms(z_matrix, plm_embedding):
        """
        @return: interactions in shape of (n_samples, n_LFs, plm_embed_dim)
        """

        # If only one dimension, need to reshape to 2D for einsum to work as expected
        if len(z_matrix.shape) == 1:
            n = z_matrix.shape[0]
            z_matrix = z_matrix.reshape(n, -1)

        if len(plm_embedding.shape) == 1:
            n = plm_embedding.shape[0]
            plm_embedding = plm_embedding.reshape(n, -1)

        assert z_matrix.shape[0] == plm_embedding.shape[0]
        return np.einsum("ij,ik->ijk", z_matrix, plm_embedding)

    @staticmethod
    def filter_knockoffs_iterative_r(z, y, fdr=0.1, niter=1, spec=0.2, **kwargs):
        """Run knockoff filter using R package via rpy2."""
        robjects = _import_rpy2()
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        knockoff = importr("knockoff")
        converter = robjects.default_converter + pandas2ri.converter

        results = []
        with converter.context():
            z_r = robjects.conversion.get_conversion().py2rpy(pd.DataFrame(z))
            y_r = robjects.conversion.get_conversion().py2rpy(pd.Series(y.flatten()))

            for _ in range(niter):
                result = knockoff.knockoff_filter(
                    X=z_r,
                    y=y_r,
                    knockoffs=knockoff.create_second_order,
                    statistic=knockoff.stat_glmnet_lambdasmax,
                    offset=0,
                    fdr=fdr,
                )
                selected = _rlist_get(result, "selected")
                results.append(robjects.conversion.get_conversion().rpy2py(selected))

        results = np.concatenate(results, axis=0)
        results = results - 1  # Convert to 0-based indexing

        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs

    @staticmethod
    def filter_knockoffs_iterative_python(
        z,
        y,
        fdr=0.1,
        niter=1,
        spec=0.2,
        method="asdp",
        shrink=False,
        offset=0,
        fstat="glmnet_lambdasmax",
        n_jobs=-1,
        **kwargs,
    ):
        """Run knockoff filter using pure Python package with parallel processing.

        Parameters
        ----------
        z : np.ndarray
            Feature matrix.
        y : np.ndarray
            Response vector.
        fdr : float
            Target false discovery rate.
        niter : int
            Number of knockoff iterations.
        spec : float
            Proportion threshold for selection frequency.
        method : str
            Knockoff construction method: 'asdp' (default), 'sdp', or 'equi'.
            - 'asdp': Approximate SDP, faster for high-dimensional data
            - 'sdp': Full semidefinite programming, more power but may fail
            - 'equi': Equicorrelated, always works but lower power
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage.
        offset : int
            Knockoff procedure offset:
            - 0: Original knockoff (more power, controls modified FDR) - R default
            - 1: Knockoff+ (conservative, controls exact FDR)
        fstat : str
            Feature statistic method:
            - 'glmnet_lambdasmax': Signed max of glmnet lasso path (matches R's stat.glmnet_lambdasmax)
            - 'glmnet_lambdadiff': Lambda difference statistic
            - 'glmnet_coefdiff': Coefficient difference at CV-selected lambda
        n_jobs : int
            Number of parallel jobs. -1 uses all available cores.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        # Map fstat names to functions
        # GLMNet-based (uses vendored Fortran glmnet — same glmnet5.f90 as R)
        # This is the default for best R compatibility since it's the same
        # coordinate descent algorithm R's knockoff package uses.
        fstat_map = {
            # GLMNet-based (default) - Fortran glmnet for R compatibility
            "glmnet_lambdasmax": stat_glmnet_lambdasmax,
            "glmnet_lambdadiff": stat_glmnet_lambdadiff,
            "glmnet_coefdiff": stat_glmnet_coefdiff,
            # Lasso-based (wrapper around glmnet with family='gaussian')
            "lasso_lambdasmax": stat_lasso_lambdasmax,
            "lasso_lambdadiff": stat_lasso_lambdadiff,
            "lasso_coefdiff": stat_lasso_coefdiff,
            # sklearn-based alternatives (use_sklearn=True to force sklearn path)
            "sklearn_lambdasmax": partial(stat_glmnet_lambdasmax, use_sklearn=True),
            "sqrt_lasso": stat_sqrt_lasso,
            "stability": stat_stability_selection,
            "random_forest": stat_random_forest,
        }
        statistic = fstat_map.get(
            fstat, stat_glmnet_lambdasmax
        )

        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y)

        # Pre-compute covariance and SDP solution (OPTIMIZATION: avoid recomputing each iteration)
        mu = np.mean(z, axis=0)

        if not shrink:
            Sigma = np.cov(z, rowvar=False)
            if Sigma.ndim == 0:
                Sigma = np.array([[Sigma]])
            if not is_posdef(Sigma):
                shrink = True

        if shrink:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(z)
            Sigma = lw.covariance_

        # Solve SDP once (OPTIMIZATION: this is expensive and doesn't change between iterations)
        p = z.shape[1]

        logger.info(
            f"Pre-computing SDP solution for {p} features using method={method}"
        )
        try:
            if method == "equi":
                diag_s = create_solve_equi(Sigma)
            elif method == "asdp":
                diag_s = create_solve_asdp(Sigma)
            else:
                diag_s = create_solve_sdp(Sigma)
        except ImportError as e:
            # Fall back to equi only if solver package not installed
            logger.warning(
                f"SDP/ASDP solver not available ({e}), falling back to equi method"
            )
            method = "equi"
            diag_s = create_solve_equi(Sigma)

        # Match R: if SDP returns all zeros, return empty (no power)
        if np.all(diag_s == 0):
            logger.warning(
                "SDP solver returned all-zero solution. Knockoffs will have no power."
            )
            return np.array([], dtype=int)

        logger.info(f"Running {niter} knockoff iterations with {n_jobs} parallel jobs")

        # Run iterations in parallel (OPTIMIZATION: embarrassingly parallel)
        # Only use multiprocessing for large niter to justify process spawning overhead
        if n_jobs == 1 or niter < 200:
            # Sequential execution for small niter or explicit single-threaded
            results = []
            for _ in range(niter):
                selected = _single_knockoff_iteration_python(
                    z, y, fdr, method, shrink, offset, statistic, mu, Sigma, diag_s
                )
                results.extend(selected)
        else:
            # Parallel execution for large niter
            results_nested = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
                delayed(_single_knockoff_iteration_python)(
                    z, y, fdr, method, shrink, offset, statistic, mu, Sigma, diag_s
                )
                for _ in range(niter)
            )
            # Flatten results
            results = [item for sublist in results_nested for item in sublist]

        if len(results) == 0:
            return np.array([], dtype=int)

        results = np.array(results)
        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs

    @staticmethod
    def _compute_glmnet_lambdasmax(X, Xk, y, nlambda=250, eps=0.0005):
        """Compute W statistics matching R's stat.glmnet_lambdasmax.

        This implements the signed maximum of lasso path statistic using a
        grid of lambda values similar to glmnet's default behavior.

        Includes the random swap symmetrization from R's implementation to
        ensure unbiased W statistics.

        Parameters
        ----------
        X : np.ndarray
            Original feature matrix (n x p).
        Xk : np.ndarray
            Knockoff feature matrix (n x p).
        y : np.ndarray
            Response vector (n,).
        nlambda : int
            Number of lambda values in the grid (default 250).
        eps : float
            Ratio of lambda_min/lambda_max (default 0.0005, matching glmnet's 1/2000).

        Returns
        -------
        np.ndarray
            W statistics for each feature (p,).
        """
        n, p = X.shape
        y = y.flatten()

        # Random swap for symmetry (matching R's stat.glmnet_lambdasmax)
        # This ensures W statistics are unbiased w.r.t. the ordering of X vs Xk
        swap = np.random.binomial(1, 0.5, size=p)
        X_swap = X * (1 - swap) + Xk * swap
        Xk_swap = X * swap + Xk * (1 - swap)

        # Combine swapped X and knockoffs: [X_swap, Xk_swap]
        X_full = np.hstack([X_swap, Xk_swap])

        # Compute lambda grid (matching glmnet's log-scale grid)
        # lambda_max is where all coefficients become zero
        lambda_max = np.max(np.abs(X_full.T @ y)) / n
        lambda_min = lambda_max * eps
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), nlambda)

        # Compute lasso path using sklearn (matches glmnet coordinate descent)
        try:
            _, coef_path, _ = lasso_path(X_full, y, alphas=lambdas, max_iter=10000)
        except Exception:
            # Fallback to simpler grid if path computation fails
            _, coef_path, _ = lasso_path(X_full, y, n_alphas=nlambda, max_iter=10000)
            lambdas = np.logspace(
                np.log10(lambda_max), np.log10(lambda_min), coef_path.shape[1]
            )

        # Find entry times (lambda at which each feature first becomes nonzero)
        Z = np.zeros(p)  # Entry times for original features
        Z_k = np.zeros(p)  # Entry times for knockoff features

        for j in range(p):
            # Original feature entry time (multiply by n to match R's glmnet behavior)
            nonzero = np.where(np.abs(coef_path[j, :]) > 1e-10)[0]
            Z[j] = lambdas[nonzero[0]] * n if len(nonzero) > 0 else 0

            # Knockoff feature entry time (multiply by n to match R's glmnet behavior)
            nonzero_k = np.where(np.abs(coef_path[p + j, :]) > 1e-10)[0]
            Z_k[j] = lambdas[nonzero_k[0]] * n if len(nonzero_k) > 0 else 0

        # Compute signed max statistic: W_j = max(Z_j, Z_k_j) * sign(Z_j - Z_k_j)
        W = np.maximum(Z, Z_k) * np.sign(Z - Z_k)

        # Adjust signs based on swap (matching R's behavior)
        W = W * (1 - 2 * swap)

        return W

    @staticmethod
    def _knockoff_threshold(W, fdr, offset=1):
        """Compute knockoff threshold with configurable offset.

        Parameters
        ----------
        W : np.ndarray
            Knockoff W statistics.
        fdr : float
            Target false discovery rate.
        offset : int
            0 for original knockoff (more power, controls modified FDR),
            1 for knockoff+ (conservative, controls FDR).

        Returns
        -------
        float
            Threshold value, or np.inf if no threshold satisfies FDR.
        """
        # Get candidate thresholds: 0 and absolute values of W (ascending order)
        # NOTE: Must include 0 to match R's knockoff.threshold behavior
        W_abs = np.abs(W)
        candidates = np.sort(np.concatenate([[0], W_abs]))

        # Find minimum threshold that satisfies FDR constraint
        threshold = np.inf
        for t in candidates:
            numerator = offset + np.sum(W <= -t)
            denominator = max(1, np.sum(W >= t))
            if numerator / denominator <= fdr:
                threshold = t
                break
        return threshold

    @staticmethod
    def filter_knockoffs_iterative(
        z,
        y,
        fdr=0.1,
        niter=1,
        spec=0.2,
        n_workers=-1,
        backend="python",
        method="asdp",
        shrink=False,
        offset=0,
        fstat="glmnet_lambdasmax",
    ):
        """
        Run knockoff filter to find significant variables.

        Parameters
        ----------
        z : np.ndarray
            Feature matrix.
        y : np.ndarray
            Response vector.
        fdr : float
            Target false discovery rate.
        niter : int
            Number of knockoff iterations.
        spec : float
            Proportion threshold for selection frequency.
        n_workers : int
            Number of parallel workers. -1 uses all available cores. Default: -1.
        backend : str
            Which knockoff implementation: 'python' (default) or 'r'.
            - 'python': Pure Python implementation (bundled knockoff-filter)
            - 'r': R knockoff package via rpy2 (requires rpy2)
        method : str
            Knockoff construction method: 'asdp' (default), 'sdp', or 'equi'.
            - 'asdp': Approximate SDP, faster for high-dimensional data
            - 'sdp': Full semidefinite programming, more power but slower
            - 'equi': Equicorrelated, always works but lower power
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage (Python backend only).
        offset : int
            Knockoff procedure offset:
            - 0: Original knockoff (more power, controls modified FDR) - R default
            - 1: Knockoff+ (conservative, controls exact FDR)
        fstat : str
            Feature statistic method:
            - 'glmnet_lambdasmax': Signed max of glmnet lasso path (default, matches R)
            - 'glmnet_lambdadiff': Lambda difference statistic
            - 'glmnet_coefdiff': Coefficient difference at CV-selected lambda

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        if backend == "python":
            return Knockoffs.filter_knockoffs_iterative_python(
                z,
                y,
                fdr=fdr,
                niter=niter,
                spec=spec,
                method=method,
                shrink=shrink,
                offset=offset,
                fstat=fstat,
                n_jobs=n_workers,
            )
        elif backend == "r":
            return Knockoffs.filter_knockoffs_iterative_r(
                z, y, fdr=fdr, niter=niter, spec=spec
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'python' or 'r'.")

    def fit_linear(self, z_matrix, y):
        """fit z-matrix in linear part to get LP"""
        reg = self.model.fit(z_matrix, y)

        LP = reg.predict(z_matrix)
        beta = reg.coef_

        return LP, beta

    @staticmethod
    def select_short_freq(
        z,
        y,
        spec=0.1,
        fdr=0.1,
        niter=1000,
        f_size=100,
        n_workers=-1,
        backend="python",
        method="asdp",
        shrink=False,
        offset=0,
        fstat="glmnet_lambdasmax",
        verbose=False,
        **kwargs,
    ):
        """
        Find significant variables using second order knockoffs across subsets of features.

        Parameters
        ----------
        z : np.ndarray or pandas.DataFrame
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray or pandas.DataFrame
            Response vector of shape (n_samples,)
        spec : float
            Proportion threshold to consider a variable frequently selected
        fdr : float
            Target false discovery rate
        niter : int
            Number of knockoff iterations
        f_size : int
            Target size for each feature subset
        n_workers : int
            Number of parallel workers. -1 uses all available cores. Default: -1.
        backend : str
            Which knockoff implementation: 'python' (default), 'r', or 'r_knockoffs'.
            - 'python': Pure Python implementation with SLIDE voting (bundled knockoff-filter)
            - 'r': Full R knockoff pipeline via rpy2 (requires rpy2)
            - 'r_knockoffs': R knockoff generation + Python SLIDE voting (best R concordance)
        method : str
            Knockoff construction method: 'asdp' (default), 'sdp', or 'equi'.
            - 'asdp': Approximate SDP, faster for high-dimensional data
            - 'sdp': Full semidefinite programming, more power but slower
            - 'equi': Equicorrelated, always works but lower power
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage (Python backend only).
        offset : int
            Knockoff procedure offset:
            - 0: Original knockoff (more power, controls modified FDR) - R default
            - 1: Knockoff+ (conservative, controls exact FDR)
        fstat : str
            Feature statistic method:
            - 'glmnet_lambdasmax': Signed max of glmnet lasso path (default, matches R)
            - 'glmnet_lambdadiff': Lambda difference statistic
            - 'glmnet_coefdiff': Coefficient difference at CV-selected lambda
        verbose : bool
            Print progress information. Default: False.
        **kwargs
            Additional arguments passed to knockoff_filter_voting_slide.

        Returns
        -------
        np.ndarray
            Array of selected variable indices

        Notes
        -----
        Backend comparison:
        - 'python': Pure Python, uses SLIDE voting with findOptIter refinement
        - 'r_knockoffs': R knockoff generation + Python SLIDE voting (best R concordance)
        - 'r': Full R pipeline (legacy, no findOptIter refinement)

        The 'r_knockoffs' backend typically achieves ~0.72 mean Jaccard similarity
        with R native, while 'python' achieves ~0.65. The difference is primarily
        in knockoff matrix generation, not the voting logic.
        """
        # Route python and r_knockoffs through select_short_freq_slide for
        # consistent SLIDE methodology (findOptIter,
        # feature chunking, two-stage screening).
        if backend in ("r_knockoffs", "python"):
            result = Knockoffs.select_short_freq_slide(
                z,
                y,
                spec=spec,
                fdr=fdr,
                niter=niter,
                f_size=f_size,
                n_jobs=n_workers,
                backend=backend,
                verbose=verbose,
                **kwargs,
            )
            return result.selected
        z = Knockoffs.scale_features(z)
        y = y.copy()

        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values

        if isinstance(z, pd.DataFrame):
            z = z.values

        n_features = z.shape[1]
        n_splits = math.ceil(n_features / f_size)
        feature_split = math.ceil(n_features / n_splits)
        feature_starts = list(range(0, n_features, feature_split))
        feature_stops = [
            min(start + feature_split, n_features) for start in feature_starts
        ]

        screen_var = []

        for start, stop in tqdm(
            zip(feature_starts, feature_stops),
            total=len(feature_starts),
            desc="Processing subsets",
        ):
            subset_z = z[:, start:stop]

            selected_indices = Knockoffs.filter_knockoffs_iterative(
                subset_z,
                y,
                fdr=fdr,
                niter=niter,
                spec=spec,
                n_workers=n_workers,
                backend=backend,
                method=method,
                shrink=shrink,
                offset=offset,
                fstat=fstat,
            )

            selected_indices = selected_indices + start

            if len(selected_indices) > 0:
                screen_var.extend(selected_indices)

        screen_var = np.array(screen_var)

        if n_splits > 1 and len(screen_var) > 1:
            subset_z = z[:, screen_var]
            final_var = Knockoffs.filter_knockoffs_iterative(
                subset_z,
                y,
                fdr=fdr,
                niter=niter,
                spec=spec,
                n_workers=n_workers,
                backend=backend,
                method=method,
                shrink=shrink,
                offset=offset,
                fstat=fstat,
            )
            final_var = screen_var[final_var]
        else:
            final_var = screen_var

        return final_var

    @staticmethod
    def select_short_freq_slide(
        z,
        y,
        spec=0.1,
        fdr=0.1,
        niter=1000,
        f_size=100,
        n_jobs=-1,
        backend="python",
        verbose=False,
        slide_selection=True,
        **kwargs,
    ) -> VotingResult:
        """
        Full SLIDE voting procedure with optional findOptIter refinement.

        This method implements the complete SLIDE voting procedure:
        1. Feature chunking (f_size parameter)
        2. Knockoff voting on each chunk
        3. Two-stage screening when multiple chunks
        4. Optional findOptIter refinement (slide_selection=True)

        Parameters
        ----------
        z : np.ndarray or pandas.DataFrame
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray or pandas.DataFrame
            Response vector of shape (n_samples,)
        spec : float
            Proportion threshold to consider a variable frequently selected.
            Default: 0.1 (matches R SLIDE).
        fdr : float
            Target false discovery rate. Default: 0.1.
        niter : int
            Number of knockoff iterations. Default: 1000 (matches R SLIDE).
        f_size : int
            Maximum features per chunk. Default: 100 (matches R SLIDE).
        n_jobs : int
            Number of parallel jobs. -1 uses all available cores.
            Note: Use n_jobs=1 when backend='r_knockoffs' (R is not thread-safe).
        backend : str
            Knockoff generation backend:
            - 'python': Pure Python knockoff generation (default)
            - 'r_knockoffs': R knockoff generation + Python voting (best R concordance)
        verbose : bool
            Print progress information. Default: False.
        slide_selection : bool
            Apply R's findOptIter refinement, which discards the frequency
            table and reports the single draw with maximal overlap with the
            frequent set. Default: True, matching native-R SLIDE. Set False to
            report the frequency (stability) set instead; that lowers
            run-to-run variance within a backend but measurably widens
            cross-backend divergence (see docs/backend-benchmark.md).
        **kwargs
            Additional arguments passed to knockoff_filter_voting_slide.

        Returns
        -------
        VotingResult
            Result object containing:
            - selection_counts: Number of times each variable was selected
            - selection_frequency: Proportion of runs where each variable was selected
            - selected: Indices of selected variables
            - threshold: The spec threshold used
            - niter: Number of iterations
            - spec: Specificity threshold
            - min_selections: Minimum selections required
            - optimal_iter: Index of the optimal iteration (from findOptIter)

        Notes
        -----
        Backend comparison:
        - 'python': Uses Python's create_second_order() for knockoff generation.
          Achieves ~0.65 mean Jaccard similarity with R native.
        - 'r_knockoffs': Uses R's knockoff::create.second_order() for knockoff
          generation, combined with Python's SLIDE voting and findOptIter.
          Achieves ~0.72 mean Jaccard similarity with R native.

        The divergence between Python and R is primarily in knockoff matrix
        generation (SDP solver differences), not in the voting logic.

        Examples
        --------
        >>> from loveslide import Knockoffs
        >>> import numpy as np
        >>> Z = np.random.randn(100, 500)
        >>> y = np.random.randn(100)
        >>> # Pure Python (default)
        >>> result = Knockoffs.select_short_freq_slide(Z, y, backend='python')
        >>> print(f"Selected: {result.selected}")
        >>> # R knockoffs + Python voting (best R concordance)
        >>> result = Knockoffs.select_short_freq_slide(Z, y, backend='r_knockoffs')
        >>> print(f"Selected: {result.selected}")
        """
        from .knockoff.stats import stat_glmnet_lambdasmax

        # Prepare data
        z = Knockoffs.scale_features(z)

        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.copy()
        else:
            y = np.asarray(y).copy()

        if isinstance(z, pd.DataFrame):
            z = z.values

        # Determine knockoffs callable based on backend
        if backend == "r_knockoffs":
            # Solve once, then use R's create.gaussian for every cached draw.
            knockoffs_func = None
            use_cache = kwargs.pop("use_cache", True)
            kwargs["sdp_solver"] = _solve_sdp_r
            kwargs["knockoff_sampler"] = _sample_gaussian_r
        elif backend == "python":
            knockoffs_func = None  # Use default create_second_order
            use_cache = kwargs.pop("use_cache", True)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'python' or 'r_knockoffs'."
            )

        # Call the full SLIDE voting implementation
        result = knockoff_filter_voting_slide(
            z,
            y,
            knockoffs=knockoffs_func,
            statistic=stat_glmnet_lambdasmax,
            fdr=fdr,
            offset=0,  # R default
            niter=niter,
            spec=spec,
            f_size=f_size,
            n_jobs=n_jobs,
            verbose=verbose,
            match_r=True,
            use_cache=use_cache,
            slide_selection=slide_selection,
            **kwargs,
        )

        return result
