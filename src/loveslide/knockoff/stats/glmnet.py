"""GLMNet-based statistics for knockoff filter."""

from typing import Dict, Optional, Tuple, Union
import logging
import math
import warnings
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import swap_columns, correct_for_swap, compute_difference_stat, compute_signed_max_stat

logger = logging.getLogger(__name__)

# Try to import vendored glmnet (Fortran-based) for better performance
try:
    from loveslide.knockoff._vendor.glmnet import ElasticNet, LogitNet
    HAS_GLMNET = True
except ImportError as e:
    HAS_GLMNET = False
    # Module import runs once per process, so this warns once.
    logger.warning(
        f"Vendored Fortran glmnet not available ({e}); knockoff statistics will use "
        "the sklearn fallback, which does not match R's glmnet. The bundled "
        "_glmnet extension is prebuilt for CPython 3.9/3.11 on Linux x86_64; rebuild "
        "it with `pixi run build-glmnet` for other platforms."
    )


def _r_scale(X: np.ndarray, *, extended_precision: bool = False) -> tuple:
    """
    Standardize X using R's scale() behavior: center and scale with ddof=1.

    R's scale() uses sample standard deviation (ddof=1), while sklearn's
    StandardScaler uses population standard deviation (ddof=0). This causes
    significant differences in knockoff W-statistics.

    Returns:
        X_scaled: Standardized array
        means: Column means
        stds: Column standard deviations (ddof=1)
    """
    if extended_precision:
        # R colMeans/sum accumulate in LDOUBLE, then return doubles.
        means = np.mean(X, axis=0, dtype=np.longdouble).astype(np.float64)
        centered = X - means
        squares = np.sum(centered * centered, axis=0, dtype=np.longdouble).astype(np.float64)
        stds = np.sqrt(squares / (X.shape[0] - 1))
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0, ddof=1)  # Preserve non-Gaussian arithmetic.
    # Handle zero std columns (constant columns)
    stds = np.where(stds == 0, 1.0, stds)
    X_scaled = (X - means) / stds
    return X_scaled, means, stds


def _lasso_max_lambda_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    nlambda: int = 500,
    intercept: bool = True,
    standardize: bool = True,  # Match R: always standardize X with scale() (ddof=1)
    family: str = 'gaussian',
    use_sklearn: bool = False,  # Force sklearn fallback for R-compatibility
    return_diagnostics: bool = False,
    _preparation_started: Optional[float] = None,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Compute the maximum lambda at which each variable enters the lasso model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    nlambda : int, default=500
        Number of lambda values.
    intercept : bool, default=True
        Whether to fit an intercept.
    standardize : bool, default=True
        Whether to standardize features.
    family : str, default='gaussian'
        Response family.
    use_sklearn : bool, default=False
        Force sklearn lasso_path instead of vendored Fortran glmnet.
        Set to True for better R-compatibility (sklearn produces
        W-statistics with positive correlation to R's knockoff package,
        while Fortran glmnet can produce negative correlation due to
        different tie-breaking behavior).

    Returns
    -------
    np.ndarray of shape (p,)
        Maximum lambda values for each variable.
    """
    try:
        from sklearn.linear_model import lasso_path, LogisticRegression
    except ImportError:
        raise ImportError("scikit-learn is required for lasso statistics")

    n, p = X.shape

    # Standardize using R's scale() behavior (ddof=1) if requested
    preparation_started = _preparation_started or time.perf_counter()
    # This matches R's knockoff package: lasso_max_lambda_glmnet
    if standardize:
        X_std, _, stds = _r_scale(X, extended_precision=family == 'gaussian')
    else:
        X_std = X
        stds = None

    if family == 'gaussian':
        # Match R's behavior: don't center y explicitly, let intercept handle it
        # R does: glmnet(scale(X), y, intercept=T, standardize=F)
        # When X is centered (via _r_scale), centering y vs using intercept gives
        # equivalent coefficient paths (only intercept term differs)
        y_work = y if intercept else (y - y.mean())

        # Generate lambda sequence (decreasing order)
        # Computed on standardized X to match R's behavior
        # Note: For centered X, X_std.T @ y == X_std.T @ (y - mean(y))
        lambda_max = np.max(np.abs(X_std.T @ y_work)) / n
        lambda_min = lambda_max / 2000
        k = np.arange(nlambda) / nlambda
        # R uses scalar libm pow; NumPy's vectorized power can round differently.
        lambdas = lambda_max * np.fromiter(
            (math.pow(lambda_min / lambda_max, exponent) for exponent in k),
            dtype=np.float64, count=nlambda,
        )
        from ..filter import _profile_stage
        _profile_stage("preparation", preparation_started)
        lasso_started = time.perf_counter()

        if HAS_GLMNET and not use_sklearn:
            # Use vendored Fortran glmnet for ~5-10x speedup
            # Pass standardized X with standardize=False (like R's knockoff package)
            # Note: Fortran glmnet can produce ties where sklearn doesn't,
            # leading to negative correlation with R's knockoff package.
            # Use use_sklearn=True for better R-compatibility.
            model = ElasticNet(
                alpha=1.0,  # Pure lasso
                n_lambda=nlambda,
                lambda_path=lambdas,  # Pre-specified, decreasing
                standardize=False,    # Already standardized above (R-style)
                fit_intercept=intercept,  # Match R: intercept=T
                n_splits=0,           # No CV
                tol=1e-7,
                max_iter=100000,
            )
            model.fit(X_std, y_work)
            alphas = model.lambda_path_
            coefs = model.coef_path_  # (p, n_lambda)
        else:
            # Fallback to sklearn
            # sklearn's lasso_path handles intercept internally
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    alphas, coefs, _ = lasso_path(X_std, y_work, alphas=lambdas, max_iter=10000)
                except Exception:
                    # Fallback: let sklearn choose alphas
                    alphas, coefs, _ = lasso_path(X_std, y_work, n_alphas=nlambda, max_iter=10000)
        _profile_stage("lasso_path", lasso_started)

        # coefs has shape (p, n_alphas)
        # Find first nonzero entry for each variable (vectorized)
        nonzero_mask = np.abs(coefs) > 0
        first_nonzero_idx = nonzero_mask.argmax(axis=1)
        has_nonzero = nonzero_mask.any(axis=1)
        # alphas are sorted in decreasing order
        lambda_entry = np.where(has_nonzero, alphas[first_nonzero_idx] * n, 0.0)

        if return_diagnostics:
            return lambda_entry, {
                'standardized_design': X_std,
                'lambda': np.asarray(alphas),
                'coefficients': np.asarray(coefs),
            }
        return lambda_entry

    elif family == 'binomial':
        # R's knockoff calls glmnet(X, y, family="binomial") with default args:
        #   nlambda=100, lambda.min.ratio = ifelse(nobs<nvars, 0.01, 1e-4)
        # It does NOT pre-compute an explicit grid like the gaussian case.
        min_lambda_ratio = 0.01 if n < p else 1e-4

        if HAS_GLMNET and not use_sklearn:
            try:
                model = LogitNet(
                    alpha=1.0,  # Pure lasso
                    n_lambda=100,  # R's glmnet default
                    min_lambda_ratio=min_lambda_ratio,
                    lambda_path=None,  # Let Fortran auto-generate (matches R)
                    standardize=False,    # Already standardized above (R-style)
                    fit_intercept=intercept,
                    n_splits=0,           # No CV
                    tol=1e-7,
                    max_iter=100000,
                )
                model.fit(X_std, y)
                alphas = model.lambda_path_
                # LogitNet coef_path_ shape: (1, p, n_lambda) for binary
                coefs = model.coef_path_.squeeze(axis=0)  # -> (p, n_lambda)

                # Find first nonzero entry for each variable (vectorized)
                nonzero_mask = np.abs(coefs) > 0
                first_nonzero_idx = nonzero_mask.argmax(axis=1)
                has_nonzero = nonzero_mask.any(axis=1)
                lambda_entry = np.where(has_nonzero, alphas[first_nonzero_idx] * n, 0.0)

                return lambda_entry
            except Exception as e:
                logger.debug(f"LogitNet failed, falling back to sklearn: {e}")
                # Fall through to sklearn fallback

        # sklearn fallback: generate lambda grid matching R's glmnet defaults
        from sklearn.linear_model import LogisticRegression
        try:
            lambda_max = np.max(np.abs(X_std.T @ (y - y.mean()))) / n
            lambda_min = lambda_max * min_lambda_ratio
            n_alphas = 100
            k = np.arange(n_alphas) / n_alphas
            lambdas = lambda_max * (lambda_min / lambda_max) ** k

            Cs = 1 / (lambdas + 1e-10)
            lambda_entry = np.zeros(p)
            for i, C in enumerate(Cs):
                clf = LogisticRegression(
                    penalty='l1', C=C, solver='saga',
                    fit_intercept=intercept, max_iter=1000
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X_std, y)

                coefs = clf.coef_.ravel()
                newly_entered = (np.abs(coefs) > 0) & (lambda_entry == 0)
                lambda_entry[newly_entered] = lambdas[i] * n

            return lambda_entry

        except Exception as e:
            warnings.warn(f"Logistic regression path failed: {e}")
            return np.zeros(p)

    else:
        raise ValueError(f"Unsupported family: {family}")


def _cv_coeffs_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    nlambda: int = 250,
    intercept: bool = True,
    family: str = 'gaussian',
    n_jobs: int = -1,
    **kwargs
) -> np.ndarray:
    """
    Compute coefficients at CV-selected lambda.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    nlambda : int, default=250
        Number of lambda values.
    intercept : bool, default=True
        Whether to fit an intercept.
    family : str, default='gaussian'
        Response family.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    np.ndarray
        Coefficients at optimal lambda.
    """
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n, p = X.shape

    if family == 'gaussian':
        from sklearn.linear_model import LassoCV

        # Generate lambda sequence
        lambda_max = np.max(np.abs(X.T @ y)) / n
        lambda_min = lambda_max / 2000
        k = np.arange(nlambda) / nlambda
        alphas = lambda_max * (lambda_min / lambda_max) ** k

        # Fit LassoCV
        cv = LassoCV(alphas=alphas, fit_intercept=intercept, n_jobs=n_jobs, cv=10, max_iter=10000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.fit(X, y)

        # Return coefficients with intercept
        if intercept:
            return np.concatenate([[cv.intercept_], cv.coef_])
        else:
            return np.concatenate([[0], cv.coef_])

    elif family == 'binomial':
        from sklearn.linear_model import LogisticRegressionCV

        # Fit LogisticRegressionCV
        cv = LogisticRegressionCV(
            penalty='l1', solver='saga',
            fit_intercept=intercept,
            n_jobs=n_jobs, cv=10, max_iter=1000
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.fit(X, y)

        # Return coefficients with intercept
        if intercept:
            return np.concatenate([cv.intercept_, cv.coef_.ravel()])
        else:
            return np.concatenate([[0], cv.coef_.ravel()])

    elif family == 'poisson':
        # Poisson regression with CV
        try:
            from sklearn.linear_model import PoissonRegressor
            from sklearn.model_selection import GridSearchCV

            # Simple grid search for alpha
            alphas = np.logspace(-4, 2, 20)
            best_score = -np.inf
            best_coef = np.zeros(p)

            for alpha in alphas:
                model = PoissonRegressor(alpha=alpha, fit_intercept=intercept, max_iter=1000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X, y)
                    score = model.score(X, y)
                    if score > best_score:
                        best_score = score
                        if intercept:
                            best_coef = np.concatenate([[model.intercept_], model.coef_])
                        else:
                            best_coef = np.concatenate([[0], model.coef_])

            return best_coef

        except Exception as e:
            warnings.warn(f"Poisson regression failed: {e}")
            return np.zeros(p + 1)

    elif family == 'multinomial':
        from sklearn.linear_model import LogisticRegressionCV

        cv = LogisticRegressionCV(
            penalty='l1', solver='saga',
            multi_class='multinomial',
            fit_intercept=intercept,
            n_jobs=n_jobs, cv=10, max_iter=1000
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.fit(X, y)

        # For multinomial, sum absolute coefficients across classes
        coefs_sum = np.sum(np.abs(cv.coef_), axis=0)
        if intercept:
            return np.concatenate([[0], coefs_sum])
        else:
            return np.concatenate([[0], coefs_sum])

    else:
        raise ValueError(f"Unsupported family: {family}")


def stat_glmnet_lambdadiff(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    use_sklearn: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GLM lambda difference statistic.

    Computes W_j = Z_j - Z_{j+p} where Z is the maximum lambda at which
    each variable enters the model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector.
    family : str, default='gaussian'
        Response family ('gaussian', 'binomial', 'poisson', 'multinomial').
    use_sklearn : bool, default=False
        Force sklearn lasso_path for better R-compatibility.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)
    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate and compute statistics
    X_combined = np.hstack([X_swap, Xk_swap])
    Z = _lasso_max_lambda_glmnet(X_combined, y, family=family, use_sklearn=use_sklearn, **kwargs)

    # Compute difference statistic
    W = compute_difference_stat(Z, p)

    # Correct for swapping
    return correct_for_swap(W, swap)


def stat_glmnet_lambdasmax(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    use_sklearn: bool = False,
    swap: Optional[np.ndarray] = None,
    return_diagnostics: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    GLM signed maximum lambda statistic.

    Computes W_j = max(Z_j, Z_{j+p}) * sign(Z_j - Z_{j+p}).

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector.
    family : str, default='gaussian'
        Response family.
    use_sklearn : bool, default=False
        Force sklearn lasso_path for better R-compatibility.
        The vendored Fortran glmnet can produce ties where sklearn doesn't,
        leading to negative W-statistic correlation with R's knockoff package.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)
    p = X.shape[1]
    preparation_started = time.perf_counter()

    # Randomly swap columns
    if swap is None:
        X_swap, Xk_swap, swap = swap_columns(X, X_k)
    else:
        swap = np.asarray(swap, dtype=bool)
        if swap.shape != (p,):
            raise ValueError(f"swap must have shape ({p},), got {swap.shape}")
        X_swap = np.where(swap[np.newaxis, :], X_k, X)
        Xk_swap = np.where(swap[np.newaxis, :], X, X_k)

    # Concatenate and compute statistics
    X_combined = np.hstack([X_swap, Xk_swap])
    result = _lasso_max_lambda_glmnet(
        X_combined,
        y,
        family=family,
        use_sklearn=use_sklearn,
        return_diagnostics=return_diagnostics,
        _preparation_started=preparation_started,
        **kwargs,
    )
    if return_diagnostics:
        Z, diagnostics = result
    else:
        Z = result

    # Compute signed max statistic
    W = compute_signed_max_stat(Z, p)

    # Correct for swapping
    W = correct_for_swap(W, swap)
    if return_diagnostics:
        diagnostics.update({
            'entry_lambda_swapped': Z,
            'swap': swap,
            'W': W,
        })
        return W, diagnostics
    return W


def stat_glmnet_coefdiff(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    cores: int = 2,
    **kwargs
) -> np.ndarray:
    """
    GLM coefficient difference statistic with cross-validation.

    Computes W_j = |Z_j| - |Z_{j+p}| where Z are coefficients at
    CV-selected lambda.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector.
    family : str, default='gaussian'
        Response family.
    cores : int, default=2
        Number of CPU cores for parallel CV.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)
    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate and compute statistics
    X_combined = np.hstack([X_swap, Xk_swap])
    glmnet_coefs = _cv_coeffs_glmnet(
        X_combined, y, family=family, n_jobs=cores, **kwargs
    )

    # Extract coefficients (skip intercept)
    if family == 'multinomial':
        # Already handled in _cv_coeffs_glmnet
        Z = np.abs(glmnet_coefs[1:2*p+1])
    elif family == 'cox':
        Z = glmnet_coefs[:2*p]
    else:
        Z = glmnet_coefs[1:2*p+1]

    # Compute absolute difference statistic
    orig = np.arange(p)
    W = np.abs(Z[orig]) - np.abs(Z[orig + p])

    # Correct for swapping
    return correct_for_swap(W, swap)
