"""GLMNet-based statistics for knockoff filter."""

from typing import Optional
import logging
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import swap_columns, correct_for_swap, compute_difference_stat, compute_signed_max_stat

logger = logging.getLogger(__name__)

# Try to import vendored glmnet (Fortran-based) for better performance
try:
    from loveslide.knockoff._vendor.glmnet import ElasticNet, LogitNet
    HAS_GLMNET = True
except ImportError:
    HAS_GLMNET = False
    logger.debug("Vendored glmnet not available, using sklearn fallback")


def _r_scale(X: np.ndarray) -> tuple:
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
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)  # R uses ddof=1 (sample std)
    # Handle zero std columns (constant columns)
    stds = np.where(stds == 0, 1.0, stds)
    X_scaled = (X - means) / stds
    return X_scaled, means, stds


def _lasso_max_lambda_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    intercept: bool = True,
    standardize: bool = True,
    family: str = 'gaussian',
    use_sklearn: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Compute the maximum lambda at which each variable enters the lasso model.

    Matches R's knockoff:::lasso_max_lambda_glmnet exactly: computes an
    explicit 500-point log-spaced lambda grid with lambda_max/2000 ratio,
    then passes it to glmnet(X, y, lambda=lambda, standardize=FALSE).

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    intercept : bool, default=True
        Whether to fit an intercept.
    standardize : bool, default=True
        Whether to standardize features using R's scale() (ddof=1).
    family : str, default='gaussian'
        Response family.
    use_sklearn : bool, default=False
        Force sklearn lasso_path instead of vendored Fortran glmnet.

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
    # This matches R's knockoff package: lasso_max_lambda_glmnet
    if standardize:
        X_std, _, stds = _r_scale(X)
    else:
        X_std = X
        stds = None

    if family == 'gaussian':
        # Match R's behavior: don't center y explicitly, let intercept handle it
        # R does: glmnet(scale(X), y, intercept=T, standardize=F)
        y_work = y if intercept else (y - y.mean())

        # Match R's knockoff:::lasso_max_lambda_glmnet exactly:
        #   lambda_max = max(abs(t(X) %*% y)) / n
        #   lambda_min = lambda_max / 2000
        #   k = (0:(nlambda-1)) / nlambda   # nlambda = 500
        #   lambda = lambda_max * (lambda_min/lambda_max)^k
        nlambda = 500
        lambda_max = np.max(np.abs(X_std.T @ y_work)) / n
        lambda_min = lambda_max / 2000
        k = np.arange(nlambda) / nlambda
        lambdas = lambda_max * (lambda_min / lambda_max) ** k

        if HAS_GLMNET and not use_sklearn:
            # Pass explicit lambda grid to match R's knockoff package.
            # R calls: glmnet(X, y, lambda=lambda, intercept=T, standardize=F)
            model = ElasticNet(
                alpha=1.0,  # Pure lasso
                lambda_path=lambdas,  # Explicit grid (matches R)
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
            # sklearn fallback with same R-matching lambda grid
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    alphas, coefs, _ = lasso_path(X_std, y_work, alphas=lambdas, max_iter=10000)
                except Exception:
                    alphas, coefs, _ = lasso_path(X_std, y_work, n_alphas=nlambda, max_iter=10000)

        # coefs has shape (p, n_alphas)
        # Find first nonzero entry for each variable (vectorized)
        nonzero_mask = np.abs(coefs) > 0
        first_nonzero_idx = nonzero_mask.argmax(axis=1)
        has_nonzero = nonzero_mask.any(axis=1)
        # alphas are sorted in decreasing order
        lambda_entry = np.where(has_nonzero, alphas[first_nonzero_idx] * n, 0.0)

        return lambda_entry

    elif family == 'binomial':
        # R's glmnet default: lambda.min.ratio = ifelse(nobs<nvars, 0.01, 1e-4)
        min_lambda_ratio = 0.01 if n < p else 1e-4
        if HAS_GLMNET and not use_sklearn:
            # Let Fortran glmnet auto-generate lambda grid (matching R)
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
                model.fit(X_std, y)  # Pass standardized X
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
            n_alphas = 100  # Match R's nlambda default
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
    **kwargs
) -> np.ndarray:
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

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate and compute statistics
    X_combined = np.hstack([X_swap, Xk_swap])
    Z = _lasso_max_lambda_glmnet(X_combined, y, family=family, use_sklearn=use_sklearn, **kwargs)

    # Compute signed max statistic
    W = compute_signed_max_stat(Z, p)

    # Correct for swapping
    return correct_for_swap(W, swap)


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
