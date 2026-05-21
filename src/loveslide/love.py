"""
Python wrapper for the LOVE algorithm.
Supports both pure Python and R backends.
"""

import os
import warnings
import numpy as np

from .love_python.love import LOVE
from .knockoffs import _rlist_get, _import_rpy2


def call_love_r(X, lbd=0.5, delta=None, thresh_fdr=0.2, rep_CV=50,
                alpha_level=0.05, verbose=False, **kwargs):
    """
    Call R LOVE implementation via rpy2.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix (n samples x p features).
    lbd : float
        Lambda parameter for thresholding. Default is 0.5.
    delta : float or list
        Delta threshold value(s). Required.
    thresh_fdr : float
        FDR threshold for correlation matrix. Default is 0.2.
    rep_CV : int
        Number of CV replicates for delta selection. Default is 50.
    alpha_level : float
        Alpha level for confidence intervals. Default is 0.05.
    verbose : bool
        Print progress. Default is False.
    **kwargs : dict
        Additional arguments (ignored for compatibility).

    Returns
    -------
    dict
        Dictionary with LOVE results matching Python format.
    """
    robjects = _import_rpy2()
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import get_conversion, localconverter

    # Use context-manager conversion (activate/deactivate removed in rpy2 >= 3.5)
    np_cv = numpy2ri.converter + get_conversion()
    with localconverter(np_cv):
        # Get path to R scripts
        r_script_dir = os.path.join(os.path.dirname(__file__), 'slide_r')

        # Source all R scripts in the slide_r directory
        r_source = robjects.r['source']
        import glob
        for script_path in glob.glob(os.path.join(r_script_dir, '*.R')):
            r_source(script_path)

        # Get the R function
        getLatentFactors = robjects.globalenv['getLatentFactors']

        # Convert delta to R vector
        if delta is None:
            delta = 0.1
        if not hasattr(delta, '__iter__'):
            delta = [delta]
        delta_r = robjects.FloatVector(delta)

        # Call R function
        result = getLatentFactors(
            x=X,
            delta=delta_r,
            thresh_fdr=thresh_fdr,
            lbd=lbd,
            rep_cv=rep_CV,
            alpha_level=alpha_level,
            verbose=verbose
        )

        # Convert R result to Python dict (version-resilient named-list access)
        result_dict = {
            'K': int(_rlist_get(result, 'K')[0]),
            'pureVec': np.array(_rlist_get(result, 'pureVec')) - 1,  # R is 1-indexed
            'pureInd': _convert_r_pure_ind(_rlist_get(result, 'purInd')),
            'group': None,  # Not returned by R function
            'A': np.array(_rlist_get(result, 'A')),
            'C': np.array(_rlist_get(result, 'C')),
            'Omega': None,  # Not returned by R function
            'Gamma': np.array(_rlist_get(result, 'Gamma')),
            'optDelta': delta[0] if len(delta) == 1 else delta[0],
        }

    return result_dict


def _convert_r_pure_ind(r_list):
    """Convert R pureInd list to Python format."""
    robjects = _import_rpy2()
    result = []
    for i in range(len(r_list)):
        item = r_list[i]
        pos_r = _rlist_get(item, 'pos')
        neg_r = _rlist_get(item, 'neg')
        pos = np.array(pos_r) - 1 if pos_r != robjects.NULL else np.array([])
        neg = np.array(neg_r) - 1 if neg_r != robjects.NULL else np.array([])
        result.append({'pos': pos.astype(int), 'neg': neg.astype(int)})
    return result


def call_love(X, lbd=0.5, mu=0.5, est_non_pure_row="HT", thresh_fdr=0.2, verbose=False,
              pure_homo=False, diagonal=False, delta=None, merge=False,
              rep_CV=50, ndelta=50, q=2, exact=False, max_pure=None, nfolds=10,
              outpath='.', backend='python', **kwargs):
    """
    Python wrapper for calling the LOVE function.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input data matrix (n samples x p features).
    lbd : float, optional
        Leading constant for lambda (precision estimation). Default is 0.5.
    mu : float, optional
        Leading constant for thresholding the loading matrix. Default is 0.5.
    est_non_pure_row : str, optional
        Method for non-pure rows: "HT", "ST", or "Dantzig". Default is "HT".
    thresh_fdr : float, optional
        Deprecated. Was used in R implementation only.
    verbose : bool, optional
        If True, print progress. Default is False.
    pure_homo : bool, optional
        If True, pure loadings have the same magnitude. Default is False.
    diagonal : bool, optional
        If True, covariance matrix of Z is diagonal. Default is False.
    delta : float, list, or np.ndarray, optional
        Grid of delta values for thresholding. Default is None.
    merge : bool, optional
        If True, take union of candidate pure variables; else intersection.
        Default is False.
    rep_CV : int, optional
        Number of repetitions for cross validation. Default is 50.
    ndelta : int, optional
        Length of delta grid. Default is 50.
    q : int, optional
        Type of score (2 or np.inf). Default is 2.
    exact : bool, optional
        Compute Inf score exactly via LP. Default is False.
    max_pure : float, optional
        Max proportion of pure variables. Default is None.
    nfolds : int, optional
        Number of folds for cross-validation. Default is 10.
    outpath : str, optional
        Output path (used by R backend only). Default is '.'.
    backend : str, optional
        Which LOVE implementation to use: 'python' (default) or 'r'.
        The 'r' backend requires rpy2 and R with the LOVE scripts.
    **kwargs : dict
        Additional keyword arguments passed to backend.

    Returns
    -------
    dict
        Dictionary containing the results from the LOVE analysis:
        - K: int, estimated number of clusters
        - pureVec: np.ndarray, indices of pure variables
        - pureInd: list of dicts with 'pos' and 'neg' indices
        - group: list of dicts with cluster assignments
        - A: np.ndarray, p x K estimated assignment matrix
        - C: np.ndarray, K x K covariance matrix of Z
        - Omega: np.ndarray or None, K x K precision matrix of Z
        - Gamma: np.ndarray, p-dim diagonal of error covariance
        - optDelta: float, selected delta value
    """

    # Extract numpy array from DataFrame if needed
    X_array = X.values if hasattr(X, 'values') else np.asarray(X)

    # Convert delta to numpy array if provided as scalar or list
    if delta is not None and not isinstance(delta, np.ndarray):
        delta = np.atleast_1d(delta)

    # Dispatch to appropriate backend
    if backend == 'r':
        if verbose:
            print("Using R LOVE backend via rpy2")
        result = call_love_r(
            X=X_array,
            lbd=lbd,
            delta=delta,
            thresh_fdr=thresh_fdr,
            rep_CV=rep_CV,
            verbose=verbose,
            **kwargs
        )
    else:
        # outpath is only used by R backend
        if outpath != '.':
            warnings.warn(
                "outpath parameter is deprecated (Python backend). It has no effect.",
                DeprecationWarning,
                stacklevel=2
            )

        if verbose:
            print("Using Python LOVE backend")

        # For pure_homo=True, use Dantzig by default to match R slide_r behavior
        # unless the user explicitly requested a different method
        actual_est_method = est_non_pure_row
        if pure_homo and est_non_pure_row == "HT":
            actual_est_method = "Dantzig"
            if verbose:
                print("Note: Using Dantzig for non-pure rows (default for pure_homo=True)")

        result = LOVE(
            X=X_array,
            lbd=lbd,
            mu=mu,
            est_non_pure_row=actual_est_method,
            verbose=verbose,
            pure_homo=pure_homo,
            diagonal=diagonal,
            delta=delta,
            merge=merge,
            rep_CV=rep_CV,
            ndelta=ndelta,
            q=q,
            exact=exact,
            max_pure=max_pure,
            nfolds=nfolds,
            thresh_fdr=thresh_fdr
        )

    return result
