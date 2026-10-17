"""
Pre-screening to detect pure noise features.
Translated from R/PreScreen.R
"""

import numpy as np
from typing import Dict, List, Optional, Union
from .utilities import partition, extract


def Screen_X(X: np.ndarray, thresh_grid: Optional[np.ndarray] = None,
             nfolds: int = 10, nthresh: int = 50,
             max_prop: float = 0.5) -> Union[np.ndarray, Dict]:
    """
    Screen features that are pure noise via k-fold cross-validation.

    Parameters
    ----------
    X : np.ndarray
        A n by p data matrix.
    thresh_grid : np.ndarray, optional
        A numeric vector of thresholds. Default is None.
    nfolds : int, optional
        Number of folds for cross-validation. Default is 10.
    nthresh : int, optional
        The length of thresh_grid when thresh_grid is None. Default is 50.
    max_prop : float, optional
        Maximal proportion of pure noise features. Default is 0.5.

    Returns
    -------
    Union[np.ndarray, Dict]
        When only one value is provided in thresh_grid, returns a vector of
        indices that are detected as pure noise. Otherwise returns a dict
        containing:
        - 'foldid': The indices of observations used for cv
        - 'thresh_min': The value of thresh_grid that has minimum cv error
        - 'thresh_1se': Largest value of thresh_grid within 1 SE of minimum
        - 'thresh_grid': The used thresh_grid sequence
        - 'cv_mean': The averaged cv errors
        - 'cv_sd': The standard errors of cv errors
        - 'noise_ind': Indices detected as pure noise using thresh_min
    """
    n_total = X.shape[0]
    p = X.shape[1]

    # Compute correlation matrix and set diagonal to 0
    R = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(R, 0)

    row_scale = np.sum(R ** 2, axis=1)

    if thresh_grid is None or (thresh_grid is not None and len(thresh_grid) > 1):
        # Multiple thresholds or no threshold provided -> use CV

        if thresh_grid is None:
            thresh_range = np.quantile(row_scale, [0, max_prop])
            thresh_grid = np.linspace(thresh_range[0], thresh_range[1], nthresh)

        # Create fold indices
        indices_shuffled = np.random.permutation(n_total)
        fold_sizes = partition(n_total, nfolds)
        indicesPerGroup = extract(indices_shuffled, fold_sizes)

        loss = np.full((nfolds, len(thresh_grid)), np.nan)

        for i in range(nfolds):
            valid_ind = indicesPerGroup[i]
            train_mask = np.ones(n_total, dtype=bool)
            train_mask[valid_ind] = False

            trainX = X[train_mask, :]
            validX = X[valid_ind, :]

            R_train = np.corrcoef(trainX, rowvar=False)
            R_valid = np.corrcoef(validX, rowvar=False)
            np.fill_diagonal(R_train, 0)
            np.fill_diagonal(R_valid, 0)

            for j, thresh in enumerate(thresh_grid):
                noise_ind = np.where(row_scale < thresh)[0]
                pred_R = R_train.copy()
                pred_R[noise_ind, :] = 0
                pred_R[:, noise_ind] = 0
                loss[i, j] = np.mean((R_valid - pred_R) ** 2)

        cv_mean = np.mean(loss, axis=0)
        cv_sd = np.std(loss, axis=0, ddof=1)

        ind_min = np.argmin(cv_mean)
        thresh_min = thresh_grid[ind_min]

        # Find largest threshold within 1 SE of minimum
        within_1se = np.where(cv_mean <= (cv_mean[ind_min] + cv_sd[ind_min]))[0]
        thresh_1se = thresh_grid[np.max(within_1se)]

        noise_ind = np.where(row_scale < thresh_min)[0]

        return {
            'foldid': indicesPerGroup,
            'thresh_min': thresh_min,
            'thresh_1se': thresh_1se,
            'thresh_grid': thresh_grid,
            'cv_mean': cv_mean,
            'cv_sd': cv_sd,
            'noise_ind': noise_ind
        }

    else:
        # Single threshold provided
        thresh = thresh_grid[0] if isinstance(thresh_grid, np.ndarray) else thresh_grid
        return np.where(row_scale < thresh)[0]
