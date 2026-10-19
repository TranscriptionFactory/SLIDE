"""
Cross-validation functions for tuning parameter selection.
Translated from R/CV.R
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from .utilities import singleton, partition, extract, offSum
from .est_pure_homo import (FindRowMax, FindPureNode, FindSignPureNode,
                            RecoverAI, EstC)
from .est_pure_hetero import Est_Pure, Est_BI_C
from .est_omega import estOmega
from .score import Score_mat


def CV_delta(X: np.ndarray, deltaGrids: np.ndarray, diagonal: bool,
             se_est: np.ndarray, merge: bool) -> float:
    """
    Cross validation to select delta for homogeneous pure loadings.

    For each value of deltaGrids, first split the data into two parts
    and calculate I, A_I and Cov(Z). Then calculate the fit
    A_I @ Cov(Z) @ A_I' to find the value which minimizes the loss criterion:
        ||Sigma - A_I @ Cov(Z) @ A_I'||_{F-off} / (|I| * (|I| - 1))

    Parameters
    ----------
    X : np.ndarray
        A n by p data matrix.
    deltaGrids : np.ndarray
        A vector of numerical constants.
    diagonal : bool
        If True, force covariance matrix of Z to be diagonal.
    se_est : np.ndarray
        Vector of standard deviations of p features.
    merge : bool
        If True, use intersection merging; else use union.

    Returns
    -------
    float
        The selected optimal delta.
    """
    n, p = X.shape

    # Split data
    sampInd = np.random.choice(n, n // 2, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[sampInd] = False

    X1 = X[sampInd, :]
    X2 = X[mask, :]

    Sigma1 = (X1.T @ X1) / X1.shape[0]
    np.fill_diagonal(Sigma1, 0)
    Sigma2 = (X2.T @ X2) / X2.shape[0]

    result_Ms = FindRowMax(np.abs(Sigma1))
    Ms = result_Ms['M']
    arg_Ms = result_Ms['arg_M']

    loss = []
    for delta in deltaGrids:
        resultFitted = CalFittedSigma(Sigma1, delta, Ms, arg_Ms, se_est,
                                      diagonal, merge)
        fittedValue = resultFitted['fitted']
        estPureVec = resultFitted['pureVec']

        if fittedValue is None:
            loss.append(np.inf)
        else:
            denom = len(estPureVec) * (len(estPureVec) - 1)
            if denom > 0:
                diff = Sigma2[np.ix_(estPureVec, estPureVec)] - fittedValue
                loss_val = 2 * offSum(diff, se_est[estPureVec]) / denom
                loss.append(loss_val)
            else:
                loss.append(np.inf)

    return deltaGrids[np.argmin(loss)]


def CalFittedSigma(Sigma: np.ndarray, delta: float, Ms: np.ndarray,
                   arg_Ms: np.ndarray, se_est: np.ndarray,
                   diagonal: bool, merge: bool) -> Dict:
    """
    Calculate the fitted value A_I @ Cov(Z) @ A_I'.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix (with diagonal zeroed).
    delta : float
        Threshold parameter.
    Ms : np.ndarray
        Row maxima vector.
    arg_Ms : np.ndarray
        Indices of row maxima.
    se_est : np.ndarray
        Standard error estimates.
    diagonal : bool
        Force diagonal covariance.
    merge : bool
        Use intersection or union merging.

    Returns
    -------
    Dict
        A dict containing:
        - 'pureVec': Vector of pure variable indices (or None)
        - 'fitted': The fitted value A_I @ C @ A_I' (or None if failed)
    """
    resultPureNode = FindPureNode(np.abs(Sigma), delta, Ms, arg_Ms, se_est, merge)
    estPureIndices = resultPureNode['pureInd']

    if singleton(estPureIndices):
        return {'pureVec': None, 'fitted': None}

    estSignPureIndices = FindSignPureNode(estPureIndices, Sigma)
    AI = RecoverAI(estSignPureIndices, len(se_est))

    # Need to restore diagonal for EstC
    Sigma_full = Sigma.copy()
    # Note: In R, diagonal was zeroed for finding, but EstC needs original
    # We'll compute C using the off-diagonal elements anyway

    C = EstC(Sigma_full, AI, diagonal)

    pureVec = list(resultPureNode['pureVec'])

    if len(estPureIndices) == 1:
        fitted = None
    else:
        subAI = AI[pureVec, :]
        fitted = subAI @ C @ subAI.T

    return {'pureVec': pureVec, 'fitted': fitted}


def KfoldCV_delta(X: np.ndarray, delta: Optional[np.ndarray] = None,
                  ndelta: int = 50, q: int = 2, exact: bool = False,
                  nfolds: int = 10,
                  max_pure: Optional[float] = None) -> Dict:
    """
    K-fold cross-validation for selecting delta (heterogeneous case).

    Parameters
    ----------
    X : np.ndarray
        A n by p data matrix.
    delta : np.ndarray, optional
        Grid of delta values. If None, will be auto-generated.
    ndelta : int, optional
        Length of delta grid when delta is None. Default is 50.
    q : int, optional
        Either 2 or np.inf for score type. Default is 2.
    exact : bool, optional
        If True, compute Inf score exactly via LP. Default is False.
    nfolds : int, optional
        Number of folds. Default is 10.
    max_pure : float, optional
        Max proportion of pure variables. Default is None.

    Returns
    -------
    Dict
        A dict containing:
        - 'foldid': Indices of observations used for cv
        - 'delta_min': Value of delta with minimum cv error
        - 'delta_1se': Delta values within 1 SE of minimum
        - 'delta': The used delta sequence
        - 'cv_mean': Averaged cv errors
        - 'cv_sd': Standard errors of cv errors
        - 'est_pure': Dict with K, I, I_part
        - 'score': The score matrix
        - 'moments': The crossproduct matrix R'R
    """
    n_total, p_total = X.shape
    R = np.corrcoef(X, rowvar=False)

    score_res = Score_mat(R, q, exact)
    score_mat = score_res['score']
    moments_mat = score_res['moments']

    if delta is not None and len(delta) == 1:
        # Single delta value provided
        delta_val = delta[0] if isinstance(delta, np.ndarray) else delta
        return {
            'foldid': None,
            'delta_min': delta_val,
            'delta_1se': delta_val,
            'delta': delta,
            'cv_mean': None,
            'cv_sd': None,
            'est_pure': Est_Pure(score_mat, delta_val),
            'score': score_mat,
            'moments': moments_mat
        }
    else:
        # Use k-fold CV
        if delta is None:
            if max_pure is None:
                max_pure = 1.0 if n_total > p_total else 0.8

            # Generate delta grid
            # Get min scores per row (excluding last row and NaN values)
            row_mins = []
            for i in range(p_total - 1):
                row_vals = score_mat[i, :]
                valid_vals = row_vals[~np.isnan(row_vals)]
                if len(valid_vals) > 0:
                    row_mins.append(np.min(valid_vals))

            if len(row_mins) > 0:
                delta_max = np.quantile(row_mins, max_pure)
                delta_min_val = np.nanmin(score_mat)
                delta = np.linspace(delta_max, delta_min_val, ndelta)
            else:
                delta = np.linspace(1, 0.01, ndelta)

        # Create fold indices
        indices_shuffled = np.random.permutation(n_total)
        fold_sizes = partition(n_total, nfolds)
        indicesPerGroup = extract(indices_shuffled, fold_sizes)

        loss = np.full((nfolds, len(delta)), np.nan)

        for i in range(nfolds):
            valid_ind = indicesPerGroup[i]
            train_mask = np.ones(n_total, dtype=bool)
            train_mask[valid_ind] = False

            trainX = X[train_mask, :]
            validX = X[valid_ind, :]

            R1 = np.corrcoef(trainX, rowvar=False)
            R2 = np.corrcoef(validX, rowvar=False)

            score_res_1 = Score_mat(R1, q, exact)
            score_mat_1 = score_res_1['score']
            moments_1 = score_res_1['moments']

            pre_I = None
            for j, delta_j in enumerate(delta):
                if j == 0:
                    pure_res = Est_Pure(score_mat_1, delta_j)
                    I = pure_res['I']
                    I_part = pure_res['I_part']
                else:
                    # Restrict to previous I
                    if pre_I is not None and len(pre_I) > 0:
                        pre_I_list = list(pre_I)
                        sub_score = score_mat_1[np.ix_(pre_I_list, pre_I_list)]
                        pure_res = Est_Pure(sub_score, delta_j)
                        # Map back to original indices
                        I = pre_I[pure_res['I']] if len(pure_res['I']) > 0 else np.array([])
                        I_part = []
                        for part in pure_res['I_part']:
                            I_part.append([pre_I_list[idx] for idx in part])
                    else:
                        I = np.array([])
                        I_part = []

                pre_I = I if len(I) > 0 else None

                if len(I_part) == 0:
                    break
                else:
                    I_list = list(I)
                    result = Est_BI_C(moments_1, R1, I_part, I)
                    B_hat = result['B']
                    C_hat = result['C']
                    B_left_inv = result['B_left_inv']

                    tmp_R1 = R1.copy()
                    B_I = B_hat[I_list, :]
                    tmp_R1[np.ix_(I_list, I_list)] = B_I @ C_hat @ B_I.T

                    if len(I) != p_total:
                        J_list = [idx for idx in range(p_total) if idx not in I_list]
                        tmp = B_left_inv @ tmp_R1[np.ix_(I_list, J_list)]
                        try:
                            tmp_prime = np.linalg.solve(C_hat, tmp)
                        except np.linalg.LinAlgError:
                            tmp_prime = np.linalg.pinv(C_hat) @ tmp
                        tmp_R1[np.ix_(J_list, J_list)] = tmp.T @ tmp_prime

                    loss[i, j] = offSum(tmp_R1 - R2, 1) / p_total / (p_total - 1)

        cv_mean = np.nanmean(loss, axis=0)
        cv_sd = np.nanstd(loss, axis=0, ddof=1)

        ind_min = np.nanargmin(cv_mean)
        delta_min = delta[ind_min]

        # Find delta values within 1 SE of minimum
        within_1se = np.where(cv_mean <= (cv_mean[ind_min] + cv_sd[ind_min]))[0]
        if len(within_1se) > 0:
            delta_1se = (delta[np.min(within_1se)], delta[np.max(within_1se)])
        else:
            delta_1se = (delta_min, delta_min)

        return {
            'foldid': indicesPerGroup,
            'delta_min': delta_min,
            'delta_1se': delta_1se,
            'delta': delta,
            'cv_mean': cv_mean,
            'cv_sd': cv_sd,
            'est_pure': Est_Pure(score_mat, delta_min),
            'score': score_mat,
            'moments': moments_mat
        }


def CV_lbd(X: np.ndarray, lbdGrids: np.ndarray, AI: np.ndarray,
           pureVec: np.ndarray, diagonal: bool) -> float:
    """
    Cross validation to select lambda for estimating the precision matrix of Z.

    Split the data into two parts. Estimate Cov(Z) on two datasets.
    For each value in lbdGrids, calculate Omega on the first dataset
    and calculate the loss on the second dataset:
        <Cov(Z), Omega> - log(det(Omega))

    Parameters
    ----------
    X : np.ndarray
        A n by p data matrix.
    lbdGrids : np.ndarray
        A vector of numerical constants.
    AI : np.ndarray
        A p by K loading matrix.
    pureVec : np.ndarray
        Estimated set of pure variables.
    diagonal : bool
        If True, force diagonal covariance.

    Returns
    -------
    float
        The selected lambda.
    """
    n = X.shape[0]

    # Split data
    sampInd = np.random.choice(n, n // 2, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[sampInd] = False

    X1 = X[sampInd, :]
    X2 = X[mask, :]

    Sigma1 = (X1.T @ X1) / X1.shape[0]
    Sigma2 = (X2.T @ X2) / X2.shape[0]

    C1 = EstC(Sigma1, AI, diagonal)
    C2 = EstC(Sigma2, AI, diagonal)

    loss = []
    for lbd in lbdGrids:
        Omega = estOmega(lbd, C1)
        det_Omega = np.linalg.det(Omega)

        if det_Omega <= 0:
            loss.append(np.inf)
        else:
            # <C2, Omega> - log(det(Omega))
            loss_val = np.sum(Omega * C2) - np.log(det_Omega)
            loss.append(loss_val)

    return lbdGrids[np.argmin(loss)]
