"""
Estimation of non-pure rows of the loading matrix A.
Translated from R/EstNonpure.R
"""

import numpy as np
from scipy.optimize import linprog
from typing import List, Optional


def EstY(Sigma: np.ndarray, AI: np.ndarray, pureVec: np.ndarray) -> np.ndarray:
    """
    Estimate Sigma_TJ via sign-adjusted averaging (matches R's estSigmaTJ).

    For each factor k, averages the sign-adjusted correlations between
    pure variables in cluster k and all non-pure variables.

    Parameters
    ----------
    Sigma : np.ndarray
        The p by p covariance/correlation matrix.
    AI : np.ndarray
        The p by K loading matrix (non-pure rows should be zero).
    pureVec : np.ndarray
        Array of pure variable indices.

    Returns
    -------
    np.ndarray
        A K by |J| matrix (where J is the set of non-pure variables).
    """
    pureVec_set = set(pureVec)
    p = Sigma.shape[0]
    K = AI.shape[1]

    # Get non-pure indices
    nonPureVec = [i for i in range(p) if i not in pureVec_set]
    n_J = len(nonPureVec)

    # Step 1: Sign-adjust sigma according to AI (R's adjustSign)
    # For each pure variable i, multiply its correlation row by the sign
    # of its non-zero loading entry. Non-pure rows stay zero.
    signed_sigma = np.zeros_like(Sigma)
    for i in range(p):
        nz = np.where(AI[i, :] != 0)[0]
        if len(nz) > 0:
            signed_sigma[i, :] = np.sign(AI[i, nz[0]]) * Sigma[i, :]

    # Step 2: Subset to non-pure columns
    sigma_J = signed_sigma[:, nonPureVec]

    # Step 3: For each factor k, average signed correlations across its
    # pure variables to get the factor-to-nonpure correlation estimate
    sigma_TJ = np.zeros((K, n_J))
    for k in range(K):
        group_k = np.where(AI[:, k] != 0)[0]  # pure nodes in cluster k
        if len(group_k) > 0:
            sigma_TJ[k, :] = np.mean(sigma_J[group_k, :], axis=0)

    return sigma_TJ


def EstAJInv(Omega: np.ndarray, Y: np.ndarray, lbd: float) -> np.ndarray:
    """
    Estimate non-pure rows via soft-thresholding.

    Estimates the |J| by K submatrix A_J by using soft thresholding.

    Parameters
    ----------
    Omega : np.ndarray
        The estimated precision matrix of Z.
    Y : np.ndarray
        A K by |J| response matrix.
    lbd : float
        Tuning parameter for soft-thresholding.

    Returns
    -------
    np.ndarray
        A |J| by K matrix.
    """
    n_J = Y.shape[1]  # Number of non-pure variables
    K = Y.shape[0]    # Number of factors
    AJ = np.zeros((n_J, K))

    for i in range(n_J):
        Atilde = Omega @ Y[:, i]
        AJ[i, :] = LP(Atilde, lbd)
        # Normalize if L1 norm > 1
        if np.sum(np.abs(AJ[i, :])) > 1:
            AJ[i, :] = AJ[i, :] / np.sum(np.abs(AJ[i, :]))

    return AJ


def LP(y: np.ndarray, lbd: float) -> np.ndarray:
    """
    Soft-thresholding via linear program.

    Solves:
        min sum(beta_pos + beta_neg)
        s.t. beta_pos - beta_neg <= lbd + y
             -beta_pos + beta_neg <= lbd - y
             beta_pos >= 0, beta_neg >= 0

    Parameters
    ----------
    y : np.ndarray
        A vector of length K.
    lbd : float
        Threshold parameter.

    Returns
    -------
    np.ndarray
        A vector of length K (beta = beta_pos - beta_neg).
    """
    K = len(y)

    # Variables layout matches R: [beta_1_pos, beta_2_pos, ..., beta_K_pos,
    #                              beta_1_neg, beta_2_neg, ..., beta_K_neg]
    # Total: 2*K variables

    # Objective: minimize sum of all variables
    c = np.ones(2 * K)

    # Build constraint matrix C where C[k, k] = 1, C[k, k+K] = -1
    # So C @ x = beta_pos - beta_neg
    # Constraints: C @ x <= lbd + y, -C @ x <= lbd - y
    C = np.zeros((K, 2 * K))
    for k in range(K):
        C[k, k] = 1        # beta_k_pos
        C[k, k + K] = -1   # beta_k_neg

    # Stack constraints: [C; -C] @ x <= [lbd + y; lbd - y]
    A_ub = np.vstack([C, -C])
    b_ub = np.concatenate([lbd + y, lbd - y])

    # Bounds: all variables >= 0
    bounds = [(0, None)] * (2 * K)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        solution = result.x
        # beta = beta_pos - beta_neg (R: LPsol[1:K] - LPsol[(K+1):(2*K)])
        beta = solution[:K] - solution[K:]
        return beta
    else:
        # Return soft-thresholded y as fallback
        return np.sign(y) * np.maximum(np.abs(y) - lbd, 0)


def EstAJDant(C_hat: np.ndarray, Y: np.ndarray, lbd: float,
              se_est_J: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate non-pure rows via the Dantzig approach.

    Parameters
    ----------
    C_hat : np.ndarray
        The estimated covariance matrix of Z.
    Y : np.ndarray
        A K by |J| response matrix.
    lbd : float
        Base tuning parameter.
    se_est_J : np.ndarray
        Estimated standard errors of the non-pure variables.

    Returns
    -------
    np.ndarray or None
        A |J| by K matrix, or None if any Dantzig LP fails.
        Matches R's estAJDant which returns NULL on failure.
    """
    n_J = Y.shape[1]
    K = Y.shape[0]
    AJ = np.zeros((n_J, K))

    for i in range(n_J):
        dantzig_sol = Dantzig(C_hat, Y[:, i], lbd * se_est_J[i])
        if dantzig_sol is None:
            return None
        AJ[i, :] = dantzig_sol
        # Normalize if L1 norm > 1
        if np.sum(np.abs(AJ[i, :])) > 1:
            AJ[i, :] = AJ[i, :] / np.sum(np.abs(AJ[i, :]))

    return AJ


def Dantzig(C_hat: np.ndarray, y: np.ndarray, lbd: float) -> Optional[np.ndarray]:
    """
    The Dantzig approach for solving one non-pure row.

    Solves:
        min sum(beta_pos + beta_neg)
        s.t. C_hat @ (beta_pos - beta_neg) - y <= lbd  (element-wise)
             -C_hat @ (beta_pos - beta_neg) + y <= lbd  (element-wise)
             beta_pos >= 0, beta_neg >= 0

    Parameters
    ----------
    C_hat : np.ndarray
        The covariance matrix estimate.
    y : np.ndarray
        Response vector.
    lbd : float
        Threshold parameter.

    Returns
    -------
    np.ndarray or None
        A vector of length K, or None if the LP solver fails.
        Matches R's dantzig() which returns NULL on failure.
    """
    K = len(y)

    # Variables layout matches R: [beta_1_pos, beta_2_pos, ..., beta_K_pos,
    #                              beta_1_neg, beta_2_neg, ..., beta_K_neg]
    # Total: 2*K variables

    # Objective: minimize sum of all variables
    c = np.ones(2 * K)

    # Build constraint matrix matching R's linprog::solveLP formulation:
    # R includes explicit non-negativity rows: A_mat = [new_C_hat; -new_C_hat; -I(2K)]
    # Python uses bounds instead, which is mathematically equivalent.
    new_C_hat = np.zeros((K, 2 * K))
    for k in range(K):
        new_C_hat[k, :K] = C_hat[k, :]      # beta_pos coefficients
        new_C_hat[k, K:] = -C_hat[k, :]     # beta_neg coefficients

    # Stack constraints: [new_C_hat; -new_C_hat] @ x <= [lbd + y; lbd - y]
    A_ub = np.vstack([new_C_hat, -new_C_hat])
    b_ub = np.concatenate([lbd + y, lbd - y])

    # Bounds: all variables >= 0
    bounds = [(0, None)] * (2 * K)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        solution = result.x
        # beta = beta_pos - beta_neg (R: LPsol[1:K] - LPsol[(K+1):(2*K)])
        beta = solution[:K] - solution[K:]
        return beta
    else:
        # Match R: return None on LP failure (R's dantzig returns NULL)
        return None
