"""
Precision matrix estimation via linear programming.
Translated from R/EstOmega.R
"""

import numpy as np
from scipy.optimize import linprog
import warnings


def estOmega(lbd: float, C: np.ndarray) -> np.ndarray:
    """
    Estimate the precision (inverse) matrix of C via regularization.

    Finds the inverse of C by solving the following linear program:
        min_{Omega} ||Omega||_{inf,1}
        subject to ||C @ Omega - I||_inf <= lambda

    The LP is solved by decoupling into several linear programs, each of
    which solves one column of Omega.

    Parameters
    ----------
    lbd : float
        A numeric constant (regularization parameter).
    C : np.ndarray
        A K by K matrix.

    Returns
    -------
    np.ndarray
        A K by K matrix (the estimated precision matrix).
    """
    K = C.shape[0]
    omega = np.zeros((K, K))
    for i in range(K):
        omega[:, i] = solve_row(i, C, lbd)
    return omega


def solve_row(col_ind: int, C_hat: np.ndarray, lbd: float) -> np.ndarray:
    """
    Estimate one column of Omega by solving a LP.

    Parameters
    ----------
    col_ind : int
        Column index (0-indexed).
    C_hat : np.ndarray
        The covariance matrix estimate.
    lbd : float
        Regularization parameter.

    Returns
    -------
    np.ndarray
        A vector of length K.
    """
    K = C_hat.shape[0]

    # Variables: [t, omega_1_pos, omega_1_neg, omega_2_pos, omega_2_neg, ...]
    # Total: 1 + 2*K variables
    # omega_j = omega_j_pos - omega_j_neg

    # Objective: minimize t (first variable)
    c = np.zeros(1 + 2 * K)
    c[0] = 1.0

    # Build constraints:
    # 1. t >= 0  (handled by bounds)
    # 2. sum_j (omega_j_pos + omega_j_neg) <= t  (L1 norm constraint)
    #    -> -t + sum_j (omega_j_pos + omega_j_neg) <= 0
    # 3. |C @ omega - e_col_ind|_inf <= lbd
    #    -> For each row k: -lbd <= (C @ omega)_k - delta_{k,col_ind} <= lbd
    #    -> (C @ omega)_k <= lbd + delta_{k,col_ind}
    #    -> -(C @ omega)_k <= lbd - delta_{k,col_ind}

    A_ub_list = []
    b_ub_list = []

    # Constraint: -t <= 0 (t >= 0, but using inequality form)
    A_ub_list.append([-1.0] + [0.0] * (2 * K))
    b_ub_list.append(0.0)

    # Constraint: -t + sum_j (omega_j_pos + omega_j_neg) <= 0
    # Equivalent to: L1 norm of omega <= t
    row = [-1.0] + [1.0] * (2 * K)
    A_ub_list.append(row)
    b_ub_list.append(0.0)

    # Constraints for |C @ omega - e_col_ind|_inf <= lbd
    # For row k of C @ omega:
    # sum_j C[k,j] * omega_j = sum_j C[k,j] * (omega_j_pos - omega_j_neg)
    e_col = np.zeros(K)
    e_col[col_ind] = 1.0

    for k in range(K):
        # (C @ omega)_k <= lbd + e_col[k]
        # sum_j C[k,j] * (omega_j_pos - omega_j_neg) <= lbd + e_col[k]
        # -lbd * t_coef + sum_j C[k,j] * omega_j_pos - sum_j C[k,j] * omega_j_neg <= e_col[k]
        # But t_coef is multiplied by lbd in original R code
        row_pos = [-lbd]
        for j in range(K):
            row_pos.append(C_hat[k, j])   # omega_j_pos coefficient
            row_pos.append(-C_hat[k, j])  # omega_j_neg coefficient
        A_ub_list.append(row_pos)
        b_ub_list.append(e_col[k])

        # -(C @ omega)_k <= lbd - e_col[k]
        row_neg = [-lbd]
        for j in range(K):
            row_neg.append(-C_hat[k, j])  # omega_j_pos coefficient
            row_neg.append(C_hat[k, j])   # omega_j_neg coefficient
        A_ub_list.append(row_neg)
        b_ub_list.append(-e_col[k])

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    # Bounds: all variables >= 0
    bounds = [(0, None)] * (1 + 2 * K)

    max_attempts = 10
    current_lbd = lbd

    for attempt in range(max_attempts):
        # Update lbd in constraints if needed
        if attempt > 0:
            # Update the lbd coefficient in constraint rows
            A_ub[2:, 0] = -current_lbd

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.success and result.x is not None and len(result.x) > 0:
            solution = result.x
            # Extract omega from solution
            # omega_j = omega_j_pos - omega_j_neg
            omega = np.zeros(K)
            for j in range(K):
                omega[j] = solution[1 + 2 * j] - solution[1 + 2 * j + 1]
            return omega
        else:
            warnings.warn(f"The penalty lambda = {current_lbd} is too small, increasing by 0.01...")
            current_lbd += 0.01

    # If all attempts fail, return zeros
    warnings.warn("LP solver failed after multiple attempts, returning zeros.")
    return np.zeros(K)
