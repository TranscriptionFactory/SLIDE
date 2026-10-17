"""
Score matrix computation functions.
Translated from R/Score.R
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict, Tuple


def Score_mat(R: np.ndarray, q: int = 2, exact: bool = False) -> Dict[str, np.ndarray]:
    """
    Calculate the score matrix for detecting parallel rows.

    Parameters
    ----------
    R : np.ndarray
        The correlation matrix.
    q : int, optional
        Either 2 or np.inf to specify the type of score. Default is 2.
    exact : bool, optional
        Only active for computing the Inf score.
        If True, compute the Inf score exactly via solving a linear program.
        Otherwise, use approximation. Default is False.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict containing:
        - 'score': A score matrix (upper triangular with NaN elsewhere)
        - 'moments': A matrix of the crossproduct of R (R'R)
    """
    p = R.shape[0]
    score_mat = np.full((p, p), np.nan)
    M = R.T @ R  # crossprod(R)

    if q == 2:
        for i in range(p - 1):
            for j in range(i + 1, p):
                # V_ij = M[c(i,j), c(i,j)] - crossprod(R[c(i,j), c(i,j)])
                idx = [i, j]
                V_ij = M[np.ix_(idx, idx)] - R[np.ix_(idx, idx)].T @ R[np.ix_(idx, idx)]

                if V_ij[0, 0] == 0 or V_ij[1, 1] == 0:
                    score_mat[i, j] = 0
                else:
                    score_ij = (min(V_ij[0, 0], V_ij[1, 1]) / (p - 2) *
                                (1 - V_ij[0, 1] ** 2 / V_ij[0, 0] / V_ij[1, 1]))
                    score_mat[i, j] = np.sqrt(np.abs(score_ij))
    elif q == np.inf:
        for i in range(p - 1):
            for j in range(i + 1, p):
                # Get R[-c(i,j), c(i,j)] - rows excluding i,j, columns i,j
                mask = np.ones(p, dtype=bool)
                mask[[i, j]] = False
                R_ij = R[mask][:, [i, j]]
                score_mat[i, j] = min(LP_Score(R_ij, 0, exact),
                                      LP_Score(R_ij, 1, exact))

    return {'score': score_mat, 'moments': M}


def LP_Score(R_ij: np.ndarray, ind: int, exact: bool = False) -> float:
    """
    Calculate the Inf score for a pair of rows.

    Parameters
    ----------
    R_ij : np.ndarray
        A (p-2) by 2 matrix.
    ind : int
        Either 0 or 1 (0-indexed, corresponding to R's 1 or 2).
    exact : bool, optional
        If True, compute exactly via LP. Otherwise, use approximation.
        Default is False.

    Returns
    -------
    float
        The computed score.
    """
    other_ind = 1 - ind  # The other column index

    if exact:
        # LP formulation from R:
        # minimize c'x subject to Ax >= b
        # Variables: [t, v1_pos, v1_neg] where v1 = v1_pos - v1_neg
        # In R: cvec = c(1, rep(0, 2))  -> minimize t
        # Constraint 1: |v1| <= 1, i.e., v1_pos + v1_neg <= 1
        # Constraint 2: t >= ||v1 * R_i + R_j||_inf
        #   -> -t + v1*R_i[k] + R_j[k] <= 0 for all k
        #   -> -t - v1*R_i[k] - R_j[k] <= 0 for all k

        n_rows = R_ij.shape[0]

        # Variables: [t, v1_pos, v1_neg]
        c = np.array([1.0, 0.0, 0.0])

        # Build inequality constraints A_ub @ x <= b_ub
        # Constraint 1: v1_pos + v1_neg <= 1
        # [0, 1, 1] @ [t, v1_pos, v1_neg] <= 1
        A_ub_list = [[0.0, 1.0, 1.0]]
        b_ub_list = [1.0]

        # Constraint 2: For each row k:
        # -t + (v1_pos - v1_neg)*R_ij[k, ind] <= -R_ij[k, other_ind]
        # -t - (v1_pos - v1_neg)*R_ij[k, ind] <= R_ij[k, other_ind]
        for k in range(n_rows):
            r_ind = R_ij[k, ind]
            r_other = R_ij[k, other_ind]
            # -t + v1*r_ind <= -r_other  =>  -t + v1_pos*r_ind - v1_neg*r_ind <= -r_other
            A_ub_list.append([-1.0, r_ind, -r_ind])
            b_ub_list.append(-r_other)
            # -t - v1*r_ind <= r_other  =>  -t - v1_pos*r_ind + v1_neg*r_ind <= r_other
            A_ub_list.append([-1.0, -r_ind, r_ind])
            b_ub_list.append(r_other)

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)

        # Bounds: t >= 0, v1_pos >= 0, v1_neg >= 0
        bounds = [(0, None), (0, None), (0, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return result.fun
        else:
            # Fallback to approximate method if LP fails
            return LP_Score(R_ij, ind, exact=False)
    else:
        # Approximate method: grid search
        v_grid = np.linspace(-1, 1, 100)
        scores = []
        for v in v_grid:
            score = np.max(np.abs(v * R_ij[:, ind] + R_ij[:, other_ind]))
            scores.append(score)
        return min(scores)
