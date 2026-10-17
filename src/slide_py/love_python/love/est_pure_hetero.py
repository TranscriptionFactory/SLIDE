"""
Estimation of pure variables for heterogeneous pure loadings.
Translated from R/EstPureHetero.R
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union


def Est_Pure(score_mat: np.ndarray, delta: float) -> Dict[str, Union[int, List, np.ndarray]]:
    """
    Estimate parallel rows using graph connected components.

    Parameters
    ----------
    score_mat : np.ndarray
        The score matrix (upper triangular with NaN elsewhere).
    delta : float
        Threshold for determining edges.

    Returns
    -------
    Dict
        A dict containing:
        - 'K': The cardinality (number) of parallel row groups
        - 'I': Array of all parallel row indices
        - 'I_part': List of lists, partition of parallel rows
    """
    # Find pairs where score <= delta (edges in graph)
    # score_mat is upper triangular, so we look for entries <= delta
    rows, cols = np.where(score_mat <= delta)

    # Build graph from edges
    G = nx.Graph()
    # Add all nodes first (in case some are isolated)
    p = score_mat.shape[0]
    G.add_nodes_from(range(p))

    # Add edges where score <= delta
    for i, j in zip(rows, cols):
        if i != j:  # Skip diagonal
            G.add_edge(i, j)

    # Find connected components
    components = list(nx.connected_components(G))

    # Filter to only include components with more than 1 node (actual groups)
    I_part = [list(comp) for comp in components if len(comp) > 1]

    # Flatten to get all parallel row indices
    I = []
    for part in I_part:
        I.extend(part)
    I = np.array(sorted(set(I)))

    return {
        'K': len(I_part),
        'I': I,
        'I_part': I_part
    }


def Est_BI_C(M: np.ndarray, R: np.ndarray, I_part: List[List[int]],
             I_set: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Estimate the submatrix B_I and Corr(Z).

    Parameters
    ----------
    M : np.ndarray
        The product matrix (R'R) returned by Score_mat.
    R : np.ndarray
        The correlation matrix.
    I_part : List[List[int]]
        Partition of parallel rows.
    I_set : np.ndarray
        Array of all parallel row indices.

    Returns
    -------
    Dict
        A dict containing:
        - 'B': The B matrix
        - 'C': The correlation matrix of Z
        - 'B_left_inv': Left inverse of B_I
        - 'Gamma': Diagonal elements of error covariance
    """
    K = len(I_part)
    p = R.shape[0]
    B_square = np.zeros((p, K))
    signs = np.ones(p)
    Gamma = np.zeros(p)

    for k in range(K):
        I_k = I_part[k]
        n_k = len(I_k)

        for ell in range(n_k):
            i = I_k[ell]
            j = I_k[(ell + 1) % n_k]  # Next element, wrap around

            idx = [i, j]
            # cross_Vij = M[c(i,j), c(i,j)] - crossprod(R[c(i,j), c(i,j)])
            cross_Vij = M[np.ix_(idx, idx)] - R[np.ix_(idx, idx)].T @ R[np.ix_(idx, idx)]

            if cross_Vij[1, 1] != 0:
                B_square[i, k] = np.abs(R[i, j]) * np.sqrt(cross_Vij[0, 0] / cross_Vij[1, 1])
            else:
                B_square[i, k] = 0

            Gamma[i] = R[i, i] - B_square[i, k]
            signs[i] = 1 if ell == 0 else np.sign(R[i, I_k[0]])

    B = np.sqrt(B_square)
    I_set_list = list(I_set)

    # BI = signs[I_set] * B[I_set, :]
    BI = signs[I_set_list, np.newaxis] * B[I_set_list, :]
    B = signs[:, np.newaxis] * B

    # cross_BI = crossprod(BI) = BI.T @ BI
    cross_BI = BI.T @ BI

    # B_left_inv = solve(cross_BI, t(BI)) = inv(cross_BI) @ BI.T
    try:
        B_left_inv = np.linalg.solve(cross_BI, BI.T)
    except np.linalg.LinAlgError:
        B_left_inv = np.linalg.pinv(cross_BI) @ BI.T

    # C_hat = B_left_inv @ (R[I_set, I_set] - diag(Gamma[I_set])) @ B_left_inv.T
    R_II = R[np.ix_(I_set_list, I_set_list)]
    Gamma_diag = np.diag(Gamma[I_set_list])
    C_hat = B_left_inv @ (R_II - Gamma_diag) @ B_left_inv.T
    np.fill_diagonal(C_hat, 1.0)

    return {
        'B': B,
        'C': C_hat,
        'B_left_inv': B_left_inv,
        'Gamma': Gamma
    }


def Re_Est_Pure(X: np.ndarray, Sigma: np.ndarray, M: np.ndarray,
                I_part: List[List[int]], Gamma: np.ndarray) -> List[List[int]]:
    """
    Re-estimate the pure variables from the selected parallel rows.

    Parameters
    ----------
    X : np.ndarray
        The n by p data matrix.
    Sigma : np.ndarray
        The covariance matrix.
    M : np.ndarray
        The product matrix (R'R).
    I_part : List[List[int]]
        Partition of parallel rows.
    Gamma : np.ndarray
        Diagonal elements of error covariance.

    Returns
    -------
    List[List[int]]
        Updated partition of pure variables.
    """
    # Find representative from each group (max row norm excluding self)
    L_hat = []
    for part in I_part:
        row_norms = []
        for i in part:
            # crossprod(M[i, -i]) = sum of squares excluding diagonal
            mask = np.ones(M.shape[0], dtype=bool)
            mask[i] = False
            row_norm = M[i, mask] @ M[i, mask]
            row_norms.append(row_norm)
        best_idx = part[np.argmax(row_norms)]
        L_hat.append(best_idx)

    L_hat = np.array(L_hat)
    Gamma_LL = Gamma[L_hat]
    K_est = Est_K(X, L_hat, Gamma_LL)

    if K_est < len(I_part) and K_est >= 1:
        # Re-select the pure variables
        I_part_tilde = Post_Est_Pure(Sigma, Gamma_LL, L_hat, I_part, K_est)
    else:
        I_part_tilde = I_part

    return I_part_tilde


def Post_Est_Pure(Sigma: np.ndarray, Gamma_LL: np.ndarray, L_hat: np.ndarray,
                  I_part: List[List[int]], K_tilde: int) -> List[List[int]]:
    """
    Post-selection of pure variables.

    Parameters
    ----------
    Sigma : np.ndarray
        The covariance matrix.
    Gamma_LL : np.ndarray
        Gamma values at representative indices.
    L_hat : np.ndarray
        Representative indices.
    I_part : List[List[int]]
        Current partition.
    K_tilde : int
        Estimated number of factors.

    Returns
    -------
    List[List[int]]
        Updated partition.
    """
    D_Sigma = np.diag(Sigma)
    Sigma_E_LL = Gamma_LL * D_Sigma[L_hat]
    n_L = len(L_hat)
    Theta_LL = Sigma[np.ix_(L_hat, L_hat)] - np.diag(Sigma_E_LL)

    D_Theta_vec = np.diag(Theta_LL)
    D_Theta_LL = np.diag(D_Theta_vec)

    L_tilde = [np.argmax(D_Theta_vec)]
    L_tilde_comp = list(set(range(len(I_part))) - set(L_tilde))

    if K_tilde > 1:
        for k in range(1, K_tilde):
            # Compute Schur complement
            L_tilde_arr = np.array(L_tilde)
            L_tilde_comp_arr = np.array(L_tilde_comp)

            Theta_sub = D_Theta_LL[np.ix_(L_tilde_comp_arr, L_tilde_comp_arr)]
            Theta_cross = Theta_LL[np.ix_(L_tilde_comp_arr, L_tilde_arr)]
            Theta_core = Theta_LL[np.ix_(L_tilde_arr, L_tilde_arr)]

            try:
                Theta_schur = Theta_sub - Theta_cross @ np.linalg.solve(Theta_core, Theta_cross.T)
            except np.linalg.LinAlgError:
                Theta_schur = Theta_sub - Theta_cross @ np.linalg.pinv(Theta_core) @ Theta_cross.T

            # Find the index with max diagonal in Schur complement
            local_idx = np.argmax(np.diag(Theta_schur))
            i_k = L_tilde_comp[local_idx]
            L_tilde.append(i_k)
            L_tilde_comp.remove(i_k)

    # Return the selected partitions
    return [I_part[i] for i in L_tilde]


def Est_K(X: np.ndarray, L_hat: np.ndarray, Gamma_LL: np.ndarray) -> int:
    """
    Estimate the number of latent factors from representative parallel rows.

    Parameters
    ----------
    X : np.ndarray
        The n by p data matrix.
    L_hat : np.ndarray
        Representative indices of parallel rows.
    Gamma_LL : np.ndarray
        Gamma values at representative indices.

    Returns
    -------
    int
        The estimated number of factors K.
    """
    n = X.shape[0]
    K_hat = len(L_hat)

    # Split data
    n_ind = np.random.choice(n, n // 2, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[n_ind] = False

    X1 = X[n_ind, :]
    X2 = X[mask, :]

    # Compute correlation matrices
    R1 = np.corrcoef(X1, rowvar=False)
    R2 = np.corrcoef(X2, rowvar=False)

    L_hat_list = list(L_hat)
    Gamma_LL_mat = np.diag(Gamma_LL)
    M1 = R1[np.ix_(L_hat_list, L_hat_list)] - Gamma_LL_mat
    M2 = R2[np.ix_(L_hat_list, L_hat_list)] - Gamma_LL_mat

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M1)
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Find K that minimizes reconstruction error
    errors = []
    for x in range(1, K_hat + 1):
        U = eigenvectors[:, :x]
        D = np.diag(eigenvalues[:x])
        M_tilde = U @ D @ U.T
        error = np.sum((M_tilde - M2) ** 2)
        errors.append(error)

    K_tilde = np.argmin(errors) + 1  # +1 because range starts at 1
    return K_tilde
