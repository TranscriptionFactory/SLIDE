"""
Estimation of pure variables for homogeneous pure loadings.
Translated from R/EstPureHomo.R
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union


def EstAI(Sigma: np.ndarray, optDelta: float, se_est: np.ndarray,
          merge: bool) -> Dict[str, Union[np.ndarray, List]]:
    """
    Estimate the submatrix of A corresponding to the pure variables.

    Parameters
    ----------
    Sigma : np.ndarray
        A p by p covariance matrix.
    optDelta : float
        Optimal delta value.
    se_est : np.ndarray
        Vector of estimated standard errors for each feature.
    merge : bool
        If True, use intersection for merging; else use union.

    Returns
    -------
    Dict
        A dict containing:
        - 'AI': The estimated p by K submatrix A_I of A
        - 'pureVec': Vector of indices of estimated pure variables
        - 'pureSignInd': List of indices with sign sub-partition
    """
    off_Sigma = np.abs(Sigma.copy())
    np.fill_diagonal(off_Sigma, 0)

    result_Ms = FindRowMax(off_Sigma)
    Ms = result_Ms['M']
    arg_Ms = result_Ms['arg_M']

    resultPure = FindPureNode(off_Sigma, optDelta, Ms, arg_Ms, se_est, merge)
    estPureIndices = resultPure['pureInd']
    estPureVec = resultPure['pureVec']

    estSignPureIndices = FindSignPureNode(estPureIndices, Sigma)
    AI = RecoverAI(estSignPureIndices, off_Sigma.shape[0])

    return {'AI': AI, 'pureVec': estPureVec, 'pureSignInd': estSignPureIndices}


def EstC(Sigma: np.ndarray, AI: np.ndarray, diagonal: bool) -> np.ndarray:
    """
    Estimate the covariance matrix of Z.

    Parameters
    ----------
    Sigma : np.ndarray
        A p by p covariance matrix.
    AI : np.ndarray
        A p by K loading matrix.
    diagonal : bool
        If True, force C to be diagonal.

    Returns
    -------
    np.ndarray
        A K by K covariance matrix.
    """
    K = AI.shape[1]
    C = np.zeros((K, K))

    for i in range(K):
        groupi = np.where(AI[:, i] != 0)[0]
        sigmai = np.abs(Sigma[np.ix_(groupi, groupi)])
        tmpEntry = np.sum(sigmai) - np.trace(sigmai)
        n_group = len(groupi)
        C[i, i] = tmpEntry / (n_group * (n_group - 1)) if n_group > 1 else 0

        if not diagonal and i < K - 1:
            for j in range(i + 1, K):
                groupj = np.where(AI[:, j] != 0)[0]
                # Adjust the sign for each row
                sigmaij = AI[groupi, i][:, np.newaxis] * Sigma[np.ix_(groupi, groupj)]
                sigmaij = (AI[groupj, j][np.newaxis, :] * sigmaij)
                C[i, j] = np.sum(sigmaij) / (len(groupi) * len(groupj))
                C[j, i] = C[i, j]

    return C


def FindRowMax(Sigma: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate the maximal absolute value for each row of the given matrix.

    Parameters
    ----------
    Sigma : np.ndarray
        A p by p matrix (typically off-diagonal covariance).

    Returns
    -------
    Dict
        A dict containing:
        - 'M': Vector of maximum values per row
        - 'arg_M': Vector of indices of maximum values per row
    """
    p = Sigma.shape[0]
    M = np.zeros(p)
    arg_M = np.zeros(p, dtype=int)

    for i in range(p):
        row_i = Sigma[i, :]
        arg_M[i] = np.argmax(row_i)
        M[i] = row_i[arg_M[i]]

    return {'arg_M': arg_M, 'M': M}


def FindPureNode(off_Sigma: np.ndarray, delta: float, Ms: np.ndarray,
                 arg_Ms: np.ndarray, se_est: np.ndarray,
                 merge: bool) -> Dict[str, Union[List, np.ndarray]]:
    """
    Estimate the pure variables for a given delta.

    Parameters
    ----------
    off_Sigma : np.ndarray
        A p by p matrix (off-diagonal absolute covariance).
    delta : float
        Threshold parameter.
    Ms : np.ndarray
        Vector of row maxima.
    arg_Ms : np.ndarray
        Vector of indices of row maxima.
    se_est : np.ndarray
        Vector of standard error estimates.
    merge : bool
        If True, use intersection merging; else use union.

    Returns
    -------
    Dict
        A dict containing:
        - 'pureInd': List of lists of pure variable indices
        - 'pureVec': Flat array of all pure variable indices
    """
    G = []
    p = off_Sigma.shape[0]

    for i in range(p):
        row_i = off_Sigma[i, :]
        Si = FindRowMaxInd(i, Ms[i], arg_Ms[i], row_i, delta, se_est)

        if len(Si) != 0:
            pureFlag = TestPure(row_i, i, Si, Ms, arg_Ms, delta, se_est)
            if pureFlag:
                new_group = list(Si) + [i]
                if merge:
                    G = Merge(G, new_group)
                else:
                    G = Merge_union(G, new_group)

    # Flatten to get pureVec
    pureVec = []
    for g in G:
        pureVec.extend(g)
    pureVec = np.array(sorted(set(pureVec)))

    return {'pureInd': G, 'pureVec': pureVec}


def FindRowMaxInd(i: int, M: float, arg_M: int, vector: np.ndarray,
                  delta: float, se_est: np.ndarray) -> np.ndarray:
    """
    Find indices of the ith row such that the absolute values are within
    2*delta difference from the given value M.

    Parameters
    ----------
    i : int
        Row index.
    M : float
        Maximum value of the row.
    arg_M : int
        Index of the maximum value.
    vector : np.ndarray
        The row vector.
    delta : float
        Threshold parameter.
    se_est : np.ndarray
        Vector of standard error estimates.

    Returns
    -------
    np.ndarray
        Array of indices.
    """
    lbd = delta * se_est[i] * se_est[arg_M] + delta * se_est[i] * se_est
    indices = np.where(M <= lbd + vector)[0]
    return indices


def TestPure(Sigma_row: np.ndarray, rowInd: int, Si: np.ndarray,
             Ms: np.ndarray, arg_Ms: np.ndarray, delta: float,
             se_est: np.ndarray) -> bool:
    """
    Check if a given row corresponds to a pure variable.

    Parameters
    ----------
    Sigma_row : np.ndarray
        A row of Sigma.
    rowInd : int
        Row index.
    Si : np.ndarray
        Candidate indices.
    Ms : np.ndarray
        Row maxima vector.
    arg_Ms : np.ndarray
        Indices of row maxima.
    delta : float
        Threshold parameter.
    se_est : np.ndarray
        Standard error estimates.

    Returns
    -------
    bool
        True if the row is pure, False otherwise.
    """
    for j in Si:
        delta_j = (se_est[rowInd] + se_est[arg_Ms[j]]) * se_est[j] * delta
        if np.abs(Sigma_row[j] - Ms[j]) > delta_j:
            return False
    return True


def FindSignPureNode(pureList: List[List[int]],
                     Sigma: np.ndarray) -> List[Dict[str, Union[List[int], np.ndarray]]]:
    """
    Estimate the sign sub-partition of the pure variables.

    If one group has no pure variables with negative sign, then an empty
    list is inserted in that position.

    Parameters
    ----------
    pureList : List[List[int]]
        A list of indices of pure variables.
    Sigma : np.ndarray
        The covariance matrix.

    Returns
    -------
    List[Dict]
        A list of dicts with 'pos' and 'neg' keys containing indices.
    """
    signPureList = []

    for purei in pureList:
        purei = sorted(purei)  # Sort to match R's findSignPureNode
        if len(purei) != 1:
            firstPure = purei[0]
            pos = [firstPure]
            neg = []

            for j in range(1, len(purei)):
                purej = purei[j]
                if Sigma[firstPure, purej] < 0:
                    neg.append(purej)
                else:
                    pos.append(purej)

            signPureList.append({'pos': pos, 'neg': neg})
        else:
            signPureList.append({'pos': purei, 'neg': []})

    return signPureList


def Merge(groupList: List[List[int]], groupVec: List[int]) -> List[List[int]]:
    """
    Merge pure variables via intersection.

    Parameters
    ----------
    groupList : List[List[int]]
        Existing list of pure variable groups.
    groupVec : List[int]
        New group of indices to merge.

    Returns
    -------
    List[List[int]]
        Updated list of groups.
    """
    if len(groupList) != 0:
        for i, group in enumerate(groupList):
            common_nodes = set(group).intersection(set(groupVec))
            if len(common_nodes) != 0:
                groupList[i] = list(common_nodes)
                return groupList

    groupList.append(list(groupVec))
    return groupList


def Merge_union(groupList: List[List[int]], groupVec: List[int]) -> List[List[int]]:
    """
    Merge pure variables via union.

    Parameters
    ----------
    groupList : List[List[int]]
        Existing list of pure variable groups.
    groupVec : List[int]
        New group of indices to merge.

    Returns
    -------
    List[List[int]]
        Updated list of groups.
    """
    if len(groupList) != 0:
        common_groups = []
        for i, group in enumerate(groupList):
            if len(set(group).intersection(set(groupVec))) > 0:
                common_groups.append(i)

        if len(common_groups) > 0:
            # Collect all nodes from common groups
            new_group = set(groupVec)
            for idx in common_groups:
                new_group.update(groupList[idx])

            # Keep remaining groups
            remain_group = [groupList[i] for i in range(len(groupList))
                            if i not in common_groups]
            remain_group.append(list(new_group))
            return remain_group

    groupList.append(list(groupVec))
    return groupList


def RecoverAI(estGroupList: List[Dict[str, Union[List[int], np.ndarray]]],
              p: int) -> np.ndarray:
    """
    Return the estimated submatrix A_I from the partition of pure variables.

    Parameters
    ----------
    estGroupList : List[Dict]
        A list of dicts with 'pos' and 'neg' keys.
    p : int
        Number of variables (rows of A).

    Returns
    -------
    np.ndarray
        A p by K matrix.
    """
    K = len(estGroupList)
    A = np.zeros((p, K))

    for i, groupi in enumerate(estGroupList):
        pos_indices = groupi['pos']
        neg_indices = groupi['neg']

        if len(pos_indices) > 0:
            A[pos_indices, i] = 1

        if len(neg_indices) > 0:
            A[neg_indices, i] = -1

    return A
