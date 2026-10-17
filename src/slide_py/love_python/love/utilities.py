"""
Utility functions for LOVE package.
Translated from R/Utilities.R
"""

import numpy as np
from typing import List, Dict, Optional, Union


def recoverGroup(A: np.ndarray) -> List[Dict[str, np.ndarray]]:
    """
    Recover clusters based on a given p by K loading matrix.

    Parameters
    ----------
    A : np.ndarray
        A p by K matrix.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        A list of group indices with sign sub-partition.
        Each element is a dict with 'pos' and 'neg' keys containing indices.
    """
    Group = []
    for i in range(A.shape[1]):
        column = A[:, i]
        posInd = np.where(column > 0)[0]
        negInd = np.where(column < 0)[0]
        Group.append({'pos': posInd, 'neg': negInd})
    return Group


def singleton(estPureIndices: List) -> bool:
    """
    Check if there exists any element in the given list that has length equal to 1.

    Parameters
    ----------
    estPureIndices : List
        A list of indices of the estimated pure variables.

    Returns
    -------
    bool
        True if exists at least one element with length 1 or list is empty,
        otherwise False.
    """
    if len(estPureIndices) == 0:
        return True
    for x in estPureIndices:
        if len(x) == 1:
            return True
    return False


def threshA(A: np.ndarray, mu: float, scale: bool = False) -> np.ndarray:
    """
    Hard-threshold the estimated A based on the given mu.

    If scale is True, then normalize each row of A such that the
    L1 norm of each row is no larger than 1.

    Parameters
    ----------
    A : np.ndarray
        Input matrix.
    mu : float
        Threshold value.
    scale : bool, optional
        Normalize the row-wise L1 norm if True. Default is False.

    Returns
    -------
    np.ndarray
        A matrix with the same dimension as A.
    """
    scaledA = A.copy()
    for i in range(A.shape[0]):
        colInd = np.abs(A[i, :]) <= mu
        scaledA[i, colInd] = 0
        if scale and np.sum(np.abs(scaledA[i, :])) > 1:
            scaledA[i, :] = scaledA[i, :] / np.sum(np.abs(scaledA[i, :]))
    return scaledA


def offSum(M: np.ndarray, weights: Union[np.ndarray, float]) -> float:
    """
    Calculate the weighted sum of squares of the upper off-diagonal elements.

    Parameters
    ----------
    M : np.ndarray
        A given symmetric matrix.
    weights : np.ndarray or float
        A vector of length equal to the number of rows of M, or a scalar.

    Returns
    -------
    float
        The weighted sum of squares.
    """
    if np.isscalar(weights):
        tmp = M / weights
        tmp = tmp / weights
    else:
        weights = np.asarray(weights)
        tmp = M / weights[:, np.newaxis]
        tmp = tmp / weights[np.newaxis, :]

    # Get upper triangular elements (excluding diagonal)
    # R: row(tmp) <= (col(tmp) - 1) means strictly upper triangular
    upper_indices = np.triu_indices(tmp.shape[0], k=1)
    return np.sum(tmp[upper_indices] ** 2)


def partition(totalNumb: int, numbGroup: int) -> List[int]:
    """
    Return a vector of numbers for each group given the total number of
    observations and the total number of groups.

    Example: divide 9 obs into 4 groups gives [3, 2, 2, 2]

    Parameters
    ----------
    totalNumb : int
        Total number of observations.
    numbGroup : int
        Number of groups.

    Returns
    -------
    List[int]
        A list of length equal to numbGroup.
    """
    remainder = totalNumb % numbGroup
    numbPerGroup = totalNumb // numbGroup
    result = [numbPerGroup] * numbGroup
    for i in range(remainder):
        result[i] += 1
    return result


def extract(preVec: np.ndarray, indices: List[int]) -> List[np.ndarray]:
    """
    Extract the indices from the previous vector and return as a list.

    Parameters
    ----------
    preVec : np.ndarray
        A vector to extract from.
    indices : List[int]
        Contains the length of each group to be extracted.

    Returns
    -------
    List[np.ndarray]
        List of length equal to length of indices.
    """
    newVec = []
    start = 0
    for length in indices:
        newVec.append(preVec[start:start + length])
        start += length
    return newVec
