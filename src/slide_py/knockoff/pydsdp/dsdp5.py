#!/usr/bin/env python
"""
DSDP5 Python wrapper with bug fixes.

This is a modified copy of pydsdp.dsdp5 that uses the local fixed convert.py
while still using the C extension from the system pydsdp package.
"""

__all__ = ['dsdp', 'dsdp_readsdpa']

from .convert import sedumi2sdpa
from numpy import array, matrix, reshape
from os import remove, path
from tempfile import NamedTemporaryFile

# Try bundled dsdp_solver first, then fall back to system package
try:
    from ...dsdp_solver.pydsdp5 import pyreadsdpa  # Bundled C extension
except ImportError:
    try:
        from pydsdp.pydsdp5 import pyreadsdpa  # System package fallback
    except ImportError:
        raise ImportError(
            "DSDP C extension not available. Either:\n"
            "  - Rebuild the bundled extension from the repository root: pip install -e . "
            "(or python setup.py build_ext --inplace)\n"
            "  - Install the system package: pip install pydsdp"
        )


def dsdp(A, b, c, K, OPTIONS=None):
    """
    Solve a semidefinite program using DSDP.

    Parameters
    ----------
    A : matrix
        Constraint matrix in SeDuMi format.
    b : matrix
        Objective coefficients.
    c : matrix
        Cone constraint data.
    K : dict
        Cone specification with keys 'l' (linear cone size) and 's' (list of SDP cone sizes).
    OPTIONS : dict, optional
        Solver options including 'gaptol', 'maxit', 'print'.

    Returns
    -------
    dict
        Solution with keys 'y' (primal solution), 'X' (dual matrices), 'STATS' (status info).
    """
    if OPTIONS is None:
        OPTIONS = {}

    tempdataF = NamedTemporaryFile(delete=False)
    data_filename = tempdataF.name
    tempdataF.close()
    sedumi2sdpa(data_filename, A, b, c, K)

    options_filename = ""
    if len(OPTIONS) > 0:
        tempoptionsF = NamedTemporaryFile(delete=False)
        options_filename = tempoptionsF.name
        tempoptionsF.close()
        write_options_file(options_filename, OPTIONS)

    # Solve the problem with DSDP5
    [y, X, STATS] = dsdp_readsdpa(data_filename, options_filename)
    # The extension returns -y for the unnegated (Rdsdp) file convention.
    y = -array(y)

    if path.isfile(data_filename):
        remove(data_filename)
    if path.isfile(options_filename):
        remove(options_filename)

    if 'l' not in K:
        K['l'] = 0
    if 's' not in K:
        K['s'] = ()

    Xout = []
    if K['l'] > 0:
        Xout.append(X[0:K['l']])

    index = K['l']
    if 's' in K:
        for d in K['s']:
            Xout.append(matrix(reshape(array(X[index:index + d * d]), [d, d])))
            index = index + d * d

    if STATS[0] == 1:
        STATS[0] = "PDFeasible"
    elif STATS[0] == 3:
        STATS[0] = "Unbounded"
    elif STATS[0] == 4:
        STATS[0] = "InFeasible"

    STATSout = {}
    if len(STATS) > 3:
        STATSout = dict(zip(
            ["stype", "dobj", "pobj", "r", "mu", "pstep", "dstep", "pnorm"],
            STATS
        ))
    else:
        STATSout = dict(zip(["stype", "dobj", "pobj"], STATS))

    return dict(zip(['y', 'X', 'STATS'], [y, Xout, STATSout]))


def dsdp_readsdpa(data_filename, options_filename):
    """Read SDPA file and solve with DSDP."""
    result = []
    if path.isfile(data_filename) and (options_filename == "" or path.isfile(options_filename)):
        result = pyreadsdpa(data_filename, options_filename)
    return result


def write_options_file(filename, OPTIONS):
    """Write DSDP options to file."""
    with open(filename, "a") as f:
        for option in OPTIONS.keys():
            f.write("-" + option + " " + str(OPTIONS[option]) + "\n")
