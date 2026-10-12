#!/usr/bin/env python
"""
convert.py - Convert SeDuMi format to SDPA sparse format.

This is a modified copy of pydsdp.convert with bug fixes:
- Added str() conversion for float values in SDPA file output (lines 73, 108)
- Use COO format for consistent row/col/data ordering in sparse matrices
- FIX (2026-01-29): Corrected row/col indexing for column-major (order='F') flattening
  Original code used row = k // blockSize, col = k % blockSize (row-major interpretation)
  Fixed to use row = k % blockSize, col = k // blockSize (column-major interpretation)
"""

__all__ = ['sedumi2sdpa']

from scipy.sparse import csc_matrix, csr_matrix
from scipy import sparse
from numpy import matrix


# Write the same file as Rdsdp's sedumi2sdpa: b and A unnegated (pydsdp
# negated both, an equivalent problem that DSDP solves along a different path)
# and R's number formatting, 15 significant digits from write.table and 7 from
# cat() for b. Near-singular knockoff SDPs amplify either difference into s.
def _r_num(x):
    return f"{x:.15g}"


def _r_cat(x):
    return f"{x:.7g}"

def sedumi2sdpa(filename, A, b, c, K):
    """Convert from SeDuMi format to SDPA sparse format.

    Arguments
    ---------
    filename : str
        Output SDPA file name.
    A, b, c : scipy matrices
        Problem data in SeDuMi format.
    K : dict
        Cone specification with keys 'l' (linear) and 's' (SDP sizes).
    """
    A = sparse.csc_matrix(A)
    b = sparse.csc_matrix(b)
    c = sparse.csc_matrix(c)
    if c.get_shape()[1] > 1:
        c = c.transpose()

    if not sparse.isspmatrix_csc(A):
        A = A.tocsc()
    if not sparse.isspmatrix_csc(b):
        b = b.tocsc()
    if not sparse.isspmatrix_csc(c):
        c = c.tocsc()

    if 'l' not in K.keys():
        K['l'] = 0

    fp = open(filename, "w")

    # write mDim
    mDim = A.get_shape()[0]
    fp.write(str(mDim) + "\n")

    # write nBlock
    if K['l'] > 0:
        fp.write(str(1 + len(K['s'])) + "\n")
    else:
        fp.write(str(len(K['s'])) + "\n")

    # write blockStruct
    if K['l'] > 0:
        fp.write(str(-K['l']) + " ")
    fp.write(" ".join([str(x) for x in K['s']]) + "\n")

    # write b
    fp.write(" ".join([_r_cat(b[i, 0]) for i in range(b.shape[0])]) + "\n")

    # write C
    len_s = sum([x ** 2 for x in K['s']])

    matnum = 0
    blocknum = 0
    curind = 0

    # block one for linear cone
    if K['l'] > 0:
        blocknum += 1
        c_l = -c[0:K['l'], :]
        list_row = [str(x + 1) for x in list(c_l.nonzero()[0])]
        list_val = [x for x in list(c_l.data)]
        length = len(list_row)
        for i in range(length):
            # BUG FIX: Added str() around list_val[i]
            fp.write(" ".join((str(matnum), str(blocknum), list_row[i],
                               list_row[i], _r_num(list_val[i]))) + "\n")
        curind = curind + K['l']

    if len_s > 0:
        offset = 0
        c_s = -c[curind:(curind + len_s), :]
        for blockSize in K['s']:
            blocknum += 1
            list_row = list(c_s[offset:(offset + blockSize * blockSize),
                                :].nonzero()[0])
            list_val = [x for x in
                        list(c_s[offset:(offset + blockSize * blockSize),
                                 :].data)]
            length = len(list_row)
            for i in range(length):
                # FIX: For column-major (order='F') flattening:
                # row = index % blockSize, col = index // blockSize
                setCol_row = (list_row[i] % blockSize) + 1
                setCol_col = (list_row[i] // blockSize) + 1
                if setCol_row <= setCol_col:
                    fp.write(" ".join(("0", str(blocknum), str(setCol_row),
                                       str(setCol_col), _r_num(list_val[i]))) + "\n")
            offset += blockSize * blockSize

    # write A
    blocknum = 0
    curind = 0

    if K['l'] > 0:
        blocknum += 1
        A_l = A[:, 0:K['l']]
        # BUG FIX: Convert to COO format for consistent row/col/data ordering
        A_l_coo = A_l.tocoo()
        list_row = [str(x + 1) for x in A_l_coo.row]
        list_col = [str(x + 1) for x in A_l_coo.col]
        list_val = [x for x in A_l_coo.data]
        length = len(list_row)
        for i in range(length):
            # BUG FIX: Added str() around list_val[i]
            fp.write(" ".join((list_row[i], str(blocknum), list_col[i],
                               list_col[i], _r_num(list_val[i]))) + "\n")
        curind = curind + K['l']

    if len_s > 0:
        offset = 0
        A_s = A[:, curind:(curind + len_s)]
        for blockSize in K['s']:
            blocknum += 1
            # BUG FIX: Convert to COO format for consistent row/col/data ordering
            A_s_block = A_s[:, offset:(offset + blockSize * blockSize)]
            A_s_coo = A_s_block.tocoo()
            list_row = [str(x + 1) for x in A_s_coo.row]
            list_col = list(A_s_coo.col)
            list_val = [x for x in A_s_coo.data]
            length = len(list_row)
            for i in range(length):
                # FIX: For column-major (order='F') flattening:
                # row = index % blockSize, col = index // blockSize
                setCol_row = (list_col[i] % blockSize) + 1
                setCol_col = (list_col[i] // blockSize) + 1
                if setCol_row <= setCol_col:
                    fp.write(" ".join((list_row[i], str(blocknum),
                                       str(setCol_row), str(setCol_col),
                                       _r_num(list_val[i]))) + "\n")
            offset += blockSize * blockSize

    fp.close()
    return
